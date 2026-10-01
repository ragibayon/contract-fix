"""CLI wiring for one repository/task workflow only."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from contractfix.config import LLMSettings
from contractfix.contracts.execution import ExecutionSettings, Executor
from contractfix.utils.artifacts import atomic_json

from .controller import discover, load_frozen, model_accounting, repair
from .languages import backend_for_language
from .languages.python.models import WorkflowSettings
from .task import Task
from .languages.python.stages import LangChainStages


def add_commands(commands, evaluations) -> None:
    command = commands.add_parser(
        "run", help="Qualify contracts for one task; patch generation remains optional"
    )
    command.add_argument("--task", type=Path)
    command.add_argument("--repo", type=Path)
    command.add_argument("--prepared", type=Path, help="Existing tasks_map.json/setup_map.json directory")
    command.add_argument("--instance")
    command.add_argument(
        "--language",
        help="Repository language backend (defaults to task metadata, historically python)",
    )
    command.add_argument("--phase", choices=["contracts", "repair"], default="contracts")
    command.add_argument("--localization", type=Path)
    command.add_argument("--workflow", type=Path)
    _common(command)

    command = commands.add_parser(
        "repair", help="Generate/select patches from one sealed guidance packet without requalification"
    )
    command.add_argument("--guidance", "--frozen", dest="frozen", type=Path, required=True,
                         help="Completed generation directory containing sealed guidance.json")
    command.add_argument("--resume", action="store_true", help="Resume patch work without resetting its budgets")
    _common(command)

    command = evaluations.add_parser(
        "gold", help="Evaluate one frozen contract against an evaluator-only gold patch"
    )
    command.add_argument("--frozen", type=Path, required=True)
    gold = command.add_mutually_exclusive_group(required=True)
    gold.add_argument("--gold-patch", type=Path)
    gold.add_argument("--gold-record", type=Path)
    command.add_argument("--execution", type=Path)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--allow-local-execution", action="store_true")


def _common(command) -> None:
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--execution", type=Path)
    command.add_argument("--prompts", type=Path)
    command.add_argument("--mode", choices=["no-contract", "context", "enforce"], default=None)
    command.add_argument("--allow-local-execution", action="store_true")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def executor(args, task: Task | None = None, frozen: dict | None = None) -> Executor:
    values = (
        read(args.execution)
        if args.execution
        else (frozen["executor"]["settings"] if frozen else {"image": task.image if task else None})
    )
    settings = ExecutionSettings(**values)
    if settings.kind == "local" and not args.allow_local_execution:
        raise ValueError(
            "local code execution is not a sandbox; use Docker or explicitly acknowledge trusted local code"
        )
    return Executor(settings)


def task_inputs(args) -> tuple[Task, Path]:
    def with_language(task: Task) -> Task:
        if not args.language:
            return task
        return Task.model_validate({**task.model_dump(), "language": args.language})

    if args.prepared:
        if args.task or args.repo or not args.instance:
            raise ValueError("use --prepared DIR --instance ID without --task/--repo")
        task = Task.model_validate(read(args.prepared / "tasks_map.json")[args.instance])
        task = with_language(task)
        setup = read(args.prepared / "setup_map.json")[args.instance]
        return task, Path(setup["repo_path"])
    if not args.task or not args.repo:
        raise ValueError("provide --task JSON --repo CHECKOUT, or --prepared DIR --instance ID")
    task = Task.model_validate(read(args.task))
    return with_language(task), args.repo


def model_settings(workflow: WorkflowSettings) -> LLMSettings:
    settings = LLMSettings.from_env()
    return settings if settings.thinking else settings.for_output(workflow.stage_output_tokens)


def dispatch(args) -> dict:
    if args.command == "eval" and args.evaluation == "gold":
        from .languages.python.offline import evaluate_gold

        frozen = load_frozen(args.frozen)
        return evaluate_gold(
            args.frozen,
            args.out,
            executor(args, frozen=frozen),
            gold_patch=args.gold_patch,
            gold_record=args.gold_record,
        )
    if args.command == "repair":
        from .languages.python.guidance import load_guidance
        started = perf_counter()
        guidance, _ = load_guidance(args.frozen)
        language_backend = backend_for_language(guidance.language)
        settings = language_backend.settings_model.model_validate(guidance.workflow)
        stages = language_backend.create_stages(settings, args.prompts)
        status = repair(args.frozen, args.out, executor(args, frozen={"executor": guidance.executor}),
                        stages, args.mode or settings.repair_variant, resume=args.resume)
        status["total_elapsed_seconds"] = perf_counter() - started
        atomic_json(args.out / "summary.json", status)
        atomic_json(args.out / "status.json", status)
        return status

    task, repo = task_inputs(args)
    language_backend = backend_for_language(task.language)
    settings = language_backend.settings_model.model_validate(
        read(args.workflow) if args.workflow else {}
    )
    repair_requested = args.phase == "repair" or settings.repair_enabled
    if repair_requested:
        settings = language_backend.settings_model.model_validate({
            **settings.model_dump(), "repair_enabled": True,
            "repair_variant": args.mode or settings.repair_variant,
        })
    stages = language_backend.create_stages(settings, args.prompts)
    execution_backend = executor(args, task=task)
    started = perf_counter()
    status = discover(
        task,
        repo,
        args.out,
        execution_backend,
        stages,
        settings,
        read(args.localization) if args.localization else None,
    )
    if repair_requested:
        qualification_status = dict(status)
        repaired = repair(args.out, args.out / "repair", execution_backend, stages, settings.repair_variant)
        status = {**status, "status": repaired["status"], "instance_id": task.instance_id,
                  "contract_status": qualification_status["status"], "qualification_result": qualification_status,
                  "repair": repaired, "requested_variant": repaired["requested_variant"],
                  "effective_mode": repaired["effective_mode"]}
    status["total_elapsed_seconds"] = perf_counter() - started
    status["model_accounting"] = model_accounting(stages)
    atomic_json(args.out / "summary.json", status)
    atomic_json(args.out / "status.json", status)
    return status


def _stages(settings: WorkflowSettings, prompts: Path | None) -> LangChainStages:
    """Compatibility wrapper for callers constructing Python stages directly."""
    return backend_for_language("python").create_stages(settings, prompts)
