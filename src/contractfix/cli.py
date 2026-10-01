"""Small CLI; live calls and repository execution require explicit commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from contractfix.config import LLMSettings
from contractfix.utils.artifacts import RunArtifacts, atomic_json, redact
from contractfix.utils.logger import configure_logging, logger


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="contractfix")
    root.add_argument(
        "--output-format",
        choices=["human", "json", "legacy-json"],
        default="human",
        help="Human Rich summary, clean JSON, or the historical mixed JSON mode",
    )
    root.add_argument(
        "--env",
        type=Path,
        help="Explicit dotenv file loaded before settings; overrides the environment",
    )
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Offline dependency inventory")

    evaluation = commands.add_parser("eval", help="Host-controlled evaluation; never an agent tool")
    evaluations = evaluation.add_subparsers(dest="evaluation", required=True)
    command = evaluations.add_parser("smoke")
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--examples", type=Path)
    command = evaluations.add_parser(
        "preflight",
        help="Zero-inference provider capability check for every workflow stage",
    )
    command.add_argument(
        "--workflow",
        type=Path,
        default=Path("configs/workflow.json"),
    )
    command.add_argument("--out", type=Path, required=True)
    command = evaluations.add_parser("contracts")
    command.add_argument("--cases", type=Path, default=Path("evals/contract-microcases/contracts.jsonl"))
    command.add_argument("--out", type=Path, required=True)
    command = evaluations.add_parser("manifest")
    command.add_argument("--dataset", choices=["lite-dev", "lite-test", "verified"], required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--revision")
    command = evaluations.add_parser("overlap")
    command.add_argument("--dev", type=Path, required=True)
    command.add_argument("--verified", type=Path, required=True)
    command = evaluations.add_parser("report")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    group = command.add_mutually_exclusive_group(required=True)
    group.add_argument("--outcomes", type=Path)
    group.add_argument("--swebench-report", type=Path)
    command = evaluations.add_parser("baseline-audit")
    command.add_argument("--workspace", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--manifest", type=Path)

    command = evaluations.add_parser("prompts", help="Offline prompt/skill/example audit and preview")
    command.add_argument("--prompts", type=Path)
    command.add_argument("--out", type=Path, required=True)
    command = evaluations.add_parser("dev-lessons", help="Review candidates only; does not update memory")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--campaign", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)

    command = commands.add_parser("generate", help="One structured NLC or EC proposal")
    command.add_argument("--kind", choices=["nlc", "ec"], required=True)
    command.add_argument("--prompt", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)

    ec = commands.add_parser("ec")
    ec_commands = ec.add_subparsers(dest="ec_action", required=True)
    pair = ec_commands.add_parser("pair", help="Local trusted-code base/candidate observation gate")
    pair.add_argument("--clauses", type=Path, required=True)
    pair.add_argument("--patch", type=Path, required=True)
    pair.add_argument(
        "--allow-edit",
        action="append",
        help="Host-approved relative Python source file; repeatable",
    )
    explore = ec_commands.add_parser(
        "explore", help="Live Deep Agents clause explorer with fixed host inputs"
    )
    explore.add_argument("--host-clause", type=Path, required=True)
    explore.add_argument("--prompt", type=Path, required=True)
    explore.add_argument("--resume", action="store_true")
    explore.add_argument("--max-evaluations", type=int, default=3)
    for item in (pair, explore):
        item.add_argument("--repo", type=Path, required=True)
        item.add_argument("--out", type=Path, required=True)
        item.add_argument("--timeout", type=float, default=120)
        item.add_argument(
            "--allow-local-execution",
            action="store_true",
            help="Acknowledge that this runner is not a security sandbox",
        )
        item.add_argument("argv", nargs=argparse.REMAINDER, help="-- followed by target command")
    from contractfix.workflow.cli import add_commands

    add_commands(commands, evaluations)
    return root


def _read(path: str | Path) -> object:
    return json.loads(Path(path).read_text())


def _print(value: object) -> None:
    print(json.dumps(redact(value), indent=2, ensure_ascii=False))


def _present(args: argparse.Namespace, report: dict) -> None:
    if args.output_format in {"json", "legacy-json"}:
        _print(report)
    else:
        logger.final_summary(report, artifacts=getattr(args, "out", None))


def _present_workflow_preflight(args: argparse.Namespace, report: dict, artifacts: RunArtifacts) -> None:
    """Keep the terminal concise; preserve the complete endpoint matrix on disk."""
    rendered = {**report, "artifacts": str(artifacts.root)}
    if args.output_format in {"json", "legacy-json"}:
        _print(rendered)
        return
    if not report.get("checked"):
        logger.warning(f"[provider preflight] not checked: {report.get('reason', 'unknown reason')}")
        return
    rows = []
    for stage in report.get("stages", []):
        policy = stage.get("reasoning_policy", {})
        capabilities = stage.get("capabilities", {})
        configuration = stage.get("model_configuration", {})
        rows.append(
            [
                stage.get("policy_stage", "unknown"),
                policy.get("intent", "unknown"),
                policy.get("resolved", "unknown"),
                configuration.get("max_tokens", "unknown"),
                capabilities.get("eligible_endpoint_count", "unknown"),
                capabilities.get("rejected_endpoint_count", "unknown"),
            ]
        )
    logger.table(
        "ContractFix · OpenRouter stage capability preflight",
        ["Stage", "Intent", "Resolved", "Completion", "Eligible", "Rejected"],
        rows,
    )
    logger.panel(
        "Preflight saved",
        "**Zero inference:** yes  \n"
        f"**Model:** `{report.get('model', 'unknown')}`  \n"
        f"**Full endpoint matrix:** `{artifacts.root / 'report.json'}`",
        style="green",
    )


def _dispatch_evaluation(args: argparse.Namespace) -> int:
    if args.evaluation == "prompts":
        from contractfix.workflow.diagnostics import audit_prompts

        _print(audit_prompts(args.out, args.prompts))
        return 0
    if args.evaluation == "dev-lessons":
        from contractfix.workflow.diagnostics import development_failures

        _print(development_failures(args.manifest, args.campaign, args.out))
        return 0
    if args.evaluation == "gold":
        from contractfix.workflow.cli import dispatch as workflow_dispatch

        report = workflow_dispatch(args)
        _present(args, report)
        return 0 if report.get("status") == "EVALUATED" else 2
    from contractfix.evaluation.manifests import download_manifest, load_manifest, overlap

    if args.evaluation == "smoke":
        from contractfix.evaluation.smoke import smoke

        report = smoke(args.out, args.examples)
        _print(report)
        return 0 if report["passed"] else 1
    if args.evaluation == "preflight":
        from contractfix.evaluation.preflight import workflow_capability_preflight
        from contractfix.workflow.cli import model_settings, read
        from contractfix.workflow.languages.python.models import WorkflowSettings

        workflow = WorkflowSettings.model_validate(read(args.workflow))
        report = workflow_capability_preflight(model_settings(workflow), workflow)
        artifacts = RunArtifacts(
            args.out,
            {
                "purpose": "workflow_capability_preflight",
                "workflow": workflow.model_dump(),
                "settings": LLMSettings.from_env().public_dict(),
            },
        )
        artifacts.save("report.json", report)
        artifacts.events.emit("preflight_finished", passed=report.get("checked", False))
        logger.attach(artifacts.root)
        _present_workflow_preflight(args, report, artifacts)
        return 0 if report.get("checked", False) else 2
    if args.evaluation == "contracts":
        from contractfix.evaluation.contracts import evaluate_contracts

        report = evaluate_contracts(LLMSettings.from_env(), args.cases, args.out)
        _print(report)
        return 0
    if args.evaluation == "manifest":
        _print(download_manifest(args.dataset, args.out, args.revision))
        return 0
    if args.evaluation == "overlap":
        _print(overlap(load_manifest(args.dev), load_manifest(args.verified)))
        return 0
    if args.evaluation == "report":
        from contractfix.evaluation.reporting import (
            outcomes_from_swebench_report,
            summarize_outcomes,
        )

        manifest = load_manifest(args.manifest)
        rows = (
            [json.loads(line) for line in args.outcomes.read_text().splitlines() if line.strip()]
            if args.outcomes
            else outcomes_from_swebench_report(manifest, _read(args.swebench_report))
        )
        report = summarize_outcomes(manifest, rows)
        atomic_json(args.out, report)
        _print(report)
        return 0
    if args.evaluation == "baseline-audit":
        from contractfix.evaluation.baselines import audit_workspace

        manifest = load_manifest(args.manifest) if args.manifest else None
        report = audit_workspace(args.workspace, manifest)
        atomic_json(args.out, report)
        _print(report)
        return 0 if report["structural_checks_passed"] else 1
    raise ValueError("unknown evaluation command")


def dispatch(args: argparse.Namespace) -> int:
    if args.env:
        from dotenv import load_dotenv

        if not args.env.is_file():
            raise ValueError("dotenv file not found")
        load_dotenv(args.env, override=True)
    if args.command in {"run", "repair"}:
        from contractfix.workflow.cli import dispatch as workflow_dispatch

        report = workflow_dispatch(args)
        _present(args, report)
        return 2 if report.get("status") == "ERROR" else 0
    if args.command == "doctor":
        from contractfix.evaluation.preflight import doctor

        report = doctor()
        _print(report)
        return 0 if report["sdk_available"] else 2
    if args.command == "eval":
        return _dispatch_evaluation(args)
    if args.command == "generate":
        from contractfix.agents.schema.contracts import ExecutableClause, NaturalClause
        from contractfix.llm.structured import generate_structured
        from contractfix.utils.artifacts import digest

        settings = LLMSettings.from_env()
        prompt = args.prompt.read_text()
        artifacts = RunArtifacts(
            args.out,
            {
                "purpose": f"structured_{args.kind}",
                "settings": settings.public_dict(),
                "prompt_sha256": digest(prompt),
            },
        )
        artifacts.save("prompt.json", {"text": prompt})
        try:
            schema = NaturalClause if args.kind == "nlc" else ExecutableClause
            parsed, usage = generate_structured(prompt, schema, settings, events=artifacts.events)
            report = {
                "proposal": parsed.model_dump(),
                "metrics": usage,
                "assurance": "schema_valid_not_semantically_qualified",
            }
            artifacts.save("report.json", report)
            _print(report)
            return 0
        except Exception as exc:
            artifacts.save(
                "error.json",
                {"error_type": type(exc).__name__, "detail": str(exc)[:800]},
            )
            raise
    if args.command == "ec":
        return _dispatch_ec(args)
    raise ValueError("unknown command")


def _dispatch_ec(args: argparse.Namespace) -> int:
    if not args.allow_local_execution:
        raise ValueError(
            "Local runner is not a sandbox. Use an isolated container; pass "
            "--allow-local-execution only for trusted code."
        )
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not argv:
        raise ValueError("provide target command after --")
    if args.out.resolve().is_relative_to(args.repo.resolve()):
        raise ValueError("output directory must be outside target repository")
    if args.timeout <= 0:
        raise ValueError("positive execution timeout required")
    if args.ec_action == "pair":
        from contractfix.contracts.core import admit_pair, seal
        from contractfix.contracts.runner import run

        bundle = seal(_read(args.clauses))
        artifacts = RunArtifacts(
            args.out,
            {
                "purpose": "observed_pair",
                "repo": str(args.repo.resolve()),
                "command": argv,
                "allowed_edit_paths": args.allow_edit,
            },
        )
        artifacts.save("bundle.json", bundle)
        base = run(
            args.repo,
            bundle,
            argv,
            timeout=args.timeout,
            allowed_edit_paths=args.allow_edit,
        )
        candidate = run(
            args.repo,
            bundle,
            argv,
            patch=args.patch,
            timeout=args.timeout,
            allowed_edit_paths=args.allow_edit,
        )
        report = admit_pair(bundle, base, candidate)
        artifacts.save("base.json", base)
        artifacts.save("candidate.json", candidate)
        artifacts.save("report.json", report)
        _print(report)
        return 0 if report["decision"] == "ADMIT_OBSERVED" else 1

    from contractfix.agents.agent import curated_texts, explore
    from contractfix.agents.session import ContractEvaluationSession
    from contractfix.utils.artifacts import EventLog, digest

    settings = LLMSettings.from_env()
    configuration = {
        "settings": settings.public_dict(),
        "curated_sha256": digest(curated_texts()),
        "prompt_sha256": digest(args.prompt.read_text()),
    }
    config_path = args.out / "agent_config.json"
    if args.resume:
        if not config_path.exists() or _read(config_path) != configuration:
            raise ValueError("resume requires unchanged model and skill configuration")
    else:
        if args.out.exists():
            raise ValueError("choose a fresh output directory or --resume")
        args.out.mkdir(parents=True)
        atomic_json(config_path, configuration)
    events = EventLog(args.out / "events.jsonl", run_id=args.out.name)
    session = ContractEvaluationSession(
        args.repo,
        argv,
        _read(args.host_clause),
        args.out,
        max_evaluations=args.max_evaluations,
        timeout=args.timeout,
        events=events,
    )
    try:
        report = explore(session, settings, args.prompt.read_text(), resume=args.resume)
        atomic_json(args.out / "report.json", redact(report))
        _print(report)
        return 0 if report["selected"] is not None else 1
    except Exception as exc:
        events.emit("agent_error", error_type=type(exc).__name__, detail=str(exc)[:800])
        raise


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    try:
        return dispatch(parser().parse_args(argv))
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(redact(f"{type(exc).__name__}: {exc}"), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
