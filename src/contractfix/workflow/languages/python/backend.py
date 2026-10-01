"""Registration adapter for Python contract qualification."""

from __future__ import annotations

from pathlib import Path

from contractfix.config import LLMSettings
from contractfix.contracts.execution import Executor

from ...task import Task
from .models import WorkflowSettings
from ...stages import Stages
from .artifacts import validate_frozen
from .qualification import discover
from .repair import repair
from .stages import LangChainStages


class PythonLanguageBackend:
    """Python backend for the single supported qualification protocol."""

    language = "python"
    aliases = frozenset({"py", "python3", "python-3"})
    settings_model = WorkflowSettings
    frozen_versions = frozenset({"contractfix-python/3"})

    def create_stages(
        self, settings: WorkflowSettings, prompts: Path | None = None
    ) -> LangChainStages:
        llm_settings = LLMSettings.from_env()
        if not llm_settings.thinking:
            llm_settings = llm_settings.for_output(settings.stage_output_tokens)
        return LangChainStages(
            llm_settings,
            prompts,
            prompt_profile=settings.prompt_profile,
            example_profile=settings.example_profile,
            stage_reasoning_tokens=settings.stage_reasoning_tokens,
            stage_reasoning_policy=settings.stage_reasoning_policy,
            stage_completion_limits=settings.stage_completion_limits,
            stage_reasoning_caps=settings.stage_reasoning_caps,
            stage_answer_tokens=settings.stage_answer_tokens,
        )

    def validate_frozen(self, frozen: dict) -> None:
        validate_frozen(frozen)

    def discover(
        self,
        task: Task,
        repo: Path,
        output: Path,
        executor: Executor,
        stages: Stages,
        settings: WorkflowSettings,
        localization_override: dict | None = None,
    ) -> dict:
        return discover(
            task, repo, output, executor, stages, settings, localization_override
        )

    def repair(
        self,
        frozen_dir: Path,
        output: Path,
        executor: Executor,
        stages: Stages,
        mode: str = "enforce",
        *, resume: bool = False,
    ) -> dict:
        return repair(frozen_dir, output, executor, stages, mode, resume=resume)
