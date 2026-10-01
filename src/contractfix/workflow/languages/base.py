"""Language backend contract for single-repository qualification.

Language backends own source parsing, contract synthesis semantics, runtime
instrumentation, and frozen-artifact formats. The workflow facade and experiment
runner only coordinate a backend through this interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from contractfix.contracts.execution import Executor
from pydantic import BaseModel

from ..task import Task
from ..stages import Stages


class UnsupportedLanguageError(ValueError):
    """Raised when no qualification backend is registered for a task language."""


@runtime_checkable
class LanguageBackend(Protocol):
    """Behavior required from a repository-language qualification backend."""

    language: str
    aliases: frozenset[str]
    frozen_versions: frozenset[str]
    settings_model: type[BaseModel]

    def create_stages(self, settings: Any, prompts: Path | None = None) -> Stages: ...

    def validate_frozen(self, frozen: dict[str, Any]) -> None: ...

    def discover(
        self,
        task: Task,
        repo: Path,
        output: Path,
        executor: Executor,
        stages: Stages,
        settings: Any,
        localization_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def repair(
        self,
        frozen_dir: Path,
        output: Path,
        executor: Executor,
        stages: Stages,
        mode: str = "enforce",
        *, resume: bool = False,
    ) -> dict[str, Any]: ...
