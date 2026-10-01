"""Language-neutral facade for qualification and guided one-repository repair.

The facade deliberately contains no parsing, instrumentation, or language-specific
contract logic. Backends own those details and remain directly testable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from contractfix.contracts.execution import Executor

from .accounting import model_accounting
from .errors import EvidenceGroundingError, LocalizationExhausted, WorkflowError
from .frozen import load_frozen
from .languages import backend_for_frozen_version, backend_for_language
from .task import Task
from .stages import Stages


def discover(
    task: Task,
    repo: Path,
    output: Path,
    executor: Executor,
    stages: Stages,
    settings: Any,
    localization_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Qualify one task with the backend selected by ``task.language``."""
    return backend_for_language(task.language).discover(
        task, repo, output, executor, stages, settings, localization_override
    )


def repair(
    frozen_dir: Path,
    output: Path,
    executor: Executor,
    stages: Stages,
    mode: str = "enforce",
    *, resume: bool = False,
) -> dict[str, Any]:
    """Repair from sealed availability, optionally resuming only patch work."""
    if (frozen_dir / "guidance.json").exists():
        import json
        envelope = json.loads((frozen_dir / "guidance.json").read_text(encoding="utf-8"))
        backend = backend_for_language(envelope["language"])
    elif (frozen_dir / "frozen.json").exists():
        frozen = load_frozen(frozen_dir)
        backend = backend_for_frozen_version(frozen["version"])
    else:
        backend = backend_for_language("python")
    return backend.repair(
        frozen_dir, output, executor, stages, mode, resume=resume
    )


_PYTHON_COMPATIBILITY_EXPORTS = frozenset(
    {
        "_candidate_failure",
        "_candidate_failure_owner",
        "_contract_recovery_feedback",
        "_derive_violation_criterion",
        "_ground_repair",
        "_index_conformance_reviews",
        "_localize",
        "_reviewable_candidate",
    }
)


def __getattr__(name: str):
    """Lazily preserve private imports from the pre-backend controller module."""
    if name not in _PYTHON_COMPATIBILITY_EXPORTS:
        raise AttributeError(name)
    from .languages.python import qualification as python

    return getattr(python, name)


__all__ = [
    "EvidenceGroundingError",
    "LocalizationExhausted",
    "WorkflowError",
    "discover",
    "load_frozen",
    "model_accounting",
    "repair",
]
