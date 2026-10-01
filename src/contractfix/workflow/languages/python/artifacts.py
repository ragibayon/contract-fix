"""Validation rules for immutable Python qualification artifacts."""

from __future__ import annotations

import hashlib

from ...errors import WorkflowError


def validate_frozen(frozen: dict) -> None:
    """Validate language-specific payloads after the shared envelope check."""
    version = frozen["version"]
    if version != "contractfix-python/3":
        raise WorkflowError(f"unsupported Python frozen artifact version: {version}")
    for candidate in frozen.get("candidates", []):
        expected = candidate["execution"]["executable_contract_sha256"]
        if hashlib.sha256(candidate["source"].encode()).hexdigest() != expected:
            raise WorkflowError("frozen executable contract source digest mismatch")
