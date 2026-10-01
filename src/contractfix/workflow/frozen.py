"""Language-neutral frozen-envelope validation."""

import json
from pathlib import Path

from contractfix.contracts.core import digest
from contractfix.contracts.snapshots import repository_identity

from .errors import WorkflowError


def load_frozen(directory: Path) -> dict:
    path = directory / "frozen.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    body = {key: item for key, item in value.items() if key != "sha256"}
    version = value.get("version")
    if not isinstance(version, str) or not version or digest(body) != value.get("sha256"):
        raise WorkflowError("frozen artifact hash/version mismatch")
    if repository_identity(directory / "base") != value["repo_sha256"]:
        raise WorkflowError("base snapshot changed after contract generation")
    # Import lazily to avoid a frozen-loader/backend import cycle. The backend owns
    # payload validation; this module owns only the immutable shared envelope.
    from .languages import UnsupportedLanguageError, backend_for_frozen_version

    try:
        backend = backend_for_frozen_version(version)
    except UnsupportedLanguageError as exc:
        raise WorkflowError("frozen artifact hash/version mismatch") from exc
    backend.validate_frozen(value)
    return value
