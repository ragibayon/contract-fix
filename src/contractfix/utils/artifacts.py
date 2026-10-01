"""Atomic artifacts and append-only, per-run JSONL events."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SENSITIVE_KEYS = re.compile(
    r"(?i)(api[_-]?key|authorization|password|secret|access[_-]?token)"
)
BEARER = re.compile(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+")
KEY = re.compile(r"\b(?:sk-or-v1-|sk-|hf_)[A-Za-z0-9_-]{12,}")


def redact(value: Any) -> Any:
    """Recursively remove common credential forms from artifact values."""
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if SENSITIVE_KEYS.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        output = BEARER.sub("Bearer [REDACTED]", value)
        output = KEY.sub("[REDACTED]", output)
        for key, secret in os.environ.items():
            if SENSITIVE_KEYS.search(key) and len(secret) >= 8:
                output = output.replace(secret, "[REDACTED]")
        return output
    return value


def canonical(value: Any) -> bytes:
    """Serialize a JSON value deterministically."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def digest(value: Any) -> str:
    """Return the SHA-256 digest of canonical JSON."""
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    """Durably replace a JSON artifact without exposing a partial file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, ensure_ascii=False, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class EventLog:
    """Append structured events from one writer process."""

    def __init__(
        self, path: str | Path, *, run_id: str, task_id: str | None = None
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.task_id = task_id
        self._lock = threading.Lock()

    def emit(self, event: str, *, stage: str = "host", **fields: Any) -> None:
        row = {
            "schema_version": 1,
            "at": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "task_id": self.task_id,
            "stage": stage,
            "event": event,
            "data": redact(fields),
        }
        serialized = canonical(row).decode() + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as output:
            output.write(serialized)
            output.flush()


class RunArtifacts:
    """Own an immutable run directory, manifest, and event stream."""

    def __init__(self, root: str | Path, manifest: dict[str, Any]) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        public = redact(manifest)
        self.manifest = {
            "schema_version": 1,
            "configuration": public,
            "sha256": digest(public),
        }
        atomic_json(self.root / "manifest.json", self.manifest)
        self.events = EventLog(
            self.root / "events.jsonl",
            run_id=self.root.name,
            task_id=manifest.get("task_id"),
        )
        self.events.emit("run_started", manifest_sha256=self.manifest["sha256"])

    def save(self, relative: str, value: Any) -> None:
        destination = self.root / relative
        if not destination.resolve().is_relative_to(self.root):
            raise ValueError("artifact path escape")
        atomic_json(destination, redact(value))
