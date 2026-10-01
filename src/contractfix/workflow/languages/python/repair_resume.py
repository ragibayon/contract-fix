"""Resume only sealed patch work; preserve paid receipts and consumed budgets."""
from __future__ import annotations

import json
from pathlib import Path

from contractfix.utils.artifacts import EventLog, RunArtifacts, digest, redact


def open_repair_artifacts(output: Path, configuration: dict, *, resume: bool) -> RunArtifacts:
    if not resume or not output.exists():
        return RunArtifacts(output, configuration)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    public = redact(configuration)
    if manifest.get("sha256") != digest(public) or manifest.get("configuration") != public:
        raise ValueError("repair resume identity mismatch: guidance, mode, model, or policy changed")
    artifacts = RunArtifacts.__new__(RunArtifacts)
    artifacts.root = output.resolve()
    artifacts.manifest = manifest
    artifacts.events = EventLog(output / "events.jsonl", run_id=output.name,
                                task_id=configuration.get("task_id"))
    artifacts.events.emit("repair_resumed", manifest_sha256=manifest["sha256"])
    return artifacts


def restore_model_receipts(stages, artifacts: RunArtifacts) -> None:
    """Continue request numbering and accounting without overwriting prior calls."""
    path = artifacts.root / "model/artifacts.json"
    if not path.exists() or not hasattr(stages, "model_artifacts"):
        return
    entries = [entry for entry in json.loads(path.read_text(encoding="utf-8")) if entry.get("stage") == "patch"]
    if getattr(stages, "model_artifacts", []):
        raise ValueError("resume requires a fresh model-stage session")
    stages.model_artifacts = entries
    stages.calls = len(entries)
    stages.call_sequence_offset = max((entry["call"] for entry in entries), default=0) - len(entries)
    if not hasattr(stages, "_account_usage"):
        return
    for entry in entries:
        location = entry.get("response") or entry.get("failure")
        if not location:
            continue
        receipt = artifacts.root / location
        if not receipt.resolve().is_relative_to(artifacts.root):
            raise ValueError("model receipt path escape")
        data = json.loads(receipt.read_text(encoding="utf-8"))
        usage = data.get("metrics") or {}
        request = json.loads((artifacts.root / entry["request"]).read_text(encoding="utf-8"))
        stages._account_usage(usage, policy_stage="patch",
                              thinking_enabled=request.get("model_configuration", {}).get("thinking", True))
        stages.cumulative_elapsed_seconds += sum(
            call.get("latency_seconds") or 0 for call in usage.get("calls", []))
