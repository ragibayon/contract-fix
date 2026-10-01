"""Seal pre-patch availability without turning advisory model judgments into facts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from contractfix.contracts.core import digest
from contractfix.contracts.snapshots import repository_identity
from contractfix.utils.artifacts import atomic_json
from ...frozen import load_frozen
from ...task import Task
from .models import WorkflowSettings
from .repair_models import RepairGuidance
from .repository import production_path, safe_path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seal_guidance(run_dir: Path, status: dict, *, persist: bool = True) -> dict | None:
    """Read only generation receipts. No gold/evaluation result is ever consulted."""
    localization_path = run_dir / "localization" / "final.json"
    frozen = load_frozen(run_dir) if (run_dir / "frozen.json").exists() else None
    if not localization_path.exists() and frozen is None:
        return None
    configuration = _read(run_dir / "manifest.json")["configuration"]
    if frozen:
        localization = frozen["localization"]
        allowed = frozen["allowed_edit_paths"]
        task, repo_sha, executor = frozen["task"], frozen["repo_sha256"], frozen["executor"]
    else:
        receipt = _read(localization_path)
        localization, allowed = receipt["localization"], receipt["localized_edit_paths"]
        task = configuration["task"]
        repo_sha = _read(run_dir / "base_identity.json")["repo_sha256"]
        executor = _read(run_dir / "executor.json")
    grounded = run_dir / "evidence" / "grounded_nlc.json"
    nlc = _read(grounded) if grounded.exists() else frozen.get("obligation") if frozen else None
    reasons = []
    # Advisory qualification is intentionally unchanged. Repair admission is explicit:
    # a grounded but reviewer-rejected clause is NOT passed as trusted guidance.
    accepted = bool(nlc and nlc.get("nlc_review", {}).get("accepted") is True)
    if not accepted:
        nlc = None
        reasons.append("NLC_UNAVAILABLE_OR_SELF_REVIEW_NOT_ACCEPTED")
    ec = None
    if frozen and nlc and frozen.get("qualification", {}).get("state") == "PRE_GOLD_QUALIFIED":
        if frozen["qualification"].get("gold_consulted") is not False:
            raise ValueError("repair requires pre-gold contract qualification")
        primary = next(c for c in frozen["candidates"] if c["id"] == frozen["primary_candidate_id"])
        ec = {
            "candidate_id": primary["id"],
            "source": primary["source"],
            "source_sha256": hashlib.sha256(primary["source"].encode()).hexdigest(),
            "contracted_operation": frozen["contracted_operation"],
            "frozen_sha256": frozen["sha256"],
        }
    if not ec:
        reasons.append("EC_NOT_AVAILABLE_FOR_ENFORCEMENT")
    body = RepairGuidance(
        task=Task.model_validate(task).model_dump(), repo_sha256=repo_sha, executor=executor,
        workflow=configuration.get("workflow", frozen["workflow"] if frozen else {}),
        generator=configuration.get("generator", {}), localization=localization,
        localized_edit_paths=sorted(set(allowed)),
        nlc_status="QUALIFIED" if nlc else "UNAVAILABLE", qualified_nlc=nlc,
        ec_status="QUALIFIED" if ec else "UNAVAILABLE", qualified_ec=ec,
        qualification_status=status["status"], unavailability_reasons=reasons,
    ).model_dump()
    envelope = {**body, "sha256": digest(body)}
    target = run_dir / "guidance.json"
    if target.exists() and _read(target) != envelope:
        raise ValueError("repair guidance is immutable; create a new run")
    if persist and not target.exists():
        atomic_json(target, envelope)
    return envelope


def load_guidance(run_dir: Path) -> tuple[RepairGuidance, str]:
    path = run_dir / "guidance.json"
    if path.exists():
        payload = _read(path)
    elif (run_dir / "frozen.json").exists():
        # Read-only migration for existing /3 runs. Existing artifacts are never
        # rewritten; the derived packet is captured in the new repair output.
        payload = seal_guidance(run_dir, {"status": "FROZEN"}, persist=False)
    else:
        raise ValueError("run has no sealed guidance or current frozen contract")
    claimed = payload.pop("sha256")
    if digest(payload) != claimed:
        raise ValueError("repair guidance hash mismatch")
    guidance = RepairGuidance.model_validate(payload)
    Task.model_validate(guidance.task)  # Reject evaluator fields even in a hand-edited packet.
    WorkflowSettings.model_validate(guidance.workflow)
    if repository_identity(run_dir / "base") != guidance.repo_sha256:
        raise ValueError("base snapshot identity changed after guidance was sealed")
    for name in guidance.localized_edit_paths:
        if safe_path(name) != name or not production_path(name):
            raise ValueError("invalid initial edit scope")
    if guidance.qualified_ec:
        ec = guidance.qualified_ec
        frozen = load_frozen(run_dir)
        if frozen["sha256"] != ec["frozen_sha256"]:
            raise ValueError("frozen contract differs from guidance")
        primary = next(c for c in frozen["candidates"] if c["id"] == frozen["primary_candidate_id"])
        if primary["source"] != ec["source"] or frozen["contracted_operation"] != ec["contracted_operation"]:
            raise ValueError("executable contract differs from frozen primary")
        if hashlib.sha256(ec["source"].encode()).hexdigest() != ec["source_sha256"]:
            raise ValueError("executable contract hash mismatch")
    return guidance, claimed
