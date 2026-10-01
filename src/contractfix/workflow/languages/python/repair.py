"""Three gold-blind APR variants over one sealed guidance packet.

Only patch actions are retried. No path in this module regenerates NLC/EC or
weakens a qualified constraint after observing a candidate failure.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError

from contractfix.contracts.execution import Executor
from contractfix.utils.artifacts import RunArtifacts, digest
from contractfix.utils.logger import logger
from ...accounting import model_accounting
from ...stages import Stages
from .executable_contracts import evaluate_executable_contract
from .guidance import load_guidance
from .models import WorkflowSettings
from .patch_context import PatchContext
from .patching import (
    EVIDENCE_SELECTOR,
    SemanticNoopError,
    materialize_edits,
    rank_patches,
    select_patch,
    validation_assurance,
)
from .repair_execution import (
    environment_receipt,
    importable_package,
    ordinary_checks,
    repository_import_check,
    run_command,
)
from .repair_models import PatchNLCConformance, PatchProposal, RepairGuidance, effective_mode
from .repair_resume import open_repair_artifacts, restore_model_receipts
from .repository_validation import discover_visible_test_command


def patch_packet(
    guidance: RepairGuidance,
    context: PatchContext,
    environment: dict,
    mode: str,
    *,
    attempt: int,
    feedback: dict | None,
    rounds_left: int,
) -> dict:
    """Context and enforcement have byte-identical first-call information.

    EC availability, source, candidate ID, and validation receipts are intentionally
    not exposed. Later enforcement feedback is the online intervention under study.
    """
    nlc = None
    if mode != "ISSUE_ONLY" and guidance.qualified_nlc:
        nlc = {
            key: guidance.qualified_nlc[key]
            for key in (
                "precondition",
                "precondition_clauses",
                "normal_postcondition",
                "exceptional_postcondition",
                "evidence",
                "rationale",
            )
            if key in guidance.qualified_nlc
        }
    return {
        "task": {key: value for key, value in guidance.task.items() if key != "image"},
        "environment": environment,
        **context.packet(),
        "qualified_nlc": nlc,
        "attempt": attempt,
        "remaining_context_rounds": rounds_left,
        "previous_failure": feedback,
    }


def _accounting_delta(before: dict, after: dict) -> dict:
    numeric = {
        key: after[key] - before[key]
        for key in after
        if isinstance(after.get(key), (int, float)) and isinstance(before.get(key), (int, float))
    }
    return {**numeric, "assurance": "provider_reported_when_available; stage calls are not SDK retry counts"}


def _show_patch_context(round_number: int, receipt: dict) -> None:
    """Render bounded context actions while retaining full source only in artifacts."""
    rows = []
    for outcome in receipt.get("outcomes", []):
        request = outcome.get("request", {})
        target = request.get("target", "")
        if request.get("start_line") is not None:
            target = f"{target}:{request['start_line']}"
        rows.append([request.get("kind", "unknown"), target, outcome.get("status", "unknown")])
    logger.table(
        f"Patch context · round {round_number}",
        ["Request", "Target", "Result"],
        rows or [["none", "none", "NO_PROGRESS"]],
    )
    logger.info(
        f"[patch context {round_number}] progress={str(bool(receipt.get('progress'))).lower()} | "
        f"retrieved={len(receipt.get('retrieved_context', []))} | "
        f"allowed_edit_paths={len(receipt.get('allowed_edit_paths', []))}"
    )


def _show_command_receipt(title: str, receipt: dict) -> None:
    command = " ".join(str(value) for value in receipt.get("command", [])) or "not recorded"
    output = receipt.get("log_tail") or "(no command output)"
    if len(output) > 8000:
        output = output[-8000:] + "\n[bounded to the last 8000 characters]"
    tests = receipt.get("test_results") or {}
    test_summary = (
        f"**Visible tests:** passed={len(tests.get('passed', []))} · "
        f"failed={len(tests.get('failed', []))} · "
        f"skipped={len(tests.get('skipped', []))}  \n"
        if tests
        else "**Visible tests:** unstructured/not applicable  \n"
    )
    logger.panel(
        title,
        f"**Command:** `{command}`  \n"
        f"**Exit code:** `{receipt.get('exit_code')}`  \n"
        f"**Passed:** `{receipt.get('passed')}`  \n"
        f"**Infrastructure error:** `{receipt.get('infrastructure_error') or 'none'}`  \n"
        f"**Repository mutated:** `{receipt.get('repository_mutated', False)}`  \n"
        f"{test_summary}\n"
        f"```text\n{output}\n```",
        style="green" if receipt.get("passed") else "red",
    )


def _show_patch_candidate(attempt: int, patch_text: str, row: dict, candidate_dir: Path) -> None:
    """Show the generated diff and every generation-time gate in chronological order."""
    maximum = 16000
    rendered = patch_text
    if len(rendered) > maximum:
        rendered = rendered[:maximum] + "\n... [truncated; full diff is saved in candidate.patch]\n"
    logger.code(f"Patch candidate {attempt}", rendered, lexer="diff")
    ordinary = row.get("ordinary_checks") or {}
    syntax = ordinary.get("syntax") or {}
    ec = row.get("ec_execution") or {}
    logger.table(
        f"Patch validation · candidate {attempt}",
        ["Gate", "Result", "Scope / detail"],
        [
            [
                "Applicable diff",
                "PASS" if row.get("applicable") else "FAIL",
                ", ".join(row.get("changed_files", [])) or "none",
            ],
            [
                "Syntax",
                "PASS" if syntax.get("passed") else "FAIL",
                ordinary.get("validation_scope", "not run"),
            ],
            [
                "Repository checks",
                ordinary.get("repository_check_status", "NOT_RUN"),
                f"introduced={len(ordinary.get('introduced_failures', []))} · "
                f"repaired={len(ordinary.get('repaired_visible_failures', []))} · "
                f"persistent={len(ordinary.get('persistent_failures', []))}",
            ],
            [
                "NLC semantic gate",
                "PASS"
                if row.get("nlc_pass") is True
                else "FAIL"
                if row.get("nlc_pass") is False
                else "NOT_REQUIRED",
                row.get("nlc_conformance_status", "not run"),
            ],
            [
                "Frozen EC",
                "PASS"
                if row.get("ec_pass") is True
                else "FAIL"
                if row.get("ec_pass") is False
                else "NOT_REQUIRED",
                ec.get("outcome", "not run"),
            ],
            [
                "Candidate",
                "ELIGIBLE"
                if select_patch([row], enforce_ec=row.get("ec_pass") is not None)
                else row.get("failure", "REJECTED"),
                f"{row.get('changed_lines', 0)} changed line(s)",
            ],
        ],
    )
    if syntax:
        _show_command_receipt(f"Syntax execution · candidate {attempt}", syntax)
    for number, receipt in enumerate(ordinary.get("repository_checks", []), start=1):
        _show_command_receipt(f"Repository check {number} · candidate {attempt}", receipt)
    if ec:
        trace = "\n".join(ec.get("execution_trace", [])) or "(no execution trace)"
        command_output = ec.get("command_log_tail") or "(no command output)"
        logger.panel(
            f"Frozen EC execution · patch candidate {attempt}",
            f"**Outcome:** `{ec.get('outcome', 'UNKNOWN')}`  \n"
            f"**Contracted operation reached:** `{ec.get('contracted_operation_reached')}`  \n"
            f"**All runtime assertions exercised:** `{ec.get('all_runtime_assertions_exercised')}`  \n"
            f"**Infrastructure error:** `{ec.get('infrastructure_error') or 'none'}`\n\n"
            f"**Execution trace**\n```text\n{trace}\n```\n\n"
            f"**Command output**\n```text\n{command_output[-8000:]}\n```",
            style="green" if row.get("ec_pass") else "red",
        )
    logger.info(f"[patch candidate {attempt}] full validation: {candidate_dir / 'validation.json'}")


def _review_patch_nlc(
    row: dict,
    *,
    artifacts: RunArtifacts,
    guidance: RepairGuidance,
    stages: Stages,
    checks: dict,
    review_context: str,
) -> None:
    patch_path = artifacts.root / row["patch_path"]
    packet = {
        "task": {key: value for key, value in guidance.task.items() if key != "image"},
        "qualified_nlc": guidance.qualified_nlc,
        "candidate_patch": patch_path.read_text(encoding="utf-8"),
        "changed_files": list(row.get("changed_files") or []),
        "repository_validation": {
            "status": checks.get("repository_check_status"),
            "introduced_failures": checks.get("introduced_failures") or [],
            "repaired_failures": checks.get("repaired_visible_failures") or [],
            "validation_scope": checks.get("validation_scope"),
        },
    }
    if review_context != "NLC_ONLY":
        packet["review_context"] = review_context
    try:
        review = stages.generate("patch_nlc_conformance", PatchNLCConformance, packet)
    except Exception as exc:
        row.update(
            nlc_pass=False,
            nlc_conformance_status="ERROR",
            nlc_conformance={
                "verdict": "INCONCLUSIVE",
                "reason": f"NLC conformance review failed: {type(exc).__name__}: {str(exc)[:1000]}",
            },
            failure="NLC_CONFORMANCE_INCONCLUSIVE",
        )
        return
    receipt = review.model_dump()
    row["nlc_conformance"] = receipt
    row["nlc_conformance_status"] = review.verdict
    row["nlc_pass"] = review.verdict == "CONFORMS"
    if not row["nlc_pass"]:
        row["failure"] = (
            "NLC_CONFORMANCE_VIOLATION"
            if review.verdict == "VIOLATES"
            else "NLC_CONFORMANCE_INCONCLUSIVE"
        )


def _save_patch_certificates(
    artifacts: RunArtifacts,
    guidance: RepairGuidance,
    rows: list[dict],
    selected: dict | None,
    selection_mode: str,
) -> None:
    """Freeze gate evidence and paths for later gold-blind candidate analysis."""
    records = []
    for row in rows:
        patch_path = row.get("patch_path")
        if patch_path:
            patch_bytes = (artifacts.root / patch_path).read_bytes()
            if hashlib.sha256(patch_bytes).hexdigest() != row["patch_sha256"]:
                raise ValueError("candidate patch changed before certification")
        checks = row.get("ordinary_checks") or {}
        records.append({
            "attempt": row.get("attempt"),
            "patch_path": patch_path,
            "patch_sha256": row.get("patch_sha256"),
            "validation_path": f"candidates/{row['attempt']:03d}/validation.json",
            "nlc_qualification_status": guidance.nlc_status,
            "ec_qualification_status": guidance.ec_status,
            "applicable": row.get("applicable") is True,
            "ordinary_pass": row.get("ordinary_pass") is True,
            "repository_check_status": checks.get("repository_check_status", "NOT_RUN"),
            "patch_nlc_review_status": row.get("nlc_conformance_status", "NOT_RUN"),
            "patch_ec_status": row.get("ec_validation_status", "NOT_RUN"),
            "failure": row.get("failure"),
            "submitted": selected is row,
            "selection_mode": selection_mode if selected is row else None,
        })
    artifacts.save("patch_certificates.json", {
        "schema_version": "contractfix-patch-certificates/1",
        "records": records,
    })


def _validate_candidate(
    row: dict,
    *,
    base: Path,
    artifacts: RunArtifacts,
    executor: Executor,
    commands: list[list[str]],
    baseline: list[dict],
    baseline_policy: dict,
    validation_plan: dict,
    settings: WorkflowSettings,
    guidance: RepairGuidance,
    mode: str,
    stages: Stages,
    import_module: str | None = None,
) -> None:
    """Apply all post-generation gates to one already materialized candidate."""
    patch_path = artifacts.root / row["patch_path"]
    checks = ordinary_checks(
        base,
        patch_path,
        list(row["changed_files"]),
        executor,
        commands,
        settings.patch_missing_checks,
        baseline,
        syntax_scope=settings.patch_syntax_scope,
    )
    repository_unavailable = (
        checks.get("repository_check_status") == "NOT_COMPARABLE"
        and checks.get("syntax", {}).get("passed") is True
    )
    if (
        repository_unavailable
        and settings.repository_validation_failure_policy == "ignore"
    ):
        ignored_failure = checks.pop("failure", None)
        ignored_infrastructure_error = checks.get("infrastructure_error")
        checks.update(
            passed=True,
            validation_scope="syntax_only_repository_check_ignored",
            repository_check_status="IGNORED_NOT_COMPARABLE",
            regression_safe=None,
            ignored_failure=ignored_failure,
            ignored_infrastructure_error=ignored_infrastructure_error,
        )
        checks["infrastructure_error"] = None
        logger.warning(
            f"[repair] repository validation for patch candidate {row['attempt']} "
            "is not comparable; preserving the receipt and continuing under the "
            "configured ignore policy"
        )
    if commands and checks.get("repository_check_status") != "IGNORED_NOT_COMPARABLE":
        checks["validation_scope"] = validation_plan["validation_scope"]
    elif baseline_policy.get("selection_effect") == "IGNORE_AND_CONTINUE":
        checks.update(
            validation_scope="syntax_only_repository_baseline_ignored",
            repository_check_status="IGNORED_NOT_COMPARABLE",
            regression_safe=None,
            baseline_issues=baseline_policy.get("issues", []),
        )
    row.update(
        ordinary_pass=checks["passed"],
        ordinary_checks=checks,
        infrastructure_error=checks.get("infrastructure_error"),
    )
    row["validation_assurance"] = validation_assurance(row)
    if not checks["passed"]:
        row["failure"] = checks.get("failure", "VISIBLE_REGRESSION")
        return
    if import_module:
        # Advisory evidence: a pre-existing import blocker must not prevent
        # producing a patch, but the model and researcher should see whether
        # the candidate also removes it. This is not SWE-bench evaluation.
        row["repository_import"] = repository_import_check(
            base, executor, import_module, patch=patch_path
        )
        logger.info(
            f"[patch {row['attempt']}] plain repository import "
            f"{import_module}: {row['repository_import']['status']} (advisory)"
        )
    # When no executable EC is available, NLC_CONTEXT still needs a semantic
    # acceptance gate. Syntax or non-comparable repository tests alone are not
    # sufficient evidence that the patch implements the recovered behavior.
    if mode == "NLC_CONTEXT":
        row["ec_validation_status"] = "NOT_REQUIRED"
        if not settings.patch_nlc_review_enabled:
            row["nlc_conformance_status"] = "SKIPPED_BY_ABLATION"
            row["nlc_pass"] = None
            return
        _review_patch_nlc(
            row, artifacts=artifacts, guidance=guidance, stages=stages,
            checks=checks, review_context="NLC_ONLY",
        )
        return
    if mode != "EC_ENFORCED":
        row["ec_validation_status"] = "NOT_REQUIRED"
        row["nlc_conformance_status"] = "NOT_REQUIRED"
        return
    row["nlc_conformance_status"] = "NOT_REQUIRED_EC_ENFORCED"
    ec = guidance.qualified_ec
    try:
        observation = evaluate_executable_contract(
            base,
            ec["source"],
            ec["contracted_operation"],
            executor,
            patch=patch_path,
        )
    except (OSError, ValueError, SyntaxError) as exc:
        observation = {
            "outcome": "ERROR",
            "contracted_operation_reached": False,
            "all_runtime_assertions_exercised": False,
            "infrastructure_error": "EC_EXECUTION_ERROR",
            "detail": str(exc)[:1000],
        }
    row["ec_execution"] = observation
    row["ec_pass"] = (
        observation["outcome"] == "SATISFIED"
        and observation.get("contracted_operation_reached") is True
        and observation.get("all_runtime_assertions_exercised") is True
        and not observation.get("infrastructure_error")
    )
    if observation.get("infrastructure_error"):
        row.update(
            infrastructure_error=observation["infrastructure_error"],
            ec_validation_status="EC_EXECUTION_ERROR",
            failure="EC_EXECUTION_ERROR",
        )
    elif observation.get("outcome") == "VIOLATED":
        row["ec_validation_status"] = "EC_VIOLATION"
        row["failure"] = "EC_VIOLATION"
    elif row["ec_pass"]:
        row["ec_validation_status"] = "EC_PASS"
    else:
        row.update(ec_validation_status="EC_NOT_COMPARABLE", failure="EC_NOT_COMPARABLE")


def _refinement_parent(rows: list[dict]) -> dict | None:
    """Choose a reproducibly EC-rejected, otherwise eligible parent without gold."""
    rejected = [
        row for row in rows
        if row.get("applicable") is True
        and row.get("ordinary_pass") is True
        and validation_assurance(row) != "REGRESSION"
        and row.get("nlc_pass") is not False
        and row.get("ec_validation_status") == "EC_VIOLATION"
        and row.get("patch_sha256")
    ]
    ranked = rank_patches(rejected, enforce_ec=False, version=EVIDENCE_SELECTOR)
    return ranked[0] if ranked else None


def _human_review_candidate(rows: list[dict]) -> dict | None:
    """Pick a materialized attempt for inspection without calling it eligible."""
    materialized = [
        row for row in rows
        if row.get("applicable") is True and row.get("patch_path")
    ]
    if not materialized:
        return None
    return min(
        materialized,
        key=lambda row: (
            row.get("ordinary_pass") is not True,
            row.get("nlc_pass") is False,
            row.get("ec_pass") is False,
            len((row.get("ordinary_checks") or {}).get("introduced_failures") or []),
            row.get("attempt", 0),
        ),
    )


def _refine_rejected_patch(
    *,
    rows: list[dict],
    pool_sha256: str,
    base: Path,
    artifacts: RunArtifacts,
    executor: Executor,
    stages: Stages,
    settings: WorkflowSettings,
    guidance: RepairGuidance,
    context: PatchContext,
    environment: dict,
    commands: list[list[str]],
    baseline: list[dict],
    baseline_policy: dict,
    validation_plan: dict,
    import_module: str | None,
) -> tuple[dict | None, dict]:
    """Spend at most one post-freeze model call; preserve the base pool on resume."""
    parent = _refinement_parent(rows)
    decision = {
        "schema_version": "contractfix-ec-refinement/1",
        "initial_candidate_pool_sha256": pool_sha256,
        "parent_attempt": parent["attempt"] if parent else None,
        "parent_patch_sha256": parent["patch_sha256"] if parent else None,
        "child_attempt": max((row["attempt"] for row in rows), default=0) + 1,
        "status": "PARENT_SELECTED" if parent else "NO_REPRODUCIBLE_EC_REJECTION",
    }
    decision_path = artifacts.root / "refinement/decision.json"
    if decision_path.exists():
        if json.loads(decision_path.read_text(encoding="utf-8")) != decision:
            raise ValueError("EC refinement decision changed during resume")
    else:
        artifacts.save("refinement/decision.json", decision)
    if parent is None:
        return None, decision
    edits_path = artifacts.root / f"candidates/{parent['attempt']:03d}/edits.json"
    if not edits_path.is_file():
        raise ValueError("refinement parent has no saved edits")
    previous_edits = json.loads(edits_path.read_text(encoding="utf-8"))["edits"]
    observation = parent["ec_execution"]
    feedback = {
        "failure": "EC_VIOLATION",
        "parent_attempt": parent["attempt"],
        "parent_patch_sha256": parent["patch_sha256"],
        "previous_edits": previous_edits,
        "contract_observation": {
            "outcome": observation["outcome"],
            "operation_reached": observation.get("contracted_operation_reached"),
            "all_assertions_exercised": observation.get("all_runtime_assertions_exercised"),
            "violations": (observation.get("assertion_execution") or {}).get(
                "observed_violations", []
            ),
        },
    }
    packet = patch_packet(
        guidance, context, environment, "EC_ENFORCED",
        attempt=decision["child_attempt"], feedback=feedback, rounds_left=0,
    )
    input_path = artifacts.root / "refinement/patch_input.json"
    response_path = artifacts.root / "refinement/patch_response.json"
    previously_requested = input_path.exists()
    if previously_requested:
        if json.loads(input_path.read_text(encoding="utf-8")) != packet:
            raise ValueError("EC refinement prompt changed during resume")
    else:
        artifacts.save("refinement/patch_input.json", packet)
    if response_path.exists():
        response = PatchProposal.model_validate(json.loads(response_path.read_text(encoding="utf-8")))
        model_calls = 1
    elif previously_requested:
        # A pre-existing request without a response may have been billed. Do not
        # issue it again; a fresh request uses the one-shot pending marker below.
        return None, {**decision, "status": "INTERRUPTED_MODEL_CALL", "model_calls": 1}
    else:
        try:
            response = stages.generate("patch", PatchProposal, packet)
            response = PatchProposal.model_validate(response.model_dump())
        except (ValidationError, ValueError) as exc:
            return None, {**decision, "status": "INVALID_RESPONSE", "detail": str(exc)[:1000],
                          "model_calls": 1}
        except Exception as exc:
            return None, {**decision, "status": "MODEL_ERROR", "error_type": type(exc).__name__,
                          "detail": str(exc)[:1000], "model_calls": 1}
        artifacts.save("refinement/patch_response.json", response.model_dump())
        model_calls = 1
    if response.action != "submit_edits":
        return None, {**decision, "status": "REFINEMENT_ABSTAINED"
                      if response.action == "abstain" else "REFINEMENT_CONTEXT_REQUESTED",
                      "model_calls": model_calls}
    proposal = response.model_dump()
    attempt = decision["child_attempt"]
    row = {
        "attempt": attempt,
        "phase": "EC_REFINEMENT",
        "parent_attempt": parent["attempt"],
        "parent_patch_sha256": parent["patch_sha256"],
        "proposal_sha256": digest(proposal),
        "applicable": False,
        "ordinary_pass": False,
        "validation_assurance": "NOT_COMPARABLE",
        "ec_pass": None,
        "ec_validation_status": "NOT_EVALUATED",
        "infrastructure_error": None,
        "allowed_edit_paths": sorted(context.allowed),
        "context_rounds": 0,
    }
    candidate_dir = artifacts.root / "candidates" / f"{attempt:03d}"
    validation_path = candidate_dir / "validation.json"
    if validation_path.exists():
        saved = json.loads(validation_path.read_text(encoding="utf-8"))
        if saved.get("proposal_sha256") != row["proposal_sha256"]:
            raise ValueError("saved EC refinement proposal changed")
        return saved, {**decision, "status": "VALIDATED", "model_calls": model_calls,
                       "child_patch_sha256": saved.get("patch_sha256")}
    try:
        patch = materialize_edits(
            base, [edit.materialized() for edit in response.edits], sorted(context.allowed)
        )
    except (OSError, ValueError, SyntaxError) as exc:
        row.update(failure="REFINEMENT_EDIT_NOT_APPLICABLE", detail=str(exc)[:1000])
    else:
        candidate_dir.mkdir(parents=True, exist_ok=True)
        patch_path = candidate_dir / "candidate.patch"
        if patch_path.exists() and patch_path.read_bytes() != patch.text.encode():
            raise ValueError("saved EC refinement patch changed")
        patch_path.write_bytes(patch.text.encode())
        row.update(applicable=True, patch_sha256=patch.sha256,
                   changed_files=list(patch.files), changed_lines=patch.changed_lines,
                   patch_path=str(patch_path.relative_to(artifacts.root)))
        if patch.sha256 in {item.get("patch_sha256") for item in rows}:
            row.update(failure="NO_PROGRESS", detail="Refinement repeated an initial patch")
        else:
            _validate_candidate(
                row, base=base, artifacts=artifacts, executor=executor,
                commands=commands, baseline=baseline, baseline_policy=baseline_policy,
                validation_plan=validation_plan, settings=settings, guidance=guidance,
                mode="EC_ENFORCED", stages=stages, import_module=import_module,
            )
    artifacts.save(f"candidates/{attempt:03d}/edits.json", proposal)
    artifacts.save(f"candidates/{attempt:03d}/validation.json", row)
    return row, {**decision, "status": "VALIDATED", "model_calls": model_calls,
                 "child_patch_sha256": row.get("patch_sha256")}


def repair(
    guidance_dir: Path,
    output: Path,
    executor: Executor,
    stages: Stages,
    mode: str = "enforce",
    *,
    resume: bool = False,
) -> dict:
    if mode not in {"no-contract", "context", "enforce"}:
        raise ValueError("unknown ablation mode")
    if output.resolve().is_relative_to((guidance_dir / "base").resolve()):
        raise ValueError("patch output must be outside the immutable base")
    started = perf_counter()
    before = copy.deepcopy(model_accounting(stages))
    if not (guidance_dir / "guidance.json").exists() and not (guidance_dir / "frozen.json").exists():
        manifest = json.loads((guidance_dir / "manifest.json").read_text())["configuration"]
        artifacts = RunArtifacts(
            output,
            {
                "purpose": "patch_generation",
                "task_id": manifest.get("task_id"),
                "generator": stages.identity,
                "requested_variant": mode,
            },
        )
        result = {
            "status": "NO_PATCH",
            "reason": "NO_LOCALIZATION_OR_REPAIR_GUIDANCE",
            "instance_id": manifest.get("task_id"),
            "requested_variant": mode,
            "effective_mode": None,
            "patch_attempts": 0,
            "selected_patch": None,
        }
        _finish(artifacts, result, "", stages, before, started)
        return result
    guidance, guidance_sha = load_guidance(guidance_dir)
    if executor.identity()["sha256"] != guidance.executor["sha256"]:
        raise ValueError("patch executor differs from the sealed qualification environment")
    settings = WorkflowSettings.model_validate(guidance.workflow)
    selected_mode = effective_mode(mode, guidance)
    artifacts = open_repair_artifacts(
        output,
        {
            "purpose": "patch_generation",
            "task_id": guidance.task["instance_id"],
            "task": guidance.task,
            "generator": stages.identity,
            "workflow": guidance.workflow,
            "guidance_sha256": guidance_sha,
            "requested_variant": mode,
            "effective_mode": selected_mode,
        },
        resume=resume,
    )
    if resume:
        restore_model_receipts(stages, artifacts)
        completed = artifacts.root / "status.json"
        if completed.exists():
            old_result = json.loads(completed.read_text())
            if old_result.get("status") in {"PATCH_SELECTED", "NO_PATCH"}:
                return old_result
    artifacts.save("guidance_snapshot.json", {**guidance.model_dump(), "sha256": guidance_sha})
    if mode == "enforce" and settings.require_ec_for_patch and guidance.qualified_ec is None:
        result = {
            "status": "NO_PATCH",
            "instance_id": guidance.task["instance_id"],
            "requested_variant": mode,
            "effective_mode": "EC_REQUIRED_UNAVAILABLE",
            "nlc_available": guidance.qualified_nlc is not None,
            "ec_available": False,
            "patch_attempts": 0,
            "patch_model_calls": 0,
            "reason": "QUALIFIED_EC_REQUIRED",
        }
        _finish(artifacts, result, "", stages, before, started)
        return result
    old_artifacts = getattr(stages, "artifacts", None)
    old_model_artifacts = getattr(stages, "model_artifacts", None)
    if hasattr(stages, "model_artifacts") and not resume:
        # The reused model may already own qualification receipts. Patch indices
        # must not point at files in the other stage's artifact directory.
        stages.model_artifacts = []
    if hasattr(stages, "artifacts"):
        stages.artifacts = artifacts
    logger.attach(artifacts.root)
    logger.info(f"[repair] {guidance.task['instance_id']} | requested={mode} | effective={selected_mode}")
    artifacts.save(
        "guidance_reference.json",
        {"sha256": guidance_sha, "source": str(guidance_dir.resolve()), "base_sha256": guidance.repo_sha256},
    )
    try:
        result, text = _repair(
            guidance_dir, guidance, guidance_sha, artifacts, executor, stages, settings, mode, selected_mode
        )
        _finish(artifacts, result, text, stages, before, started)
        return result
    except Exception as exc:
        result = {
            "status": "ERROR",
            "instance_id": guidance.task["instance_id"],
            "requested_variant": mode,
            "effective_mode": selected_mode,
            "error_type": type(exc).__name__,
            "detail": str(exc)[:1500],
        }
        _finish(artifacts, result, "", stages, before, started)
        raise
    finally:
        if hasattr(stages, "artifacts"):
            stages.artifacts = old_artifacts
        if old_model_artifacts is not None and not resume:
            stages.model_artifacts = old_model_artifacts


def _repair(guidance_dir, guidance, guidance_sha, artifacts, executor, stages, settings, variant, mode):
    frozen_pool = settings.patch_candidate_policy in {"budgeted", "budgeted_ec_retry"}
    base = guidance_dir / "base"
    commands = list(settings.regression_commands)
    validation_plan = {
        "schema_version": "contractfix-repository-validation-plan/1",
        "mode": settings.repository_check_mode,
        "source": "repository",
        "validation_scope": (
            settings.repository_validation_scope
            if settings.repository_check_mode == "auto_visible"
            else "configured"
            if settings.repository_check_mode == "configured"
            else "none"
        ),
        "commands": commands,
        "timeout_seconds": executor.settings.timeout,
        "evaluator_metadata_consulted": False,
    }
    if settings.repository_check_mode == "auto_visible":
        commands, discovery = discover_visible_test_command(
            base,
            guidance.localized_edit_paths,
            limit=settings.repository_test_file_limit,
            broader_limit=settings.repository_broader_test_file_limit,
            scope=settings.repository_validation_scope,
        )
        validation_plan.update(commands=commands, discovery=discovery)
    validation_plan["repository_validation_status"] = (
        "NOT_AVAILABLE"
        if not commands
        else "PARTIAL"
        if settings.repository_check_mode == "auto_visible"
        and settings.repository_validation_scope != "repository_suite"
        else "AVAILABLE"
    )
    plan_path = artifacts.root / "validation_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != validation_plan:
            raise ValueError("repository validation plan changed during resume")
    else:
        artifacts.save("validation_plan.json", validation_plan)
    context = PatchContext(
        base,
        guidance,
        executor,
        char_budget=settings.patch_context_chars,
        additions=settings.patch_context_additions,
        allow_expansion=settings.allow_edit_scope_expansion,
    )
    environment = environment_receipt(base, executor)
    import_module = importable_package(base, guidance.localized_edit_paths)
    import_path = artifacts.root / "repository_import_baseline.json"
    if import_path.exists():
        import_baseline = json.loads(import_path.read_text(encoding="utf-8"))
        if import_baseline.get("module") != import_module:
            raise ValueError("repository import target changed during resume")
    else:
        import_baseline = (
            repository_import_check(base, executor, import_module)
            if import_module
            else {"module": None, "status": "NOT_AVAILABLE", "scope": "no_localized_package"}
        )
        artifacts.save("repository_import_baseline.json", import_baseline)
    if import_baseline["status"] == "BLOCKED":
        logger.warning(
            f"[repair] plain repository import {import_module} is blocked in the "
            "task runtime; this is advisory evidence, not a patch-generation gate"
        )
    environment["repository_import"] = {
        "module": import_module,
        "status": import_baseline["status"],
        "log_tail": (import_baseline.get("execution") or {}).get("log_tail", "")[-3000:],
        "scope": import_baseline["scope"],
    }
    if not (artifacts.root / "environment.json").exists():
        artifacts.save("environment.json", environment)
    result = {
        "status": "NO_PATCH",
        "instance_id": guidance.task["instance_id"],
        "requested_variant": variant,
        "effective_mode": mode,
        "effective_repair_mode": (
            "EC_ENFORCED" if mode == "EC_ENFORCED" else
            "NLC_CONTEXT_FALLBACK" if mode == "NLC_CONTEXT" else
            "ISSUE_CONTEXT_FALLBACK"
        ),
        "ec_status": "QUALIFIED" if guidance.ec_status == "QUALIFIED" else "NO_QUALIFIED_EC",
        "guidance_sha256": guidance_sha,
        "nlc_available": guidance.nlc_status == "QUALIFIED",
        "ec_available": guidance.ec_status == "QUALIFIED",
        "fallback_reasons": guidance.unavailability_reasons if variant != "no-contract" else [],
        "initial_localized_edit_paths": guidance.localized_edit_paths,
        "patch_attempts": 0,
        "patch_model_calls": 0,
        "patch_model_call_limit": settings.max_patch_model_calls,
        "context_rounds": 0,
        "candidates": [],
        "selected_patch": None,
        "regression_commands": commands,
        "repository_check_mode": settings.repository_check_mode,
        "repository_validation_status": validation_plan[
            "repository_validation_status"
        ],
        "repository_validation_failure_policy": (
            settings.repository_validation_failure_policy
        ),
        "missing_checks_policy": settings.patch_missing_checks,
        "repository_import_baseline_status": import_baseline["status"],
        "selection_policy": (
            "hard_gates_then_comparable_checks_repaired_failures_import_readiness_attempt"
        ),
        "selector_version": EVIDENCE_SELECTOR,
        "candidate_policy": settings.patch_candidate_policy,
        "ec_conflict_policy": settings.ec_conflict_policy,
        "reason": "PATCH_BUDGET_EXHAUSTED",
    }
    if environment["status"] != "OBSERVED":
        result["reason"] = "TASK_ENVIRONMENT_UNAVAILABLE"
        return result, ""
    # Execute the immutable base exactly once. A nonzero pytest exit is legitimate
    # baseline evidence when structured failing test identities were captured.
    baseline_path = artifacts.root / "baseline_results.json"
    if baseline_path.exists():
        baseline_document = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_document.get("validation_plan") != validation_plan:
            raise ValueError("repository validation baseline does not match the sealed plan")
        baseline = baseline_document["checks"]
    else:
        baseline = [
            run_command(base, executor, command, capture_tests=True)
            for command in commands
        ]
        baseline_document = {
            "schema_version": "contractfix-repository-baseline/1",
            "validation_plan": validation_plan,
            "checks": baseline,
            "configured": bool(baseline),
            "failed_test_ids": sorted(
                {
                    f"command-{number}::{test_id}"
                    for number, check in enumerate(baseline, start=1)
                    for test_id in (check.get("test_results") or {}).get("failed", [])
                }
            ),
            "summary": {
                "passed": sum(
                    len((check.get("test_results") or {}).get("passed", []))
                    for check in baseline
                ),
                "failed": sum(
                    len((check.get("test_results") or {}).get("failed", []))
                    for check in baseline
                ),
                "skipped": sum(
                    len((check.get("test_results") or {}).get("skipped", []))
                    for check in baseline
                ),
            },
        }
        artifacts.save("baseline_results.json", baseline_document)
    baseline_issues = []
    if any(check.get("infrastructure_error") for check in baseline):
        baseline_issues.append("BASE_CHECK_INFRASTRUCTURE_ERROR")
    if any(check.get("repository_mutated") for check in baseline):
        baseline_issues.append("BASE_CHECK_MUTATED_REPOSITORY")
    if any(
        not check.get("passed")
        and (
            check.get("test_results") is None
            or not check["test_results"].get("observed")
            or check["test_results"].get("exit_status") not in {0, 1}
        )
        for check in baseline
    ):
        baseline_issues.append("BASE_RESULTS_UNSTRUCTURED")
    if not commands:
        selection_effect = "NOT_CONFIGURED"
    elif not baseline_issues:
        selection_effect = "DIFFERENTIAL_GATE"
    elif settings.repository_validation_failure_policy == "abstain":
        selection_effect = "ABSTAIN"
    else:
        selection_effect = "IGNORE_AND_CONTINUE"
    baseline_policy = {
        "schema_version": "contractfix-repository-baseline-policy/1",
        "comparable": bool(commands) and not baseline_issues,
        "issues": baseline_issues,
        "policy": settings.repository_validation_failure_policy,
        "selection_effect": selection_effect,
    }
    baseline_policy_path = artifacts.root / "baseline_policy.json"
    if baseline_policy_path.exists():
        if json.loads(baseline_policy_path.read_text(encoding="utf-8")) != baseline_policy:
            raise ValueError("repository baseline policy changed during resume")
    else:
        artifacts.save("baseline_policy.json", baseline_policy)
    if baseline_issues and settings.repository_validation_failure_policy == "abstain":
        result["reason"] = baseline_issues[0]
        result["repository_validation_status"] = "NOT_COMPARABLE"
        return result, ""
    selection_commands = commands
    selection_baseline = baseline
    if baseline_issues:
        selection_commands = []
        selection_baseline = []
        result["repository_validation_status"] = "IGNORED_NOT_COMPARABLE"
        logger.warning(
            "[repair] base-visible validation is not comparable; continuing patch "
            "generation under the configured ignore policy. Selection will not be "
            "labeled repository-regression-validated."
        )
    elif commands:
        logger.info(
            "[repair] base-visible validation captured comparable evidence; "
            "pre-existing failures are allowed and only newly introduced failures "
            "will reject a candidate"
        )
    if baseline:
        for number, receipt in enumerate(baseline, start=1):
            _show_command_receipt(f"Repository baseline check {number}", receipt)
    else:
        logger.warning(
            "[repair] repository regression checks: NOT_CONFIGURED; "
            "selection can use syntax and the frozen EC gate, but not repository-test evidence"
        )
    feedback = None
    seen = set()
    rows = result["candidates"]
    # Semantic repair attempts, source-context requests, and deterministic
    # edit-realization corrections have separate budgets. A malformed exact
    # replacement does not consume a semantic repair candidate unless its
    # bounded materialization-correction budget is exhausted.
    materialization_retry = 0
    max_patch_calls = (
        settings.max_patch_context_rounds
        + settings.max_patch_attempts * settings.max_patch_materialization_attempts
    )
    if settings.max_patch_model_calls is not None:
        max_patch_calls = min(max_patch_calls, settings.max_patch_model_calls)
    for call in range(1, max_patch_calls + 1):
        if result["patch_attempts"] >= settings.max_patch_attempts:
            break
        attempt = result["patch_attempts"] + 1
        packet = patch_packet(
            guidance,
            context,
            environment,
            mode,
            attempt=attempt,
            feedback=feedback,
            rounds_left=max(0, settings.max_patch_context_rounds - result["context_rounds"]),
        )
        input_path = artifacts.root / f"patch_calls/{call:03d}_input.json"
        response_path = artifacts.root / f"patch_calls/{call:03d}_response.json"
        previously_requested = input_path.exists()
        if previously_requested:
            previous_packet = json.loads(input_path.read_text(encoding="utf-8"))
            # Environment timing is not a model input; source, semantics and scope
            # are deterministically replayed from sealed guidance and saved actions.
            if previous_packet.get("attempt") != attempt:
                raise ValueError("repair call/attempt history is inconsistent")
        else:
            artifacts.save(f"patch_calls/{call:03d}_input.json", packet)
        artifacts.events.emit(
            "patch_call_replayed" if previously_requested else "patch_call_started",
            stage="patch",
            attempt=attempt,
            call=call,
        )
        result["patch_model_calls"] += 1
        try:
            if response_path.exists():
                response = PatchProposal.model_validate(json.loads(response_path.read_text(encoding="utf-8")))
            elif previously_requested:
                # A killed provider call may have been billed. Reserve its slot;
                # never replay it at the same attempt number or reset the budget.
                result["patch_attempts"] += 1
                validation = artifacts.root / f"candidates/{attempt:03d}/validation.json"
                row = (
                    json.loads(validation.read_text())
                    if validation.exists()
                    else {
                        "attempt": attempt,
                        "applicable": False,
                        "ordinary_pass": False,
                        "failure": "INTERRUPTED_MODEL_CALL",
                        "detail": "No complete saved artifact; attempt consumed",
                    }
                )
                rows.append(row)
                if not validation.exists():
                    artifacts.save(f"candidates/{attempt:03d}/validation.json", row)
                feedback = (
                    None if frozen_pool else
                    {"failure": row["failure"], "detail": row.get("detail")}
                )
                continue
            else:
                response = stages.generate("patch", PatchProposal, packet)
                response = PatchProposal.model_validate(
                    response.model_dump() if hasattr(response, "model_dump") else response
                )
        except (ValidationError, ValueError) as exc:
            result["patch_attempts"] += 1
            failure = {"failure": "SCHEMA_ERROR", "detail": str(exc)[:1500]}
            feedback = None if frozen_pool else failure
            rows.append(
                {
                    "attempt": attempt,
                    "applicable": False,
                    "ordinary_pass": False,
                    **failure,
                }
            )
            artifacts.save(f"candidates/{attempt:03d}/validation.json", rows[-1])
            continue
        except Exception as exc:
            # Transport/output-budget failures do not prove a hard repair. No
            # automatic HIGH escalation or full-pipeline restart is performed.
            result.update(reason="PATCH_MODEL_ERROR", error_type=type(exc).__name__, detail=str(exc)[:1500])
            break
        if not response_path.exists():
            artifacts.save(f"patch_calls/{call:03d}_response.json", response.model_dump())
        if response.action == "abstain":
            result.update(reason="PATCH_ABSTAINED", abstention_reason=response.reason)
            break
        if response.action == "request_context":
            if result["context_rounds"] >= settings.max_patch_context_rounds:
                result["reason"] = "PATCH_CONTEXT_BUDGET_EXHAUSTED"
                break
            result["context_rounds"] += 1
            receipt = context.request(response.context_requests, result["context_rounds"])
            if not (artifacts.root / f"context/round_{result['context_rounds']:02d}.json").exists():
                artifacts.save(f"context/round_{result['context_rounds']:02d}.json", receipt)
            _show_patch_context(result["context_rounds"], receipt)
            feedback = {"failure": "CONTEXT_REQUEST_RESULT", "outcomes": receipt["outcomes"]}
            if not receipt["progress"] and result["context_rounds"] >= settings.max_patch_context_rounds:
                feedback = {
                    "failure": "CONTEXT_EXHAUSTED",
                    "outcomes": receipt["outcomes"],
                    "next_action": "submit_edits_or_abstain",
                }
                logger.warning(
                    "[repair] context exhausted; requesting a final patch proposal "
                    "or explicit abstention"
                )
            continue
        proposal = response.model_dump()
        proposal_sha256 = digest(proposal)
        row = {
            "attempt": attempt,
            "proposal_sha256": proposal_sha256,
            "applicable": False,
            "ordinary_pass": False,
            "validation_assurance": "NOT_COMPARABLE",
            "ec_pass": None,
            "ec_validation_status": "NOT_EVALUATED",
            "infrastructure_error": None,
            "allowed_edit_paths": sorted(context.allowed),
            "context_rounds": result["context_rounds"],
        }

        # Exact edit realization is a deterministic transport/mechanics gate.
        # Correcting a bad anchor is not a new semantic repair candidate.
        try:
            patch = materialize_edits(
                base,
                [edit.materialized() for edit in response.edits],
                sorted(context.allowed),
            )
        except SemanticNoopError as exc:
            # A comment/whitespace/noqa-only change is a semantic repair failure,
            # not an edit-transport failure. Consume one semantic candidate and
            # ask for a behavior-changing repair next.
            result["patch_attempts"] += 1
            candidate_dir = artifacts.root / "candidates" / f"{attempt:03d}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            row.update(
                failure="SEMANTIC_NOOP",
                detail=str(exc)[:1500],
                materialization_attempts=materialization_retry + 1,
            )
            materialization_retry = 0
            if not (candidate_dir / "edits.json").exists():
                artifacts.save(f"candidates/{attempt:03d}/edits.json", proposal)
            validation_path = candidate_dir / "validation.json"
            if validation_path.exists():
                row = json.loads(validation_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, SyntaxError) as exc:
            materialization_retry += 1
            failure = (
                "SYNTAX_ERROR"
                if isinstance(exc, SyntaxError)
                else "EDIT_NOT_APPLICABLE"
            )
            detail = str(exc)[:1500]

            if materialization_retry < settings.max_patch_materialization_attempts:
                feedback = {
                    "failure": failure,
                    "detail": detail,
                    "previous_edits": proposal["edits"],
                    "materialization_attempt": materialization_retry,
                    "materialization_attempts_remaining": (
                        settings.max_patch_materialization_attempts
                        - materialization_retry
                    ),
                    "next_action": (
                        "Correct the deterministic edit realization using exact "
                        "current repository source while preserving the same intended "
                        "repair. If the required exact source is not supplied, request "
                        "that source before resubmitting."
                    ),
                }
                logger.info(
                    f"[patch {attempt}/{settings.max_patch_attempts}] "
                    f"{failure}; materialization retry "
                    f"{materialization_retry}/"
                    f"{settings.max_patch_materialization_attempts}"
                )
                continue

            # Only after the bounded realization budget is exhausted does this
            # become one consumed semantic repair candidate.
            result["patch_attempts"] += 1
            candidate_dir = artifacts.root / "candidates" / f"{attempt:03d}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            row.update(
                failure=failure,
                detail=detail,
                materialization_attempts=materialization_retry,
            )
            materialization_retry = 0

            if not (candidate_dir / "edits.json").exists():
                artifacts.save(f"candidates/{attempt:03d}/edits.json", proposal)
            validation_path = candidate_dir / "validation.json"
            if validation_path.exists():
                row = json.loads(validation_path.read_text(encoding="utf-8"))

        else:
            # A successfully materialized patch is one semantic repair candidate.
            result["patch_attempts"] += 1
            row["materialization_attempts"] = materialization_retry + 1
            materialization_retry = 0

            candidate_dir = artifacts.root / "candidates" / f"{attempt:03d}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            if not (candidate_dir / "edits.json").exists():
                artifacts.save(f"candidates/{attempt:03d}/edits.json", proposal)

            validation_path = candidate_dir / "validation.json"
            if validation_path.exists():
                row = json.loads(validation_path.read_text(encoding="utf-8"))
                if row.get("patch_sha256"):
                    from hashlib import sha256

                    if (
                        sha256((artifacts.root / row["patch_path"]).read_bytes()).hexdigest()
                        != row["patch_sha256"]
                    ):
                        raise ValueError("saved candidate patch changed before resume")
                    seen.add(row["patch_sha256"])
            else:
                patch_path = candidate_dir / "candidate.patch"
                patch_path.write_bytes(patch.text.encode())
                row.update(
                    applicable=True,
                    patch_sha256=patch.sha256,
                    changed_files=list(patch.files),
                    changed_lines=patch.changed_lines,
                    patch_path=str(patch_path.relative_to(artifacts.root)),
                )
                if patch.sha256 in seen:
                    row.update(
                        failure="NO_PROGRESS",
                        detail="Repeated identical patch against the same base",
                    )
                else:
                    seen.add(patch.sha256)
                    if not frozen_pool:
                        _validate_candidate(
                            row,
                            base=base,
                            artifacts=artifacts,
                            executor=executor,
                            commands=selection_commands,
                            baseline=selection_baseline,
                            baseline_policy=baseline_policy,
                            validation_plan=validation_plan,
                            settings=settings,
                            guidance=guidance,
                            mode=mode,
                            stages=stages,
                            import_module=(
                                import_module
                                if import_baseline["status"] == "BLOCKED"
                                else None
                            ),
                        )

        rows.append(row)
        if frozen_pool:
            # Generation completes before any candidate receives test/EC
            # execution. Persist terminal generation defects immediately; valid
            # materialized patches are validated together after the pool freezes.
            if row.get("failure") and not validation_path.exists():
                artifacts.save(f"candidates/{attempt:03d}/validation.json", row)
            feedback = None
            continue
        if not validation_path.exists():
            artifacts.save(f"candidates/{attempt:03d}/validation.json", row)
        if row.get("patch_path"):
            patch_text = (artifacts.root / row["patch_path"]).read_text(encoding="utf-8")
            _show_patch_candidate(attempt, patch_text, row, candidate_dir)
        eligible = select_patch([row], enforce_ec=mode == "EC_ENFORCED") is not None
        artifacts.events.emit(
            "patch_candidate_validated",
            stage="patch",
            attempt=attempt,
            eligible=eligible,
            failure=row.get("failure"),
        )
        logger.info(
            f"[patch {attempt}/{settings.max_patch_attempts}] {'eligible' if eligible else row.get('failure')}"
        )
        if eligible and settings.patch_candidate_policy == "first_eligible":
            break
        if row.get("failure") == "NO_PROGRESS":
            result["reason"] = "NO_PROGRESS"
            logger.warning(
                "[repair] identical patch proposed again; stopping this "
                "no-progress branch without spending another model call"
            )
            break
        ordinary_infrastructure = (row.get("ordinary_checks") or {}).get(
            "infrastructure_error"
        )
        if ordinary_infrastructure or row.get("failure") == "REPOSITORY_CHECKS_REQUIRED":
            result["reason"] = row.get("failure") or "INFRASTRUCTURE_ERROR"
            break
        # Previous edits and concrete failures are data, never a license to change NLC.
        feedback = {
            "failure": row.get("failure", "VALID_CANDIDATE"),
            "detail": row.get("detail"),
            "previous_edits": response.model_dump()["edits"],
        }
        if row.get("ordinary_checks") and not row["ordinary_pass"]:
            feedback["ordinary_checks"] = row["ordinary_checks"]
        if row.get("nlc_conformance") and row.get("nlc_pass") is False:
            feedback["nlc_conformance"] = row["nlc_conformance"]
        if row.get("ec_execution") and not row["ec_pass"]:
            observed = row["ec_execution"]
            feedback["contract_observation"] = {
                "outcome": observed["outcome"],
                "operation_reached": observed.get("contracted_operation_reached"),
                "all_assertions_exercised": observed.get("all_runtime_assertions_exercised"),
                "violations": observed.get("assertion_execution", {}).get("observed_violations", []),
                "detail": observed.get("detail"),
                "operation_observations": [
                    {key: event[key] for key in ("event", "exception_type", "detail") if key in event}
                    for event in observed.get("events", [])
                    if event.get("event") in {"contracted_operation_raised", "contracted_operation_returned"}
                ],
            }
        if row.get("repository_import", {}).get("status") == "BLOCKED":
            feedback["repository_import"] = {
                "module": import_module,
                "status": "BLOCKED",
                "log_tail": row["repository_import"]["execution"].get("log_tail", "")[-3000:],
            }
    if (
        result["patch_model_calls"] >= max_patch_calls
        and result["patch_attempts"] < settings.max_patch_attempts
        and result["reason"] == "PATCH_BUDGET_EXHAUSTED"
    ):
        result["reason"] = "PATCH_MODEL_CALL_BUDGET_EXHAUSTED"
    candidate_pool = {
        "schema_version": "contractfix-patch-candidate-pool/1",
        "policy": settings.patch_candidate_policy,
        "requested_candidates": settings.max_patch_attempts,
        "patch_model_call_limit": settings.max_patch_model_calls,
        "candidate_records": len(rows),
        "materialized_candidates": sum(bool(row.get("patch_sha256")) for row in rows),
        "validation_feedback_exposed": not frozen_pool,
        "candidates": [
            {
                "attempt": row.get("attempt"),
                "proposal_sha256": row.get("proposal_sha256"),
                "patch_sha256": row.get("patch_sha256"),
                "patch_path": row.get("patch_path"),
            }
            for row in rows
        ],
    }
    candidate_pool["sha256"] = digest(candidate_pool)
    pool_path = artifacts.root / "candidate_pool.json"
    if pool_path.exists():
        if json.loads(pool_path.read_text(encoding="utf-8")) != candidate_pool:
            raise ValueError("frozen candidate pool changed during resume")
    else:
        artifacts.save("candidate_pool.json", candidate_pool)
    artifacts.events.emit(
        "candidate_pool_frozen",
        stage="patch",
        sha256=candidate_pool["sha256"],
        candidate_records=len(rows),
        materialized_candidates=candidate_pool["materialized_candidates"],
        validation_feedback_exposed=candidate_pool["validation_feedback_exposed"],
    )

    if frozen_pool:
        logger.info(
            f"[repair] candidate pool frozen | records={len(rows)} | "
            f"materialized={candidate_pool['materialized_candidates']} | "
            f"sha256={candidate_pool['sha256']}"
        )
        for row in rows:
            if not row.get("patch_path"):
                continue
            candidate_dir = artifacts.root / "candidates" / f"{row['attempt']:03d}"
            if not row.get("failure") and not row.get("ordinary_checks"):
                _validate_candidate(
                    row,
                    base=base,
                    artifacts=artifacts,
                    executor=executor,
                    commands=selection_commands,
                    baseline=selection_baseline,
                    baseline_policy=baseline_policy,
                    validation_plan=validation_plan,
                    settings=settings,
                    guidance=guidance,
                    mode=mode,
                    stages=stages,
                    import_module=(
                        import_module if import_baseline["status"] == "BLOCKED" else None
                    ),
                )
                artifacts.save(
                    f"candidates/{row['attempt']:03d}/validation.json", row
                )
            patch_text = (artifacts.root / row["patch_path"]).read_text(encoding="utf-8")
            _show_patch_candidate(row["attempt"], patch_text, row, candidate_dir)
            eligible = select_patch([row], enforce_ec=mode == "EC_ENFORCED") is not None
            artifacts.events.emit(
                "patch_candidate_validated",
                stage="patch",
                attempt=row["attempt"],
                eligible=eligible,
                failure=row.get("failure"),
            )
            logger.info(
                f"[patch {row['attempt']}/{settings.max_patch_attempts}] "
                f"{'eligible' if eligible else row.get('failure')}"
            )
    # Verify all immutable inputs again, including EC hash and base contents.
    _, final_sha = load_guidance(guidance_dir)
    if final_sha != guidance_sha:
        raise ValueError("repair mutated its sealed guidance")
    result["allowed_edit_paths"] = sorted(context.allowed)
    result["edit_scope_expansions"] = context.expansions
    result["candidate_pool_sha256"] = candidate_pool["sha256"]
    v1_available = mode != "ISSUE_ONLY" and guidance.qualified_nlc is not None
    context_selected = select_patch(rows, enforce_ec=False) if v1_available else None
    paired_available = mode == "EC_ENFORCED" and guidance.qualified_ec is not None
    enforced_selected = select_patch(rows, enforce_ec=True) if paired_available else None
    refinement_selected = None
    if settings.patch_candidate_policy == "budgeted_ec_retry":
        if paired_available and enforced_selected is None:
            child, refinement = _refine_rejected_patch(
                rows=rows, pool_sha256=candidate_pool["sha256"], base=base,
                artifacts=artifacts, executor=executor, stages=stages,
                settings=settings, guidance=guidance, context=context,
                environment=environment, commands=selection_commands,
                baseline=selection_baseline, baseline_policy=baseline_policy,
                validation_plan=validation_plan,
                import_module=(
                    import_module if import_baseline["status"] == "BLOCKED" else None
                ),
            )
            result["patch_model_calls"] += refinement.get("model_calls", 0)
            if child is not None:
                rows.append(child)
                refinement_selected = select_patch([child], enforce_ec=True)
                if child.get("patch_path"):
                    _show_patch_candidate(
                        child["attempt"],
                        (artifacts.root / child["patch_path"]).read_text(encoding="utf-8"),
                        child,
                        artifacts.root / "candidates" / f"{child['attempt']:03d}",
                    )
        else:
            refinement = {
                "schema_version": "contractfix-ec-refinement/1",
                "initial_candidate_pool_sha256": candidate_pool["sha256"],
                "status": (
                    "NOT_TRIGGERED_V2_SELECTED" if enforced_selected is not None
                    else "NOT_APPLICABLE_NO_QUALIFIED_EC"
                ),
                "model_calls": 0,
            }
        refinement["sha256"] = digest(refinement)
        lineage_path = artifacts.root / "refinement/lineage.json"
        if lineage_path.exists():
            if json.loads(lineage_path.read_text(encoding="utf-8")) != refinement:
                raise ValueError("EC refinement lineage changed during resume")
        else:
            artifacts.save("refinement/lineage.json", refinement)
        result["ec_refinement"] = refinement
        result["refinement_lineage_sha256"] = refinement["sha256"]
    strict_selected = enforced_selected or refinement_selected
    reviewed_fallback = None
    if (
        mode == "EC_ENFORCED"
        and strict_selected is None
        and settings.ec_conflict_policy == "fallback_after_nlc_review"
        and v1_available
    ):
        for row in rank_patches(rows, enforce_ec=False):
            _review_patch_nlc(
                row, artifacts=artifacts, guidance=guidance, stages=stages,
                checks=row["ordinary_checks"], review_context="EC_CONFLICT_FALLBACK",
            )
            artifacts.save(f"candidates/{row['attempt']:03d}/validation.json", row)
            if row["nlc_pass"]:
                reviewed_fallback = row
                break
    if mode == "EC_ENFORCED":
        # Keep V2 and V3 strict for measurement. The overall system can still
        # return an ordinarily eligible attempt when the frozen EC rejects all
        # candidates; identify that conflict instead of certifying EC success.
        overall_selected = strict_selected or (
            context_selected
            if settings.ec_conflict_policy == "fallback_to_nlc_context"
            else reviewed_fallback
        )
        if refinement_selected is not None:
            overall_mode = "EC_REFINED"
            overall_reason = "V3_REFINEMENT_SELECTED"
        elif enforced_selected is not None:
            overall_mode = "EC_ENFORCED"
            overall_reason = "V2_SELECTED"
        else:
            overall_mode = (
                "EC_CONFLICT_FALLBACK" if overall_selected is not None
                else "EC_ENFORCED_NO_SELECTION"
            )
            ec_states = {
                row.get("ec_validation_status") for row in rows
                if row.get("ordinary_pass") and row.get("applicable")
            }
            if "EC_EXECUTION_ERROR" in ec_states:
                overall_reason = "V2_EC_EXECUTION_UNAVAILABLE"
            elif "EC_NOT_COMPARABLE" in ec_states:
                overall_reason = "V2_EC_NOT_COMPARABLE"
            else:
                overall_reason = "ALL_V2_CANDIDATES_EC_REJECTED"
    elif context_selected is not None:
        overall_selected = context_selected
        overall_mode = "NLC_CONTEXT_FALLBACK"
        overall_reason = "V1_SELECTED"
    else:
        overall_selected = select_patch(rows, enforce_ec=False)
        overall_mode = "ISSUE_CONTEXT_FALLBACK"
        overall_reason = "V0_SELECTED" if overall_selected else "NO_ELIGIBLE_CANDIDATE"
    if overall_selected is None and settings.submit_best_attempt:
        overall_selected = _human_review_candidate(rows)
        if overall_selected is not None:
            overall_mode = "UNVALIDATED_BEST_ATTEMPT"
            overall_reason = "NO_CANDIDATE_PASSED_DECLARED_GATES"
            checks = overall_selected.get("ordinary_checks") or {}
            result["best_attempt"] = {
                "schema_version": "contractfix-best-attempt/1",
                "attempt": overall_selected["attempt"],
                "patch_sha256": overall_selected.get("patch_sha256"),
                "status": "SUBMITTED_WITH_FAILED_OR_MISSING_GATES",
                "failure": overall_selected.get("failure"),
                "ordinary_pass": overall_selected.get("ordinary_pass") is True,
                "repository_check_status": checks.get("repository_check_status"),
                "validation_assurance": overall_selected.get("validation_assurance"),
                "introduced_visible_failures": checks.get("introduced_failures") or [],
                "ec_validation_status": overall_selected.get("ec_validation_status"),
                "nlc_conformance_status": overall_selected.get("nlc_conformance_status"),
                "strict_ec_selection": False,
                "submitted_for_official_evaluation": True,
            }
            artifacts.save("best_attempt.json", result["best_attempt"])
    result["paired_selections"] = {
        "V1_NLC_CONTEXT": {
            "selected_attempt": context_selected.get("attempt") if context_selected else None,
            "patch_sha256": context_selected.get("patch_sha256") if context_selected else None,
            "available": v1_available,
        },
        "V2_EC_ENFORCED": {
            "selected_attempt": enforced_selected.get("attempt") if enforced_selected else None,
            "patch_sha256": enforced_selected.get("patch_sha256") if enforced_selected else None,
            "available": paired_available,
        },
        "CONTRACTFIX_OVERALL": {
            "selected_attempt": overall_selected.get("attempt") if overall_selected else None,
            "patch_sha256": overall_selected.get("patch_sha256") if overall_selected else None,
            "available": overall_selected is not None,
            "selection_mode": overall_mode,
            "reason": overall_reason,
            "strict_v2": overall_mode in {"EC_ENFORCED", "EC_REFINED"},
        },
        "common_candidate_pool_sha256": candidate_pool["sha256"],
        "refinement_lineage_sha256": result.get("refinement_lineage_sha256"),
        "common_ranking": "comparable_checks_repaired_failures_import_readiness_attempt",
        "selector_version": EVIDENCE_SELECTOR,
        "ec_conflict_policy": settings.ec_conflict_policy,
    }
    paired_predictions = {
        "schema_version": "contractfix-paired-repair-selections/1",
        "selector_version": EVIDENCE_SELECTOR,
        "instance_id": guidance.task["instance_id"],
        "requested_variant": variant,
        "candidate_pool_sha256": candidate_pool["sha256"],
        "selections": {},
    }
    if settings.patch_candidate_policy == "budgeted_ec_retry":
        result["paired_selections"]["V3_EC_RETRY"] = {
            "selected_attempt": strict_selected.get("attempt") if strict_selected else None,
            "patch_sha256": strict_selected.get("patch_sha256") if strict_selected else None,
            "available": True,
        }
    selection_rows = [
        ("V1_NLC_CONTEXT", context_selected),
        ("V2_EC_ENFORCED", enforced_selected),
    ]
    if settings.patch_candidate_policy == "budgeted_ec_retry":
        selection_rows.append(("V3_EC_RETRY", strict_selected))
    selection_rows.append(("CONTRACTFIX_OVERALL", overall_selected))
    for name, row in selection_rows:
        patch_text = (
            (artifacts.root / row["patch_path"]).read_text(encoding="utf-8")
            if row
            else ""
        )
        paired_predictions["selections"][name] = {
            "available": result["paired_selections"][name]["available"],
            "selected_attempt": row.get("attempt") if row else None,
            "patch_sha256": row.get("patch_sha256") if row else None,
            "model_patch": patch_text,
        }
    artifacts.save("paired_selections.json", paired_predictions)
    selected = overall_selected
    _save_patch_certificates(artifacts, guidance, rows, selected, overall_mode)
    if selected:
        text = (artifacts.root / selected["patch_path"]).read_bytes().decode("utf-8")
        if overall_mode == "EC_CONFLICT_FALLBACK":
            result["ec_conflict_fallback"] = {
                "schema_version": "contractfix-ec-conflict-fallback/1",
                "reason": overall_reason,
                "selected_attempt": selected["attempt"],
                "selected_patch_sha256": selected["patch_sha256"],
                "ec_validation_status": selected.get("ec_validation_status"),
                "ordinary_pass": selected.get("ordinary_pass") is True,
                "nlc_conformance_status": selected.get("nlc_conformance_status"),
                "strict_ec_selection": None,
            }
        selected_checks = selected.get("ordinary_checks") or {}
        selected_repository_status = selected_checks.get(
            "repository_check_status"
        )
        result.update(
            status="PATCH_SELECTED",
            reason=(
                overall_reason if overall_mode in {"EC_CONFLICT_FALLBACK", "UNVALIDATED_BEST_ATTEMPT"}
                else "SELECTED_BY_DECLARED_GATES"
            ),
            selection_mode=overall_mode,
            strict_v2=overall_mode in {"EC_ENFORCED", "EC_REFINED"},
            selected_attempt=selected["attempt"],
            selected_patch="selected.patch",
            selected_patch_sha256=selected["patch_sha256"],
            validation_scope=selected_checks.get("validation_scope", "NOT_RUN"),
            repository_check_status=selected_repository_status,
            repository_validation_status=(
                "IGNORED_NOT_COMPARABLE"
                if selected_repository_status == "IGNORED_NOT_COMPARABLE"
                else result["repository_validation_status"]
            ),
            introduced_failures=selected_checks.get("introduced_failures", []),
            repaired_visible_failures=selected_checks.get(
                "repaired_visible_failures", []
            ),
            persistent_failures=selected_checks.get("persistent_failures", []),
            ec_pass=selected["ec_pass"],
            repository_import_status=(
                selected["repository_import"]["status"]
                if selected.get("repository_import")
                else "NOT_CHECKED_CANDIDATE"
            ),
        )
        if result["repository_import_status"] == "BLOCKED":
            logger.warning(
                "[repair] selected patch still has a plain repository-import blocker; "
                "the frozen EC result does not prove official evaluator tests can collect"
            )
        logger.panel(
            "Best attempt submitted · unvalidated"
            if overall_mode == "UNVALIDATED_BEST_ATTEMPT" else "Patch selected",
            f"**Attempt:** {selected['attempt']}  \n"
            f"**Changed files:** {', '.join(selected.get('changed_files', [])) or 'none'}  \n"
            f"**Changed lines:** {selected.get('changed_lines', 0)}  \n"
            f"**Validation:** {selected_checks.get('validation_scope', 'NOT_RUN')}  \n"
            f"**Repository checks:** {selected_checks.get('repository_check_status', 'NOT_CONFIGURED')}  \n"
            f"**Introduced visible failures:** {len(selected_checks.get('introduced_failures', []))}  \n"
            f"**Repaired visible failures:** {len(selected_checks.get('repaired_visible_failures', []))}  \n"
            f"**Plain repository import:** {result['repository_import_status']} (advisory)  \n"
            f"**Frozen EC gate:** {selected.get('ec_validation_status', 'NOT_REQUIRED')}  \n"
            f"**SHA-256:** `{selected['patch_sha256']}`",
            style="yellow" if overall_mode == "UNVALIDATED_BEST_ATTEMPT" else "green",
        )
        return result, text
    if mode == "EC_ENFORCED":
        # Preserve an inspectable candidate without converting a failed strict
        # gate into an automatic benchmark prediction.
        diagnostic = select_patch(rows, enforce_ec=False)
        if diagnostic is not None:
            review = {
                "status": "NOT_SELECTED_FOR_STRICT_EC",
                "reason": overall_reason,
                "attempt": diagnostic["attempt"],
                "patch_path": diagnostic["patch_path"],
                "patch_sha256": diagnostic["patch_sha256"],
                "ec_validation_status": diagnostic.get("ec_validation_status"),
                "validation_assurance": diagnostic.get("validation_assurance"),
                "repository_check_status": (
                    diagnostic.get("ordinary_checks") or {}
                ).get("repository_check_status"),
            }
            result["diagnostic_candidate"] = review
            artifacts.save("diagnostic_candidate.json", review)
            logger.info(
                f"[repair] no strict EC selection; reviewable candidate "
                f"attempt={diagnostic['attempt']} saved at {review['patch_path']}"
            )
    candidate = _human_review_candidate(rows)
    if candidate is not None:
        patch_path = artifacts.root / candidate["patch_path"]
        (artifacts.root / "human_review.patch").write_bytes(patch_path.read_bytes())
        result["human_review"] = {
            "status": "MATERIALIZED_NOT_AUTOMATICALLY_SELECTED",
            "attempt": candidate["attempt"],
            "patch_path": "human_review.patch",
            "patch_sha256": candidate.get("patch_sha256"),
            "failure": candidate.get("failure"),
            "ordinary_pass": candidate.get("ordinary_pass") is True,
            "ec_validation_status": candidate.get("ec_validation_status"),
            "nlc_conformance_status": candidate.get("nlc_conformance_status"),
            "automatic_selection_eligible": False,
        }
    else:
        proposals = sorted((artifacts.root / "patch_calls").glob("*_response.json"))
        result["human_review"] = {
            "status": "NO_MATERIALIZED_PATCH",
            "last_proposal_path": str(proposals[-1].relative_to(artifacts.root)) if proposals else None,
            "automatic_selection_eligible": False,
        }
    artifacts.save("human_review.json", result["human_review"])
    return result, ""


def _finish(artifacts, result, text, stages, before, started):
    if text:
        (artifacts.root / "selected.patch").write_bytes(text.encode())
    model = stages.identity.get("settings", {}).get("model", stages.identity.get("kind", "contractfix"))
    prediction = {"instance_id": result.get("instance_id"), "model_name_or_path": model, "model_patch": text}
    artifacts.save("prediction.json", prediction)
    artifacts.save("predictions_for_swebench.json", [prediction])
    after = model_accounting(stages)
    result.update(
        total_elapsed_seconds=perf_counter() - started,
        model_accounting=after,
        patch_accounting=_accounting_delta(before, after),
    )
    artifacts.save(
        "selection.json",
        {
            key: result.get(key)
            for key in (
                "status",
                "reason",
                "selected_attempt",
                "selected_patch_sha256",
                "requested_variant",
                "effective_mode",
                "effective_repair_mode",
                "ec_status",
                "selection_policy",
                "selector_version",
                "selection_mode",
                "strict_v2",
                "candidate_policy",
                "ec_conflict_policy",
                "validation_scope",
                "repository_check_status",
                "repository_validation_status",
                "introduced_failures",
                "repaired_visible_failures",
                "persistent_failures",
                "ec_pass",
                "repository_import_baseline_status",
                "repository_import_status",
                "candidate_pool_sha256",
                "paired_selections",
                "diagnostic_candidate",
                "ec_conflict_fallback",
            )
        },
    )
    artifacts.save("summary.json", result)
    artifacts.save("status.json", result)
    (artifacts.root / "report.md").write_text(
        "# Repair result\n\n"
        + "\n".join(
            f"- **{key}:** `{result.get(key)}`"
            for key in (
                "instance_id",
                "status",
                "requested_variant",
                "effective_mode",
                "effective_repair_mode",
                "ec_status",
                "selection_mode",
                "strict_v2",
                "reason",
                "patch_attempts",
                "context_rounds",
                "selected_patch_sha256",
                "validation_scope",
                "repository_check_status",
                "repository_validation_status",
                "introduced_failures",
                "repaired_visible_failures",
                "persistent_failures",
                "candidate_pool_sha256",
            )
        )
        + "\n\nPATCH_SELECTED means generation-time gates passed, not independently verified repair correctness.\n",
        encoding="utf-8",
    )
    artifacts.events.emit(
        "repair_finished", stage="patch", status=result["status"], reason=result.get("reason")
    )
    if text:
        logger.success(f"[repair] selected diff: {artifacts.root / 'selected.patch'}")
    logger.info(f"[repair] complete receipt: {artifacts.root / 'report.md'}")
