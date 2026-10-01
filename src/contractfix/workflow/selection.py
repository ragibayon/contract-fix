"""Evaluator-blind decisions over a frozen pool of patch validation receipts.

The EC can rank candidates only when a reached, exercised assertion provides
evidence. Probe and equivalence evidence is deliberately explicit: absence of
an adapter or a domain certificate is UNKNOWN, never inferred equivalence.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .languages.python.patching import EVALUATOR_ONLY_FIELDS, rank_patches

SELECTOR_VERSION = "ec-decision/1"


def ec_observation(row: Mapping[str, Any]) -> str:
    """Classify one frozen assertion execution without conflating host failures."""
    observation = row.get("ec_execution") or {}
    if row.get("infrastructure_error") or observation.get("infrastructure_error"):
        return "INFRASTRUCTURE_ERROR"
    if not observation:
        return "UNKNOWN"
    if observation.get("outcome") == "ERROR":
        return "EXECUTION_ERROR"
    if observation.get("evaluation_harness_entered") is not True:
        return "INVALID_FIXTURE"
    if observation.get("contracted_operation_reached") is not True:
        return "UNREACHED_OPERATION"
    assertions = observation.get("assertion_execution") or {}
    if assertions.get("all_executed") is not True:
        return "UNEXERCISED_ASSERTION"
    if observation.get("outcome") == "SATISFIED" and row.get("ec_pass") is True:
        return "ACCEPTED"
    if observation.get("outcome") == "VIOLATED" and assertions.get("observed_violations"):
        return "ASSERTION_VIOLATION"
    return "UNKNOWN"


def _safe_candidates(candidates: Sequence[dict]) -> None:
    for row in candidates:
        if EVALUATOR_ONLY_FIELDS.intersection(row) or EVALUATOR_ONLY_FIELDS.intersection(
            row.get("ordinary_checks") or {}
        ):
            raise ValueError("official evaluator fields cannot enter patch selection")


def _groups(rows: Sequence[dict]) -> list[list[int]]:
    """Only identical patch bytes on one frozen base establish equivalence here."""
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        patch_hash = row.get("patch_sha256")
        key = f"sha256:{patch_hash}" if patch_hash else f"attempt:{row['attempt']}"
        groups[key].append(row["attempt"])
    return list(groups.values())


def frozen_witness_probes(rows: Sequence[dict], *, limit: int = 16) -> list[dict]:
    """Reuse only comparable executions of the same frozen EC witness."""
    if limit < 0:
        raise ValueError("probe budget must be nonnegative")
    probes = []
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            if len(probes) >= limit:
                return probes
            a, b = left.get("ec_execution") or {}, right.get("ec_execution") or {}
            same_contract = bool(a.get("executable_contract_sha256")) and (
                a.get("executable_contract_sha256") == b.get("executable_contract_sha256")
            )
            left_status, right_status = ec_observation(left), ec_observation(right)
            comparable = same_contract and left_status in {"ACCEPTED", "ASSERTION_VIOLATION"} and (
                right_status in {"ACCEPTED", "ASSERTION_VIOLATION"}
            )
            if comparable:
                probes.append({
                    "left_attempt": left["attempt"],
                    "right_attempt": right["attempt"],
                    "different": left_status != right_status,
                    "ec_applies": True,
                    "domain": "frozen EC witness and assertion inputs",
                })
    return probes


def select_with_ec(
    candidates: Sequence[dict], *,
    probe_evidence: Sequence[dict] | None = None,
    max_probes: int = 16,
) -> dict:
    """Return a selection and auditable EC scope from one frozen candidate pool.

    Probe records are produced by a language adapter, never by this function.
    A probe can expose a difference but cannot make an EC obligation apply by
    itself. The current rule keeps the NLC fallback when the EC rejects all.
    """
    if max_probes < 0 or (probe_evidence is not None and len(probe_evidence) > max_probes):
        raise ValueError("probe budget exceeded")
    _safe_candidates(candidates)
    ranked = rank_patches(list(candidates), enforce_ec=False)
    if probe_evidence is None:
        probe_evidence = frozen_witness_probes(ranked, limit=max_probes)
    baseline = ranked[0]["attempt"] if ranked else None
    statuses = {row["attempt"]: ec_observation(row) for row in ranked}
    accepted = [row for row in ranked if statuses[row["attempt"]] == "ACCEPTED"]
    rejected = [row for row in ranked if statuses[row["attempt"]] == "ASSERTION_VIOLATION"]
    differences = []
    for probe in probe_evidence:
        if probe.get("different") is not True:
            continue
        left, right = probe.get("left_attempt"), probe.get("right_attempt")
        if left not in statuses or right not in statuses:
            raise ValueError("probe references an ineligible candidate")
        differences.append({
            "left_attempt": left,
            "right_attempt": right,
            "domain": probe.get("domain"),
            "ec_applies": probe.get("ec_applies") is True,
        })
    conflict = any(
        item["ec_applies"]
        and statuses[item["left_attempt"]] == "ACCEPTED"
        and statuses[item["right_attempt"]] == "ACCEPTED"
        for item in differences
    )
    if conflict:
        classification = "EC_CONFLICT"
        selected = baseline
    elif accepted and rejected:
        classification = "MIXED"
        selected = accepted[0]["attempt"]
    elif accepted and len(accepted) == len(ranked):
        classification = "ACCEPTS_ALL"
        selected = baseline
    elif rejected and len(rejected) == len(ranked):
        classification = "REJECTS_ALL"
        selected = baseline
    else:
        classification = "UNKNOWN"
        selected = baseline
    return {
        "version": SELECTOR_VERSION,
        "selected_attempt": selected,
        "fallback_attempt": baseline,
        "classification": classification,
        "eligible_attempts": [row["attempt"] for row in ranked],
        "ec_statuses": statuses,
        "equivalence_domain": "identical patch bytes on the same frozen base",
        "equivalence_groups": _groups(ranked),
        "differences": differences,
        "probe_count": len(probe_evidence),
        "symbolic_patch_coverage": "UNSUPPORTED",
        "selection_changed": selected != baseline,
    }
