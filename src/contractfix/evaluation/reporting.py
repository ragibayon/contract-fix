"""Manifest-denominator reports. Unreported outcomes remain unknown, never disappear."""

from __future__ import annotations
import math
from collections import Counter


def wilson(successes: int, n: int, z: float = 1.959963984540054):
    if not n:
        return None
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0, center - half), min(1, center + half)]


def summarize_outcomes(manifest: dict, outcomes: list[dict]):
    ids = manifest["task_ids"]
    allowed = set(ids)
    by_id = {}
    valid = {"resolved", "unresolved", "error", "timeout", "no_patch", "not_run"}
    for row in outcomes:
        iid = row["instance_id"]
        if iid not in allowed:
            raise ValueError("out-of-manifest outcome: " + iid)
        if iid in by_id:
            raise ValueError("duplicate outcome: " + iid)
        if row.get("status") not in valid:
            raise ValueError("invalid outcome status")
        by_id[iid] = row
    counts = Counter(by_id.get(i, {"status": "not_run"})["status"] for i in ids)
    n = len(ids)
    solved = counts["resolved"]
    complete = counts["not_run"] == 0
    return {
        "manifest_sha256": manifest["sha256"],
        "denominator": n,
        "counts": dict(counts),
        "resolved_fraction_over_manifest": solved / n if n else None,
        "resolution_rate_status": "complete" if complete else "observed_lower_bound_incomplete_campaign",
        "resolution_rate_wilson95": wilson(solved, n) if complete else None,
        "not_run_ids": [i for i in ids if i not in by_id or by_id[i]["status"] == "not_run"],
        "assurance": "official_test_resolution_is_not_semantic_equivalence",
    }


def outcomes_from_swebench_report(manifest: dict, report: dict) -> list[dict]:
    # Explicit support for standard aggregate report ID lists; reject a guessed format.
    fields = {
        "resolved_ids": "resolved",
        "unresolved_ids": "unresolved",
        "error_ids": "error",
        "empty_patch_ids": "no_patch",
    }
    if not any(k in report for k in fields):
        raise ValueError("unrecognized SWE-bench aggregate report")
    seen = {}
    allowed = set(manifest["task_ids"])
    for field, status in fields.items():
        values = report.get(field, [])
        if not isinstance(values, list):
            raise ValueError("expected an ID list: " + field)
        for iid in values:
            if iid not in allowed:
                raise ValueError("report contains a task outside the frozen manifest")
            if iid in seen and seen[iid] != status:
                raise ValueError("overlapping outcome classes in report")
            seen[iid] = status
    return [{"instance_id": i, "status": s} for i, s in seen.items()]


def compare_paired(manifest, left, right):
    """Report paired wins/losses only when all tasks have terminal outcomes in each arm."""
    left_summary = summarize_outcomes(manifest, left)
    right_summary = summarize_outcomes(manifest, right)
    if left_summary["not_run_ids"] or right_summary["not_run_ids"]:
        raise ValueError("paired comparison requires complete arms")
    a = {x["instance_id"]: x["status"] == "resolved" for x in left}
    b = {x["instance_id"]: x["status"] == "resolved" for x in right}
    cells = Counter((a[i], b[i]) for i in manifest["task_ids"])
    return {
        "both_resolved": cells[(True, True)],
        "left_only": cells[(True, False)],
        "right_only": cells[(False, True)],
        "neither": cells[(False, False)],
    }
