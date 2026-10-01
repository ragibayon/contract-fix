"""Stable human-readable reports derived from structured run receipts."""

from __future__ import annotations

from pathlib import Path


def write_report(root: Path, frozen: dict) -> None:
    obligation = frozen["obligation"]
    candidates = frozen["candidates"]
    lines = [
        f"# ContractFix report: {frozen['task']['instance_id']}",
        "",
        f"- Status: `{frozen['qualification']['state']}`",
        f"- Artifact: `{frozen['version']}`",
        f"- Frozen SHA-256: `{frozen['sha256']}`",
        "- Contracted operation: "
        f"`{frozen['contracted_operation']['file']}:{frozen['contracted_operation']['symbol']}`",
        f"- Model calls: `{frozen['model_accounting']['calls']}`",
        f"- Recorded API cost: `${frozen['model_accounting']['provider_reported_cost_usd'] or 0:.8f}`",
        f"- Cost accounting: `{frozen['model_accounting'].get('cost_assurance', 'unreported')}`",
        f"- NLC review mode: `{frozen['qualification']['nlc_review']['mode']}`",
        f"- NLC review outcome: `{frozen['qualification']['nlc_review']['outcome']}`",
        f"- NLC/EC disagreement: `{frozen['qualification'].get('nlc_review_ec_disagreement', False)}`",
        "",
        "## Grounded REPAIR obligation",
        "",
        f"**Precondition:** {obligation['precondition']}",
        "",
        f"**Normal postcondition:** {obligation.get('normal_postcondition') or 'not specified'}",
        "",
        f"**Exceptional postcondition:** {obligation.get('exceptional_postcondition') or 'not specified'}",
        "",
        f"**Dynamic violation criterion:** {obligation['violation_criterion']['description']}",
        "",
        "### Evidence",
        "",
    ]
    lines.extend(f"- `{item['id']}` ({item['supports']}): {item['quote']}" for item in obligation["evidence"])
    lines += [
        "",
        "## Candidate results",
        "",
        "| Candidate | Outcome | Contracted operation | Runtime assertions | Runtime gate | Conformance gate | Review | Role |",
        "|---|---|---:|---:|---|---|---|---|",
    ]
    for candidate in candidates:
        execution = candidate["execution"]
        lines.append(
            "| {id} | {outcome} | {operation} | {assertions} | {runtime_gate} | {conformance_gate} | {conformance} | {role} |".format(
                id=candidate["id"],
                outcome=execution["outcome"],
                operation=("yes" if execution["contracted_operation_reached"] else "no"),
                assertions=("yes" if execution["runtime_assertions_exercised"] else "no"),
                runtime_gate=("pass" if not candidate["execution_failures"] else "fail"),
                conformance_gate=(
                    "skipped"
                    if candidate["conformance_status"] == "NOT_RUN_RUNTIME_FAILURE"
                    else (
                        "pass"
                        if candidate["conformance_status"] == "QUALIFIED"
                        else "fail"
                    )
                ),
                conformance=candidate["conformance_status"],
                role=candidate["selection_role"],
            )
        )
    lines += [
        "",
        "## Primary executable contract",
        "",
        "```python",
        next(item["source"] for item in candidates if item["selection_role"] == "PRIMARY").rstrip(),
        "```",
        "",
        "## Gold",
        "",
        "Gold is intentionally absent from qualification. See a separate post-freeze diagnostic.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_failure_report(root: Path, task_id: str, status: dict) -> None:
    lines = [
        f"# ContractFix report: {task_id}",
        "",
        f"- Status: `{status['status']}`",
        "",
        "## Diagnostic",
        "",
        "```json",
        __import__("json").dumps(status, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_gold_report(root: Path, result: dict) -> None:
    """Render the post-freeze diagnostic without changing qualification."""
    lines = [
        f"# ContractFix gold diagnostic: {result['instance_id']}",
        "",
        f"- Result: `{result['diagnostic']}`",
        f"- Frozen SHA-256: `{result['frozen_sha256']}`",
        f"- Executable-contract SHA-256: `{result['executable_contract_sha256']}`",
        f"- Same frozen executable contract: `{result['same_frozen_executable_contract']}`",
        f"- Qualification modified: `{result['qualification_modified']}`",
        f"- Model calls: `{result['model_calls']}`",
        "",
        "The gold patch was consulted only after the contract was frozen.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_gold_campaign_report(root: Path, summary: dict) -> None:
    """Render the latest durable campaign checkpoint for overnight inspection."""
    lines = [
        "# ContractFix post-freeze evaluation",
        "",
        f"- Attempted: `{summary['attempted']}/{summary['denominator']}`",
        f"- Remaining: `{summary['not_run']}`",
        f"- Status counts: `{summary['status_counts']}`",
        f"- Diagnostic counts: `{summary['diagnostic_counts']}`",
        "",
        "| Instance | Status | Diagnostic | Detail |",
        "|---|---|---|---|",
    ]
    for row in summary["outcomes"]:
        detail = str(row.get("detail") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['instance_id']} | {row['status']} | {row.get('diagnostic') or ''} | {detail} |")
    lines.append("")
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
