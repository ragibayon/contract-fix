"""Gold-patch evaluation: this module has no model, prompt, or generation dependency."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from contractfix.contracts.execution import Executor
from contractfix.utils.artifacts import RunArtifacts
from contractfix.utils.logger import logger
from ...frozen import load_frozen
from .repository import production_path
from .executable_contracts import evaluate_executable_contract


def _gold_paths(patch: bytes) -> list[str]:
    result = subprocess.run(["git", "apply", "--numstat", "-"], input=patch, capture_output=True, timeout=30)
    if result.returncode:
        raise ValueError("gold patch could not be parsed by git apply")
    paths = []
    for line in result.stdout.decode().splitlines():
        added, removed, name = line.split("\t", 2)
        if added == "-" or removed == "-" or not production_path(name):
            raise ValueError("v1 gold evaluation supports production Python text patches only")
        paths.append(name)
    if not paths:
        raise ValueError("gold patch is empty")
    return sorted(set(paths))


def read_gold_record(path: Path, task: dict) -> str:
    """Load an evaluator-side JSON/JSONL record and validate its task identity."""
    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
        rows = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    matches = [row for row in rows if row.get("instance_id") == task["instance_id"]]
    if len(matches) != 1:
        raise ValueError("gold record must identify exactly one matching instance")
    row = matches[0]
    for field in ("repo", "base_commit"):
        if row.get(field) != task[field]:
            raise ValueError("gold record identity mismatch: " + field)
    if not isinstance(row.get("patch"), str) or not row["patch"].strip():
        raise ValueError("gold record has no nonempty patch")
    # test_patch, FAIL_TO_PASS, PASS_TO_PASS are deliberately never consumed here.
    return row["patch"]


def evaluate_gold(
    frozen_dir: Path,
    output: Path,
    executor: Executor,
    *,
    gold_patch: Path | None = None,
    gold_record: Path | None = None,
) -> dict:
    """Evaluate a current frozen contract without exposing gold during generation."""
    frozen = load_frozen(frozen_dir)
    if (gold_patch is None) == (gold_record is None):
        raise ValueError("provide exactly one evaluator-side gold patch or gold record")
    artifacts = RunArtifacts(
        output,
        {
            "purpose": "offline_gold_contract_quality",
            "frozen_sha256": frozen["sha256"],
            "task_id": frozen["task"]["instance_id"],
        },
    )
    logger.attach(artifacts.root)
    logger.banner(
        "ContractFix · Post-freeze gold diagnostic",
        f"**Task:** \u0060{frozen['task']['instance_id']}\u0060  \\n**Frozen:** \u0060{frozen['sha256']}\u0060",
    )
    return _evaluate_python_gold(
        frozen_dir,
        frozen,
        artifacts,
        executor,
        gold_patch=gold_patch,
        gold_record=gold_record,
    )
def _evaluate_python_gold(
    frozen_dir: Path,
    frozen: dict,
    artifacts: RunArtifacts,
    executor: Executor,
    *,
    gold_patch: Path | None,
    gold_record: Path | None,
) -> dict:
    """Evaluate the byte-identical frozen executable contract without revising it."""
    if executor.identity()["sha256"] != frozen["executor"]["sha256"]:
        raise ValueError("gold evaluation executor differs from frozen generation environment")
    try:
        text = (
            read_gold_record(gold_record, frozen["task"])
            if gold_record
            else gold_patch.read_text(encoding="utf-8")
        )
        patch = artifacts.root / "gold.patch"
        patch.write_text(text, encoding="utf-8")
        _gold_paths(text.encode())
        primary = next(item for item in frozen["candidates"] if item["id"] == frozen["primary_candidate_id"])
        contract_sha = hashlib.sha256(primary["source"].encode()).hexdigest()
        if contract_sha != primary["execution"]["executable_contract_sha256"]:
            raise ValueError("frozen primary executable contract digest mismatch")
        report = evaluate_executable_contract(
            frozen_dir / "base",
            primary["source"],
            frozen["contracted_operation"],
            executor,
            patch=patch,
            entrypoint="contractfix_contract",
        )
        passed = (
            report["outcome"] == "SATISFIED"
            and report["contracted_operation_reached"]
            and not report["infrastructure_error"]
        )
        result = {
            "status": "EVALUATED",
            "diagnostic": "GOLD_PASS" if passed else "GOLD_FAIL",
            "instance_id": frozen["task"]["instance_id"],
            "frozen_sha256": frozen["sha256"],
            "executable_contract_sha256": contract_sha,
            "same_frozen_executable_contract": True,
            "pre_gold_qualification": frozen["qualification"],
            "qualification_modified": False,
            "gold": report,
            "model_calls": 0,
            "gold_patch_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        artifacts.save("diagnostic.json", result)
        (artifacts.root / "execution.log").write_text(report["command_log_tail"], encoding="utf-8")
        artifacts.save(
            "status.json",
            {
                "status": result["status"],
                "diagnostic": result["diagnostic"],
                "instance_id": result["instance_id"],
            },
        )
        from ...reporting import write_gold_report

        write_gold_report(artifacts.root, result)
        logger.panel(
            "Gold diagnostic",
            f"**Result:** `{result['diagnostic']}`\n\n"
            "**Same frozen executable contract:** "
            f"`{result['same_frozen_executable_contract']}`\n\n"
            f"**Executable-contract SHA-256:** `{contract_sha}`",
            style="green" if passed else "red",
        )
        logger.panel(
            "Frozen EC on gold patch",
            f"**Runtime outcome:** `{report.get('outcome')}`  \n"
            f"**Contracted operation reached:** `{report.get('contracted_operation_reached')}`  \n"
            f"**All runtime assertions exercised:** `{report.get('all_runtime_assertions_exercised')}`  \n"
            f"**Infrastructure error:** `{report.get('infrastructure_error') or 'none'}`\n\n"
            "**Execution trace**\n```text\n"
            + ("\n".join(report.get("execution_trace", [])) or "(no execution trace)")
            + "\n```\n\n**Command output**\n```text\n"
            + ((report.get("command_log_tail") or "(no command output)")[-8000:])
            + "\n```\n\n"
            f"**Full diagnostic:** `{artifacts.root / 'diagnostic.json'}`",
            style="green" if passed else "red",
        )
        return result
    except Exception as exc:
        result = {
            "status": "EVALUATION_ERROR",
            "error_type": type(exc).__name__,
            "detail": str(exc)[:1500],
            "instance_id": frozen["task"]["instance_id"],
            "model_calls": 0,
            "qualification_modified": False,
        }
        artifacts.save("diagnostic.json", result)
        return result
