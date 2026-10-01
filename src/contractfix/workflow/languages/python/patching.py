"""Deterministic edit materialization and selection. No model or evaluator imports."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import difflib
import hashlib
import os
from pathlib import Path
import subprocess

from .models import Edit
from .repository import production_path, safe_path

LEGACY_SELECTOR = "repair-selector/1"
ASSURANCE_SELECTOR = "repair-selector/2"
EVIDENCE_SELECTOR = "repair-selector/3"
EVALUATOR_ONLY_FIELDS = frozenset({
    "official_outcome", "official_evaluation", "FAIL_TO_PASS", "PASS_TO_PASS",
    "test_patch", "gold_patch", "tests_status",
})


class SemanticNoopError(ValueError):
    """The textual diff changes no Python AST semantics."""


@dataclass(frozen=True)
class MaterializedPatch:
    text: str
    sha256: str
    files: tuple[str, ...]
    changed_lines: int


def checked_file(root: Path, name: str) -> Path:
    """Do not let symlinks alias another file or escape the permitted edit scope."""
    name = safe_path(name)
    path = root / name
    current = root
    for part in Path(name).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink paths are not editable/readable: " + name)
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("not an existing repository file: " + name)
    return path


def materialize_edits(root: Path, edits: list[Edit], allowed: list[str]) -> MaterializedPatch:
    """Apply all edits or none, keeping original bytes/snapshot unchanged.

    Existing UTF-8 production Python files only. Exact matching is intentional;
    zero/multiple matches request a new model edit rather than fuzzy application.
    Git verifies the emitted diff before any repository code is executed.
    """
    if not edits:
        raise ValueError("empty edit proposal")
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    for edit_number, edit in enumerate(edits, start=1):
        name = safe_path(edit.file)
        if name not in allowed or not production_path(name):
            raise ValueError("patch changed a file outside the production edit allowlist: " + name)
        if name not in before:
            path = checked_file(root, name)
            if path.stat().st_size > 2_000_000:
                raise ValueError("edit file exceeds size limit")
            # read_bytes avoids universal-newline translation of historical files.
            before[name] = path.read_bytes().decode("utf-8")
            after[name] = before[name]
        if not edit.old.strip():
            raise ValueError(f"edit {edit_number}: old text is empty in {name}")
        match_count = after[name].count(edit.old)
        if match_count != 1:
            preview = edit.old.replace("\n", "\\n")[:400]
            raise ValueError(
                f"edit {edit_number}: old text match count={match_count}, "
                f"expected exactly 1 in {name}; old={preview!r}"
            )
        after[name] = after[name].replace(edit.old, edit.new, 1)
    chunks: list[str] = []
    changed: list[str] = []
    count = 0
    for name in sorted(before):
        if before[name] == after[name]:
            continue
        ast.parse(after[name], filename=name)
        changed.append(name)
        for line in difflib.unified_diff(
            before[name].splitlines(keepends=True), after[name].splitlines(keepends=True),
            fromfile="a/" + name, tofile="b/" + name,
        ):
            if line[:1] in {"+", "-"} and not line.startswith(("+++", "---")):
                count += 1
            if not line.endswith("\n"):
                line += "\n\\ No newline at end of file\n"
            chunks.append(line)
    text = "".join(chunks)
    if not text:
        raise ValueError("empty patch; edits do not change the base")
    # Reject comment/whitespace/noqa-only edits before they can become eligible
    # repairs. All editable production files are Python, so AST equivalence is
    # a deterministic, evaluator-blind semantic-no-op check.
    if changed and all(
        ast.dump(ast.parse(before[name], filename=name), include_attributes=False)
        == ast.dump(ast.parse(after[name], filename=name), include_attributes=False)
        for name in changed
    ):
        raise SemanticNoopError(
            "semantic no-op: textual changes leave the Python AST unchanged"
        )
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env["GIT_CEILING_DIRECTORIES"] = str(root.resolve().parent)
    result = subprocess.run(
        ["git", "apply", "--check", "--whitespace=nowarn", "-"], input=text.encode(),
        cwd=root, capture_output=True, timeout=30, env=env,
    )
    if result.returncode:
        raise ValueError("host diff is not applicable: " + result.stderr.decode(errors="replace")[-1500:])
    return MaterializedPatch(text, hashlib.sha256(text.encode()).hexdigest(), tuple(changed), count)


def validation_assurance(row: dict) -> str:
    """Classify visible-test evidence without treating missing checks as a pass."""
    checks = row.get("ordinary_checks") or {}
    if checks.get("introduced_failures") or checks.get("regression_safe") is False:
        return "REGRESSION"
    if checks.get("regression_safe") is True and checks.get("repository_check_status") == "PASSED":
        return "REGRESSION_SAFE"
    return "NOT_COMPARABLE"


def _selection_key(row: dict, version: str) -> tuple:
    repaired = len((row.get("ordinary_checks") or {}).get("repaired_visible_failures") or [])
    size = (len(row.get("changed_files") or []), row.get("changed_lines") or 0, row["attempt"])
    if version == LEGACY_SELECTOR:
        return (-repaired, -int((row.get("repository_import") or {}).get("status") == "AVAILABLE"), *size)
    if version == ASSURANCE_SELECTOR:
        return (0 if validation_assurance(row) == "REGRESSION_SAFE" else 1, -repaired, *size)
    if version == EVIDENCE_SELECTOR:
        # A smaller diff is not evidence of correct behavior. When checks do
        # not distinguish candidates, preserve the model's proposal order.
        return (
            0 if validation_assurance(row) == "REGRESSION_SAFE" else 1,
            -repaired,
            -int((row.get("repository_import") or {}).get("status") == "AVAILABLE"),
            row["attempt"],
        )
    raise ValueError(f"unknown patch selector version: {version}")


def rank_patches(candidates: list[dict], *, enforce_ec: bool, version: str = EVIDENCE_SELECTOR) -> list[dict]:
    """Rank generation receipts only; evaluator data is forbidden input."""
    if version not in {LEGACY_SELECTOR, ASSURANCE_SELECTOR, EVIDENCE_SELECTOR}:
        raise ValueError(f"unknown patch selector version: {version}")
    for row in candidates:
        if EVALUATOR_ONLY_FIELDS.intersection(row) or EVALUATOR_ONLY_FIELDS.intersection(
            row.get("ordinary_checks") or {}
        ):
            raise ValueError("official evaluator fields cannot enter patch selection")
    eligible = [
        row for row in candidates
        if row.get("applicable") is True
        and row.get("ordinary_pass") is True
        and validation_assurance(row) != "REGRESSION"
        and row.get("nlc_pass") is not False
        and (not enforce_ec or (row.get("ec_pass") is True and not row.get("infrastructure_error")))
    ]
    return sorted(eligible, key=lambda row: _selection_key(row, version))


def select_patch(
    candidates: list[dict], *, enforce_ec: bool, version: str = EVIDENCE_SELECTOR
) -> dict | None:
    """Select only validated candidates, with a stable, declared tie-break.

    EC outcomes are intentionally ignored by context/no-contract selection.
    An enforced failure is never used as justification to fall back to context.
    """
    ranked = rank_patches(candidates, enforce_ec=enforce_ec, version=version)
    return ranked[0] if ranked else None
