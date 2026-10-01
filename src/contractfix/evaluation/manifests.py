"""Freeze identities and generation-only records. Never export hidden evaluator fields."""

from __future__ import annotations
import json
import re
from pathlib import Path
from contractfix.utils.artifacts import atomic_json, digest

SAFE_FIELDS = ("instance_id", "repo", "base_commit", "version", "problem_statement")
FORBIDDEN = {"patch", "gold_patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "hints_text"}
DATASETS = {
    "lite-dev": ("SWE-bench/SWE-bench_Lite", "dev", 23),
    "lite-test": ("SWE-bench/SWE-bench_Lite", "test", 300),
    "verified": ("SWE-bench/SWE-bench_Verified", "test", 500),
}


def generation_task(raw: dict) -> dict:
    row = {k: raw[k] for k in SAFE_FIELDS if k in raw}
    if set(row) != set(SAFE_FIELDS) or any(not isinstance(v, str) or not v.strip() for v in row.values()):
        raise ValueError("task requires five nonempty string fields: " + ", ".join(SAFE_FIELDS))
    if row["instance_id"] in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", row["instance_id"]):
        raise ValueError("unsafe instance ID")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", row["repo"]):
        raise ValueError("invalid repo")
    if not re.fullmatch(r"[0-9a-f]{40}", row["base_commit"]):
        raise ValueError("base_commit must be a full SHA")
    return row


def freeze_tasks(rows, output_dir, *, dataset: str, split: str, revision: str, expected_count: int):
    tasks = [generation_task(dict(r)) for r in rows]
    ids = [r["instance_id"] for r in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task IDs")
    if len(ids) != expected_count:
        raise ValueError(f"expected {expected_count} tasks; received {len(ids)}")
    if not revision:
        raise ValueError("dataset revision is required")
    tasks = sorted(tasks, key=lambda r: r["instance_id"])
    ids = [r["instance_id"] for r in tasks]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "tasks.txt").write_text("\n".join(ids) + "\n")
    (out / "generation_tasks.jsonl").write_text(
        "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks)
    )
    body = {
        "schema_version": 1,
        "dataset": dataset,
        "split": split,
        "dataset_revision": revision,
        "task_count": len(tasks),
        "task_ids": ids,
        "generation_sha256": digest(tasks),
        "source_identities": [[t["repo"], t["base_commit"]] for t in tasks],
        "generation_fields": list(SAFE_FIELDS),
        "hints_policy": "excluded",
    }
    manifest = {**body, "sha256": digest(body)}
    atomic_json(out / "manifest.json", manifest)
    return manifest


def load_manifest(path) -> dict:
    p = Path(path)
    p = p / "manifest.json" if p.is_dir() else p
    m = json.loads(p.read_text())
    body = {k: v for k, v in m.items() if k != "sha256"}
    if m.get("sha256") != digest(body):
        raise ValueError("manifest hash mismatch")
    ids = m["task_ids"]
    if len(ids) != len(set(ids)) or len(ids) != m["task_count"]:
        raise ValueError("invalid manifest membership")
    return m


def overlap(left: dict, right: dict) -> dict:
    common = sorted(set(left["task_ids"]) & set(right["task_ids"]))
    bases = sorted(set(map(tuple, left["source_identities"])) & set(map(tuple, right["source_identities"])))
    return {
        "shared_instance_ids": common,
        "shared_repo_base_pairs": bases,
        "id_disjoint": not common,
        "base_pair_disjoint": not bases,
        "assurance": "identity_overlap_only_not_proof_of_no_training_or_semantic_contamination",
    }


def download_manifest(alias, output_dir, revision=None):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    dataset, split, count = DATASETS[alias]
    rev = HfApi().dataset_info(dataset, revision=revision).sha
    rows = load_dataset(dataset, split=split, revision=rev)
    return freeze_tasks(rows, output_dir, dataset=dataset, split=split, revision=rev, expected_count=count)
