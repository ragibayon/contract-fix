"""Offline prompt inspection and development-only failure aggregation.

These operations make no model calls, do not read gold records, and never update
skills or memory. A reviewed new prompt pack is a separate, explicit artifact.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from contractfix.evaluation.manifests import DATASETS, load_manifest
from contractfix.utils.artifacts import atomic_json
from .languages.python.promptbook import PromptBook, SCHEMAS


def audit_prompts(output: Path, prompts: Path | None = None) -> dict:
    book = PromptBook(prompts)
    output.mkdir(parents=True, exist_ok=False)
    report = book.audit()
    atomic_json(output / "assets.json", book.assets)
    for stage in SCHEMAS:
        # This is an inspected synthetic packet, not a SWE-bench prompt measurement.
        example = next(item for item in book.examples if item["stage"] == stage)
        request = book.assemble(stage, example["packet"])
        request["schema"] = book.schemas[stage].model_json_schema()
        request["preview_only"] = True
        atomic_json(output / f"{stage}.json", request)
    atomic_json(output / "audit.json", report)
    return report


def development_failures(manifest_path: Path, campaign: Path, output: Path) -> dict:
    manifest = load_manifest(manifest_path)
    dataset, split, count = DATASETS["lite-dev"]
    if (manifest["dataset"], manifest["split"], manifest["task_count"]) != (dataset, split, count):
        raise ValueError("lesson review accepts only the official 23-task Lite dev manifest")
    identity = json.loads((campaign / "campaign_identity.json").read_text())
    if set(identity["task_ids"]) != set(manifest["task_ids"]):
        raise ValueError("campaign and development manifest membership differ")
    linked = identity.get("manifest")
    if linked is None or linked["sha256"] != manifest["sha256"]:
        raise ValueError("campaign must have been bound to this manifest at creation")
    data = json.loads((campaign / "campaign.json").read_text())
    rows = data["outcomes"]
    if len({row["instance_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate campaign outcome")
    if set(row["instance_id"] for row in rows) - set(manifest["task_ids"]):
        raise ValueError("outcome outside the development manifest")
    counts, references = Counter(), {}
    for row in rows:
        status = row["status"]
        labels = status.get("qualification", {}).get("failures", [])
        if status.get("error_type"):
            labels = [status["error_type"]]
        if not labels:
            labels = [status["status"]]
        for label in set(labels):
            counts[label] += 1
            references.setdefault(label, []).append(row["instance_id"])
    report = {"manifest_sha256": manifest["sha256"], "campaign_sha256": identity["sha256"],
              "denominator": count, "observed_tasks": len(rows), "not_run": count - len(rows),
              "failure_counts": dict(sorted(counts.items())),
              "review_candidates": [{"category": label, "source_instance_ids": sorted(references[label]),
                                     "review_status": "pending", "promoted_to_memory": False}
                                    for label in sorted(counts)],
              "model_calls": 0, "skills_modified": False, "memory_modified": False,
              "assurance": "diagnostic_categories_not_validated_lessons"}
    if output.exists():
        raise ValueError("diagnostic output already exists")
    atomic_json(output, report)
    return report
