"""Validate frozen study cohorts and download generation-safe task records."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from statistics import NormalDist


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
SAFE_FIELDS = ("instance_id", "repo", "base_commit", "version", "problem_statement")


def load_cohorts() -> tuple[dict, dict[str, list[str]]]:
    manifest = json.loads((TASKS / "cohorts.json").read_text(encoding="utf-8"))
    cohorts = {}
    for name, spec in manifest["cohorts"].items():
        path = TASKS / f"{name}.txt"
        ids = path.read_text(encoding="utf-8").splitlines()
        if len(ids) != spec["count"] or len(ids) != len(set(ids)) or not all(ids):
            raise ValueError(f"invalid frozen cohort: {name}")
        cohorts[name] = ids
    for left, right in (
        ("lite_development", "lite_validation"),
        ("multilingual_development", "multilingual_evaluation"),
        ("javascript_development", "javascript_evaluation"),
    ):
        if set(cohorts[left]) & set(cohorts[right]):
            raise ValueError(f"overlapping cohorts: {left}, {right}")
    return manifest, cohorts


def verified_sample(ids: list[str], seed: int = 42) -> list[str]:
    """Reproduce the original finite-population, repository-stratified draw."""
    if len(ids) != 500 or len(ids) != len(set(ids)):
        raise ValueError("Verified source must contain 500 distinct tasks")
    z = NormalDist().inv_cdf(0.975)
    n0 = z * z * 0.25 / 0.05**2
    size = min(len(ids), math.ceil(n0 * len(ids) / (n0 + len(ids) - 1)))
    groups: dict[str, list[str]] = defaultdict(list)
    for task_id in ids:
        groups[task_id.split("__", 1)[0]].append(task_id)
    quotas = {repo: size * len(group) / len(ids) for repo, group in groups.items()}
    allocation = {repo: max(1, math.floor(quota)) for repo, quota in quotas.items()}
    remaining = size - sum(allocation.values())
    ranking = sorted(groups, key=lambda repo: (-(quotas[repo] - allocation[repo]), repo))
    if remaining < 0:
        for repo in reversed(ranking):
            if remaining == 0:
                break
            if allocation[repo] > 1:
                allocation[repo] -= 1
                remaining += 1
    else:
        for repo in ranking[:remaining]:
            allocation[repo] += 1
    rng = random.Random(seed)
    selected = set()
    for repo in sorted(groups):
        selected.update(rng.sample(sorted(groups[repo]), allocation[repo]))
    return sorted(selected)


def javascript_development(ids: list[str], rows: dict[str, dict]) -> set[str]:
    """Reproduce the outcome-blind 13-task JavaScript allocation."""
    repos = {"axios/axios", "babel/babel", "facebook/docusaurus",
             "immutable-js/immutable-js", "mrdoob/three.js",
             "preactjs/preact", "vuejs/core"}
    groups: dict[str, list[str]] = defaultdict(list)
    for task_id in ids:
        repo = rows[task_id]["repo"]
        if repo not in repos:
            raise ValueError(f"unexpected JavaScript repository: {repo}")
        groups[repo].append(task_id)
    if len(ids) != 43 or set(groups) != repos:
        raise ValueError("JavaScript source cohort must have 43 tasks in seven repositories")
    target = 13
    remaining = target - len(groups)
    capacities = {repo: len(group) - 1 for repo, group in groups.items()}
    capacity = sum(capacities.values())
    extras = {repo: remaining * count // capacity for repo, count in capacities.items()}
    seats = remaining - sum(extras.values())
    ranking = sorted(groups, key=lambda repo: (-(remaining * capacities[repo] % capacity), repo))
    for repo in ranking[:seats]:
        extras[repo] += 1
    selected = set()
    for repo in sorted(groups):
        ordered = sorted(groups[repo])
        random.Random(f"20260929:javascript:{repo}").shuffle(ordered)
        selected.update(ordered[:1 + extras[repo]])
    return selected


def download(manifest: dict, cohorts: dict[str, list[str]], output: Path) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install download dependency: uv run --with 'datasets==5.0.1' "
                         "python scripts/prepare_tasks.py --download") from exc
    sources = {}
    resolved = {}
    for alias, spec in manifest["datasets"].items():
        revision = spec["revision"]
        dataset = load_dataset(spec["name"], split=spec["split"], revision=revision)
        rows = {row["instance_id"]: row for row in dataset}
        if len(rows) != len(dataset):
            raise ValueError(f"duplicate IDs in {alias} dataset")
        sources[alias] = rows
        resolved[alias] = {"name": spec["name"], "split": spec["split"], "revision": revision}
    verified = sources["verified"]
    if verified_sample(list(verified)) != sorted(cohorts["verified_evaluation"]):
        raise ValueError("frozen Verified sample differs from source sampling")
    lite_ids = set(cohorts["lite_development"]) | set(cohorts["lite_validation"])
    if lite_ids & set(verified):
        raise ValueError("Lite development or validation overlaps Verified")
    js_ids = cohorts["javascript_development"] + cohorts["javascript_evaluation"]
    if javascript_development(js_ids, sources["multilingual"]) != set(cohorts["javascript_development"]):
        raise ValueError("frozen JavaScript split differs from source sampling")
    output.mkdir(parents=True, exist_ok=True)
    for name, ids in cohorts.items():
        source = sources[manifest["cohorts"][name]["dataset"]]
        missing = set(ids) - set(source)
        if missing:
            raise ValueError(f"{name}: missing source IDs: {sorted(missing)}")
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for task_id in ids:
                row = source[task_id]
                record = {key: row[key] for key in SAFE_FIELDS}
                if name.startswith(("multilingual_", "javascript_")):
                    record["image"] = row["image"]
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output / "sources.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cohorts)} generation-safe cohorts to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="download and verify source rows")
    parser.add_argument("--output", type=Path, default=ROOT / "downloads" / "tasks")
    args = parser.parse_args()
    manifest, cohorts = load_cohorts()
    if args.download:
        download(manifest, cohorts, args.output)
    else:
        print("Validated " + ", ".join(f"{name}={len(ids)}" for name, ids in cohorts.items()))


if __name__ == "__main__":
    main()
