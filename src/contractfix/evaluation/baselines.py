"""Non-executing audit of prepared native baseline inputs. No hidden tests are read."""

from pathlib import Path
import json
import subprocess
from .manifests import SAFE_FIELDS, generation_task


def audit_workspace(workspace, manifest=None):
    root = Path(workspace).resolve()
    issues = []
    tasks = json.loads((root / "tasks_map.json").read_text())
    setups = json.loads((root / "setup_map.json").read_text())
    if not isinstance(tasks, dict) or not isinstance(setups, dict):
        raise ValueError("maps must be objects")
    if not tasks:
        issues.append({"error": "empty task map"})
    if manifest:
        if set(tasks) != set(manifest["task_ids"]):
            issues.append({"error": "frozen manifest membership mismatch"})
        identities = dict(zip(manifest["task_ids"], manifest["source_identities"]))
        for iid, task in tasks.items():
            if iid in identities and [task.get("repo"), task.get("base_commit")] != identities[iid]:
                issues.append({"task": iid, "error": "frozen source identity mismatch"})
    for iid, task in tasks.items():
        if set(task) != set(SAFE_FIELDS):
            issues.append(
                {
                    "task": iid,
                    "error": "non_generation_fields",
                    "fields": sorted(set(task) - set(SAFE_FIELDS)),
                }
            )
        try:
            generation_task(task)
        except ValueError as exc:
            issues.append({"task": iid, "error": str(exc)})
        if task.get("instance_id") != iid:
            issues.append({"task": iid, "error": "ID mismatch"})
        s = setups.get(iid, {})
        for field in ("repo_path", "env_name", "install", "test_cmd"):
            if not isinstance(s.get(field), str) or not s[field].strip():
                issues.append({"task": iid, "error": "missing_native_environment_field", "field": field})
        p = Path(s.get("repo_path", "/nonexistent"))
        if p.is_dir():

            def git(*args):
                out = subprocess.run(["git", "-C", str(p), *args], capture_output=True, text=True, timeout=30)
                if out.returncode:
                    raise ValueError(out.stderr[:300])
                return out.stdout.strip()

            try:
                if git("rev-parse", "HEAD") != task.get("base_commit"):
                    issues.append({"task": iid, "error": "base revision mismatch"})
                if git("status", "--porcelain"):
                    issues.append({"task": iid, "error": "dirty worktree"})
                if git("remote"):
                    issues.append({"task": iid, "error": "agent-visible repository remote"})
                if git("rev-list", "--all", "--count") != "1":
                    issues.append({"task": iid, "error": "history contains more than base"})
            except (ValueError, subprocess.TimeoutExpired) as exc:
                issues.append({"task": iid, "error": "git_check_failed", "detail": str(exc)[:300]})
        else:
            issues.append({"task": iid, "error": "checkout missing"})
    if set(tasks) != set(setups):
        issues.append({"error": "setup/task membership mismatch"})
    return {
        "task_count": len(tasks),
        "structural_checks_passed": not issues,
        "issues": issues,
        "native_execution_verified": False,
        "next_gate": "run pinned native framework in its environment; verify model adapter, tests and prediction export",
    }
