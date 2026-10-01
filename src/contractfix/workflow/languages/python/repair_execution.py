"""Execute host-owned commands in fresh task-environment snapshots."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from contractfix.contracts.execution import Executor
from contractfix.contracts.snapshots import IGNORED, repository_identity
from contractfix.utils.artifacts import redact
from .repository import safe_path


PYTEST_RECEIPT_PLUGIN = r'''
import json
import os
from pathlib import Path

_outcomes = {}
_collection_failures = set()


def pytest_runtest_logreport(report):
    nodeid = report.nodeid
    current = _outcomes.get(nodeid)
    if report.failed:
        _outcomes[nodeid] = "failed"
    elif report.when == "call" and current != "failed":
        _outcomes[nodeid] = "skipped" if report.skipped else "passed"
    elif report.skipped and current is None:
        _outcomes[nodeid] = "skipped"


def pytest_collectreport(report):
    if report.failed:
        _collection_failures.add("collection::" + report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    failed = sorted(
        {nodeid for nodeid, outcome in _outcomes.items() if outcome == "failed"}
        | _collection_failures
    )
    passed = sorted(nodeid for nodeid, outcome in _outcomes.items() if outcome == "passed")
    skipped = sorted(nodeid for nodeid, outcome in _outcomes.items() if outcome == "skipped")
    destination = Path(os.environ["CONTRACTFIX_EC_EVENTS"]) / "visible-tests.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "schema_version": "contractfix-visible-test-results/1",
        "exit_status": int(exitstatus),
        "failed": failed,
        "passed": passed,
        "skipped": skipped,
        "observed": len(failed) + len(passed) + len(skipped),
    }, sort_keys=True), encoding="utf-8")
'''


def _protected_inputs(root: Path, names: list[str] | None = None) -> dict:
    """Guard existing source/test inputs, not disposable coverage/build outputs."""
    if names is None:
        names = [p.relative_to(root).as_posix() for p in root.rglob("*")
                 if not (set(p.relative_to(root).parts) & IGNORED)
                 and (p.is_file() or p.is_symlink())]
    result = {}
    for name in names:
        path = root / name
        if any(parent.is_symlink() for parent in path.parents if parent != root and parent.is_relative_to(root)):
            result[name] = {"unsafe_ancestor": True}
        elif path.is_symlink():
            result[name] = {"symlink": os.readlink(path)}
        elif path.is_file():
            with path.open("rb") as stream:
                result[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        else:
            result[name] = {"missing_or_not_regular": True}
    return result


def _instrument_pytest(command: list[str]) -> list[str] | None:
    """Return a pytest argv that loads our result-only plugin, if recognizable."""
    values = list(command)
    is_module = len(values) >= 3 and values[1:3] == ["-m", "pytest"]
    is_executable = bool(values) and Path(values[0]).name in {"pytest", "py.test"}
    if not (is_module or is_executable):
        return None
    if "contractfix_visible_tests" not in values:
        values.extend(["-p", "contractfix_visible_tests"])
    # A local snapshot under repository tmp/ must not inherit its parent's
    # rootdir, which makes otherwise identical node IDs differ per snapshot.
    if not any(value == "--rootdir" or value.startswith("--rootdir=") for value in values):
        values.append("--rootdir=.")
    return values


def _sympy_test_results(command: list[str], output: str, exit_code: int | None) -> dict | None:
    """Parse the checked-out SymPy runner's verbose, named test outcomes."""
    if command[:2] != ["@python", "bin/test"] or "-v" not in command:
        return None
    if exit_code not in {0, 1} or "tests finished:" not in output:
        return None
    passed, failed, skipped = set(), set(), set()
    current_file = None
    pending = None
    expected_count = 0
    outcome = re.compile(r"^(ok|E|F|T|K|X|f|s|w)$")
    for raw in output.splitlines():
        line = raw.strip()
        heading = re.match(r"^(sympy/\S+\.py)\[(\d+)\]", line)
        if heading:
            current_file = heading.group(1)
            expected_count += int(heading.group(2))
            pending = None
            continue
        named = re.match(
            r"^(test_[A-Za-z0-9_]+)\s+(ok|E|F|T|K|X|f|s|w)"
            r"(?:\s+\[(?:FAIL|OK)\])?$", line,
        )
        if named and current_file:
            test_id = f"{current_file}::{named.group(1)}"
            bucket = failed if named.group(2) in {"E", "F", "T", "K"} else (
                passed if named.group(2) == "ok" else skipped
            )
            bucket.add(test_id)
            pending = None
            continue
        start = re.match(r"^(test_[A-Za-z0-9_]+)\s+", line)
        if start and current_file:
            pending = f"{current_file}::{start.group(1)}"
        elif pending and outcome.fullmatch(line):
            bucket = failed if line in {"E", "F", "T", "K"} else (
                passed if line == "ok" else skipped
            )
            bucket.add(pending)
            pending = None
    observed = len(passed) + len(failed) + len(skipped)
    if not observed or observed != expected_count:
        return None
    return {
        "schema_version": "contractfix-visible-test-results/1",
        "exit_status": exit_code,
        "failed": sorted(failed),
        "passed": sorted(passed),
        "skipped": sorted(skipped),
        "observed": observed,
    }


def run_command(
    base: Path,
    executor: Executor,
    command: list[str],
    *,
    patch: Path | None = None,
    capture_tests: bool = False,
) -> dict:
    """No test/patch process ever receives the host guidance directory or credentials."""
    with tempfile.TemporaryDirectory(prefix="contractfix-patch-check-") as temporary:
        root = Path(temporary)
        work = root / "repo"
        shutil.copytree(base, work, symlinks=True)
        if patch:
            applied = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", str(patch.resolve())],
                cwd=work, capture_output=True, timeout=30,
                env={**{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
                     "GIT_CEILING_DIRECTORIES": str(work.parent)},
            )
            if applied.returncode:
                return {"command": command, "exit_code": None, "infrastructure_error": None,
                        "passed": False, "failure": "EDIT_NOT_APPLICABLE",
                        "log_tail": applied.stderr.decode(errors="replace")[-3000:]}
        repository_identity(work)  # validate all initial symlink targets
        initial = _protected_inputs(work)
        runtime = root / "runtime"
        runtime.mkdir()
        executed = list(command)
        structured = _instrument_pytest(command) if capture_tests else None
        if structured:
            (runtime / "contractfix_visible_tests.py").write_text(
                PYTEST_RECEIPT_PLUGIN, encoding="utf-8"
            )
            executed = structured
        bundle = root / "bundle.json"
        bundle.write_text("{}", encoding="utf-8")
        log = root / "command.log"
        try:
            code, error = executor.run(executed, work, runtime, bundle, root / "outputs", log)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            code, error = None, f"{type(exc).__name__}: {exc}"
        try:
            mutated = _protected_inputs(work, list(initial)) != initial
        except (OSError, ValueError):
            mutated = True
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        receipt_path = root / "outputs/events/visible-tests.json"
        test_results = None
        if receipt_path.is_file():
            try:
                test_results = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                test_results = None
        if test_results is None and capture_tests:
            test_results = _sympy_test_results(command, text, code)
        return redact({
            "command": command,
            "executed_command": executed,
            "exit_code": code, "infrastructure_error": error,
            "passed": code == 0 and error is None and not mutated,
            "repository_mutated": mutated,
            "mutation_policy": "preexisting_inputs_unchanged; new_outputs_discarded",
            "log_tail": text[-8000:],
            "test_results": test_results,
        })


def environment_receipt(base: Path, executor: Executor, packages: list[str] | None = None) -> dict:
    """Inspect the TASK interpreter, not the host or a remembered package release."""
    packages = packages or []
    script = '''import json, sys, platform
names = json.loads(sys.argv[1])
try:
    from importlib import metadata
except ImportError:
    try:
        import importlib_metadata as metadata
    except ImportError:
        metadata = None
versions = {}
for name in names:
    try:
        if metadata is not None:
            versions[name] = metadata.version(name)
        else:
            import pkg_resources
            versions[name] = pkg_resources.get_distribution(name).version
    except Exception:
        versions[name] = None
print(json.dumps({"python": platform.python_version(), "implementation": platform.python_implementation(),
                  "executable": sys.executable, "packages": versions}))
'''
    result = run_command(base, executor, ["@python", "-c", script, json.dumps(packages)])
    if not result["passed"]:
        return {"status": "UNAVAILABLE", "execution": result}
    try:
        data = json.loads(result["log_tail"].strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"status": "UNAVAILABLE", "execution": result}
    return {"status": "OBSERVED", **data, "executor_sha256": executor.identity()["sha256"]}


def importable_package(base: Path, paths: list[str]) -> str | None:
    """Find a localized repository package without importing it on the host."""
    for name in paths:
        parts = Path(name).parts
        if not parts or Path(name).suffix != ".py":
            continue
        source_parts = parts[1:] if parts[0] == "src" else parts
        if len(source_parts) < 2:
            continue
        package = source_parts[0]
        prefix = base / ("src" if parts[0] == "src" else "")
        if package.isidentifier() and (prefix / package / "__init__.py").is_file():
            return package
    return None


def repository_import_check(
    base: Path, executor: Executor, module: str, *, patch: Path | None = None
) -> dict:
    """Probe an unmodified package import in the task runtime, without EC shims."""
    if not module.isidentifier():
        raise ValueError("repository import target must be a top-level package")
    script = "import importlib, sys; importlib.import_module(sys.argv[1])"
    receipt = run_command(base, executor, ["@python", "-c", script, module], patch=patch)
    return {
        "module": module,
        "status": (
            "AVAILABLE" if receipt["passed"] else
            "UNAVAILABLE" if receipt.get("infrastructure_error") else "BLOCKED"
        ),
        "execution": receipt,
        "scope": "task_runtime_plain_import_not_official_evaluator_setup",
    }


def ordinary_checks(
    base: Path,
    patch: Path,
    files: list[str],
    executor: Executor,
    commands: list[list[str]],
    missing_policy: str,
    baseline_results: list[dict] | None = None,
    *,
    syntax_scope: str = "changed_files",
) -> dict:
    if syntax_scope not in {"changed_files", "touched_packages"}:
        raise ValueError("unknown patch syntax scope")
    syntax_files = {safe_path(name) for name in files if name.endswith(".py")}
    if syntax_scope == "touched_packages":
        for name in tuple(syntax_files):
            parts = Path(name).parts
            if not parts:
                continue
            if parts[0] == "src" and len(parts) >= 3:
                package = base / "src" / parts[1]
            elif len(parts) >= 2 and (base / parts[0] / "__init__.py").is_file():
                package = base / parts[0]
            else:
                continue
            if not package.is_dir() or package.is_symlink():
                continue
            syntax_files.update(
                path.relative_to(base).as_posix()
                for path in package.rglob("*.py")
                if path.is_file() and not path.is_symlink()
                and not set(path.relative_to(base).parts) & {".tox", ".nox", ".venv", "venv", "build", "dist"}
            )
    # compile(bytes, ...) honors a source coding declaration and writes no pyc.
    # f-strings, match, etc. are checked by the pinned historical interpreter.
    compile_script = (
        "import sys\n"
        "for p in sys.argv[1:]:\n"
        "    with open(p, 'rb') as f: compile(f.read(), p, 'exec')\n"
    )
    syntax = run_command(base, executor, ["@python", "-c", compile_script, *sorted(syntax_files)], patch=patch)
    result = {"syntax": syntax, "syntax_scope": syntax_scope, "syntax_files": sorted(syntax_files),
              "repository_checks": [], "passed": False,
              "validation_scope": "repository_checks" if commands else "syntax_only",
              "infrastructure_error": syntax.get("infrastructure_error")}
    if not syntax["passed"]:
        result["failure"] = "INFRASTRUCTURE_ERROR" if syntax.get("infrastructure_error") else "SYNTAX_ERROR"
        return result
    if not commands:
        result.update(passed=missing_policy == "syntax_only", repository_check_status="NOT_CONFIGURED")
        if not result["passed"]:
            result["failure"] = "REPOSITORY_CHECKS_REQUIRED"
        return result
    if baseline_results is None or len(baseline_results) != len(commands):
        result.update(
            failure="BASELINE_RESULTS_REQUIRED",
            repository_check_status="NOT_COMPARABLE",
        )
        return result
    introduced: set[str] = set()
    repaired: set[str] = set()
    persistent: set[str] = set()
    base_failed_all: set[str] = set()
    patched_failed_all: set[str] = set()
    base_passed_all: set[str] = set()
    patched_passed_all: set[str] = set()
    for number, (command, baseline) in enumerate(zip(commands, baseline_results), start=1):
        receipt = run_command(base, executor, command, patch=patch, capture_tests=True)
        result["repository_checks"].append(receipt)
        if receipt.get("infrastructure_error") or receipt.get("repository_mutated"):
            result.update(
                failure="INFRASTRUCTURE_ERROR",
                infrastructure_error=(
                    receipt.get("infrastructure_error")
                    or "repository-visible check mutated a protected input"
                ),
                repository_check_status="NOT_COMPARABLE",
            )
            return result
        before = baseline.get("test_results")
        after = receipt.get("test_results")
        prefix = f"command-{number}::"
        if before is not None and after is not None:
            if (
                not before.get("observed")
                or not after.get("observed")
                or before.get("exit_status") not in {0, 1}
                or after.get("exit_status") not in {0, 1}
            ):
                result.update(
                    failure="VISIBLE_TEST_RESULTS_INCOMPLETE",
                    repository_check_status="NOT_COMPARABLE",
                )
                return result
            base_failed = {prefix + item for item in before.get("failed", [])}
            patched_failed = {prefix + item for item in after.get("failed", [])}
            base_failed_all.update(base_failed)
            patched_failed_all.update(patched_failed)
            base_passed_all.update(prefix + item for item in before.get("passed", []))
            patched_passed_all.update(prefix + item for item in after.get("passed", []))
            introduced.update(patched_failed - base_failed)
            repaired.update(base_failed - patched_failed)
            persistent.update(base_failed & patched_failed)
            continue
        # A clean unstructured baseline is still comparable by process status:
        # any candidate failure is necessarily newly introduced. A failing
        # unstructured baseline cannot identify which tests changed. Report that
        # loss of comparability; the workflow policy decides whether to ignore it
        # or abstain.
        if baseline.get("passed"):
            if not receipt.get("passed"):
                introduced.add(prefix + "unstructured-command-failure")
            continue
        result.update(
            failure="BASELINE_RESULTS_UNSTRUCTURED",
            repository_check_status="NOT_COMPARABLE",
        )
        return result
    result.update(
        introduced_failures=sorted(introduced),
        repaired_visible_failures=sorted(repaired),
        persistent_failures=sorted(persistent),
        baseline={
            "passed": len(base_passed_all),
            "failed": sorted(base_failed_all),
        },
        candidate={
            "passed": len(patched_passed_all),
            "failed": sorted(patched_failed_all),
        },
        regression_safe=not introduced,
        passed=not introduced,
        repository_check_status="PASSED" if not introduced else "REGRESSION",
    )
    if introduced:
        result["failure"] = "VISIBLE_REGRESSION"
    return result
