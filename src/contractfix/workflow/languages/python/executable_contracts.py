"""Validate and evaluate Python executable contracts in a disposable checkout."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import warnings

from contractfix.contracts.execution import Executor
from contractfix.contracts.snapshots import repository_identity, validate_relative_symlink

IGNORED = (".git", ".venv", "__pycache__", ".pytest_cache")
RUNTIME_MONITOR = Path(__file__).with_name("runtime_monitor.py")


def _execution_trace(result: dict) -> list[str]:
    """Render host-observed EC events without relying on model-authored prints."""
    trace = []
    for event in result.get("events", []):
        name = event.get("event", "unknown_event")
        if name == "evaluation_harness_entered":
            trace.append("HARNESS entered")
        elif name in {
            "contracted_operation_reached",
            "contracted_operation_returned",
        }:
            action = "reached" if name.endswith("reached") else "returned"
            trace.append(f"OPERATION {action}: {event.get('target', 'unknown target')}")
        elif name == "contracted_operation_raised":
            trace.append(
                "OPERATION raised: "
                f"{event.get('exception_type', 'unknown exception')}"
                + (f" | {event['detail']}" if event.get("detail") else "")
            )
        elif name == "contract_violation":
            identity = event.get("runtime_assertion_id") or event.get(
                "runtime_assertion_line", "unknown assertion"
            )
            trace.append(
                f"ASSERTION failed: {identity}"
                + (f" | {event['detail']}" if event.get("detail") else "")
            )
        elif name == "evaluation_harness_error":
            trace.append(
                "HARNESS error: "
                f"{event.get('exception_type', 'unknown exception')}"
                + (f" | {event['detail']}" if event.get("detail") else "")
            )
        elif name == "contract_satisfied":
            trace.append("CONTRACT satisfied")
        elif name == "runtime_assertions_observed":
            trace.append(
                "ASSERTIONS declared_lines="
                f"{event.get('runtime_assertion_lines', [])} "
                "executed_lines="
                f"{event.get('executed_runtime_assertion_lines', [])}"
            )
    if result.get("infrastructure_error"):
        trace.append(f"INFRASTRUCTURE error: {result['infrastructure_error']}")
    trace.append(
        f"RESULT outcome={result.get('outcome')} command_exit={result.get('command_exit')}"
    )
    return trace


def _assertion_execution_receipt(result: dict) -> dict:
    """Map assertion source lines to stable IDs and explicit execution status."""
    declared = result.get("validation", {}).get("runtime_assertion_ids", [])
    executed_lines = set(result.get("executed_runtime_assertion_lines", []))
    assertions = [
        {
            "id": item["id"],
            "source_line": item["line"],
            "executed": item["line"] in executed_lines,
        }
        for item in declared
    ]
    violations = [
        {
            "id": event.get("runtime_assertion_id"),
            "source_line": event.get("runtime_assertion_line"),
            "detail": event.get("detail", ""),
        }
        for event in result.get("events", [])
        if event.get("event") == "contract_violation"
    ]
    return {
        "declared_count": len(assertions),
        "executed_count": sum(item["executed"] for item in assertions),
        "violation_count": len(violations),
        "all_executed": bool(assertions) and all(
            item["executed"] for item in assertions
        ),
        "assertions": assertions,
        "observed_violations": violations,
    }


def validate_executable_contract(source: str, *, entrypoint: str = "contractfix_contract") -> dict:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SyntaxWarning)
        tree = ast.parse(source, filename="contractfix_contract.py")
    syntax_warnings = [item for item in caught if issubclass(item.category, SyntaxWarning)]
    if syntax_warnings:
        details = "; ".join(
            f"line {item.lineno}: {item.message}" for item in syntax_warnings
        )
        raise ValueError(f"executable contract emits SyntaxWarning: {details}")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == entrypoint]
    if len(functions) != 1:
        raise ValueError(f"executable contract must define exactly one zero-argument {entrypoint}")
    args = functions[0].args
    if args.args or args.posonlyargs or args.vararg or any(
        default is None for default in args.kw_defaults
    ):
        raise ValueError(f"executable contract must define exactly one zero-argument {entrypoint}")
    assertions = sorted(node.lineno for node in ast.walk(functions[0]) if isinstance(node, ast.Assert))
    if not assertions:
        raise ValueError("executable contract must contain at least one runtime contract assertion")
    forbidden = {"subprocess", "socket"}
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(name=node.module or "")])
    }
    if imports & forbidden:
        raise ValueError("SECURITY_POLICY_VIOLATION: process and network modules are forbidden")
    imported_names = {
        alias.asname or alias.name.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_names.update(
        alias.asname or alias.name
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        for alias in node.names if alias.name != "*"
    )

    def imported_root(node: ast.AST) -> bool:
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        return isinstance(node, ast.Name) and node.id in imported_names

    def mutates_import(target: ast.AST) -> bool:
        if isinstance(target, (ast.Tuple, ast.List)):
            return any(mutates_import(item) for item in target.elts)
        return isinstance(target, (ast.Attribute, ast.Subscript)) and imported_root(target)

    for node in ast.walk(tree):
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = node.targets
        for target in targets:
            if mutates_import(target):
                raise ValueError(
                    "CONTRACT_POLICY_VIOLATION: executable contracts cannot mutate imported modules"
                )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"setattr", "delattr"}
            and node.args
            and imported_root(node.args[0])
        ):
            raise ValueError(
                "CONTRACT_POLICY_VIOLATION: executable contracts cannot mutate imported modules"
            )
    compile(tree, "contractfix_contract.py", "exec")
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    runtime_assertion_ids = [
        {"id": f"runtime_assertion_{position:03d}", "line": line}
        for position, line in enumerate(assertions, start=1)
    ]
    return {
        "runtime_assertion_lines": assertions,
        "runtime_assertion_ids": runtime_assertion_ids,
        "ast_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
    }


def equivalent_contract_sources(first: str, second: str) -> bool:
    try:
        return (
            validate_executable_contract(first)["ast_sha256"]
            == validate_executable_contract(second)["ast_sha256"]
        )
    except (SyntaxError, ValueError):
        # Invalid proposals are handled as candidate failures, not duplicates.
        return False


def _instrument(source: str, relative: str, symbol: str, accessor: str | None = None) -> str:
    tree = ast.parse(source)
    found: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

    def visit(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, prefix + node.name + ".")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and prefix + node.name == symbol:
                found.append(node)

    visit(tree.body)
    if accessor is not None:
        if accessor not in {"getter", "setter", "deleter"}:
            raise ValueError("unknown contracted operation accessor")

        def matches_accessor(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Attribute) and decorator.attr == accessor:
                    if isinstance(decorator.value, ast.Name) and decorator.value.id == node.name:
                        return True
                if (accessor == "getter" and isinstance(decorator, ast.Name)
                        and decorator.id == "property"):
                    return True
            return False

        found = [node for node in found if matches_accessor(node)]
    if len(found) != 1:
        raise ValueError("contracted operation missing or ambiguous")
    operation = found[0]
    decorator_line = min(
        [operation.lineno, *(item.lineno for item in operation.decorator_list)]
    )
    lines = source.splitlines(keepends=True)
    indentation = lines[decorator_line - 1][
        : len(lines[decorator_line - 1]) - len(lines[decorator_line - 1].lstrip())
    ]
    decorator = f'{indentation}@_contractfix_monitor({relative + ":" + symbol!r})\n'

    position = 0
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        position = tree.body[0].end_lineno or tree.body[0].lineno
    future_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    ]
    if future_imports:
        position = max(node.end_lineno or node.lineno for node in future_imports)
    # Target repositories execute in isolated environments where ContractFix is
    # deliberately not installed. The runner materializes this small monitor as
    # part of the generated harness instead of coupling target code to our package.
    import_line = (
        "from __cf_executable_contract__.runtime_monitor import "
        "observe as _contractfix_monitor\n"
    )
    public_lines = [
        item.lineno for item in operation.decorator_list
        if (isinstance(item, ast.Name) and item.id == "public")
        or (isinstance(item, ast.Attribute) and item.attr == "public")
    ]
    # A public/export decorator may inspect function.__globals__. If it sees
    # the monitor wrapper instead of the repository function, registration
    # happens in the monitor module and the repository import can fail.
    monitor_line = min(public_lines) if public_lines else operation.lineno
    for index, text in sorted(
        # Normally observe the function body before decorators capture it
        # (e.g. Click commands). Export decorators are the exception above.
        [(monitor_line - 1, decorator), (position, import_line)],
        key=lambda insertion: insertion[0],
        reverse=True,
    ):
        lines.insert(index, text)
    return "".join(lines)


LAUNCHER = """import ast, importlib.util, json, os, sys, traceback
from pathlib import Path
events = Path(os.environ["CONTRACTFIX_EC_EVENTS"])
events.mkdir(parents=True, exist_ok=True)
stream = events / ("executable-contract.%s.jsonl" % os.getpid())
def emit(event, **data):
    with stream.open("a", encoding="utf-8") as out:
        out.write(json.dumps({"event": event, **data}, default=str) + "\\n")
source_path = Path(__file__).with_name("candidate.py")
tree = ast.parse(source_path.read_text(encoding="utf-8"))
assertions = {n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert)}
explicit_raises = {n.lineno for n in ast.walk(tree) if isinstance(n, ast.Raise)}
executed = set()
def trace(frame, event, arg):
    if frame.f_code.co_filename == str(source_path) and event == "line":
        executed.add(frame.f_lineno)
    return trace
try:
    spec = importlib.util.spec_from_file_location("contractfix_generated_contract", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    emit("evaluation_harness_entered")
    sys.settrace(trace)
    entrypoint = (
        "contractfix_contract"
        if hasattr(module, "contractfix_contract")
        else "contractfix_probe"
    )
    getattr(module, entrypoint)()
except AssertionError as exc:
    types = sorted({f"{type(v).__module__}.{type(v).__qualname__}" for tb in traceback.walk_tb(exc.__traceback__) for v in tb[0].f_locals.values() if isinstance(v, BaseException)})
    frames = list(traceback.walk_tb(exc.__traceback__))
    origin_frame, origin_line = frames[-1]
    failed = origin_line if origin_frame.f_code.co_filename == str(source_path) and origin_line in (assertions | explicit_raises) else None
    if failed is not None and failed in executed:
        emit("contract_violation", violation_kind="runtime_contract_assertion", detail=str(exc), observed_exception_types=types, runtime_assertion_line=failed)
    else:
        emit("evaluation_harness_error", exception_type="builtins.AssertionError", detail=str(exc), failure_origin="not_generated_assertion")
except BaseException as exc:
    emit("evaluation_harness_error", exception_type=f"{type(exc).__module__}.{type(exc).__qualname__}", detail=str(exc), traceback=traceback.format_exc())
else:
    emit("contract_satisfied")
finally:
    sys.settrace(None)
    emit("runtime_assertions_observed", runtime_assertion_lines=sorted(assertions), executed_runtime_assertion_lines=sorted(assertions & executed))
"""


IMPORT_CHECK_LAUNCHER = """import importlib, json, os, sys, traceback
from pathlib import Path
receipt = {}
try:
    module = importlib.import_module(sys.argv[1])
except BaseException as exc:
    receipt = {"status": "IMPORT_FAILED", "exception_type": type(exc).__module__ + "." + type(exc).__qualname__, "detail": str(exc), "traceback": traceback.format_exc()}
else:
    actual = getattr(module, '__file__', None)
    expected = Path(os.environ['CONTRACTFIX_EC_REPO']).joinpath(sys.argv[2]).resolve()
    matched = actual is not None and Path(actual).resolve() == expected
    receipt = {"status": "IMPORT_OK" if matched else "IMPORT_FAILED",
               "module_path": actual, "expected_module_path": str(expected),
               "source_matches_snapshot": matched}
    if not matched:
        receipt['detail'] = 'import resolved outside the requested snapshot module'
Path(os.environ["CONTRACTFIX_EC_EVENTS"]).joinpath("base-import.json").write_text(json.dumps(receipt), encoding="utf-8")
"""


def check_base_import(repo: Path, target: dict, executor: Executor) -> dict:
    """Corroborate a pre-operation failure without running any model-authored witness.

    A matching failure establishes a base-import blocker, not whether the root
    cause is a dependency mismatch or an import-time defect in the repository.
    No packages, repository files, or model policies are repaired here.
    """
    relative = Path(target["file"])
    if relative.suffix != ".py":
        return {"status": "INCONCLUSIVE", "reason": "non_python_target"}
    parts = list(relative.with_suffix("").parts)
    if parts and parts[0] in {"src", "lib"} and not (repo / parts[0] / "__init__.py").exists():
        parts.pop(0)
    if parts and parts[-1] == "__init__":
        parts.pop()
    if not parts or not all(part.isidentifier() for part in parts):
        return {"status": "INCONCLUSIVE", "reason": "unknown_module_name"}
    module_name = ".".join(parts)
    with tempfile.TemporaryDirectory(prefix="contractfix-base-import-") as temporary:
        temp = Path(temporary)
        work = temp / "repo"
        shutil.copytree(repo, work, symlinks=True, ignore=shutil.ignore_patterns(*IGNORED))
        for path in work.rglob("*"):
            if path.is_symlink():
                validate_relative_symlink(path.relative_to(work).as_posix(), path.readlink().as_posix())
        launcher = work / "__cf_import_health__.py"
        if launcher.exists():
            return {"status": "INCONCLUSIVE", "reason": "reserved_path_collision"}
        launcher.write_text(IMPORT_CHECK_LAUNCHER, encoding="utf-8")
        expected = repository_identity(work)
        bundle = temp / "bundle.json"
        bundle.write_text("{}", encoding="utf-8")
        runtime_root = temp / "runtime"
        runtime_root.mkdir()
        outputs = temp / "outputs"
        logfile = temp / "command.log"
        exit_code, infrastructure = executor.run(
            ["@python", launcher.name, module_name, relative.as_posix()], work,
            runtime_root, bundle, outputs, logfile,
        )
        if repository_identity(work) != expected:
            infrastructure = "REPOSITORY_MUTATION"
        receipt_path = outputs / "events" / "base-import.json"
        receipt = (
            json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt_path.exists() else {"status": "INCONCLUSIVE"}
        )
        if infrastructure or exit_code:
            receipt["status"] = "INCONCLUSIVE"
        return {
            **receipt, "module": module_name, "command_exit": exit_code,
            "infrastructure_error": infrastructure, "candidate_source_executed": False,
            "instrumented": False, "executor_sha256": executor.identity()["sha256"],
            "command_log_tail": logfile.read_text(encoding="utf-8", errors="replace")[-4000:]
            if logfile.exists() else "",
        }


def evaluate_executable_contract(
    repo: Path,
    source: str,
    target: dict,
    executor: Executor,
    *,
    patch: Path | None = None,
    entrypoint: str = "contractfix_contract",
) -> dict:
    validation = validate_executable_contract(source, entrypoint=entrypoint)
    with tempfile.TemporaryDirectory(prefix="contractfix-executable-contract-") as temporary:
        temp = Path(temporary)
        work = temp / "repo"
        shutil.copytree(repo, work, symlinks=True, ignore=shutil.ignore_patterns(*IGNORED))
        for path in work.rglob("*"):
            if path.is_symlink():
                validate_relative_symlink(
                    path.relative_to(work).as_posix(),
                    path.readlink().as_posix(),
                )
        if patch:
            before_patch = repository_identity(work)
            subprocess.run(
                ["git", "apply", str(patch.resolve())], cwd=work, check=True,
                capture_output=True, timeout=30,
                env={**{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
                     "GIT_CEILING_DIRECTORIES": str(work.parent)},
            )
            if repository_identity(work) == before_patch:
                raise ValueError("patch did not change the isolated EC checkout")
        operation_file = work / target["file"]
        operation_file.write_text(
            _instrument(
                operation_file.read_text(encoding="utf-8"),
                target["file"],
                target["symbol"],
                target.get("accessor"),
            ),
            encoding="utf-8",
        )
        contract_dir = work / "__cf_executable_contract__"
        contract_dir.mkdir()
        (contract_dir / "__init__.py").write_text("", encoding="utf-8")
        shutil.copyfile(RUNTIME_MONITOR, contract_dir / "runtime_monitor.py")
        (contract_dir / "candidate.py").write_text(source, encoding="utf-8")
        (contract_dir / "launcher.py").write_text(LAUNCHER, encoding="utf-8")
        expected = repository_identity(work)
        bundle = temp / "bundle.json"
        bundle.write_text("{}", encoding="utf-8")
        runtime_root = temp / "runtime"
        runtime_root.mkdir()
        outputs = temp / "outputs"
        exit_code, infrastructure = executor.run(
            ["@python", "__cf_executable_contract__/launcher.py"],
            work,
            runtime_root,
            bundle,
            outputs,
            temp / "command.log",
        )
        if repository_identity(work) != expected:
            infrastructure = "REPOSITORY_MUTATION"
        events = []
        for path in sorted(outputs.rglob("executable-contract.*.jsonl")):
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                events.append(json.loads(line))
        log = (temp / "command.log").read_text(encoding="utf-8", errors="replace")[-16000:]
    names = [event["event"] for event in events]
    assertion_by_line = {item["line"]: item["id"] for item in validation["runtime_assertion_ids"]}
    for event in events:
        if (
            event["event"] == "contract_violation"
            and event.get("runtime_assertion_line") in assertion_by_line
        ):
            event["runtime_assertion_id"] = assertion_by_line[event["runtime_assertion_line"]]
    assertion_event = next(
        (event for event in events if event["event"] == "runtime_assertions_observed"),
        {},
    )
    runtime_assertions_exercised = (
        bool(assertion_event.get("executed_runtime_assertion_lines")) or "contract_violation" in names
    )
    declared_assertion_lines = set(assertion_event.get("runtime_assertion_lines", []))
    executed_assertion_lines = set(
        assertion_event.get("executed_runtime_assertion_lines", [])
    )
    all_runtime_assertions_exercised = bool(declared_assertion_lines) and (
        declared_assertion_lines <= executed_assertion_lines
    )
    result = {
        "executable_contract_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "validation": validation,
        "executor_sha256": executor.identity()["sha256"],
        "events": events,
        "evaluation_harness_entered": "evaluation_harness_entered" in names,
        "contracted_operation_reached": "contracted_operation_reached" in names,
        "runtime_assertions_exercised": runtime_assertions_exercised,
        "all_runtime_assertions_exercised": all_runtime_assertions_exercised,
        "executed_runtime_assertion_lines": sorted(executed_assertion_lines),
        "outcome": "VIOLATED"
        if "contract_violation" in names
        else ("SATISFIED" if "contract_satisfied" in names else "ERROR"),
        "infrastructure_error": infrastructure,
        "command_exit": exit_code,
        "command_log_tail": log,
    }
    result["assertion_execution"] = _assertion_execution_receipt(result)
    result["execution_trace"] = _execution_trace(result)
    return result


def intended_violation(report: dict, violation_criterion: dict) -> bool:
    if report["infrastructure_error"]:
        return False
    if violation_criterion["kind"] == "reported_exception":
        operation_reached = report.get(
            "contracted_operation_reached",
            any(
                event.get("event") in {"contracted_operation_reached", "contracted_operation_raised"}
                for event in report["events"]
            ),
        )
        if not operation_reached:
            return False
        expected = violation_criterion["exception_type"]
        observed = {
            event.get("exception_type")
            for event in report["events"]
            if event["event"] == "contracted_operation_raised"
        }
        expected_observed = expected in observed or any(
            name and name.endswith("." + expected) for name in observed
        )
        assertion_violation = any(
            event.get("event") == "contract_violation" for event in report["events"]
        )
        if expected_observed and assertion_violation:
            return True
        # A traceback label can name an outer wrapper, while the grounded
        # contract requires a normal result at the selected operation's caller.
        # Such an assertion still needs semantic conformance review before an
        # EC can freeze; an uncaught harness error never qualifies here.
        return bool(
            violation_criterion.get("normal_assertion_fallback")
            and report.get("outcome") == "VIOLATED"
            and report.get("runtime_assertions_exercised")
            and assertion_violation
        )
    return report["outcome"] == "VIOLATED"
