"""Read only the requested commit; build a transparent, replaceable AST localizer."""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict, dataclass
import io
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
from typing import Any
import warnings

from contractfix.contracts.snapshots import repository_identity, validate_relative_symlink
from ...task import Task
from .models import ContextRequest, Edit

SKIP = {".git", ".venv", "__pycache__", "node_modules", ".tox"}
PUBLIC_TEST_PACKAGES = {("django", "test")}


def legacy_repair_index() -> bool:
    """Reproduce the pre-extension repository index for saved repair runs."""
    return os.getenv("CONTRACTFIX_REPAIR_INDEX_COMPAT") == "legacy"


def _parse_repository_source(source: str, filename: str) -> tuple[ast.Module, list[dict[str, Any]]]:
    """Parse immutable repository source while retaining non-fatal compiler warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SyntaxWarning)
        tree = ast.parse(source, filename=filename)
    diagnostics = [
        {
            "file": filename,
            "line": item.lineno,
            "category": item.category.__name__,
            "message": str(item.message),
        }
        for item in caught
        if issubclass(item.category, SyntaxWarning)
    ]
    return tree, diagnostics


def safe_path(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or any(part in SKIP for part in path.parts)):
        raise ValueError(f"unsafe repository path: {name}")
    return path.as_posix()


def production_path(name: str) -> bool:
    path = PurePosixPath(safe_path(name))
    if legacy_repair_index():
        return (
            path.suffix == ".py"
            and not any(part.lower() in {"test", "tests", "testing"} for part in path.parts)
            and not path.name.startswith("test_")
            and not path.name.endswith("_test.py")
            and not name.startswith("__cf_")
        )
    parts = path.parts
    excluded_test_parts = [
        part for position, part in enumerate(parts[:-1])
        if part.lower() in {"test", "tests", "testing"}
        and (position != 1 or parts[:2] not in PUBLIC_TEST_PACKAGES)
    ]
    return (path.suffix == ".py" and not excluded_test_parts
            and not path.name.startswith("test_")
            and not path.name.endswith("_test.py") and not name.startswith("__cf_"))


def repository_execution_examples(
    root: Path,
    query: str,
    operation_symbol: str,
    *,
    limit: int = 6,
    excerpt_chars: int = 6000,
) -> list[dict[str, Any]]:
    """Return bounded base-repository tests/examples useful for EC witness construction.

    These snippets are mechanical API usage context, not semantic evidence.  The
    snapshot contains only the task's base commit, so this never reads a gold patch
    or generated test.  Keeping this index separate from ``RepositoryIndex`` also
    prevents tests from becoming production repair locations.
    """
    candidates: list[tuple[float, str, int, dict[str, Any]]] = []
    query_terms = words(query + " " + operation_symbol)
    operation_name = operation_symbol.split(".")[-1]
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        parts = {part.lower() for part in PurePosixPath(relative).parts}
        is_usage_context = (
            bool(parts & {"test", "tests", "testing", "example", "examples"})
            or path.name.startswith("test_")
            or path.name.endswith("_test.py")
        )
        if (
            not is_usage_context
            or any(part in SKIP for part in path.parts)
            or path.stat().st_size > 1_000_000
        ):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree, _ = _parse_repository_source(source, relative)
        except (OSError, ValueError, SyntaxError, UnicodeError):
            continue
        lines = source.splitlines(keepends=True)
        imports = "".join(
            "".join(lines[node.lineno - 1 : (node.end_lineno or node.lineno)])
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        )[:2000]
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            start = min([node.lineno] + [item.lineno for item in node.decorator_list])
            snippet = "".join(lines[start - 1 : node.end_lineno])
            terms = words(relative + " " + node.name + " " + snippet[:excerpt_chars])
            score = float(
                sum(min(count, terms[token]) for token, count in query_terms.items())
            )
            if operation_name.lower() in snippet.lower():
                score += 40
            if operation_name.lower() in node.name.lower():
                score += 20
            if score <= 0:
                continue
            candidates.append(
                (
                    -score,
                    relative,
                    start,
                    {
                        "file": relative,
                        "symbol": node.name,
                        "start": start,
                        "module_imports": imports,
                        "source": snippet[:excerpt_chars],
                        "truncated": len(snippet) > excerpt_chars,
                        "usage_only": True,
                    },
                )
            )
    candidates.sort(key=lambda item: item[:3])
    examples = [item[3] for item in candidates[:limit]]
    for example in examples:
        example["fixture_context"] = _fixture_context(root, example)
    return examples


def _fixture_context(root: Path, example: dict[str, Any], *, max_chars: int = 3000) -> list[dict]:
    """Include bounded pytest setup, not just tests with unexplained injected args."""
    path = root / example["file"]
    paths = [path]
    parent = path.parent
    while parent == root or root in parent.parents:
        conftest = parent / "conftest.py"
        if conftest != path and conftest.is_file() and not conftest.is_symlink():
            paths.append(conftest)
        if parent == root:
            break
        parent = parent.parent
    definitions = {}
    requested = []
    for candidate in paths:
        if candidate.stat().st_size > 1_000_000:
            continue
        source = candidate.read_text(errors="replace")
        try:
            tree, _ = _parse_repository_source(source, candidate.name)
        except (SyntaxError, ValueError):
            continue
        lines = source.splitlines(keepends=True)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if candidate == path and node.name == example["symbol"]:
                requested = [arg.arg for arg in node.args.args]
            fixture = any(
                isinstance(decorator.func if isinstance(decorator, ast.Call) else decorator, (ast.Name, ast.Attribute))
                and (ast.unparse(decorator.func if isinstance(decorator, ast.Call) else decorator)
                     .split(".")[-1] == "fixture")
                for decorator in node.decorator_list
            )
            if fixture and node.name not in definitions:
                start = min([node.lineno] + [item.lineno for item in node.decorator_list])
                definitions[node.name] = (candidate, node,
                    "".join(lines[start - 1:node.end_lineno]))
    context = []
    seen = set()
    for name in requested:
        if name in seen or name not in definitions:
            continue
        seen.add(name)
        candidate, node, source = definitions[name]
        if len(source) > max_chars:
            continue
        context.append({"file": candidate.relative_to(root).as_posix(), "symbol": name,
                        "source": source, "usage_only": True})
        max_chars -= len(source)
        requested.extend(arg.arg for arg in node.args.args if arg.arg not in seen)
    return context


def git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=120)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace")[-1500:])
    return result.stdout


def snapshot(repo: Path, task: Task, destination: Path) -> str:
    """Export tracked base-commit files, never history, untracked files, or gold data."""
    repo = repo.resolve()
    actual = git(repo, "rev-parse", "HEAD").decode().strip()
    expected = git(repo, "rev-parse", task.base_commit + "^{commit}").decode().strip()
    if actual != expected:
        raise ValueError("checkout HEAD is not the task base_commit")
    if git(repo, "status", "--porcelain", "--untracked-files=no").strip():
        raise ValueError("tracked worktree changes are not allowed")
    destination.mkdir(parents=True, exist_ok=False)
    archive = git(repo, "archive", "--format=tar", expected)
    with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
        for member in contents.getmembers():
            name = safe_path(member.name)
            path = destination / name
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = contents.extractfile(member)
                if stream is None:
                    raise ValueError("unreadable archive member")
                path.write_bytes(stream.read())
                path.chmod(0o755 if member.mode & 0o111 else 0o644)
            elif member.issym():
                target = validate_relative_symlink(name, member.linkname)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.symlink_to(target)
            else:
                raise ValueError("submodules/special files are unsupported in v1")
    return repository_identity(destination)


def words(text: str) -> Counter:
    split = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return Counter(w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", split))


@dataclass(frozen=True)
class Location:
    id: str
    file: str
    symbol: str
    start: int
    end: int
    parameters: list[str]
    docstring: str
    source: str
    calls: list[str]
    kind: str = "function"

    def packet(self, limit: int = 2400) -> dict[str, Any]:
        result = asdict(self)
        if legacy_repair_index():
            result.pop("kind")
        result["source"] = self.source[:limit]
        result["docstring"] = self.docstring[:600]
        result.pop("calls")
        result["truncated"] = len(self.source) > limit
        return result

    def target(self, *, include_accessor: bool = False) -> dict[str, str]:
        target = {"file": self.file, "symbol": self.symbol}
        if include_accessor:
            name = re.escape(self.symbol.rsplit(".", 1)[-1])
            match = re.search(rf"(?m)^\s*@{name}\.(getter|setter|deleter)\s*$", self.source)
            if match:
                target["accessor"] = match.group(1)
            elif re.search(r"(?m)^\s*@property\s*$", self.source):
                target["accessor"] = "getter"
        return target


class RepositoryIndex:
    """Lexical ranking plus one-hop call neighbors, not a claimed SOTA localizer."""

    def __init__(self, root: Path):
        self.root = root
        self.locations: dict[str, Location] = {}
        self.parse_failures: list[str] = []
        self.parse_warnings: list[dict[str, Any]] = []
        module_assignments: list[tuple[str, ast.stmt, list[str]]] = []
        module_classes: list[tuple[str, ast.ClassDef, list[str]]] = []
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            if not production_path(relative) or path.stat().st_size > 1_000_000:
                continue
            try:
                source = path.read_text(encoding="utf-8")
                tree, diagnostics = _parse_repository_source(source, relative)
            except (ValueError, SyntaxError, UnicodeError):
                self.parse_failures.append(relative)
                continue
            self.parse_warnings.extend(diagnostics)
            lines = source.splitlines(keepends=True)
            self._collect(tree.body, relative, lines)
            module_assignments.extend(
                (relative, node, lines)
                for node in tree.body
                if isinstance(node, (ast.Assign, ast.AnnAssign))
            )
            module_classes.extend(
                (relative, node, lines)
                for node in tree.body if isinstance(node, ast.ClassDef)
            )
        # The original index exposed functions only. Preserve that exact repair
        # context for older saved Full attempts without changing newer runs.
        for relative, node, lines in ([] if legacy_repair_index() else module_assignments):
            names = (
                [target.id for target in node.targets if isinstance(target, ast.Name)]
                if isinstance(node, ast.Assign)
                else [node.target.id] if isinstance(node.target, ast.Name) else []
            )
            if len(names) != 1 or node.end_lineno is None:
                continue
            ident = "L" + str(len(self.locations) + 1)
            self.locations[ident] = Location(
                ident, relative, names[0], node.lineno, node.end_lineno,
                [], "", "".join(lines[node.lineno - 1:node.end_lineno]), [], "assignment",
            )
        for relative, node, lines in ([] if legacy_repair_index() else module_classes):
            ident = "L" + str(len(self.locations) + 1)
            self.locations[ident] = Location(
                ident, relative, node.name, node.lineno, node.end_lineno or node.lineno,
                [], ast.get_docstring(node) or "",
                "".join(lines[node.lineno - 1:(node.end_lineno or node.lineno)]),
                [], "class",
            )

    def _collect(self, body: list[ast.stmt], relative: str, lines: list[str], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                self._collect(node.body, relative, lines, prefix + node.name + ".")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ident = "L" + str(len(self.locations) + 1)
                params = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
                params += [a.arg for a in (node.args.vararg, node.args.kwarg) if a]
                calls = [n.func.id if isinstance(n.func, ast.Name) else n.func.attr
                         for n in ast.walk(node) if isinstance(n, ast.Call)
                         and isinstance(n.func, (ast.Name, ast.Attribute))]
                start = min([node.lineno] + [n.lineno for n in node.decorator_list])
                self.locations[ident] = Location(ident, relative, prefix + node.name,
                    start, node.end_lineno or node.lineno, params, ast.get_docstring(node) or "",
                    "".join(lines[start - 1:node.end_lineno]), calls)

    def search(self, issue: str, limit: int) -> list[Location]:
        query = words(issue)
        issue_lower = issue.lower()
        named_lines = {
            (filename.lower(), int(line))
            for filename, line in re.findall(
                r"([A-Za-z0-9_.-]+\.py)[\"']?\s*,?\s*(?:at\s+)?line\s+(\d+)",
                issue,
                flags=re.IGNORECASE,
            )
        }

        def score(item: Location) -> float:
            terms = words(item.file + " " + item.symbol + " " + item.docstring + " " + item.source[:4000])
            value = sum(min(count, terms[token]) for token, count in query.items())
            if (
                item.symbol.split(".")[-1].lower() in issue_lower
                if legacy_repair_index()
                else re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(item.symbol.split('.')[-1])}(?![A-Za-z0-9_])",
                    issue,
                    flags=re.IGNORECASE,
                )
            ):
                value += 30
            file_lower = item.file.lower()
            basename = PurePosixPath(item.file).name.lower()
            if file_lower in issue_lower:
                value += 30
            elif basename in issue_lower:
                value += 30
            elif not legacy_repair_index():
                module_parts = PurePosixPath(item.file[:-3]).parts
                if len(module_parts) >= 2:
                    module_suffix = ".".join(module_parts[-2:]).lower()
                    if re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(module_suffix)}(?![A-Za-z0-9_])",
                        issue_lower,
                    ):
                        value += 40
            if any(
                basename == filename and item.start <= line <= item.end
                for filename, line in named_lines
            ):
                value += 60
            return value
        ranked = sorted(self.locations.values(), key=lambda loc: (-score(loc), loc.id))
        selected = ranked[:max(1, limit - 2)]
        names = {loc.symbol.split(".")[-1] for loc in selected[:3]}
        called = {name for loc in selected[:3] for name in loc.calls}
        neighbors = [loc for loc in ranked if loc not in selected and
                     (set(loc.calls) & names or loc.symbol.split(".")[-1] in called)]
        return (selected + neighbors[:2] + [loc for loc in ranked if loc not in selected and loc not in neighbors[:2]])[:limit]

    def requested_context(
        self,
        requests: list[ContextRequest],
        known: set[str],
        limit: int,
    ) -> tuple[list[Location], set[str]]:
        """Resolve bounded model requests through host-controlled repository queries."""
        additions: list[Location] = []
        expanded: set[str] = set()
        queues: list[list[Location]] = []

        for request in requests:
            if request.kind == "search":
                symbol_query = re.fullmatch(
                    r"(?:(?:class|def)\s+)?([A-Za-z_][A-Za-z0-9_]*)",
                    request.target.strip(),
                )
                if symbol_query:
                    symbol = symbol_query.group(1)
                    exact = [
                        loc for loc in self.locations.values()
                        if symbol in loc.symbol.split(".") and loc.id not in known
                    ]
                    if exact:
                        ranked_exact = sorted(exact, key=lambda item: (item.file, item.start))
                        if legacy_repair_index():
                            queues.append(ranked_exact)
                        else:
                            methods = [item for item in ranked_exact if item.kind == "function"]
                            classes = [item for item in ranked_exact if item.kind == "class"]
                            others = [item for item in ranked_exact
                                      if item.kind not in {"function", "class"}]
                            queues.append(methods[:1] + classes + methods[1:] + others)
                        continue
                # Known hits must not consume the new-context allowance.
                queues.append([loc for loc in self.search(request.target, limit + len(known))
                               if loc.id not in known])
                continue
            location = self.locations.get(request.target)
            if location is None or location.id not in known:
                raise ValueError(f"context request targets unavailable location: {request.target}")
            if request.kind == "expand":
                expanded.add(location.id)
            elif request.kind == "callers":
                name = location.symbol.split(".")[-1]
                queues.append(sorted(
                    [candidate for candidate in self.locations.values()
                     if name in candidate.calls and candidate.id not in known],
                    key=lambda item: (item.file != location.file, item.file, item.start),
                ))
            elif request.kind == "callees":
                names = set(location.calls)
                queues.append(sorted(
                    [candidate for candidate in self.locations.values()
                     if candidate.symbol.split(".")[-1] in names and candidate.id not in known],
                    key=lambda item: (item.file != location.file, item.file, item.start),
                ))
        # A broad first search must not consume every slot and starve the
        # model's other targeted requests in the same round.
        while any(queues) and len(additions) < limit:
            for queue in queues:
                while queue and queue[0] in additions:
                    queue.pop(0)
                if queue and len(additions) < limit:
                    additions.append(queue.pop(0))
        return additions, expanded


def evidence_packet(task: Task, selected: list[Location]) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {
        "E1": {"kind": "issue", "text": task.problem_statement, "source": "problem_statement"}}
    for location in selected:
        evidence[f"DOC_{location.id}"] = {"kind": "docstring", "text": location.docstring,
            "source": location.file, "symbol": location.symbol, "start": location.start}
        evidence[f"CODE_{location.id}"] = {"kind": "observation_of_buggy_code", "text": location.source,
            "source": location.file, "symbol": location.symbol, "start": location.start}
    return evidence


def make_patch(root: Path, edits: list[Edit], allowed: list[str]) -> str:
    """Compatibility wrapper for strict deterministic patch materialization."""
    from .patching import materialize_edits
    return materialize_edits(root, edits, allowed).text
