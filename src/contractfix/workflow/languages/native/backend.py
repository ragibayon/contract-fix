"""Java, C, and C++ qualification over prepared repositories and native toolchains."""

from __future__ import annotations

import difflib
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
from time import perf_counter

from contractfix.config import LLMSettings
from contractfix.contracts.execution import Executor
from contractfix.contracts.snapshots import repository_identity, validate_relative_symlink
from contractfix.llm.structured import ReasoningOnlyOutputBudgetError
from contractfix.utils.artifacts import RunArtifacts, digest, redact
from contractfix.utils.logger import logger

from ...accounting import model_accounting
from ...selection import select_with_ec
from ...task import Task
from ...stages import Stages
from ..python.stages import LangChainStages
from ..python.repository import git, snapshot
from ..python.repair_models import PatchContextRequest
from .models import (
    NativeEC, NativeLocalization, NativeNLC, NativeNLCReview, NativeSettings,
    NativePatchProposal, PatchProposal,
)
from .prompts import NativePromptBook
from .prompt_packs import prompt_pack_for
from .javascript import build_commands as _javascript_build_commands
from .javascript import prepare_work as _prepare_javascript_work
from .javascript import requires_witness_build as _javascript_requires_witness_build

SOURCE_SUFFIXES = {
    "java": frozenset({".java"}),
    "c": frozenset({".c", ".h", ".y"}),
    "cpp": frozenset({".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh"}),
    "javascript": frozenset({".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"}),
}
SKIP = frozenset({".git", ".venv", "build", "target", "node_modules", ".gradle"})
JS_SKIP = frozenset({"__tests__", "__mocks__", "__fixtures__", "fixtures",
                     "fixture", "coverage", "dist", "vendor"})
TOKEN = re.compile(r"[a-z][a-z0-9_]+")


def _native_generate(stages: Stages, stage: str, schema: type, packet: dict):
    """Retry one reasoning-only completion without changing evidence or policy."""
    try:
        return stages.generate(stage, schema, packet)
    except ReasoningOnlyOutputBudgetError:
        retry_packet = {
            **packet,
            "reasoning_only_budget_retry": 1,
            "retry_instruction": (
                "Return the requested structured answer now. Use the same evidence "
                "and keep the response concise; do not add free-form discussion."
            ),
        }
        logger.warning(f"[native {stage}] reasoning-only output exhausted; retrying once")
        return stages.generate(stage, schema, retry_packet)


def _source_path(root: Path, name: str, language: str) -> Path:
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or ".." in relative.parts or "\\" in name
            or set(relative.parts) & SKIP or relative.suffix not in SOURCE_SUFFIXES[language]
            or any(part.lower() in {"test", "tests", "testing"} for part in relative.parts)
            or relative.stem.lower().startswith("test")
            or (language == "javascript" and (
                name.endswith(".d.ts")
                or any(part.lower() in JS_SKIP for part in relative.parts[:-1])
                or relative.stem.lower().endswith((".test", ".spec"))
            ))):
        raise ValueError(f"invalid production {language} source path: {name}")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink source path: {name}")
    if not current.is_file() or not current.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"source file missing: {name}")
    return current


def _index(root: Path, language: str, issue: str, limit: int = 8) -> list[dict]:
    terms = set(TOKEN.findall(issue.lower()))
    opening_terms = set(TOKEN.findall(issue[:1000].lower()))
    scored = []
    for path in _walk_source_candidates(root):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES[language]:
            continue
        name = path.relative_to(root).as_posix()
        try:
            _source_path(root, name, language)
            if path.stat().st_size > 500_000:
                continue
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError, ValueError):
            continue
        source_terms = set(TOKEN.findall((name + " " + source[:6000]).lower()))
        score = len(terms & source_terms)
        if language == "c":
            # Long CLI transcripts contain many incidental words. Give the
            # opening issue description and the project's source tree extra
            # weight over bundled third-party dependencies.
            score += 2 * len(opening_terms & source_terms)
            if name.startswith(("src/", "py/")):
                score += 4
        if path.stem.lower() in terms:
            score += 20
        scored.append((-score, name, {"path": name, "source": source[:2500]}))
    scored.sort(key=lambda row: (row[0], row[1]))
    return [row[2] for row in scored[:limit]]


def _walk_source_candidates(root: Path):
    """Avoid descending into installed dependencies and generated build trees."""
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = [name for name in names if name not in SKIP]
        for name in files:
            yield Path(directory) / name


def _ground_localization_paths(
    root: Path, language: str, sources: list[dict], localization: NativeLocalization,
) -> tuple[list[dict], dict | None]:
    """Retrieve a real base source named by localization but missed by ranking."""
    indexed = {row["path"] for row in sources}
    missing = set(localization.repair_paths) | {localization.operation_file}
    missing -= indexed
    if not missing:
        return sources, None
    if len(missing) > 2:
        raise ValueError("localization selected too many paths outside supplied source")
    recovered = []
    for name in sorted(missing):
        path = _source_path(root, name, language)
        if path.stat().st_size > 500_000:
            raise ValueError("localization selected oversized source")
        recovered.append({"path": name, "source": path.read_text(encoding="utf-8")[:2500]})
    return [*sources, *recovered], {
        "reason": "localization named existing production source omitted by initial ranking",
        "recovered_paths": sorted(missing),
    }


def _native_snapshot(repo: Path, task: Task, destination: Path, language: str) -> str:
    """Export the base commit and retained benchmark build instrumentation.

    Some official images have tracked Gradle/Ant harness edits. They are kept
    in this disposable native snapshot, but production source must still be
    exactly the task base commit. The prepared image itself is never changed.
    """
    actual = git(repo, "rev-parse", "HEAD").decode().strip()
    expected = git(repo, "rev-parse", task.base_commit + "^{commit}").decode().strip()
    if actual != expected:
        raise ValueError("checkout HEAD is not the task base_commit")
    gitlinks = {}
    for entry in git(repo, "ls-tree", "-r", "-z", "HEAD").split(b"\0"):
        if entry.startswith(b"160000 commit "):
            metadata, raw_name = entry.split(b"\t", 1)
            name = raw_name.decode()
            gitlinks[name] = metadata.split()[-1].decode()
    dirty = set(git(repo, "diff", "--name-only", "HEAD").decode().splitlines())
    if not dirty and not gitlinks and language != "javascript":
        return snapshot(repo, task, destination)
    for name in dirty - gitlinks.keys():
        path = repo / _native_archive_path(name)
        if (path.suffix in SOURCE_SUFFIXES[language] or not path.is_file()
                or path.is_symlink()):
            raise ValueError(f"tracked production or unsafe worktree changes: {name}")
    destination.mkdir(parents=True, exist_ok=False)
    archive = git(repo, "archive", "--format=tar", expected)
    with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
        for member in contents.getmembers():
            name = _native_archive_path(member.name)
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
                raise ValueError("submodules/special files are unsupported in native snapshot")
    omitted_submodules = []
    for name, commit in sorted(gitlinks.items()):
        _native_archive_path(name)
        submodule = repo / name
        if submodule.is_symlink() or not submodule.is_dir():
            raise ValueError(f"missing or unsafe submodule: {name}")
        checked_out = git(submodule, "rev-parse", "HEAD").decode().strip()
        pinned_available = subprocess.run(
            ["git", "-C", str(submodule), "cat-file", "-e", commit + "^{commit}"],
            capture_output=True, timeout=30,
        ).returncode == 0
        if not pinned_available:
            omitted_submodules.append({"path": name, "pinned_commit": commit,
                                       "checked_out_commit": checked_out})
            continue
        changed = (git(submodule, "diff", "--name-only", "HEAD").decode().splitlines()
                   if checked_out == commit else [])
        for relative in changed:
            source = submodule / _native_archive_path(relative)
            if (source.suffix in SOURCE_SUFFIXES[language] or not source.is_file()
                    or source.is_symlink()):
                raise ValueError(f"tracked production or unsafe worktree changes: {name}/{relative}")
        sub_archive = git(submodule, "archive", "--format=tar", commit)
        with tarfile.open(fileobj=io.BytesIO(sub_archive)) as contents:
            for member in contents.getmembers():
                relative = _native_archive_path(member.name)
                target = destination / name / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    stream = contents.extractfile(member)
                    if stream is None:
                        raise ValueError("unreadable submodule archive member")
                    target.write_bytes(stream.read())
                    target.chmod(0o755 if member.mode & 0o111 else 0o644)
                elif member.issym():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(validate_relative_symlink(
                        PurePosixPath(name) / relative, member.linkname,
                    ))
                else:
                    raise ValueError("nested submodules/special files are unsupported")
        for relative in changed:
            target = destination / name / relative
            if target.is_symlink() or not target.is_file():
                raise ValueError(f"unsafe instrumented path: {name}/{relative}")
            target.write_bytes((submodule / relative).read_bytes())
    for name in sorted(dirty - gitlinks.keys()):
        target = destination / name
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"unsafe instrumented path: {name}")
        target.write_bytes((repo / name).read_bytes())
    if omitted_submodules:
        (destination / ".contractfix_omitted_submodules.json").write_text(
            json.dumps(omitted_submodules, sort_keys=True) + "\n", encoding="utf-8",
        )
    return repository_identity(destination)


def _native_archive_path(name: str) -> str:
    """Validate tracked archive paths without excluding fixture node_modules."""
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or ".." in relative.parts
            or "\\" in name or "\x00" in name or ".git" in relative.parts):
        raise ValueError(f"unsafe native archive path: {name}")
    return relative.as_posix()


def _operation_excerpt(source: str, symbol: str, *, limit: int = 14000) -> str:
    """Give downstream stages the bounded method neighborhood, not file prefix."""
    if len(source) <= limit:
        return source
    name = re.split(r"::|\.", symbol)[-1]
    matches = [match for match in re.finditer(
        r"\b" + re.escape(name) + r"\s*\(", source
    ) if not source[source.rfind("\n", 0, match.start()) + 1:match.start()].lstrip().startswith(
        ("*", "//")
    )]

    def definition(match: re.Match[str]) -> bool:
        closing = source.find(")", match.end())
        if closing < 0:
            return False
        suffix = source[closing + 1:closing + 240]
        before_brace, separator, _ = suffix.partition("{")
        return bool(separator) and ";" not in before_brace

    definitions = [match for match in matches if definition(match)]
    candidates = definitions or matches
    owner = re.search(r"<(?:(?:detail|std)::)?([A-Za-z_]\w*)", symbol)
    if owner:
        preceding = [(match, source.rfind(owner.group(1), 0, match.start()))
                     for match in candidates]
        owned = [(match, match.start() - position) for match, position in preceding
                 if position >= 0 and match.start() - position < 3000]
        center = min(owned, key=lambda item: item[1])[0].start() if owned else (
            candidates[0].start() if candidates else 0
        )
    else:
        center = candidates[0].start() if candidates else 0
    start = max(0, center - limit // 3)
    end = min(len(source), start + limit)
    start = max(0, end - limit)
    if start:
        start = source.find("\n", start) + 1
    if end < len(source):
        end = source.rfind("\n", start, end)
    return source[start:end]


def _localized_sources(root: Path, language: str, localization: NativeLocalization) -> list[dict]:
    rows = []
    for name in dict.fromkeys([localization.operation_file, *localization.repair_paths]):
        source = _source_path(root, name, language).read_text(encoding="utf-8")
        rows.append({
            "path": name,
            "source": _operation_excerpt(source, localization.operation_symbol)
            if name == localization.operation_file else source[:6000],
        })
    return rows


class NativePatchContext:
    """Serve bounded source requests from the frozen base, with explicit edit scope."""

    def __init__(self, base: Path, language: str, allowed_paths: list[str], *,
                 max_excerpts: int = 8, max_chars: int = 14000):
        self.base = base
        self.language = language
        self.allowed_paths = list(allowed_paths)
        self.max_excerpts = max_excerpts
        self.max_chars = max_chars
        self.read_paths = set(allowed_paths)
        self.excerpts: list[dict] = []

    def _add(self, name: str, line: int, end_line: int | None = None) -> bool:
        path = _source_path(self.base, name, self.language)
        if path.stat().st_size > 500_000:
            raise ValueError("requested source exceeds size limit")
        lines = path.read_text(encoding="utf-8").splitlines()
        if line < 1 or line > len(lines):
            raise ValueError("requested source line is outside file")
        if end_line is not None and (end_line < line or end_line - line >= 120):
            raise ValueError("expand range must contain 1 to 120 lines")
        covered_ranges = [
            (row["start_line"], row["end_line"])
            for row in self.excerpts if row["path"] == name
        ]
        if end_line is None:
            if any(first <= line <= last for first, last in covered_ranges):
                return False
        else:
            requested_end = min(len(lines), end_line)
            missing = next((number for number in range(line, requested_end + 1)
                            if not any(first <= number <= last
                                       for first, last in covered_ranges)), None)
            if missing is None:
                return False
            line = missing
            end_line = next((number - 1 for number in range(line + 1, requested_end + 1)
                             if any(first <= number <= last
                                    for first, last in covered_ranges)), requested_end)
        remaining = self.max_chars - sum(len(row["source"]) for row in self.excerpts)
        if remaining <= 0 or len(self.excerpts) >= self.max_excerpts:
            raise ValueError("native context budget exhausted")
        start = line if end_line is not None else max(1, line - 20)
        stop = min(len(lines), end_line if end_line is not None else line + 60)
        limit = min(2400, remaining)
        def complete_lines(first: int) -> list[str]:
            selected: list[str] = []
            used = 0
            for source_line in lines[first - 1:stop]:
                extra = len(source_line) + bool(selected)
                if used + extra > limit:
                    break
                selected.append(source_line)
                used += extra
            return selected

        selected = complete_lines(start)
        if start + len(selected) - 1 < line:
            start = line
            selected = complete_lines(start)
        if not selected:
            raise ValueError("requested source line exceeds remaining context budget")
        excerpt = "\n".join(selected)
        covered = start + len(selected) - 1
        self.read_paths.add(name)
        self.excerpts.append({"path": name, "start_line": start, "end_line": covered,
                              "source": excerpt,
                              "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return True

    def _search(self, target: str) -> bool:
        scoped_path = None
        if ":" in target:
            prefix, query = target.split(":", 1)
            try:
                _source_path(self.base, prefix.strip(), self.language)
            except (OSError, ValueError):
                pass
            else:
                scoped_path, target = prefix.strip(), query
        tokens = sorted(set(re.findall(r"[A-Za-z_][A-Za-z_0-9]{3,}", target)),
                        key=lambda token: (-len(token), token))[:8]
        if not tokens:
            raise ValueError("search contains no source identifier")
        previously_excerpted = {row["path"] for row in self.excerpts}
        matches = []
        for path in _walk_source_candidates(self.base):
            name = path.relative_to(self.base).as_posix()
            if scoped_path is not None and name != scoped_path:
                continue
            try:
                source = _source_path(self.base, name, self.language)
                if source.stat().st_size > 500_000:
                    continue
                lines = source.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError, ValueError):
                continue
            for line_number, line in enumerate(lines, 1):
                candidate_start = max(1, line_number - 20)
                if any(
                    row["path"] == name
                    and candidate_start < row["start_line"] + 80
                    and row["start_line"] < candidate_start + 80
                    for row in self.excerpts
                ):
                    continue
                hits = [token for token in tokens
                        if re.search(r"\b" + re.escape(token) + r"\b", line)]
                if hits:
                    score = max(map(len, hits))
                    if any(re.search(
                        r"\b" + re.escape(token) + r"\s*\([^)]*\)\s*\{",
                        line,
                    ) for token in hits):
                        score += 20
                    if path.suffix in {".c", ".cc", ".cpp", ".cxx", ".java", ".js", ".ts"}:
                        score += 3
                    if name in previously_excerpted:
                        score += 5
                    if "/test" in name or name.startswith("test"):
                        score -= 10
                    matches.append((-score, name, line_number))
        for _, name, line_number in sorted(matches):
            if self._add(name, line_number):
                return True
        return False

    def request(self, requests: list[PatchContextRequest]) -> dict:
        outcomes = []
        for item in requests[:3]:
            try:
                if item.kind == "search":
                    added = self._search(item.target)
                elif item.kind == "read_file":
                    added = self._add(item.target, item.start_line)
                elif item.kind == "expand":
                    path, separator, location = item.target.partition(":")
                    if not separator:
                        raise ValueError("expand requires path:line, path:start-end, or path:symbol")
                    if location.strip().isdigit():
                        added = self._add(path.strip(), int(location.strip()))
                    elif re.fullmatch(r"\s*\d+\s*-\s*\d+\s*", location):
                        first, last = (int(value.strip()) for value in location.split("-", 1))
                        added = self._add(path.strip(), first, last)
                    else:
                        added = self._search(item.target)
                elif item.kind == "allow_edit":
                    _source_path(self.base, item.target, self.language)
                    if item.target not in self.read_paths:
                        raise ValueError("read source before requesting edit scope")
                    if item.target in self.allowed_paths:
                        added = False
                    elif len(self.allowed_paths) >= 4:
                        raise ValueError("native edit scope file budget exhausted")
                    else:
                        self.allowed_paths.append(item.target)
                        added = True
                else:
                    raise ValueError("unsupported native context request")
                outcome = {"request": item.model_dump(),
                           "status": "ADDED" if added else "NO_NEW_CONTEXT"}
                if added and item.kind != "allow_edit" and self.excerpts:
                    found = self.excerpts[-1]["path"]
                    outcome["source_path"] = found
                    if found not in self.allowed_paths:
                        outcome["suggested_next_request"] = {
                            "kind": "allow_edit", "target": found,
                        }
                outcomes.append(outcome)
            except (OSError, UnicodeError, ValueError) as exc:
                outcomes.append({"request": item.model_dump(), "status": "REJECTED",
                                 "detail": str(exc)[:300]})
        return {"outcomes": outcomes, "allowed_paths": list(self.allowed_paths),
                "retrieved_context": list(self.excerpts)}


def _c_command_target(root: Path, command: str) -> tuple[str, str, str] | None:
    """Resolve one command to a unique defining function from checked-in data."""
    definition = root / "src/commands" / f"{command}.json"
    if not command or not definition.is_file() or definition.is_symlink():
        return None
    metadata = json.loads(definition.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or len(metadata) != 1:
        return None
    entry = next(iter(metadata.values()))
    actual = entry.get("function") if isinstance(entry, dict) else None
    if not isinstance(actual, str) or not re.fullmatch(r"[A-Za-z_]\w*", actual):
        return None
    matches = [name for name in sorted((root / "src").glob("*.c"))
               if re.search(r"\b" + re.escape(actual) + r"\s*\([^)]*\)\s*\{",
                            name.read_text(encoding="utf-8", errors="replace"))]
    if len(matches) != 1:
        return None
    return actual, matches[0].relative_to(root).as_posix(), definition.relative_to(root).as_posix()


def _ground_c_command_operation(
    root: Path, localization: NativeLocalization,
) -> tuple[NativeLocalization, dict | None]:
    """Resolve a Redis command alias using its checked-in command definition."""
    source = (root / localization.operation_file).read_text(encoding="utf-8")
    if re.search(r"\b" + re.escape(localization.operation_symbol) + r"\s*\(", source):
        return localization, None
    command = localization.operation_symbol.removesuffix("Command").lower()
    target = _c_command_target(root, command)
    if target is None:
        return localization, None
    actual, operation_file, definition = target
    updated = localization.model_copy(update={
        "operation_symbol": actual, "operation_file": operation_file,
        "repair_paths": list(dict.fromkeys(
            operation_file if name == localization.operation_file else name
            for name in localization.repair_paths
        )),
    })
    return updated, {
        "reason": "command metadata maps alias to function in selected base source",
        "definition": definition,
        "declared_symbol": localization.operation_symbol,
        "grounded_symbol": actual,
        "grounded_file": operation_file,
    }


def _ground_c_issue_command(
    root: Path, issue: str, localization: NativeLocalization,
) -> tuple[NativeLocalization, dict | None]:
    """Correct a wrong operation when the issue title names one checked-in command."""
    title = re.sub(r"^\s*(?:\[[^]]+\]\s*)+", "", issue.splitlines()[0])
    commands = re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", title)
    if len(commands) != 1:
        return localization, None
    target = _c_command_target(root, commands[0].lower())
    if target is None:
        return localization, None
    actual, operation_file, definition = target
    if (localization.operation_symbol == actual
            and localization.operation_file == operation_file):
        return localization, None
    updated = localization.model_copy(update={
        "operation_symbol": actual, "operation_file": operation_file,
        "repair_paths": list(dict.fromkeys(
            operation_file if name == localization.operation_file else name
            for name in localization.repair_paths
        )),
    })
    return updated, {
        "reason": "issue title names one command with a unique checked-in implementation",
        "issue_command": commands[0], "definition": definition,
        "declared_symbol": localization.operation_symbol,
        "grounded_symbol": actual, "grounded_file": operation_file,
    }


def _native_ec_links_selected_operation(
    source: str, localization: NativeLocalization, language: str = "c",
) -> bool:
    """Reject standalone C/C++ models that never compile selected repository code."""
    selected = localization.operation_file
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL)
    include = re.compile(r"(?m)^\s*#\s*include\s*[\"<]([^\">]+)[\">]")
    if not any(name == selected or selected.endswith("/" + name)
               for name in include.findall(code)):
        return False
    symbol = localization.operation_symbol.rsplit("::", 1)[-1]
    if re.search(r"\b" + re.escape(symbol) + r"\s*\(", code):
        return True
    if language == "cpp" and PurePosixPath(selected).suffix in {".h", ".hh", ".hpp"}:
        # A public namespace API may exercise an internal template operation
        # in the included repository header. The later semantic review and
        # buggy-base execution still decide whether it is a valid EC.
        return bool(re.search(r"\b[A-Za-z_]\w*(?:::[A-Za-z_]\w*)+\s*\(", code))
    return False


def _ground_cpp_operation(
    root: Path, localization: NativeLocalization, indexed: set[str],
) -> tuple[NativeLocalization, dict | None]:
    """Correct a C++ template owner only when one supplied file proves it."""
    symbol = localization.operation_symbol
    template = re.search(r"<([^<>]+)>", symbol)
    if template is None or "::" not in symbol:
        return localization, None
    owner = template.group(1).strip()
    method = symbol.rsplit("::", 1)[-1]
    if not owner or not method:
        return localization, None

    def contains(name: str) -> bool:
        source = (root / name).read_text(encoding="utf-8", errors="replace")
        return owner in source and re.search(r"\b" + re.escape(method) + r"\s*\(", source) is not None

    if contains(localization.operation_file):
        return localization, None
    matches = [name for name in sorted(indexed) if contains(name)]
    if len(matches) != 1:
        return localization, None
    old, new = localization.operation_file, matches[0]
    updated = localization.model_copy(update={
        "operation_file": new,
        "repair_paths": list(dict.fromkeys(new if name == old else name
                                          for name in localization.repair_paths)),
    })
    return updated, {"reason": "unique supplied template owner", "from": old, "to": new,
                     "owner": owner, "method": method}


def _normalize_evidence_paths(paths: list[str], supplied: set[str]) -> tuple[list[str], dict[str, str]]:
    """Reduce source-qualified citations to their supplied file path."""
    normalized = []
    changes = {}
    for citation in paths:
        if citation == "task.problem_statement" and "problem_statement" in supplied:
            normalized.append("problem_statement")
            changes[citation] = "problem_statement"
            continue
        if citation in supplied:
            normalized.append(citation)
            continue
        matches = [name for name in supplied
                   if citation.startswith((name + "::", name + ": ", name + "#"))]
        if len(matches) != 1:
            raise ValueError("NLC evidence references unsupplied source")
        normalized.append(matches[0])
        changes[citation] = matches[0]
    return list(dict.fromkeys(normalized)), changes


def materialize_native_edits(root: Path, proposal: PatchProposal, allowed: list[str], language: str) -> tuple[str, dict[str, str]]:
    """Materialize exact replacements without modifying the frozen base."""
    if proposal.action != "submit_edits":
        raise ValueError("patch proposal did not submit edits")
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    for edit in proposal.edits:
        name = edit.file
        if name not in allowed:
            raise ValueError(f"edit outside localized paths: {name}")
        path = _source_path(root, name, language)
        if name not in before:
            before[name] = path.read_text(encoding="utf-8")
            after[name] = before[name]
        old = "\n".join(edit.old_lines)
        new = "\n".join(edit.new_lines)
        if not old.strip() or after[name].count(old) != 1:
            raise ValueError(f"exact edit does not match once: {name}")
        after[name] = after[name].replace(old, new, 1)
    chunks = []
    for name in sorted(after):
        if before[name] == after[name]:
            continue
        chunks.extend(difflib.unified_diff(
            before[name].splitlines(keepends=True), after[name].splitlines(keepends=True),
            fromfile="a/" + name, tofile="b/" + name,
        ))
    patch = "".join(chunks)
    if not patch:
        raise ValueError("empty native patch")
    check = subprocess.run(["git", "apply", "--check", "-"], input=patch.encode(),
                           cwd=root, capture_output=True, timeout=30)
    if check.returncode:
        raise ValueError("native diff failed git apply check: " + check.stderr.decode(errors="replace")[-300:])
    return patch, after


def _exact_edit_context(root: Path, proposal: PatchProposal, language: str) -> list[dict]:
    """Return bounded, exact base lines after a model's edit text misses."""
    contexts = []
    for edit in proposal.edits:
        if len(contexts) == 2:
            break
        try:
            source = _source_path(root, edit.file, language).read_text(encoding="utf-8")
        except (OSError, UnicodeError, ValueError):
            continue
        lines = source.splitlines()
        anchors = [line for line in edit.old_lines if len(line.strip()) >= 8]
        position = None
        for anchor in anchors:
            exact = [index for index, line in enumerate(lines) if line == anchor]
            if len(exact) == 1:
                position = exact[0]
                break
            stripped = [index for index, line in enumerate(lines)
                        if line.strip() == anchor.strip()]
            if len(stripped) == 1:
                position = stripped[0]
                break
        if position is None:
            contexts.append({"file": edit.file, "status": "NO_UNIQUE_ANCHOR",
                             "instruction": "Request exact source context before resubmitting the edit."})
            continue
        excerpt = []
        for index in range(max(0, position - 3), min(len(lines), position + 22)):
            row = {"line": index + 1, "text": lines[index]}
            if len(json.dumps([*excerpt, row])) > 2400:
                break
            excerpt.append(row)
        contexts.append({"file": edit.file, "status": "EXACT_BASE_LINES",
                         "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                         "source_lines": excerpt})
    return contexts


def _native_modules(root: Path, paths: list[str], marker: str) -> list[str]:
    """Select build modules from localized production paths, not benchmark tests."""
    modules = []
    for name in paths:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid localized build path")
        for parent in relative.parents:
            if str(parent) == ".":
                break
            if (root / parent / marker).is_file():
                modules.append(parent.as_posix())
                break
    return list(dict.fromkeys(modules))


def _native_commands(root: Path, language: str,
                     source_paths: list[str] | None = None) -> list[list[str]]:
    source_paths = source_paths or []
    if language == "javascript":
        return _javascript_build_commands(root, source_paths)
    if language == "java":
        if (root / "mvnw").is_file():
            modules = _native_modules(root, source_paths, "pom.xml")
            return [["./mvnw", "-B", "-DskipTests",
                     *(["-pl", ",".join(modules), "-am"] if modules else []),
                     "install" if len(modules) > 1 else "compile"]]
        if (root / "pom.xml").is_file():
            modules = _native_modules(root, source_paths, "pom.xml")
            return [["mvnd", "-B", "-DskipTests",
                     *(["-pl", ",".join(modules), "-am"] if modules else []),
                     "install" if len(modules) > 1 else "compile"]]
        if (root / "gradlew").is_file():
            modules = _native_modules(root, source_paths, "build.gradle")
            targets = [":" + module.replace("/", ":") + ":compileJava"
                       for module in modules] or ["compileJava"]
            return [["./gradlew", "--no-daemon", *targets]]
        if (root / "build.xml").is_file():
            return [["ant", "test.compile"]]
    else:
        if (root / "CMakeLists.txt").is_file():
            return [["cmake", "-S", ".", "-B", "build"],
                    ["cmake", "--build", "build", "-j2"]]
        if (root / "Makefile").is_file():
            return [["make", "-j2"]]
        if (root / "configure").is_file():
            return [["./configure"], ["make", "-j2"]]
        if (root / "configure.ac").is_file():
            return [["autoreconf", "-fi"], ["./configure"], ["make", "-j2"]]
        if language == "c" and (root / "ports/unix/Makefile").is_file():
            return [["make", "-C", "mpy-cross", "-j2"],
                    ["make", "-C", "ports/unix", "-j2"]]
    return []


def _build_failure(check: dict) -> str | None:
    if check.get("error") == "NO_NATIVE_BUILD_COMMAND":
        return "HARNESS_UNAVAILABLE"
    if check.get("error") == "timeout":
        return "TIMEOUT"
    if check.get("error"):
        return "INFRASTRUCTURE_ERROR"
    if check.get("exit_code") == 0:
        return None
    output = check.get("log_tail", "").lower()
    if (re.search(r"(?m)^/testbed/[^\n]+:\d+(?::\d+)?: error:", output)
            or "error: cannot find symbol" in output
            or "compilation failure" in output):
        return "COMPILER_ERROR"
    if ("could not resolve dependencies" in output or "cannot access" in output
            or "could not find" in output or "not found" in output
            or "unknownhostexception" in output
            or "org.gradle.wrapper.download" in output):
        return "HARNESS_DEPENDENCY_UNAVAILABLE"
    if "permission denied" in output or "no such file or directory" in output:
        return "HARNESS_ERROR"
    if "error:" in output or "compilation failure" in output:
        return "COMPILER_ERROR"
    return "BUILD_ERROR"


def _first_native_diagnostic(output: str) -> str | None:
    """Keep the actionable compiler error when a long build log hides it."""
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if re.search(r"\b(?:fatal )?error:\s|\bSyntaxError:\s|\berror TS\d+:\s",
                     line, flags=re.IGNORECASE):
            return "\n".join(lines[max(0, index - 1):index + 3])[:1200]
    return None


def _execute(executor: Executor, command: list[str], work: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / "bundle.json"
    bundle.write_text("{}\n", encoding="utf-8")
    log = output / "command.log"
    start = perf_counter()
    try:
        code, error = executor.run(
            command, work, Path(__file__).resolve().parents[4], bundle, output, log,
        )
    except Exception as exc:
        code, error = None, f"INFRASTRUCTURE_ERROR:{type(exc).__name__}"
        log.write_text(str(redact(str(exc)))[:1000], encoding="utf-8")
    log_text = log.read_text(encoding="utf-8", errors="replace")
    return {
        "command": command, "exit_code": code, "error": error,
        "log_tail": log_text[-4000:],
        "first_diagnostic": _first_native_diagnostic(log_text),
        "seconds": round(perf_counter() - start, 3),
    }


def _native_build(executor: Executor, work: Path, output: Path, language: str,
                  source_paths: list[str] | None = None) -> dict:
    if language == "javascript":
        _prepare_javascript_work(work)
    _make_disposable_writable(work)
    commands = _native_commands(work, language, source_paths)
    if not commands:
        return {"error": "NO_NATIVE_BUILD_COMMAND", "exit_code": None, "steps": []}
    steps = []
    for index, command in enumerate(commands, 1):
        receipt = _execute(executor, command, work, output / f"step-{index:02d}")
        steps.append(receipt)
        if (receipt["error"] is None and receipt["exit_code"] != 0
                and command[:3] == ["cmake", "--build", "build"]):
            diagnostic = receipt.get("first_diagnostic") or receipt.get("log_tail", "")
            missing = re.search(
                r"fatal error:\s*([^\s:]+): No such file or directory", diagnostic
            )
            if missing and any(path.is_file() for path in work.rglob(
                PurePosixPath(missing.group(1)).name
            )):
                retry = _execute(
                    executor, ["cmake", "--build", "build", "-j1"],
                    work, output / f"step-{index:02d}-serial-retry",
                )
                retry["retry_reason"] = "generated_header_build_order"
                steps.append(retry)
                receipt = retry
        if receipt["error"] or receipt["exit_code"] != 0:
            break
    return {**steps[-1], "steps": steps}


def _make_disposable_writable(work: Path) -> None:
    """Let a capability-restricted container build in this disposable copy."""
    for directory, names, files in os.walk(work, followlinks=False):
        root = Path(directory)
        try:
            root.chmod(root.stat().st_mode | 0o777)
        except PermissionError:
            # Docker may leave compiled outputs owned by the container user.
            # They need not be rewritten by the host for witness execution.
            pass
        for name in names + files:
            path = root / name
            if not path.is_symlink():
                try:
                    path.chmod(path.stat().st_mode | (0o777 if path.is_dir() else 0o666))
                except PermissionError:
                    pass


def _java_class_directories(work: Path) -> list[str]:
    directories = []
    for path in work.rglob("*"):
        if not path.is_dir():
            continue
        relative = path.relative_to(work).as_posix()
        if (relative.endswith(("target/classes", "build/classes/java/main", "build/classes"))
                and not path.is_symlink()):
            directories.append(relative)
    return sorted(set(directories))


def _witness(executor: Executor, work: Path, output: Path, language: str, ec: dict,
             *, operation_file: str | None = None) -> dict:
    """Compile and execute a bounded model witness in a disposable snapshot."""
    source = ec["source"]
    if ("CONTRACTFIX_OPERATION_REACHED" not in source
            or "CONTRACTFIX_ASSERTION_PASS" not in source
            or "CONTRACTFIX_ASSERTION_FAIL" not in source):
        return {"outcome": "ERROR", "failure": "INVALID_FIXTURE"}
    class_paths = ec.get("class_path") or []
    if any(path != "." and (PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
                            or ":" in path or "\\" in path or "\x00" in path)
           for path in class_paths):
        return {"outcome": "ERROR", "failure": "INVALID_FIXTURE"}
    repository_build = None
    if (language == "java" and not _java_class_directories(work)) or (
        language == "javascript"
        and _javascript_requires_witness_build(work, operation_file)
    ):
        repository_build = _native_build(
            executor, work, output / "repository_build", language,
            [operation_file] if operation_file else [],
        )
        if _build_failure(repository_build):
            return {"outcome": "ERROR", "failure": "REPOSITORY_BUILD_FAILURE",
                    "build_failure": _build_failure(repository_build),
                    "repository_build": repository_build}
    folder = work / ".contractfix_witness"
    folder.mkdir(exist_ok=True)
    if language == "javascript":
        _prepare_javascript_work(work)
    if language == "java":
        source_path = folder / "ContractFixWitness.java"
        source_path.write_text(source, encoding="utf-8")
        class_path = ":".join(dict.fromkeys([
            ".", ".contractfix_witness", *class_paths,
            *_java_class_directories(work),
        ]))
        compile_command = ["javac", "-cp", class_path, ".contractfix_witness/ContractFixWitness.java"]
        run_command = ["java", "-cp", class_path, "ContractFixWitness"]
    elif language == "c":
        source_path = folder / "contractfix_witness.c"
        source_path.write_text(source, encoding="utf-8")
        compile_command = ["cc", "-std=c11", "-I.", "-Isrc", "-Iinclude",
                           ".contractfix_witness/contractfix_witness.c",
                           "-o", ".contractfix_witness/witness"]
        run_command = ["./.contractfix_witness/witness"]
    elif language == "javascript":
        source_path = folder / "contractfix_witness.cjs"
        source_path.write_text(source, encoding="utf-8")
        compile_command = ["node", "--check", ".contractfix_witness/contractfix_witness.cjs"]
        run_command = ["node", ".contractfix_witness/contractfix_witness.cjs"]
    else:
        source_path = folder / "contractfix_witness.cpp"
        source_path.write_text(source, encoding="utf-8")
        fmt_header_only = ((work / "include/fmt/format.h").is_file()
                           and re.search(r"#\s*include\s*[\"<]fmt/", source) is not None)
        compile_command = ["c++", "-std=c++17", "-I.", "-Iinclude",
                           *(["-DFMT_HEADER_ONLY"] if fmt_header_only else []),
                           ".contractfix_witness/contractfix_witness.cpp",
                           "-o", ".contractfix_witness/witness"]
        run_command = ["./.contractfix_witness/witness"]
    _make_disposable_writable(work)
    compiled = _execute(executor, compile_command, work, output / "compile")
    if compiled["error"]:
        return {"outcome": "ERROR", "failure": "INFRASTRUCTURE_ERROR", "compile": compiled}
    if compiled["exit_code"] != 0:
        return {"outcome": "ERROR", "failure": "COMPILER_ERROR", "compile": compiled}
    executed = _execute(executor, run_command, work, output / "run")
    text = executed["log_tail"]
    reached = "CONTRACTFIX_OPERATION_REACHED" in text
    pass_seen = "CONTRACTFIX_ASSERTION_PASS" in text
    fail_seen = "CONTRACTFIX_ASSERTION_FAIL" in text
    if executed["error"]:
        outcome, failure = "ERROR", "TIMEOUT" if executed["error"] == "timeout" else "INFRASTRUCTURE_ERROR"
    elif not reached:
        outcome, failure = "ERROR", "UNREACHED_OPERATION"
    elif pass_seen == fail_seen:
        outcome, failure = "ERROR", "INVALID_FIXTURE"
    else:
        outcome, failure = ("SATISFIED", None) if pass_seen else ("VIOLATED", None)
    return {
        "outcome": outcome, "failure": failure,
        "evaluation_harness_entered": True,
        "contracted_operation_reached": reached,
        "all_runtime_assertions_exercised": outcome != "ERROR",
        "assertion_execution": {
            "all_executed": outcome != "ERROR",
            "observed_violations": [{"id": "native_assertion_001"}] if outcome == "VIOLATED" else [],
        },
        "executable_contract_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "repository_build": repository_build,
        "compile": compiled, "run": executed,
    }


class NativeLanguageBackend:
    """Shared contract workflow with separately versioned language adapters."""

    settings_model = NativeSettings

    def __init__(self, language: str):
        if language not in {"java", "c", "cpp", "javascript"}:
            raise ValueError("native backend supports Java, C, C++, or JavaScript")
        self.language = language
        self.aliases = (
            frozenset({"c++", "cxx", "cc", "c-plus-plus"}) if language == "cpp" else
            frozenset({"js"}) if language == "javascript" else frozenset()
        )
        self.artifact_version = (
            4 if language == "java" else 2 if language == "c" else
            3 if language == "cpp" else 1
        )
        self.frozen_versions = frozenset(
            f"contractfix-{language}/{version}"
            for version in range(1, self.artifact_version + 1)
        )

    def create_stages(self, settings: NativeSettings, prompts: Path | None = None) -> LangChainStages:
        if prompts is not None:
            raise ValueError("native prompt pack is versioned with the backend")
        llm = LLMSettings.from_env().for_output(settings.stage_output_tokens)
        stages = LangChainStages(llm)
        stages.prompts = NativePromptBook(self.language)
        stages.stage_answer_tokens.update({name: settings.stage_output_tokens
                                           for name in stages.prompts.policy["stages"]})
        return stages

    def validate_frozen(self, frozen: dict) -> None:
        if frozen.get("version") not in self.frozen_versions:
            raise ValueError("wrong native frozen artifact version")
        if frozen.get("language") != self.language:
            raise ValueError("native artifact language mismatch")

    def discover(self, task: Task, repo: Path, output: Path, executor: Executor,
                 stages: Stages, settings: NativeSettings,
                 localization_override: dict | None = None) -> dict:
        if task.language != self.language:
            raise ValueError("task/backend language mismatch")
        artifacts = RunArtifacts(output, {"task_id": task.instance_id, "task": task.model_dump(),
                                          "language": self.language, "workflow": settings.model_dump(),
                                          "generator": stages.identity})
        if hasattr(stages, "artifacts"):
            stages.artifacts = artifacts
        base = output / "base"
        base_hash = _native_snapshot(repo, task, base, self.language)
        sources = _index(base, self.language, task.problem_statement)
        if not sources:
            raise ValueError("no native production source files indexed")
        packet = {"task": task.model_packet(), "language": self.language, "sources": sources}
        localization = (NativeLocalization.model_validate(localization_override)
                        if localization_override else _native_generate(
                            stages, "native_localize", NativeLocalization, packet,
                        ))
        if self.language == "c":
            localization, issue_adjustment = _ground_c_issue_command(
                base, task.problem_statement, localization,
            )
            if issue_adjustment:
                artifacts.save("localization/grounding_issue_command.json", issue_adjustment)
            localization, adjustment = _ground_c_command_operation(base, localization)
            if adjustment:
                artifacts.save("localization/grounding_adjustment.json", adjustment)
        sources, retrieval_adjustment = _ground_localization_paths(
            base, self.language, sources, localization,
        )
        if retrieval_adjustment:
            artifacts.save("localization/retrieval_adjustment.json", retrieval_adjustment)
        indexed = {row["path"] for row in sources}
        if self.language == "cpp":
            localization, adjustment = _ground_cpp_operation(base, localization, indexed)
            if adjustment:
                artifacts.save("localization/grounding_adjustment.json", adjustment)
        artifacts.save("localization/final.json", localization.model_dump())
        localized = _localized_sources(base, self.language, localization)
        artifacts.save("localization/context.json", {"sources": localized})
        grounded_packet = {"task": task.model_packet(), "language": self.language,
                           "sources": localized, "localization": localization.model_dump()}
        nlc = _native_generate(stages, "native_nlc", NativeNLC, grounded_packet)
        artifacts.save("nlc/generated.json", nlc.model_dump())
        try:
            evidence, normalization = _normalize_evidence_paths(
                nlc.evidence_paths, indexed | {"problem_statement"},
            )
        except ValueError:
            artifacts.save("nlc/invalid_evidence.json", {
                "reason": "NLC cites an unsupplied source",
                "evidence_paths": nlc.evidence_paths,
            })
            review = NativeNLCReview(
                accepted=False, reason="NLC cites an unsupplied source",
            )
        else:
            if normalization:
                artifacts.save("nlc/evidence_normalization.json", normalization)
                nlc = nlc.model_copy(update={"evidence_paths": evidence})
            review = _native_generate(stages, "native_nlc_review", NativeNLCReview,
                                      {**grounded_packet, "nlc": nlc.model_dump()})
        artifacts.save("nlc/review.json", review.model_dump())
        ec = None
        if review.accepted:
            ec_packet = {**grounded_packet, "nlc": nlc.model_dump()}
            attempts = 2 if self.language in {"c", "cpp", "javascript"} else 1
            for ec_attempt in range(1, attempts + 1):
                generated = _native_generate(stages, "native_ec", NativeEC, ec_packet)
                artifacts.save(
                    f"ec/attempt-{ec_attempt:03d}/generated.json", generated.model_dump()
                )
                if self.language in {"c", "cpp"} and not _native_ec_links_selected_operation(
                    generated.source, localization, self.language,
                ):
                    observation = {"outcome": "ERROR", "failure": "UNGROUNDED_EC",
                                   "reason": "witness does not include and call selected repository operation"}
                else:
                    witness_work = (output / "witness_base" if ec_attempt == 1 else
                                    output / f"witness_base_attempt_{ec_attempt:03d}")
                    shutil.copytree(base, witness_work, symlinks=True)
                    observation = _witness(
                        executor, witness_work,
                        output / "ec" / ("buggy" if ec_attempt == 1 else
                                          f"buggy-attempt-{ec_attempt:03d}"),
                        self.language, generated.model_dump(),
                        operation_file=localization.operation_file,
                    )
                artifacts.save(f"ec/attempt-{ec_attempt:03d}/buggy_observation.json",
                               observation)
                failure = observation.get("failure")
                buggy_already_satisfies = (
                    self.language == "javascript"
                    and observation["outcome"] == "SATISFIED"
                )
                if ec_attempt == attempts or (not buggy_already_satisfies and failure not in {
                    "UNGROUNDED_EC", "COMPILER_ERROR", "INVALID_FIXTURE",
                    "UNREACHED_OPERATION",
                }):
                    break
                compiled = observation.get("compile") or {}
                ec_packet = {**ec_packet, "previous_ec_feedback": {
                    "failure": "BUGGY_BASE_SATISFIED" if buggy_already_satisfies else failure,
                    "reason": (
                        "The buggy base already satisfies this witness; choose a concrete "
                        "issue scenario that violates the same accepted NLC."
                        if buggy_already_satisfies else observation.get("reason")
                    ),
                    "first_diagnostic": compiled.get("first_diagnostic"),
                    "compile_log_tail": compiled.get("log_tail", "")[-800:],
                    "run_log_tail": (observation.get("run") or {}).get(
                        "log_tail", ""
                    )[-800:],
                }}
            artifacts.save("ec/generated.json", generated.model_dump())
            artifacts.save("ec/buggy_observation.json", observation)
            if observation["outcome"] == "VIOLATED":
                ec_review = _native_generate(
                    stages, "native_nlc_review", NativeNLCReview,
                    {**grounded_packet, "nlc": nlc.model_dump(),
                     "ec": generated.model_dump(), "buggy_observation": {
                         "outcome": observation["outcome"],
                         "contracted_operation_reached": observation.get("contracted_operation_reached"),
                         "assertion_execution": observation.get("assertion_execution"),
                     }},
                )
                artifacts.save("ec/conformance_review.json", ec_review.model_dump())
                if ec_review.accepted:
                    repeat_work = output / "witness_repeat_base"
                    shutil.copytree(base, repeat_work, symlinks=True)
                    repeat = _witness(
                        executor, repeat_work, output / "ec/buggy_repeat",
                        self.language, generated.model_dump(),
                        operation_file=localization.operation_file,
                    )
                    artifacts.save("ec/buggy_repeat_observation.json", repeat)
                    if repeat["outcome"] == "VIOLATED" and repeat.get(
                        "executable_contract_sha256"
                    ) == observation.get("executable_contract_sha256"):
                        ec = generated.model_dump()
        executor_identity = executor.identity()
        body = {
            "version": f"contractfix-{self.language}/{self.artifact_version}",
            "language": self.language,
            "task": {**task.model_dump(), "language": self.language},
            "repo_sha256": base_hash, "executor": executor_identity,
            "workflow": settings.model_dump(), "localization": localization.model_dump(),
            "allowed_edit_paths": localization.repair_paths,
            "nlc": nlc.model_dump() if review.accepted else None,
            "nlc_review": review.model_dump(), "ec": ec,
        }
        frozen = {**body, "sha256": digest(body)}
        artifacts.save("frozen.json", frozen)
        artifacts.save("guidance.json", frozen)
        status = {"status": "FROZEN" if ec else "NLC_ONLY" if review.accepted else "NLC_UNAVAILABLE",
                  "instance_id": task.instance_id, "language": self.language,
                  "frozen_artifact": str(output / "frozen.json"), "model_accounting": model_accounting(stages)}
        artifacts.save("qualification.json", status)
        logger.info(f"[native {self.language}] {status['status']} | {output / 'frozen.json'}")
        return status

    def repair(self, frozen_dir: Path, output: Path, executor: Executor, stages: Stages,
               mode: str = "enforce", *, resume: bool = False) -> dict:
        if mode not in {"no-contract", "context", "enforce"}:
            raise ValueError("unknown native repair mode")
        guidance = json.loads((frozen_dir / "guidance.json").read_text(encoding="utf-8"))
        claimed = guidance.pop("sha256")
        if digest(guidance) != claimed or guidance["language"] != self.language:
            raise ValueError("native guidance identity mismatch")
        if repository_identity(frozen_dir / "base") != guidance["repo_sha256"]:
            raise ValueError("native frozen base changed")
        if output.exists():
            if resume and (output / "status.json").is_file():
                return json.loads((output / "status.json").read_text(encoding="utf-8"))
            raise ValueError("interrupted native repair needs receipt audit before retry")
        artifacts = RunArtifacts(output, {"task_id": guidance["task"]["instance_id"],
                                          "guidance_sha256": claimed, "language": self.language,
                                          "mode": mode, "generator": stages.identity})
        if hasattr(stages, "artifacts"):
            stages.artifacts = artifacts
        settings = NativeSettings.model_validate(guidance["workflow"])
        base = frozen_dir / "base"
        localization = NativeLocalization.model_validate(guidance["localization"])
        localized = _localized_sources(base, self.language, localization)
        baseline_work = output / "baseline_work"
        shutil.copytree(base, baseline_work, symlinks=True)
        baseline = _native_build(
            executor, baseline_work, output / "ordinary_baseline", self.language,
            localization.repair_paths,
        )
        baseline_failure = _build_failure(baseline)
        artifacts.save("ordinary_baseline.json", {**baseline, "failure": baseline_failure})
        if baseline_failure:
            result = {
                "status": "BASE_VALIDATION_UNAVAILABLE", "failure": baseline_failure,
                "requested_variant": mode, "effective_mode": mode,
                "language": self.language, "selected_attempt": None,
                "selected_patch_path": None, "candidate_count": 0,
                "model_accounting": model_accounting(stages),
            }
            artifacts.save("status.json", result)
            logger.warning(f"[native {self.language}] {baseline_failure} | {output / 'ordinary_baseline.json'}")
            return result
        candidates = []
        pack = prompt_pack_for(self.language)
        context = NativePatchContext(
            base, self.language, localization.repair_paths,
            max_excerpts=pack.context_max_excerpts,
            max_chars=pack.context_max_chars,
        )
        previous_feedback = []
        for attempt in range(1, settings.candidate_limit + 1):
            packet = {
                "task": guidance["task"], "language": self.language,
                "localization": guidance["localization"],
                "nlc": guidance["nlc"] if mode != "no-contract" else None,
                "ec": guidance["ec"] if mode == "enforce" else None,
                "allowed_paths": context.allowed_paths,
                "available_edit_paths": sorted(context.read_paths - set(context.allowed_paths)),
                "source": localized,
                "retrieved_context": context.excerpts,
                "previous_candidate_feedback": previous_feedback,
                "attempt": attempt,
            }
            proposal = _native_generate(stages, "native_patch", NativePatchProposal, packet)
            latest_context_feedback = None
            for round_number in range(1, 4):
                if proposal.action != "request_context":
                    break
                receipt = context.request(proposal.context_requests)
                latest_context_feedback = receipt
                artifacts.save(
                    f"context/attempt-{attempt:03d}-round-{round_number:02d}.json",
                    receipt,
                )
                packet = {**packet, "allowed_paths": context.allowed_paths,
                          "available_edit_paths": sorted(
                              context.read_paths - set(context.allowed_paths)
                          ),
                          "retrieved_context": context.excerpts,
                          "context_feedback": receipt}
                proposal = _native_generate(stages, "native_patch", NativePatchProposal, packet)
            if proposal.action == "request_context":
                packet = {**packet, "context_request_closed": True,
                          "last_context_request": proposal.model_dump()}
                proposal = _native_generate(stages, "native_patch", NativePatchProposal, packet)
            attempt_dir = output / "candidates" / f"{attempt:03d}"
            attempt_dir.mkdir(parents=True)
            artifacts.save(f"candidates/{attempt:03d}/proposal.json", proposal.model_dump())
            row = {"attempt": attempt, "applicable": False, "ordinary_pass": False,
                   "nlc_pass": None, "ec_pass": None, "ec_validation_status": "UNKNOWN"}
            if proposal.action != "submit_edits":
                row["failure"] = (
                    "CONTEXT_BUDGET_EXHAUSTED" if proposal.action == "request_context"
                    else "MODEL_ABSTAINED"
                )
                artifacts.save(f"candidates/{attempt:03d}/validation.json", row)
                candidates.append(row)
                previous_feedback.append({
                    "attempt": attempt,
                    "failure": row["failure"],
                    "context_outcomes": [
                        {key: outcome[key] for key in ("request", "status", "detail",
                                                        "suggested_next_request") if key in outcome}
                        for outcome in (latest_context_feedback or {}).get("outcomes", [])
                    ],
                })
                logger.info(
                    f"[native {self.language} patch {attempt}] {row['failure']} | {attempt_dir}"
                )
                continue
            try:
                try:
                    patch, modified = materialize_native_edits(
                        base, proposal, context.allowed_paths, self.language,
                    )
                except ValueError as mismatch:
                    if (attempt != settings.candidate_limit
                            or not str(mismatch).startswith("exact edit does not match once:")):
                        raise
                    exact_context = _exact_edit_context(base, proposal, self.language)
                    artifacts.save(f"candidates/{attempt:03d}/edit_correction_context.json",
                                   exact_context)
                    corrected = _native_generate(
                        stages, "native_patch", NativePatchProposal,
                        {**packet, "edit_correction_only": True,
                         "edit_error": str(mismatch),
                         "exact_edit_context": exact_context,
                         "rejected_proposal": proposal.model_dump()},
                    )
                    artifacts.save(f"candidates/{attempt:03d}/edit_correction.json",
                                   corrected.model_dump())
                    if corrected.action != "submit_edits":
                        raise mismatch
                    patch, modified = materialize_native_edits(
                        base, corrected, context.allowed_paths, self.language,
                    )
                    proposal = corrected
                    row["edit_correction_used"] = True
                patch_path = attempt_dir / "candidate.patch"
                patch_path.write_text(patch, encoding="utf-8")
                row.update(applicable=True, patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
                           patch_path=str(patch_path), changed_files=sorted(modified),
                           changed_lines=sum(line[:1] in {"+", "-"} and not line.startswith(("+++", "---"))
                                             for line in patch.splitlines()))
                work = attempt_dir / "work"
                shutil.copytree(base, work, symlinks=True)
                for name, content in modified.items():
                    (work / name).write_text(content, encoding="utf-8")
                check = _native_build(
                    executor, work, attempt_dir / "ordinary", self.language,
                    context.allowed_paths,
                )
                row["ordinary_checks"] = check
                failure = _build_failure(check)
                row["ordinary_pass"] = failure is None
                if failure:
                    row["failure"] = failure
                if row["ordinary_pass"] and mode != "no-contract" and guidance["nlc"]:
                    review = _native_generate(stages, "native_nlc_review", NativeNLCReview,
                                              {"task": guidance["task"], "nlc": guidance["nlc"],
                                               "patch": patch, "source": packet["source"]})
                    artifacts.save(f"candidates/{attempt:03d}/nlc_review.json", review.model_dump())
                    row["nlc_pass"] = review.accepted
                    if not review.accepted:
                        row["nlc_review_reason"] = review.reason
                if row["ordinary_pass"] and guidance["ec"] and mode == "enforce":
                    observation = _witness(
                        executor, work, attempt_dir / "ec", self.language, guidance["ec"],
                        operation_file=localization.operation_file,
                    )
                    artifacts.save(f"candidates/{attempt:03d}/ec_observation.json", observation)
                    row["ec_execution"] = observation
                    row["ec_pass"] = observation["outcome"] == "SATISFIED"
                    row["ec_validation_status"] = (
                        "EC_PASS" if row["ec_pass"] else
                        "EC_VIOLATION" if observation["outcome"] == "VIOLATED" else
                        observation.get("failure", "UNKNOWN")
                    )
            except (OSError, ValueError, UnicodeError, subprocess.TimeoutExpired) as exc:
                row["failure"] = type(exc).__name__
                row["detail"] = str(redact(str(exc)))[:500]
                if (isinstance(exc, ValueError)
                        and str(exc).startswith("exact edit does not match once:")):
                    row["exact_edit_context"] = _exact_edit_context(
                        base, proposal, self.language,
                    )
            artifacts.save(f"candidates/{attempt:03d}/validation.json", row)
            candidates.append(row)
            previous_feedback.append({
                "attempt": attempt,
                "failure": row.get("failure"),
                "detail": row.get("detail"),
                "exact_edit_context": row.get("exact_edit_context"),
                "ordinary_first_diagnostic": (row.get("ordinary_checks") or {}).get(
                    "first_diagnostic"
                ),
                "ordinary_log_tail": (row.get("ordinary_checks") or {}).get(
                    "log_tail", ""
                )[-1200:],
                "nlc_review_reason": row.get("nlc_review_reason"),
                "ec_validation_status": row["ec_validation_status"],
                "ec_failure_detail": (
                    (row.get("ec_execution") or {}).get("run") or {}
                ).get("log_tail", "")[-1200:],
            })
            logger.info(f"[native {self.language} patch {attempt}] {row.get('failure', row['ec_validation_status'])} | {attempt_dir}")
        decision = select_with_ec(candidates)
        artifacts.save("selection.json", decision)
        selected = next((row for row in candidates if row["attempt"] == decision["selected_attempt"]), None)
        result = {
            "status": "PATCH_SELECTED" if selected else "NO_ELIGIBLE_PATCH",
            "requested_variant": mode,
            "effective_mode": (
                "no-contract" if mode == "no-contract" else
                "enforce" if mode == "enforce" and guidance["ec"] else
                "context" if guidance["nlc"] else "no-contract"
            ),
            "language": self.language, "selected_attempt": decision["selected_attempt"],
            "selected_patch_path": selected.get("patch_path") if selected else None,
            "candidate_count": len(candidates), "selection": decision,
            "model_accounting": model_accounting(stages),
        }
        artifacts.save("status.json", result)
        logger.info(f"[native {self.language}] {result['status']} | {output / 'status.json'}")
        return json.loads((output / "status.json").read_text(encoding="utf-8"))
