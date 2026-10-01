"""Bounded repository-document retrieval adapted from the legacy source adapter.

Read only the already-exported base snapshot: no remote documentation, Git history,
HTML assets, symlinks or generated files. A matched document is evidence to review,
not a guarantee that its contents are authoritative or current.
"""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

STOP = {"the", "and", "with", "from", "this", "that", "when", "return", "none", "true", "false"}
SKIP = {".git", ".venv", "__pycache__", "node_modules", "_build", "build", "dist", "assets"}


def terms(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z][a-z0-9_]{2,}", text.lower()) if word not in STOP}


def retrieve_documents(root: Path, query: str, symbols: list[str], *, base_commit: str,
                       limit: int = 2, max_chars: int = 3200) -> dict[str, dict]:
    """Return stable, non-overlapping snippets with source hashes and line ranges."""
    if limit < 0 or max_chars < 0:
        raise ValueError("retrieval budgets must be nonnegative")
    if not limit or not max_chars:
        return {}
    keywords = terms(query) | {name.lower().split(".")[-1] for name in symbols}
    candidates = []
    scanned = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in SKIP or part.startswith(".") for part in relative.parts):
            continue
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root.parent):
            continue
        if path.suffix.lower() not in {".md", ".rst", ".txt"} or not path.is_file():
            continue
        if path.stat().st_size > 1_000_000:
            continue
        scanned += 1
        if scanned > 1000:
            break
        try:
            raw = path.read_bytes()
            lines = raw.decode("utf-8").splitlines()
        except UnicodeError:
            continue
        for start in range(0, len(lines), 20):
            stop = min(start + 30, len(lines))
            text = "\n".join(lines[start:stop])
            score = len(terms(text) & keywords)
            if score:
                candidates.append((-score, relative.as_posix(), start, stop, text,
                                   hashlib.sha256(raw).hexdigest()))
    selected, occupied, remaining = {}, {}, max_chars
    for score, name, start, stop, text, source_hash in sorted(candidates):
        if len(selected) == limit or remaining < 80:
            break
        if any(start < end and stop > begin for begin, end in occupied.get(name, [])):
            continue
        # Complete lines, not silent mid-sentence cuts; truncation remains explicit.
        kept = []
        for line in text.splitlines():
            if len("\n".join(kept + [line])) > remaining:
                break
            kept.append(line)
        snippet = "\n".join(kept)
        if not snippet.strip():
            continue
        selected[f"REPO_DOC_{len(selected)+1}"] = {
            "kind": "repository_document", "source": name, "start": start + 1,
            "end": start + len(kept), "text": snippet, "source_sha256": source_hash,
            "base_commit": base_commit, "snippet_truncated": len(kept) < stop - start,
            "authority": "requires_source_review", "retrieval_score": -score}
        occupied.setdefault(name, []).append((start, stop))
        remaining -= len(snippet)
    return selected


def retrieve_existing_tests(
    root: Path, query: str, symbols: list[str], *, base_commit: str,
    source_files: list[str] | None = None, limit: int = 1, max_chars: int = 3200,
) -> dict[str, dict]:
    """Retrieve bounded tests for selected symbols from the immutable base only.

    Test assertions describe established behavior and examples, but can also
    encode the reported bug. They require review against the issue before being
    treated as a desired postcondition.
    """
    if limit < 0 or max_chars < 0:
        raise ValueError("retrieval budgets must be nonnegative")
    if not limit or not max_chars:
        return {}
    aliases = {
        re.sub(r"^_?print_", "", symbol.split(".")[-1]).lower()
        for symbol in symbols
    }
    aliases = {alias for alias in aliases if len(alias) >= 5}
    if not aliases:
        return {}
    source_stems = {Path(name).stem.lower() for name in source_files or []}
    query_terms = terms(query)
    candidates = []
    scanned = 0
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(part in SKIP or part.startswith(".") for part in relative.parts):
            continue
        if (
            path.is_symlink() or any(parent.is_symlink() for parent in path.parents
                                     if parent != root.parent)
            or not path.is_file() or path.stat().st_size > 1_000_000
        ):
            continue
        if "tests" not in relative.parts and not path.name.startswith("test_"):
            continue
        scanned += 1
        if scanned > 1000:
            break
        try:
            raw = path.read_bytes()
            source = raw.decode("utf-8")
            tree = ast.parse(source, filename=relative.as_posix())
        except (UnicodeError, SyntaxError, OSError):
            continue
        lines = source.splitlines()
        source_hash = hashlib.sha256(raw).hexdigest()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test_"):
                continue
            start, stop = node.lineno - 1, node.end_lineno or node.lineno
            snippet = "\n".join(lines[start:stop])
            name = node.name.lower()
            body = snippet.lower()
            direct = max(
                ((3 if alias in name else 1 if alias in body else 0) * len(alias)
                 for alias in aliases), default=0,
            )
            if not direct:
                continue
            related_module = any(
                stem in path.stem.lower() for stem in source_stems if len(stem) >= 4
            )
            score = direct * 10 + 100 * related_module + min(
                20, len(terms(snippet) & query_terms)
            )
            candidates.append((-score, relative.as_posix(), start, stop, source_hash, lines))
    selected, remaining = {}, max_chars
    for score, name, start, stop, source_hash, lines in sorted(candidates):
        if len(selected) >= limit or remaining < 80:
            break
        kept = []
        for line in lines[start:stop]:
            if len("\n".join(kept + [line])) > remaining:
                break
            kept.append(line)
        snippet = "\n".join(kept)
        if not snippet.strip():
            continue
        selected[f"REPO_TEST_{len(selected) + 1}"] = {
            "kind": "existing_test", "source": name, "start": start + 1,
            "end": start + len(kept), "text": snippet,
            "source_sha256": source_hash, "base_commit": base_commit,
            "snippet_truncated": len(kept) < stop - start,
            "authority": "requires_issue_review", "retrieval_score": -score,
        }
        remaining -= len(snippet)
    return selected
