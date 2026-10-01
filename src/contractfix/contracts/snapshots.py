"""Content identity for ordinary source directories, including safe symlinks."""

import hashlib
import os
from pathlib import Path, PurePosixPath

from .core import digest

IGNORED = {".git", ".venv", "__pycache__", ".pytest_cache"}


def validate_relative_symlink(relative: str | PurePosixPath, target: str) -> str:
    """Accept only repository-internal relative link targets, without resolving them."""
    link = PurePosixPath(relative)
    destination = PurePosixPath(target)
    if not target or destination.is_absolute() or "\\" in target or "\x00" in target:
        raise ValueError("unsafe repository symlink")
    parts = list(link.parent.parts)
    for part in destination.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                raise ValueError("repository symlink escapes snapshot")
            parts.pop()
        else:
            parts.append(part)
    return target


def repository_identity(path: str | Path) -> str:
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError("repository not found")
    files: dict[str, object] = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if set(rel.parts) & IGNORED:
            continue
        if p.is_symlink():
            target = validate_relative_symlink(rel.as_posix(), os.readlink(p))
            files[rel.as_posix()] = {"kind": "symlink", "target": target}
            continue
        if p.is_file():
            files[rel.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return digest(files)
