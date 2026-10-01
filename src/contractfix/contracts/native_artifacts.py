"""Recover only compiled libraries hidden by the historical checkout mount."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import uuid


# Executed with the pinned task interpreter, without network access. No source,
# tests, gold patches, or Python bytecode is exported. Generated _version.py
# metadata is the sole Python exception, and only fills a missing checkout file.
EXPORT_NATIVE = r"""import os, subprocess, sys, tarfile
root = sys.argv[1]
tracked = set(subprocess.check_output(['git', '-C', root, 'ls-files', '-z']).decode().split('\x00'))
total = 0
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|', dereference=True) as archive:
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in ('.git', '.venv', '__pycache__'))
        for name in sorted(files):
            if name.endswith(('.so', '.pyd', '.dll', '.dylib')) or '.so.' in name or name == '_version.py':
                path = os.path.join(directory, name)
                if name == '_version.py' and os.path.relpath(path, root) in tracked:
                    continue
                total += os.path.getsize(path)
                if total > 512 * 1024 * 1024:
                    raise ValueError('native artifact export exceeds 512 MiB')
                archive.add(path, arcname=os.path.relpath(path, root), recursive=False)
"""


def unpack_native(archive: Path, destination: Path) -> list[Path]:
    """Validate each archive member and a bounded total before writing any files."""
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        if sum(member.size for member in members) > 512 * 1024 * 1024:
            raise ValueError("native artifact archive exceeds 512 MiB")
        for member in members:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or ".." in path.parts
                    or "\\" in member.name
                    or not (path.suffix in {".so", ".pyd", ".dll", ".dylib"}
                            or ".so." in path.name or path.name == "_version.py")):
                raise ValueError("invalid native artifact archive member")
        result = []
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            result.append(target)
        return result


class NativeArtifacts:
    """One pinned image export per executor, with temporary overlays per execution."""

    def __init__(self, image: str, python: str, repo_path: str, cache_parent: Path | None = None) -> None:
        # The cache must outlive individual disposable checkouts. Import probes
        # and contract replays reuse an executor but destroy each work directory.
        if cache_parent is None:
            cache_parent = Path(__file__).resolve().parents[3] / "tmp/native-artifacts"
        cache_parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="native-artifacts-", dir=cache_parent)
        self.root = Path(self.temporary.name)
        archive = self.root / "native.tar"
        name = "contractfix-native-" + uuid.uuid4().hex
        command = ["docker", "run", "--rm", "--name", name, "--pull=never", "--network=none",
                   "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                   "--memory=1g", "--pids-limit=64", "--entrypoint", python,
                   image, "-c", EXPORT_NATIVE, repo_path]
        try:
            with archive.open("wb") as output:
                result = subprocess.run(command, stdout=output, stderr=subprocess.PIPE, timeout=120)
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
        if result.returncode:
            raise RuntimeError("could not read pinned image native artifacts: " +
                               result.stderr.decode(errors="replace")[-1000:])
        self.files = unpack_native(archive, self.root / "files")
        archive.unlink()

    @contextmanager
    def overlay(self, work: Path) -> Iterator[None]:
        added = []
        directories = []
        try:
            for source in self.files:
                relative = source.relative_to(self.root / "files")
                target = work / relative
                if target.exists() or target.is_symlink():
                    continue
                # Never follow a repository symlink when copying trusted libraries.
                if any(parent.is_symlink() for parent in target.parents if parent != work.parent):
                    raise ValueError("native artifact path traverses a symlink")
                missing = []
                parent = target.parent
                while parent != work and not parent.exists():
                    missing.append(parent)
                    parent = parent.parent
                target.parent.mkdir(parents=True, exist_ok=True)
                directories.extend(reversed(missing))
                shutil.copyfile(source, target)
                added.append((target, hashlib.sha256(source.read_bytes()).digest()))
            yield
        finally:
            mutated = False
            for target, expected in reversed(added):
                if (target.is_symlink() or not target.is_file()
                        or hashlib.sha256(target.read_bytes()).digest() != expected):
                    mutated = True
                if target.is_file() or target.is_symlink():
                    target.unlink()
            for directory in reversed(directories):
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
            if mutated:
                raise RuntimeError("native execution artifacts were mutated")
