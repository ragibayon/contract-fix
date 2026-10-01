"""Node build commands for a disposable JavaScript/TypeScript checkout."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re
import subprocess


SOURCE_SUFFIXES = frozenset({".js", ".cjs", ".mjs", ".jsx", ".ts", ".tsx", ".mts", ".cts"})


def prepare_work(root: Path) -> None:
    """Reuse the pinned image's installed dependencies without copying them."""
    link = root / "node_modules"
    if not link.exists() and not link.is_symlink():
        link.symlink_to("/testbed/node_modules", target_is_directory=True)
    # Vue's pinned build enumerates source enums with `git grep`. Native
    # snapshots have no .git, so give this disposable worktree a source-only
    # index. No benchmark tests or reference patch enter the index.
    if ((root / "scripts/inline-enums.js").is_file()
            and (root / "packages/runtime-core").is_dir()
            and not (root / ".git").exists()):
        subprocess.run(["git", "init", "-q", str(root)], check=True,
                       capture_output=True, timeout=30)
        subprocess.run(["git", "-C", str(root), "add", "--", "packages",
                        "packages-private", "scripts"], check=True,
                       capture_output=True, timeout=60)
        subprocess.run([
            "git", "-C", str(root), "-c", "user.email=build@example.invalid",
            "-c", "user.name=Build Harness", "commit", "-qm",
            "disposable source index",
        ], check=True, capture_output=True, timeout=60)


def _package_metadata(root: Path) -> dict:
    package = root / "package.json"
    if not package.is_file():
        return {}
    try:
        body = json.loads(package.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return body if isinstance(body, dict) else {}


def _nearest_package(root: Path, name: str) -> tuple[Path, dict]:
    relative = PurePosixPath(name)
    if (relative.is_absolute() or ".." in relative.parts or "\\" in name):
        return root, _package_metadata(root)
    for parent in relative.parents:
        if str(parent) == ".":
            break
        folder = root / parent
        metadata = _package_metadata(folder)
        if metadata:
            return folder, metadata
    return root, _package_metadata(root)


def requires_witness_build(root: Path, operation_file: str | None) -> bool:
    """Build when the selected package entry is absent or source is TypeScript."""
    if operation_file is None:
        return False
    package_root, metadata = _nearest_package(root, operation_file)
    entry = metadata.get("main")
    if isinstance(entry, str) and entry:
        path = PurePosixPath(entry)
        if not path.is_absolute() and ".." not in path.parts:
            entry_file = package_root / path
            if not entry_file.is_file():
                return True
            if PurePosixPath(operation_file).suffix in {".ts", ".tsx", ".mts", ".cts"}:
                # A tracked CommonJS wrapper may exist while its generated
                # dist module does not (Vue's runtime packages use this form).
                try:
                    wrapper = entry_file.read_text(encoding="utf-8")[:5000]
                except (OSError, UnicodeError):
                    return True
                for relative in re.findall(
                    r"require\s*\(\s*['\"](\.[^'\"]+)['\"]\s*\)", wrapper
                ):
                    dependency = entry_file.parent / relative
                    if not any(candidate.is_file() for candidate in (
                        dependency, dependency.with_suffix(".js"),
                        dependency / "index.js",
                    )):
                        return True
            return False
    return PurePosixPath(operation_file).suffix in {".ts", ".tsx", ".mts", ".cts"}


def build_commands(root: Path, source_paths: list[str]) -> list[list[str]]:
    metadata = _package_metadata(root)
    if not metadata:
        return []
    scripts = metadata.get("scripts")
    if not isinstance(scripts, dict):
        scripts = {}
    package_manager = str(metadata.get("packageManager", "")).split("@", 1)[0]
    if package_manager not in {"npm", "yarn", "pnpm"}:
        package_manager = (
            "pnpm" if (root / "pnpm-lock.yaml").is_file() else
            "yarn" if (root / "yarn.lock").is_file() else "npm"
        )
    manager_command = [package_manager]
    if package_manager == "yarn":
        # Corepack may try to download an otherwise bundled Yarn release in
        # these network-isolated benchmark images.
        yarnrc = root / ".yarnrc.yml"
        if yarnrc.is_file():
            for line in yarnrc.read_text(encoding="utf-8").splitlines():
                if not line.startswith("yarnPath:"):
                    continue
                relative = line.partition(":")[2].strip().strip("\"'")
                path = PurePosixPath(relative)
                if (not path.is_absolute() and ".." not in path.parts
                        and path.suffix == ".cjs" and (root / path).is_file()):
                    manager_command = ["node", path.as_posix()]
                break
    if (metadata.get("name") == "babel" and (root / "Gulpfile.mjs").is_file()
            and isinstance(scripts.get("build"), str)
            and scripts["build"].strip() == "make build"):
        # The full standalone bundle fails on the pinned image dependencies.
        # build-dev compiles package sources and the vendor modules they import.
        return [[*manager_command, "gulp", "build-dev"]]
    if (metadata.get("name") == "immutable"
            and isinstance(scripts.get("build:dist"), str)):
        # Root build appends a network-dependent statistics step.
        return [["npm", "run", "build:dist"]]
    if (metadata.get("name") == "preact"
            and str(scripts.get("build", "")).startswith("npm-run-all --parallel")):
        modules = {PurePosixPath(name).parts[0] for name in source_paths
                   if PurePosixPath(name).parts}
        targets = ["core" if name == "src" else name for name in sorted(modules)]
        if not targets:
            targets = ["core"]
        if all(isinstance(scripts.get(f"build:{target}"), str) for target in targets):
            return [["npm", "run", f"build:{target}"] for target in targets]
    if ((root / "scripts/inline-enums.js").is_file()
            and scripts.get("build") == "node scripts/build.js"):
        targets = sorted({parts[1] for name in source_paths
                          if len(parts := PurePosixPath(name).parts) >= 3
                          and parts[0] in {"packages", "packages-private"}
                          and (root / parts[0] / parts[1] / "package.json").is_file()})
        if targets:
            return [["npm", "run", "build", "--", target] for target in targets]
    if package_manager == "yarn" and manager_command == ["yarn"] and source_paths:
        local_commands = []
        for name in source_paths:
            folder, package_metadata = _nearest_package(root, name)
            if folder == root:
                local_commands = []
                break
            package_scripts = package_metadata.get("scripts") or {}
            if not isinstance(package_scripts, dict):
                local_commands = []
                break
            stage = next((stage for stage in ("typecheck", "build", "compile")
                          if isinstance(package_scripts.get(stage), str)
                          and package_scripts[stage].strip()), None)
            if stage is None:
                local_commands = []
                break
            local_commands.append(["npm", "--prefix", folder.relative_to(root).as_posix(),
                                   "run", stage])
        if local_commands:
            return [list(command) for command in dict.fromkeys(
                tuple(command) for command in local_commands
            )]
    if package_manager in {"yarn", "pnpm"} and manager_command == [package_manager]:
        # npm runs already-installed scripts without Corepack downloading a
        # package manager. The command itself remains repository-defined.
        manager_command = ["npm"]
    for name in ("build", "compile", "typecheck", "check:types"):
        if isinstance(scripts.get(name), str) and scripts[name].strip():
            return [[*manager_command, "run", name]]

    # A syntax check is a valid local gate for plain JS when the repository
    # supplies no build script. JSX and TypeScript require a project toolchain.
    paths = []
    for name in source_paths:
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts
                or relative.suffix not in SOURCE_SUFFIXES
                or not (root / relative).is_file()):
            return []
        paths.append(relative.as_posix())
    if paths and all(PurePosixPath(name).suffix in {".js", ".cjs", ".mjs"}
                     for name in paths):
        return [["node", "--check", name] for name in dict.fromkeys(paths)]
    return []
