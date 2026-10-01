"""Discover bounded tests already visible in a Python repository's base commit."""

from __future__ import annotations

from pathlib import Path

from .repository import safe_path


CONFIGURATION_FILES = (
    "pyproject.toml",
    "pytest.ini",
    "tox.ini",
    "setup.cfg",
    "noxfile.py",
    "Makefile",
)
IGNORED_PARTS = {".git", ".tox", ".nox", ".venv", "venv", "build", "dist"}


def _is_test_file(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix == ".py" and (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name in {"test.py", "tests.py"}
    )


def _pytest_command(files: list[str]) -> list[str]:
    # Task repositories are mounted read-only. Some suites treat pytest's
    # cache-write warning as an error, obscuring otherwise comparable tests.
    return ["@python", "-m", "pytest", "-q", "-p", "no:cacheprovider", *files]


def _visible_command(root: Path, files: list[str]) -> tuple[list[str], str]:
    """Use a repository's checked-out native runner when its layout matches."""
    if (root / "django/__init__.py").is_file() and (root / "tests/runtests.py").is_file():
        if all(name.startswith("tests/") and name.endswith(".py") for name in files):
            labels = [".".join(Path(name).with_suffix("").parts[1:]) for name in files]
            return (["@python", "tests/runtests.py", "--noinput", "--parallel", "1",
                     *labels], "django-runtests")
    if (root / "sympy/__init__.py").is_file() and (root / "bin/test").is_file():
        if all(name.startswith("sympy/") and name.endswith(".py") for name in files):
            return ["@python", "bin/test", "-v", "--no-colors", *files], "sympy-runtests"
    return _pytest_command(files), "pytest"


def discover_visible_test_command(
    root: Path,
    production_files: list[str],
    *,
    limit: int = 3,
    broader_limit: int = 8,
    scope: str = "related_visible_tests",
) -> tuple[list[list[str]], dict]:
    """Select deterministic, base-visible tests related to localized code.

    No SWE-bench task bundle, evaluator metadata, generated test, or gold artifact is
    accepted as input. Exact filename relationships win; bounded source references
    provide a fallback for repositories whose tests use a different directory shape.
    """
    modules = []
    stems = set()
    for name in production_files:
        checked = safe_path(name)
        path = Path(checked)
        parts = list(path.with_suffix("").parts)
        if parts and parts[0] in {"src", "lib"}:
            parts = parts[1:]
        if path.name == "__init__.py":
            parts = parts[:-1]
        if parts:
            stems.add(parts[-1].lower())
            modules.append(".".join(parts))
    if scope not in {"related_visible_tests", "related_and_package", "repository_suite"}:
        raise ValueError("unknown repository validation scope")
    scored: list[tuple[int, str]] = []
    visible_tests: list[str] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if (
            set(relative.parts) & IGNORED_PARTS
            or not _is_test_file(relative)
            or path.stat().st_size > 1_000_000
        ):
            continue
        visible_tests.append(relative.as_posix())
        name = path.stem.lower()
        score = 0
        if any(name in {f"test_{stem}", f"{stem}_test"} for stem in stems):
            score = 100
        elif any(stem in name for stem in stems):
            score = 80
        else:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if any(module in text for module in modules):
                score = 60
            elif any(stem in text for stem in stems):
                score = 40
        if score:
            scored.append((score, relative.as_posix()))
    ranked = [name for _, name in sorted(scored, key=lambda item: (-item[0], item[1]))]
    selected = ranked[:limit]
    tiers = []
    commands = []
    framework = None
    if selected:
        command, framework = _visible_command(root, selected)
        commands.append(command)
        tiers.append({"name": "related_visible_tests", "files": selected, "command": command})
    if scope == "related_and_package" and selected:
        parents = {str(Path(name).parent) for name in selected}
        broader = [
            name
            for name in visible_tests
            if name not in selected
            and any(
                Path(name).is_relative_to(Path(parent))
                or Path(parent).is_relative_to(Path(name).parent)
                for parent in parents
            )
        ][:broader_limit]
        if broader:
            command, framework = _visible_command(root, broader)
            commands.append(command)
            tiers.append({"name": "bounded_package_tests", "files": broader, "command": command})
    elif scope == "repository_suite" and visible_tests:
        roots = sorted(
            {
                Path(name).parts[0]
                for name in visible_tests
                if Path(name).parts
            }
        )
        command, framework = _visible_command(root, roots)
        commands = [command]
        tiers = [{"name": "repository_suite", "files": roots, "command": command}]
    configuration = [name for name in CONFIGURATION_FILES if (root / name).is_file()]
    workflow_root = root / ".github/workflows"
    if workflow_root.is_dir():
        configuration.extend(
            path.relative_to(root).as_posix()
            for path in sorted(workflow_root.iterdir())
            if path.is_file() and path.suffix.lower() in {".yml", ".yaml"}
        )
    return commands, {
        "schema_version": "contractfix-python-visible-tests/1",
        "strategy": "localized_filename_then_base_source_reference",
        "framework": framework,
        "source": "repository",
        "validation_scope": scope,
        "configuration_files": configuration,
        "production_files": sorted(set(production_files)),
        "selected_test_files": selected,
        "candidate_count": len(scored),
        "visible_test_file_count": len(visible_tests),
        "tiers": tiers,
        "evaluator_metadata_consulted": False,
    }
