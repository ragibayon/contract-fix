"""One shared base/candidate runner with isolated copies and pluggable execution."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .core import verify, summarize, digest
from .execution import ExecutionSettings, Executor
from .expressions import Unsupported
from .instrumentation import instrument  # backwards-compatible public import

IGNORED = ('.git', '.venv', '__pycache__', '.pytest_cache')
CONTRACT_STATES = {'SATISFIED', 'VIOLATED', 'NOT_APPLICABLE', 'CHECKER_ERROR', 'INCONCLUSIVE'}


def _hashes(folder: Path) -> dict[str, str]:
    return {path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.rglob('*')) if path.is_file() and not path.is_symlink()}


def _safe(name: str) -> str:
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError('unsafe relative path')
    return path.as_posix()


def run(repo, bundle, command, *, patch=None, timeout=120, allowed_edit_paths=None,
        executor: Executor | None = None, workload_files: dict[str, str] | None = None,
        plain: bool = False):
    """Evaluate a frozen workload. No model calls, provider credentials, or gold lookups.

    Legacy callers use the explicit trusted-local CLI. New workflow callers must
    supply an Executor. A target process is assumed non-adversarial: logs are not
    cryptographic attestations against malicious code that forges observer events.
    """
    verify(bundle)
    root = Path(repo).resolve()
    if not root.is_dir() or not command:
        raise ValueError('repository directory and command are required')
    executor = executor or Executor(ExecutionSettings(kind='local', python=sys.executable, timeout=timeout))
    package_root = Path(__file__).resolve().parents[2]
    patch_bytes = Path(patch).read_bytes() if patch else None
    allowed = set(allowed_edit_paths) if allowed_edit_paths is not None else {c['target']['file'] for c in bundle['clauses']}
    for name in allowed:
        if _safe(name) != name or Path(name).suffix != '.py' or name.startswith('__cf_'):
            raise ValueError('allowed edits must be relative production Python files')
    files = workload_files or {}
    for name in files:
        if not _safe(name).startswith('__cf_witness__/'):
            raise ValueError('workload files must stay in the reserved witness directory')
    edit_sha = digest(sorted(allowed))
    runtime_sha = digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in Path(__file__).parent.glob('*.py')})
    events, other_events = [], []
    error, exit_code, log_text = None, None, ''
    repo_sha = witness_sha = executor_sha = None
    source_hashes = {}
    with tempfile.TemporaryDirectory(prefix='contractfix-ec-') as temporary:
        temp = Path(temporary)
        work = temp / 'repo'
        shutil.copytree(root, work, symlinks=True, ignore=shutil.ignore_patterns(*IGNORED))
        bundle_path = temp / 'bundle.json'
        bundle_path.write_text(json.dumps(bundle), encoding='utf-8')
        outputs = temp / 'outputs'
        try:
            if any(p.is_symlink() for p in work.rglob('*')):
                raise Unsupported('symlink_in_repository_copy')
            before = _hashes(work)
            repo_sha = digest(before)
            witness_sha = digest({'protected_files': {k: v for k, v in before.items() if k not in allowed},
                                  'workload_files': files, 'command': command})
            if patch_bytes:
                patch_path = temp / 'candidate.patch'
                patch_path.write_bytes(patch_bytes)
                for options in (['--check'], []):
                    subprocess.run(['git', 'apply', *options, str(patch_path)], cwd=work,
                                   check=True, capture_output=True, timeout=30,
                                   env={**{k: v for k, v in os.environ.items() if not k.startswith('GIT_')},
                                        'GIT_CEILING_DIRECTORIES': str(work.parent)})
                if any(p.is_symlink() for p in work.rglob('*')):
                    raise Unsupported('patch_created_symlink')
                after = _hashes(work)
                changed = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
                if not changed:
                    raise Unsupported('patch_did_not_change_isolated_checkout')
                if changed - allowed:
                    raise Unsupported('patch_changed_protected_file:' + repr(sorted(changed - allowed)))
            for name, content in files.items():
                path = work / name
                if path.exists():
                    raise Unsupported('witness_path_collision')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding='utf-8')
            by_file = {}
            for clause in bundle['clauses']:
                by_file.setdefault(clause['target']['file'], []).append(clause)
            for name, clauses in by_file.items():
                path = work / _safe(name)
                raw = path.read_bytes()
                source_hashes[name] = hashlib.sha256(raw).hexdigest()
                if not plain:
                    path.write_text(instrument(raw.decode('utf-8'), name, clauses), encoding='utf-8')
            expected = _hashes(work)
            executor_sha = executor.identity()['sha256']
            exit_code, error = executor.run(list(command), work, package_root,
                                            bundle_path, outputs, temp / 'command.log')
            # Runtime changes to tracked source or frozen drivers invalidate the run.
            for name, value in expected.items():
                path = work / name
                if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != value:
                    error = 'runtime_modified_frozen_file:' + name
                    break
            # A workload must not manufacture Python stubs or replacement modules.
            if error is None:
                created = {p.relative_to(work).as_posix() for p in work.rglob('*.py')}
                new_python = created - expected.keys()
                if new_python:
                    error = 'runtime_created_python_source:' + repr(sorted(new_python)[:5])
        except Exception as exc:
            error = type(exc).__name__ + ':' + str(exc)[:700]
        for path in sorted(outputs.glob('events.*.jsonl')):
            if path.stat().st_size > 32 * 1024 * 1024:
                error = 'event_log_budget_exceeded'
                continue
            for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
                try:
                    event = json.loads(line)
                    (events if event.get('state') in CONTRACT_STATES else other_events).append(event)
                except (json.JSONDecodeError, AttributeError):
                    error = 'malformed_event_log'
        if (temp / 'command.log').exists():
            with (temp / 'command.log').open('rb') as log:
                log.seek(max(0, (temp / 'command.log').stat().st_size - 16000))
                log_text = log.read().decode('utf-8', errors='replace')
    summary = summarize(events, bundle['clauses'], exit_code, error)
    if plain:
        summary['disposition'] = 'UNINSTRUMENTED_OK' if not error and exit_code == 0 else 'INCONCLUSIVE'
    return {'bundle_sha256': bundle['sha256'], 'command': list(command), 'plain': plain,
        'repo_sha256': repo_sha, 'witness_sha256': witness_sha, 'runtime_sha256': runtime_sha,
        'executor_sha256': executor_sha, 'edit_policy_sha256': edit_sha,
        'patch_sha256': hashlib.sha256(patch_bytes).hexdigest() if patch_bytes else None,
        'source_hashes': source_hashes, 'events': events, 'execution_events': other_events,
        'command_log_tail': log_text, 'summary': summary}
