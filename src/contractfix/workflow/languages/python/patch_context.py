"""Bounded, read-only patch-stage repository/version retrieval."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

from contractfix.contracts.execution import Executor
from .models import ContextRequest
from .patching import checked_file
from .repair_execution import environment_receipt
from .repair_models import PatchContextRequest, RepairGuidance
from .repository import RepositoryIndex, SKIP, production_path, safe_path

_TEXT_SUFFIXES = {".py", ".pyi", ".md", ".rst", ".txt", ".toml", ".cfg", ".ini", ".json", ".yaml", ".yml"}


def readable_name(name: str) -> bool:
    path = Path(safe_path(name))
    return (not any(part.startswith("__cf_") or part in SKIP for part in path.parts)
            and not any(part.lower() in {".env", ".ssh", "credentials", "secrets"} for part in path.parts)
            and not path.name.startswith(".env")
            and path.suffix.lower() in _TEXT_SUFFIXES)


class PatchContext:
    def __init__(self, base: Path, guidance: RepairGuidance, executor: Executor,
                 *, char_budget: int, additions: int, allow_expansion: bool):
        self.base = base
        self.guidance = guidance
        self.executor = executor
        self.index = RepositoryIndex(base)
        self.known = set(guidance.localization["repair_locations"])
        self.known.update(guidance.localization.get("contracted_operation_locations", []))
        if self.known - self.index.locations.keys():
            raise ValueError("localization IDs differ from the sealed base index")
        self.allowed = set(guidance.localized_edit_paths)
        self.read_files = set(guidance.localized_edit_paths)
        self.char_budget, self.additions = char_budget, additions
        self.allow_expansion = allow_expansion
        self.receipts: list[dict] = []
        self.expansions: list[dict] = []
        self.version_receipts: dict[str, dict] = {}
        self._keys: set[tuple] = set()
        self.initial = [self.index.locations[k].packet(2400) for k in sorted(self.known)]

    def _add_file(self, name: str, start: int, round_number: int) -> bool:
        if not readable_name(name):
            raise ValueError("file not eligible for patch-context reading")
        path = checked_file(self.base, name)
        if path.stat().st_size > 2_000_000:
            raise ValueError("context source exceeds size limit")
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        lines = text.splitlines(keepends=True)
        if start > max(1, len(lines)):
            raise ValueError("requested start line exceeds source length")
        used = sum(len(item["source"]) for item in self.receipts)
        remaining = min(12000, self.char_budget - used)
        if remaining <= 0:
            raise ValueError("patch context character budget exhausted")
        excerpt = "".join(lines[start - 1:])[:remaining]
        key = (name, start, hashlib.sha256(excerpt.encode()).hexdigest())
        if key in self._keys:
            return False
        self._keys.add(key)
        self.read_files.add(name)
        self.receipts.append({
            "id": f"P{len(self.receipts) + 1}", "file": name, "start_line": start,
            "source": excerpt, "source_sha256": hashlib.sha256(raw).hexdigest(),
            "excerpt_sha256": key[2], "base_commit": self.guidance.task["base_commit"],
            "introduced_round": round_number, "truncated": len("".join(lines[start - 1:])) > len(excerpt),
            "role": "repository_visible_context_not_contract_revision",
        })
        return True

    def _add_location(self, location, round_number: int) -> bool:
        added = self._add_file(location.file, location.start, round_number)
        self.known.add(location.id)
        return added

    def request(self, requests: list[PatchContextRequest], round_number: int) -> dict:
        """Each failed request is recorded; no model request executes shell commands."""
        outcomes = []
        progressed = False
        for request in requests[:self.additions]:
            try:
                changed = False
                target = request.target
                if request.kind == "version":
                    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", target):
                        raise ValueError("invalid distribution name")
                    if target not in self.version_receipts:
                        self.version_receipts[target] = environment_receipt(self.base, self.executor, [target])
                        changed = True
                elif request.kind == "allow_edit":
                    name = safe_path(target)
                    if not self.allow_expansion:
                        raise ValueError("edit-scope expansion disabled")
                    if name not in self.read_files or not production_path(name):
                        raise ValueError("read the production source before requesting edit permission")
                    checked_file(self.base, name)
                    if name not in self.allowed:
                        if len(self.allowed) >= 20:
                            raise ValueError("edit scope file budget exhausted")
                        self.allowed.add(name)
                        self.expansions.append({"file": name, "round": round_number,
                                                "rationale": request.rationale})
                        changed = True
                elif request.kind == "read_file":
                    changed = self._add_file(target, request.start_line, round_number)
                else:
                    if request.kind == "expand":
                        location = self.index.locations.get(target)
                        if location is None or target not in self.known:
                            raise ValueError("unknown/unseen location ID")
                        changed = self._add_location(location, round_number)
                    else:
                        locations, _ = self.index.requested_context(
                            [ContextRequest(kind=request.kind, target=target)], self.known, self.additions
                        )
                        for location in locations:
                            changed = self._add_location(location, round_number) or changed
                        # Explicit search can also reveal repository-visible tests/docs.
                        if request.kind == "search":
                            terms = [t.lower() for t in re.findall(r"[\w.]{3,}", target)][:6]
                            hits = 0
                            for path in sorted(self.base.rglob("*")):
                                name = path.relative_to(self.base).as_posix()
                                if not path.is_file() or path.is_symlink() or not readable_name(name):
                                    continue
                                if path.stat().st_size > 500000 or name in self.read_files:
                                    continue
                                text = path.read_text(encoding="utf-8", errors="replace")
                                first = next((i for i, line in enumerate(text.splitlines(), 1)
                                              if terms and any(t in line.lower() for t in terms)), None)
                                if first:
                                    changed = self._add_file(name, max(1, first - 4), round_number) or changed
                                    hits += 1
                                    if hits >= 2:
                                        break
                progressed = progressed or changed
                outcomes.append({"request": request.model_dump(), "status": "ADDED" if changed else "NO_NEW_CONTEXT"})
            except (OSError, ValueError) as exc:
                outcomes.append({"request": request.model_dump(), "status": "REJECTED", "detail": str(exc)[:800]})
        return {"round": round_number, "progress": progressed, "outcomes": outcomes,
                "allowed_edit_paths": sorted(self.allowed), "scope_expansions": list(self.expansions),
                "retrieved_context": list(self.receipts), "dependency_versions": self.version_receipts}

    def packet(self) -> dict:
        return {"initial_localization": self.initial,
                "available_location_ids": {k: {"file": self.index.locations[k].file,
                                                "symbol": self.index.locations[k].symbol}
                                           for k in sorted(self.known)},
                "retrieved_context": self.receipts, "allowed_edit_paths": sorted(self.allowed),
                "dependency_versions": self.version_receipts,
                "remaining_context_chars": max(
                    0, self.char_budget - sum(len(item["source"]) for item in self.receipts)
                )}
