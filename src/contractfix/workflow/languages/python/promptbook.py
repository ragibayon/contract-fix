"""Immutable, stage-specific prompts with reviewed memory and typed demonstrations.

Selection is deterministic lexical/tag matching, not embedding retrieval. No run
can append memory or change an example after this snapshot has been constructed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from contractfix.contracts.core import digest
from .models import (
    CanonicalRepairObligations,
    ExecutableContractCandidates,
    ExecutableContractConformanceBatch,
    Localization,
    RepairObligationReview,
)

from .repair_models import PatchNLCConformance, PatchProposal

SCHEMAS = {
    "patch": PatchProposal,
    "patch_nlc_conformance": PatchNLCConformance,
    "localize": Localization,
    "repair_obligations": CanonicalRepairObligations,
    "repair_obligation_review": RepairObligationReview,
    "executable_contract_synthesis": ExecutableContractCandidates,
    "executable_contract_conformance": ExecutableContractConformanceBatch,
}

PROMPT_FILES = {
    "patch": "patch_generation.j2",
    "patch_nlc_conformance": "patch_nlc_conformance.j2",
    "localize": "repository_localization.j2",
    "repair_obligations": "repair_obligation_derivation.j2",
    "repair_obligation_review": "repair_obligation_review.j2",
    "executable_contract_synthesis": "executable_contract_synthesis.j2",
    "executable_contract_conformance": "executable_contract_conformance.j2",
}
CONTROL_PROMPT_FILES = {
    "system": "qualification_system.j2",
    "schema_retry": "structured_output_retry.j2",
}


def _terms(value: object) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9_]+", json.dumps(value, sort_keys=True).lower()))


def _without_rationales(value: object) -> object:
    """Remove optional prose rationales from a demonstration recursively."""
    if isinstance(value, dict):
        return {
            key: _without_rationales(item)
            for key, item in value.items()
            if key != "rationale"
        }
    if isinstance(value, list):
        return [_without_rationales(item) for item in value]
    return value


class PromptBook:
    """Load and validate one prompt release, then render without further disk reads."""

    def __init__(
        self,
        directory: Path | None = None,
        *,
        profile: str = "contract-first-v3",
        example_profile: str = "qualified-successes-v1",
    ):
        root = directory or Path(__file__).with_name("prompts")
        if profile != "contract-first-v3":
            raise ValueError("only contract-first-v3 is supported")
        self.profile = profile
        if example_profile != "qualified-successes-v1":
            raise ValueError("only qualified-successes-v1 is supported")
        self.example_profile = example_profile
        self.schemas = dict(SCHEMAS)
        self.assets = {}
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("prompt assets must not be symlinks")
            if path.is_file() and path.suffix in {".j2", ".md", ".json", ".jsonl"}:
                if path.stat().st_size > 200_000:
                    raise ValueError("prompt asset exceeds size limit")
                self.assets[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8")
        required = set(PROMPT_FILES.values()) | set(CONTROL_PROMPT_FILES.values()) | {
            "policy.json",
            "examples.jsonl",
            "memory.json",
        }
        if required - self.assets.keys():
            raise ValueError("missing prompt assets: " + str(sorted(required - self.assets.keys())))
        self.texts = {
            stage: self.assets[name]
            for stage, name in {**PROMPT_FILES, **CONTROL_PROMPT_FILES}.items()
        }
        self.policy = json.loads(self.assets["policy.json"])
        if set(self.policy["stages"]) != set(SCHEMAS):
            raise ValueError("prompt policy must cover exactly the supported stages")
        if self.policy.get("selection") != "deterministic_tag_overlap":
            raise ValueError("unsupported demonstration selector")
        count = self.policy.get("max_examples", 1)
        if type(count) is not int or not 0 <= count <= 3:
            raise ValueError("max_examples must be an integer between zero and three")
        for rule in self.policy["stages"].values():
            if type(rule["answer_tokens"]) is not int or not 128 <= rule["answer_tokens"] <= 8192:
                raise ValueError("invalid stage answer budget")
            for name in rule["skills"]:
                if f"skills/{name}.md" not in self.assets:
                    raise ValueError("missing stage skill: " + name)
        synthetic_examples = [
            json.loads(line)
            for line in self.assets["examples.jsonl"].splitlines()
            if line.strip() and json.loads(line)["stage"] in SCHEMAS
        ]
        qualified_examples = [
            json.loads(line)
            for line in self.assets.get("examples.qualified.jsonl", "").splitlines()
            if line.strip()
        ]
        # Pre-gold EC examples were frozen under the former runtime rule that
        # treated an uncaught reported exception as a qualified violation. Keep
        # their provenance in the asset, but do not teach that obsolete protocol.
        protocol_sensitive_stages = {
            "executable_contract_synthesis",
            "executable_contract_conformance",
        }
        active_qualified_examples = [
            item
            for item in qualified_examples
            if not (
                item["stage"] in protocol_sensitive_stages
                and item.get("origin") == "qualified_pre_gold"
            )
        ]
        qualified_stages = {item["stage"] for item in active_qualified_examples}
        self.synthetic_examples = synthetic_examples
        self.excluded_source_instances = {
            item["source_instance_id"]
            for item in qualified_examples
            if item.get("source_instance_id")
        }
        self.examples = active_qualified_examples + [
            item for item in synthetic_examples if item["stage"] not in qualified_stages
        ]
        ids = set()
        for example in self.examples:
            if example["id"] in ids or example["stage"] not in SCHEMAS:
                raise ValueError("duplicate example ID or unknown stage")
            ids.add(example["id"])
            origin = example.get("origin")
            if origin not in {"independently_authored_synthetic", "qualified_pre_gold"}:
                raise ValueError(
                    "unsupported demonstration origin; synthetic examples and explicitly "
                    "qualified pre-gold examples are the only accepted sources"
                )
            if origin == "qualified_pre_gold":
                required_provenance = {
                    "source_instance_id",
                    "source_run_id",
                    "artifact_sha256",
                    "frozen_sha256",
                    "gold_diagnostic",
                }
                if required_provenance - set(example):
                    raise ValueError("qualified demonstration lacks provenance")
                if example["gold_diagnostic"] != "GOLD_PASS":
                    raise ValueError("qualified demonstration requires GOLD_PASS provenance")
            self.schemas[example["stage"]].model_validate(example["answer"])
        memory = json.loads(self.assets["memory.json"])
        self.lessons = memory["lessons"]
        for lesson in self.lessons:
            if lesson.get("review_status") != "approved" or lesson.get("split") != "lite-dev":
                raise ValueError("reusable lessons require approved Lite-dev provenance")
            if not lesson.get("source_run_ids") or not lesson.get("text", "").strip():
                raise ValueError("memory lesson requires provenance and text")
            if len(lesson["text"]) > 1200 or set(lesson["stages"]) - set(SCHEMAS):
                raise ValueError("memory lesson too large or stage unknown")
        self.env = Environment(undefined=StrictUndefined, autoescape=False)

    @property
    def sha256(self) -> str:
        return digest(self.assets)

    def _skills(self, stage: str) -> str:
        return "\n\n".join(self.assets[f"skills/{name}.md"]
                           for name in self.policy["stages"][stage]["skills"])

    def _memory(self, stage: str) -> str:
        return "\n".join(lesson["text"] for lesson in self.lessons if stage in lesson["stages"])

    def render(self, stage: str, packet: dict, *, component: str = "localize") -> str:
        return self.env.from_string(self.texts[stage]).render(
            packet=packet, stage=component, skill_text=self._skills(component),
            memory_text=self._memory(component), prompt_profile=self.profile)

    def demonstration_messages(self, stage: str, packet: dict) -> tuple[list[dict], list[str]]:
        terms = _terms(packet)
        instance_id = (packet.get("task") or {}).get("instance_id")
        available = [
            item
            for item in self.examples
            if item["stage"] == stage
            and (
                not instance_id
                or item.get("source_instance_id") != instance_id
            )
        ]
        # A qualified regression fixture must never demonstrate itself. When it
        # is the only qualified example for this stage, fall back to an
        # independently authored synthetic demonstration rather than aborting
        # the workflow or leaking the expected contract.
        if not available:
            available = [
                item for item in self.synthetic_examples if item["stage"] == stage
            ]
        available.sort(key=lambda item: (
            -3 * len(terms & set(item["tags"])) - len(terms & _terms(item["packet"])), item["id"]))
        selected = available[:self.policy["max_examples"]]
        if instance_id and any(
            item.get("source_instance_id") == instance_id for item in selected
        ):
            raise ValueError("prompt demonstration provenance guard failed")
        messages = []
        for example in selected:
            answer = example["answer"]
            include_rationales = bool(packet.get("include_rationales", False))
            if not include_rationales:
                answer = _without_rationales(answer)
            example_packet = {
                **example["packet"],
                "include_rationales": include_rationales,
            }
            # Qualified examples predate the candidate-limit terminology.
            # Normalize only the model-facing copy; keep the immutable stored
            # provenance untouched and never teach an exact requested count.
            legacy_candidate_count = example_packet.pop("candidate_count", None)
            if "candidate_limit" not in example_packet and legacy_candidate_count is not None:
                example_packet["candidate_limit"] = legacy_candidate_count
            messages.extend([
                {
                    "role": "user",
                    "content": (
                        "# Independent demonstration (not the current task)\n"
                        + self.render(stage, example_packet, component=stage)
                    ),
                },
                {"role": "assistant", "content": json.dumps(answer, sort_keys=True)},
            ])
        return messages, [item["id"] for item in selected]

    def assemble(self, stage: str, packet: dict) -> dict:
        messages, selected = self.demonstration_messages(stage, packet)
        return {"system": self.render("system", {}, component=stage),
                "prompt": self.render(stage, packet, component=stage),
                "demonstrations": messages, "example_ids": selected,
                "skills": self.policy["stages"][stage]["skills"],
                "prompts_sha256": self.sha256}

    def audit(self) -> dict:
        return {"version": self.policy["version"], "sha256": self.sha256,
                "profile": self.profile,
                "example_profile": self.example_profile,
                "assets": len(self.assets), "examples": len(self.examples),
                "reviewed_lessons": len(self.lessons), "selection": self.policy["selection"],
                "stages": {
                    stage: {
                        "schema": self.schemas[stage].__name__,
                        **self.policy["stages"][stage],
                    }
                    for stage in SCHEMAS
                },
                "live_model_tested": False}
