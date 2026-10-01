"""Shared native prompt contract with versioned Java, C, and C++ packs."""

from __future__ import annotations

from dataclasses import asdict
import json

from contractfix.utils.artifacts import digest

from .prompt_packs import prompt_pack_for


STAGES = (
    "native_localize",
    "native_nlc",
    "native_nlc_review",
    "native_ec",
    "native_patch",
)

COMMON_DIRECTIONS = {
    "native_localize": (
        "Select only supplied production source paths. operation_file must equal "
        "one of the supplied file paths exactly; put the method or function name "
        "in operation_symbol."
    ),
    "native_nlc": (
        "State the issue-grounded natural-language precondition and postconditions. "
        "Cite supplied source paths. A citation may add #symbol to a supplied path; "
        "do not cite files that have not been supplied."
    ),
    "native_nlc_review_contract": (
        "Review the NLC against the issue and supplied source. Reject unsupported or added behavior."
    ),
    "native_nlc_review_patch": (
        "Review the patch against the accepted NLC and issue. Reject missing or added behavior."
    ),
    "native_nlc_review_ec": (
        "Review whether the executable witness invokes the real repository operation, "
        "directly or through a caller-visible API that executes it, and "
        "checks the accepted NLC's precondition and caller-visible postcondition. "
        "Reject copied stand-in implementations, setup-only failures, unexercised assertions, "
        "missing contract meaning, and unsupported strengthening. Use only the issue, "
        "supplied source, accepted NLC, and witness; never use a reference patch or official tests."
    ),
    "native_ec": (
        "Write a native witness for the issue. Print CONTRACTFIX_OPERATION_REACHED "
        "after invoking the contracted operation; print CONTRACTFIX_ASSERTION_PASS "
        "only when the postcondition holds, or CONTRACTFIX_ASSERTION_FAIL when it "
        "does not. Do not use official tests, reference patches, shell commands, "
        "or dependency changes."
    ),
    "native_patch": (
        "Propose exact source-line replacements in allowed production paths only. "
        "Preserve existing dependencies and tests. Each old_lines and new_lines "
        "array item must contain one physical source line without a newline. "
        "Use request_context to search or read missing source. If a newly read "
        "production file contains the real repair, request allow_edit for that "
        "exact file before submitting edits; never edit outside allowed_paths. "
        "available_edit_paths lists read production files ready for an allow_edit "
        "request. When the implementation is in one of those files, request "
        "allow_edit for its exact path instead of abstaining over edit scope. "
        "For exact source ranges use expand with target path:start-end; use "
        "path:identifier for a scoped search. Follow context_feedback "
        "suggested_next_request when it points to the implementation file. "
        "If previous_candidate_feedback reports exhausted or repeated context, "
        "use the source already supplied before requesting more. If it reports "
        "an exact-edit mismatch, use exact_edit_context source_lines to copy "
        "old_lines exactly, including punctuation and indentation; modify "
        "only new_lines. If no unique anchor was found, request exact source "
        "context before resubmitting. "
        "When edit_correction_only is true, correct the rejected proposal's "
        "old_lines from the supplied exact base lines and submit edits now; "
        "do not request more context or change the issue requirement. "
        "For a failed build, address ordinary_first_diagnostic and then the "
        "recorded NLC review or EC failure before proposing another patch. "
        "When ec_validation_status is EC_VIOLATION, use ec_failure_detail to "
        "refine the implementation against the same frozen obligation. "
        "If context_request_closed is true, no further source requests are "
        "available in this attempt: submit a grounded edit from the supplied "
        "source or abstain with a reason. "
        "Use abstain only with a reason and no edits or context requests."
    ),
}


class NativePromptBook:
    profile = "native-contract/10"
    example_profile = "none"

    def __init__(self, language: str):
        self.pack = prompt_pack_for(language)
        self.language = language
        self.texts = {
            "schema_retry": "Return one valid object matching the supplied JSON schema.",
        }
        self.policy = {"stages": {stage: {"answer_tokens": 1024} for stage in STAGES}}
        self.assets = {
            "language": language,
            "prompt_version": self.profile,
            "language_pack": self.pack.version,
        }
        self.system = (
            f"You are repairing a {self.pack.label} repository from issue and base-source evidence only."
        )
        self.sha256 = digest(
            {
                "profile": self.profile,
                "pack": asdict(self.pack),
                "system": self.system,
                "common_directions": COMMON_DIRECTIONS,
                "policy": self.policy,
                "texts": self.texts,
            }
        )

    def assemble(self, stage: str, packet: dict) -> dict:
        if stage not in self.policy["stages"]:
            raise ValueError(f"unknown native stage: {stage}")
        direction_key = stage
        if stage == "native_nlc_review":
            direction_key = (
                "native_nlc_review_patch" if "patch" in packet else
                "native_nlc_review_ec" if "ec" in packet else
                "native_nlc_review_contract"
            )
        directions = COMMON_DIRECTIONS[direction_key]
        guidance = self.pack.guidance(stage)
        if stage == "native_patch":
            guidance += " " + self.pack.context_navigation
        return {
            "system": self.system,
            "prompt": (directions + " " + guidance + "\n\n" + json.dumps(packet, ensure_ascii=False)),
            "demonstrations": [],
        }
