"""Typed repair guidance and patch actions; model output never grants permissions."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from .models import Edit, StrictModel

RepairMode = Literal["ISSUE_ONLY", "NLC_CONTEXT", "EC_ENFORCED"]
RepairVariant = Literal["no-contract", "context", "enforce"]


class PatchContextRequest(StrictModel):
    kind: Literal["search", "expand", "callers", "callees", "read_file", "version", "allow_edit"]
    target: str = Field(min_length=1, max_length=300)
    start_line: int = Field(default=1, ge=1)
    rationale: str = Field(default="", max_length=500)


class PatchEdit(StrictModel):
    """Transport-safe exact replacement expressed as physical source lines."""

    file: str
    old_lines: list[str] = Field(min_length=1, max_length=400)
    new_lines: list[str] = Field(default_factory=list, max_length=400)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_strings(cls, value):
        """Keep stored/tests readable while exposing only line arrays to live models."""
        if isinstance(value, dict) and ("old" in value or "new" in value):
            value = dict(value)
            if "old_lines" not in value and "old" in value:
                value["old_lines"] = str(value.pop("old")).splitlines()
            if "new_lines" not in value and "new" in value:
                value["new_lines"] = str(value.pop("new")).splitlines()
        return value

    @field_validator("old_lines", "new_lines")
    @classmethod
    def one_physical_line_per_item(cls, lines: list[str]) -> list[str]:
        if any("\n" in line or "\r" in line for line in lines):
            raise ValueError("patch edit line arrays require one physical source line per item")
        return lines

    def materialized(self) -> Edit:
        return Edit(
            file=self.file,
            old="\n".join(self.old_lines),
            new="\n".join(self.new_lines),
        )


class PatchNLCConformance(StrictModel):
    """Patch-to-NLC semantic gate for NLC-only or EC-conflict selection."""

    verdict: Literal["CONFORMS", "VIOLATES", "INCONCLUSIVE"]
    satisfied_clauses: list[str] = Field(default_factory=list, max_length=12)
    missing_meaning: list[str] = Field(default_factory=list, max_length=12)
    contradicted_meaning: list[str] = Field(default_factory=list, max_length=12)
    unsupported_added_behavior: list[str] = Field(default_factory=list, max_length=12)
    reason: str = Field(min_length=1, max_length=2000)


class PatchProposal(StrictModel):
    """One complete patch OR bounded context requests OR explicit abstention."""

    action: Literal["submit_edits", "request_context", "abstain"]
    edits: list[PatchEdit] = Field(default_factory=list, max_length=12)
    context_requests: list[PatchContextRequest] = Field(default_factory=list, max_length=6)
    reason: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def validate_action(self):
        if self.action == "submit_edits":
            if not self.edits or self.context_requests:
                raise ValueError("submit_edits requires edits and no context requests")
        elif self.action == "request_context":
            if not self.context_requests or self.edits:
                raise ValueError("request_context requires requests and no edits")
        elif self.edits or self.context_requests or not self.reason.strip():
            raise ValueError("abstain requires a reason and no edits/context requests")
        return self


class RepairGuidance(StrictModel):
    version: Literal["contractfix-repair-guidance/1"] = "contractfix-repair-guidance/1"
    language: Literal["python"] = "python"
    task: dict
    repo_sha256: str
    executor: dict
    workflow: dict
    generator: dict
    localization: dict
    localized_edit_paths: list[str]
    nlc_status: Literal["QUALIFIED", "UNAVAILABLE"]
    qualified_nlc: dict | None = None
    ec_status: Literal["QUALIFIED", "UNAVAILABLE"]
    qualified_ec: dict | None = None
    qualification_status: str
    unavailability_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_availability(self):
        if (self.nlc_status == "QUALIFIED") != (self.qualified_nlc is not None):
            raise ValueError("NLC status and payload disagree")
        if (self.ec_status == "QUALIFIED") != (self.qualified_ec is not None):
            raise ValueError("EC status and payload disagree")
        if self.qualified_ec and not self.qualified_nlc:
            raise ValueError("an enforceable EC requires qualified NLC")
        if not self.localized_edit_paths:
            raise ValueError("repair guidance requires localized production files")
        return self


def effective_mode(variant: RepairVariant, guidance: RepairGuidance) -> RepairMode:
    if variant not in {"no-contract", "context", "enforce"}:
        raise ValueError("unknown repair variant")
    if variant == "no-contract" or guidance.qualified_nlc is None:
        return "ISSUE_ONLY"
    if variant == "enforce" and guidance.qualified_ec is not None:
        return "EC_ENFORCED"
    return "NLC_CONTEXT"
