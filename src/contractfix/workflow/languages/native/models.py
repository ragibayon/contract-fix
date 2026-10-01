"""Versioned Java/C/C++ stage contracts, independent of Python artifact schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..python.repair_models import PatchEdit, PatchProposal


class NativePatchEdit(PatchEdit):
    """Accept unambiguous multiline array items as physical source lines."""

    @field_validator("old_lines", "new_lines", mode="before")
    @classmethod
    def split_physical_lines(cls, value):
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            return value
        flattened = []
        for item in value:
            if "\n" not in item and "\r" not in item:
                flattened.append(item)
                continue
            lines = item.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            if lines[-1] == "":
                lines.pop()
            flattened.extend(lines)
        return flattened


class NativePatchProposal(PatchProposal):
    """Native-only transport normalization before the shared action checks."""

    edits: list[NativePatchEdit] = Field(default_factory=list, max_length=12)

    @model_validator(mode="before")
    @classmethod
    def preserve_context_request_intent(cls, value):
        if (isinstance(value, dict) and value.get("action") == "abstain"
                and value.get("context_requests") and not value.get("edits")):
            return {**value, "action": "request_context"}
        return value


class NativeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repair_enabled: bool = False
    repair_variant: Literal["no-contract", "context", "enforce"] = "enforce"
    stage_output_tokens: int = Field(default=1024, ge=128, le=4096)
    candidate_limit: int = Field(default=2, ge=1, le=4)


class NativeLocalization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repair_paths: list[str] = Field(min_length=1, max_length=4)
    operation_file: str = Field(description="Exact supplied production source file path containing the operation")
    operation_symbol: str
    rationale: str = ""


class NativeNLC(BaseModel):
    model_config = ConfigDict(extra="forbid")

    precondition: str = Field(min_length=1)
    normal_postcondition: str = Field(min_length=1)
    exceptional_postcondition: str | None = None
    evidence_paths: list[str] = Field(min_length=1)
    rationale: str = ""


class NativeNLCReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    reason: str


class NativeEC(BaseModel):
    """A model-authored native witness with explicit operation/assertion markers."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, max_length=20000)
    class_path: list[str] = Field(default_factory=list, max_length=8)
    rationale: str = ""


__all__ = [
    "NativeSettings", "NativeLocalization", "NativeNLC", "NativeNLCReview",
    "NativeEC", "NativePatchProposal", "PatchProposal",
]
