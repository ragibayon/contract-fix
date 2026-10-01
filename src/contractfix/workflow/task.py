"""Language-neutral repository task envelope."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    repo: str
    base_commit: str = Field(pattern=r"^[a-fA-F0-9]{7,40}$")
    version: str
    problem_statement: str = Field(min_length=1)
    # Python is the compatibility default for historical SWE-bench task records.
    # Excluding this routing field preserves their serialized packets and hashes.
    language: str = Field(default="python", min_length=1, max_length=40, exclude=True)
    image: str | None = None

    @field_validator("language")
    @classmethod
    def normalize_language(cls, value: str) -> str:
        normalized = value.strip().lower().replace("_", "-")
        if not normalized:
            raise ValueError("task language must not be empty")
        return normalized

    def model_packet(self) -> dict[str, Any]:
        return self.model_dump(exclude={"image"})
