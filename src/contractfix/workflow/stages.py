"""Language-neutral stage protocol with compatibility exports."""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class Stages(Protocol):
    @property
    def identity(self) -> dict: ...

    def generate(self, stage: str, schema: type[T], packet: dict) -> T: ...


def __getattr__(name: str):
    """Preserve imports from the former combined Python stage module."""
    if name not in {"LangChainStages", "PromptBook", "_rationale"}:
        raise AttributeError(name)
    from .languages.python import stages as python_stages

    return getattr(python_stages, name)


__all__ = ["Stages"]
