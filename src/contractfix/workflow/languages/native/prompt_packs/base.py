"""Versioned language guidance used by the shared native prompt workflow."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NativeLanguagePromptPack:
    language: str
    label: str
    version: str
    localize: str
    nlc: str
    review: str
    ec: str
    patch: str
    context_navigation: str = ""
    context_max_excerpts: int = 8
    context_max_chars: int = 14000

    def guidance(self, stage: str) -> str:
        return {
            "native_localize": self.localize,
            "native_nlc": self.nlc,
            "native_nlc_review": self.review,
            "native_ec": self.ec,
            "native_patch": self.patch,
        }.get(stage, "")
