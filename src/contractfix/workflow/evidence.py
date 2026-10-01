"""Host-owned canonical evidence spans and semantic model views."""

from __future__ import annotations

import hashlib
import re
from typing import Any


def _chunks(text: str, maximum: int = 1800) -> list[tuple[int, int]]:
    """Return stable contiguous ranges without rewriting source bytes."""
    ranges: list[tuple[int, int]] = []
    for match in re.finditer(r"\S.*?(?=\n[^\S\n]*\n|\Z)", text, re.DOTALL):
        start, end = match.span()
        # A trailing newline must not discard the entire final paragraph.
        while end > start and text[end - 1].isspace():
            end -= 1
        while end - start > maximum:
            window = text[start : start + maximum]
            split = max(window.rfind("\n"), window.rfind(". "))
            if split < maximum // 3:
                split = maximum
            elif window[split : split + 2] == ". ":
                split += 1
            ranges.append((start, start + split))
            start += split
            while start < end and text[start].isspace():
                start += 1
        if start < end:
            ranges.append((start, end))
    return ranges


def canonical_spans(evidence: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    spans: dict[str, dict[str, Any]] = {}
    for evidence_id, record in evidence.items():
        text = str(record.get("text", ""))
        for position, (start, end) in enumerate(_chunks(text), start=1):
            span_id = f"{evidence_id}.S{position:03d}"
            quote = text[start:end]
            spans[span_id] = {
                **{key: value for key, value in record.items() if key != "text"},
                "span_id": span_id,
                "evidence_id": evidence_id,
                "start": start,
                "end": end,
                "text": quote,
                "sha256": hashlib.sha256(quote.encode()).hexdigest(),
            }
    return spans


def semantic_obligation(obligation: dict[str, Any]) -> dict[str, Any]:
    """Remove control IDs and diagnostic hypotheses from model-visible semantics."""
    return {
        "precondition": obligation["precondition"],
        "normal_postcondition": obligation.get("normal_postcondition"),
        "exceptional_postcondition": obligation.get("exceptional_postcondition"),
        "dynamic_violation_criterion": obligation["violation_criterion"],
        "support": [{"role": item["supports"], "text": item["quote"]} for item in obligation["evidence"]],
    }


def semantic_location(location: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in location.items() if key != "id"}
