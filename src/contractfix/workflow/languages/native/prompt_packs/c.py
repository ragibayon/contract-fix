"""C source, contract, witness, and repair guidance."""

from __future__ import annotations

from .base import NativeLanguagePromptPack


PACK = NativeLanguagePromptPack(
    language="c",
    label="C",
    version="native-c/1",
    localize=(
        "Trace command aliases and dispatch tables to the implementing function. "
        "Prefer a .c definition over a header prototype unless the behavior is in a "
        "macro or inline function. Follow callers when the selected file only dispatches."
    ),
    nlc=(
        "State return values, error codes, NULL and sentinel behavior precisely. "
        "Mention ownership or mutation only when the issue and source establish it."
    ),
    review=(
        "Check each claimed return value, sentinel, and state transition against "
        "the source. Reject unsupported behavior or changes to ownership and errors."
    ),
    ec=(
        "Use int main(void) for C. Include the selected repository source and call its actual "
        "operation symbol. Do not copy the operation into a proxy or model. Print "
        "the reachability marker after the call and one assertion marker for its result."
    ),
    patch=(
        "Edit the .c implementation that owns the behavior; a header-only change must "
        "be justified by an inline or macro implementation or an API change. Preserve "
        "ownership, integer bounds, return codes, and existing command aliases."
    ),
)
