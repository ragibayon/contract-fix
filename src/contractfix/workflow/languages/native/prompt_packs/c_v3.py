"""C development pack: command ownership, state, and direct repository witness."""

from __future__ import annotations

from .base import NativeLanguagePromptPack


PACK = NativeLanguagePromptPack(
    language="c",
    label="C",
    version="native-c/3",
    localize=(
        "Trace command names and aliases through checked-in command metadata and "
        "dispatch to the defining .c function. A header declaration or dispatch "
        "branch alone is not an implementation. For stateful behavior, identify "
        "the code that mutates or serializes the state, including related callers. "
        "For generated parsers, select the checked-in .y grammar when the "
        "reported behavior is defined there; generated parser .c files are "
        "derived outputs. Trace grammar actions into semantic compilation "
        "and execution; do not assign runtime behavior to the parser driver."
    ),
    nlc=(
        "State observable return values, error codes, NULL and sentinel handling, "
        "and state transitions separately. Add ownership, lifetime, or mutation "
        "requirements only when the issue and supplied source support them. Cite "
        "the supplied implementing file, optionally with #function."
    ),
    review=(
        "Verify each return, error, sentinel, and state claim against issue and "
        "source evidence. Reject unsupported ownership or mutation assumptions. "
        "When reviewing a patch, check the accepted contract without replacing "
        "the repository's command or function with a local model."
    ),
    ec=(
        "Use int main(void). Include the selected repository source and call the "
        "actual operation symbol with a minimal valid fixture. Do not copy or "
        "reimplement the operation. Print the reachability marker after the call "
        "and exactly one assertion marker for the observed NLC postcondition. "
        "If a standalone witness cannot link to the real operation, leave the EC "
        "unqualified rather than fabricating behavior. If previous_ec_feedback "
        "is present, repair the cited include, call, fixture, or compiler error "
        "without changing the accepted NLC or replacing the operation."
    ),
    patch=(
        "Edit the .c definition that owns the behavior. A header-only edit needs "
        "an inline or macro implementation or a justified API change. Preserve "
        "ownership, integer bounds, return codes, and command aliases. After a "
        "failed build or EC check, correct the source or abstain; do not weaken "
        "the contract to accept an invalid patch."
    ),
    context_navigation=(
        "If a header or command dispatcher is supplied, search the concrete "
        "function and inspect the returned source_path. Request allow_edit for "
        "that file after reading it. Use expand path:start-end for adjacent "
        "implementation lines, and do not repeat a range already in "
        "retrieved_context."
    ),
    context_max_excerpts=8,
    context_max_chars=14000,
)
