"""JavaScript and TypeScript repository guidance for the shared contract flow."""

from __future__ import annotations

from .base import NativeLanguagePromptPack


PACK = NativeLanguagePromptPack(
    language="javascript",
    label="JavaScript/TypeScript",
    version="native-javascript/1",
    localize=(
        "Trace the issue's public API to the source function that implements it. "
        "Distinguish source under src, lib, and packages from generated dist output, "
        "declarations, fixtures, tests, and wrapper exports. Follow re-exports "
        "and package entry points when needed. For component behavior, identify "
        "the owning runtime or renderer function. Keep exact repository paths."
    ),
    nlc=(
        "Describe the observable behavior established by the issue and supplied "
        "source. Specify relevant inputs, return or Promise behavior, thrown or "
        "rejected errors, and side effects. Distinguish undefined, null, falsy "
        "values, and missing properties where the issue does. Do not infer a "
        "universal guarantee from one caller or an undocumented implementation detail."
    ),
    review=(
        "Check the claimed precondition and postconditions against the issue and "
        "the implementing function. Reject unsupported claims about async timing, "
        "exceptions, identity, mutability, or browser versus Node behavior. "
        "A narrow passing witness does not justify a broader contract."
    ),
    ec=(
        "Write a self-contained Node CommonJS witness. Require the real repository "
        "API from a relative source path or its built package entry point; do not "
        "copy the implementation. Await Promise results in an async main when needed. "
        "Print CONTRACTFIX_OPERATION_REACHED after the operation is invoked and "
        "then exactly one of CONTRACTFIX_ASSERTION_PASS or "
        "CONTRACTFIX_ASSERTION_FAIL according to the accepted postcondition. "
        "Treat an expected throw or rejection as an observation to assert, not "
        "as an unhandled process failure. Avoid browser-only globals unless the "
        "repository's installed environment supplies them."
    ),
    patch=(
        "Edit the implementation source rather than generated bundles, lockfiles, "
        "tests, or declarations. Preserve CommonJS/ESM exports, TypeScript types, "
        "Promise timing, and existing public behavior. Use build diagnostics and "
        "frozen NLC/EC feedback to refine a subsequent candidate without weakening "
        "the contract. For a re-export, request the implementing file before editing."
    ),
    context_navigation=(
        "Search for the exported symbol through index files and package entry "
        "points. Request exact source ranges with expand path:start-end, then "
        "allow_edit on the implementing source path before changing it."
    ),
    context_max_excerpts=6,
    context_max_chars=12000,
)
