"""C++ development pack: template ownership, platform semantics, and builds."""

from __future__ import annotations

from .base import NativeLanguagePromptPack


PACK = NativeLanguagePromptPack(
    language="cpp",
    label="C++",
    version="native-cpp/2",
    localize=(
        "Resolve namespaces, overloads, and template specialization ownership. "
        "Definitions may be in headers well below an early declaration or usage. "
        "Select the defining source or header and inspect the relevant method "
        "body before choosing repair paths. Distinguish a formatter or trait "
        "specialization from its callers."
    ),
    nlc=(
        "State only the issue-scoped observable behavior. Preserve supported "
        "exception, const/reference, template, and path semantics. Do not infer "
        "cross-platform guarantees or a specific implementation strategy from "
        "one test. Cite supplied source paths, optionally with #symbol."
    ),
    review=(
        "Check the contract or patch against the selected template instantiation "
        "and issue evidence. Reject unsupported platform-specific guarantees, "
        "changed overload meaning, or a patch that only passes a narrow target "
        "while violating the accepted NLC."
    ),
    ec=(
        "Use int main(). Include the selected repository header or source and call "
        "the actual operation with valid namespace and template arguments. Assert "
        "an observable NLC postcondition on the buggy behavior. Never copy the "
        "operation into a proxy. If the witness cannot compile in the prepared "
        "environment, report an EC error rather than claiming qualification. "
        "If previous_ec_feedback is present, fix the exact compiler or fixture "
        "error while preserving the accepted NLC and real operation call."
    ),
    patch=(
        "Preserve overload resolution, template constraints, const qualifiers, "
        "and the repository's C++ standard. Edit the owning definition, including "
        "a header when that is where the template lives. A candidate must pass "
        "the declared local build before selection; do not replace that gate "
        "with a narrow official target result."
    ),
    context_navigation=(
        "When the supplied excerpt ends before a template or formatter body, "
        "request expand path:start-end around the definition. Search the owner "
        "symbol when only a call site is shown, then request allow_edit for the "
        "returned implementation path. Use context_feedback to avoid repeating "
        "already covered line ranges."
    ),
    context_max_excerpts=8,
    context_max_chars=14000,
)
