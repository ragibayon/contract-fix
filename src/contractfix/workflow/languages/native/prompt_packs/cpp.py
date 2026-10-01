"""C++ source, contract, witness, and repair guidance."""

from __future__ import annotations

from .base import NativeLanguagePromptPack


PACK = NativeLanguagePromptPack(
    language="cpp",
    label="C++",
    version="native-cpp/1",
    localize=(
        "Resolve namespaces, overloads, and template owners before selecting the "
        "operation. Template definitions may live in headers; distinguish a call site "
        "or declaration from the definition that must change."
    ),
    nlc=(
        "Preserve documented exception, const, reference, and template behavior. "
        "Do not infer cross-platform path or formatting guarantees without source "
        "or issue evidence."
    ),
    review=(
        "Check template instantiations and exception or path semantics against "
        "the issue and source. Reject unsupported platform-specific guarantees."
    ),
    ec=(
        "Use int main(). Include the selected repository source and invoke its actual "
        "operation with the correct namespace and template arguments. Do not copy "
        "the operation into a proxy or model."
    ),
    patch=(
        "Preserve overload resolution, template constraints, const qualifiers, and "
        "the repository's C++ standard. Check the full local build before accepting "
        "a change that passes only a narrow target."
    ),
)
