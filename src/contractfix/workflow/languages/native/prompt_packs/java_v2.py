"""Java development pack: implementation ownership, evidence, and witness use."""

from __future__ import annotations

from .base import NativeLanguagePromptPack


PACK = NativeLanguagePromptPack(
    language="java",
    label="Java",
    version="native-java/2",
    localize=(
        "Trace the reported public API through wrappers to the method that implements "
        "the behavior. Prefer a method body over a call site or declaration. For "
        "annotations, locate the processor or handler rather than the annotation "
        "type; for generated code, identify the generator. Keep module and package "
        "paths exact. If the issue names both read and write behavior, inspect both."
    ),
    nlc=(
        "Describe only behavior established by the issue and supplied production "
        "source. Separate normal and exceptional paths; distinguish null input, "
        "iterator exhaustion, and generated behavior when relevant. Cite exact "
        "supplied paths, optionally with #method. Do not turn an observation at one "
        "call site into a universal API guarantee."
    ),
    review=(
        "Check normal and exceptional paths in every NLC or patch claim against "
        "the reported behavior and supplied method body. Reject an unsupported "
        "exception, iterator, null, "
        "or generated-code guarantee. A passing narrow test does not excuse a "
        "contradiction of the accepted contract."
    ),
    ec=(
        "Use class ContractFixWitness with public static void main. Invoke the "
        "actual repository class and method using classes built from the sealed "
        "base; give relative class_path entries only when needed. Make the buggy "
        "behavior observable and assert the accepted NLC's postcondition. Never "
        "copy the method into a stand-in implementation or assert only construction."
    ),
    patch=(
        "Edit the implementing class in its owning module. Preserve overloads, "
        "checked exceptions, iterator lifetime, and annotation-processing effects. "
        "Use build or NLC review feedback to repair a candidate, not to weaken the "
        "accepted NLC. If the issue spans paired read/write operations, inspect "
        "both paths before editing."
    ),
    context_navigation=(
        "When only an annotation declaration, interface, or call site is visible, "
        "search for its implementation symbol. Read the returned source_path and "
        "request allow_edit for that exact path before changing it. For a late "
        "method body, request expand with path:start-end; avoid repeatedly asking "
        "for an already supplied range."
    ),
    context_max_excerpts=6,
    context_max_chars=12000,
)
