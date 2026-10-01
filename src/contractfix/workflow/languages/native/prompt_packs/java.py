"""Java source, contract, witness, and repair guidance."""

from __future__ import annotations

from .base import NativeLanguagePromptPack


PACK = NativeLanguagePromptPack(
    language="java",
    label="Java",
    version="native-java/1",
    localize=(
        "Follow the issue from API or call site to the production method that implements "
        "the behavior. Distinguish an annotation declaration from its processor or "
        "handler, and a method declaration from its implementation. Keep package and "
        "module paths exact."
    ),
    nlc=(
        "Describe normal and exceptional behavior separately. Include null handling, "
        "iteration lifetime, or generated behavior only when the issue and source "
        "support them; do not add a guarantee inferred only from a method name."
    ),
    review=(
        "Check both the normal and exceptional paths against the issue and source. "
        "Reject a patch that passes a target test by changing unrelated iterator, "
        "null, or generated-code behavior."
    ),
    ec=(
        "Use class ContractFixWitness with public static void main. Invoke the actual "
        "repository class and method with the required class path; do not create a "
        "stand-in implementation. Exercise the relevant normal or exceptional case."
    ),
    patch=(
        "Edit the implementing Java class in its owning module. Preserve overloads, "
        "checked exceptions, and annotation-processing behavior. If the issue spans "
        "paired operations such as read and write, inspect both before proposing edits."
    ),
)
