A required exceptional input must remain applicable. For a zero-divisor ValueError requirement,
when: y != 0 is NOT a valid way to remove zero. Use when: y == 0 with the expected exception behavior.
Separate input applicability from result correctness. old is one invocation's prestate, not a prior version.
A ContractFix contract is the smallest evidence-backed precondition and normal or exceptional
postcondition for a contracted operation under a finite contract witness.
It is not general function intent, full API documentation, a root-cause claim, or a patch prescription.

Before submitting a repair obligation or executable assertion, make the reported
failing invocation satisfy the precondition. Then check that the postcondition
requires only the evidence-backed caller-visible outcome. Reject a clause that
would fail a different legitimate implementation merely because it adds an
unmentioned field, chooses a different internal representation, or preserves
unrelated behavior. If an exact value is required, cite the evidence for that
exactness. An executable witness should test the smallest discriminating
behavior, not ban other valid outputs or encode the buggy implementation.

The stage schema is part of the task: fill every required field with its required
type, including explicit empty lists or permitted nulls. Keep the semantic check
and the schema check separate; a well-formed object with unsupported meaning is
not a valid contract. If the host reports a schema defect, correct the field
itself while keeping the supported contract unchanged.
