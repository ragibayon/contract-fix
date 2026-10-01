# Review patch conformance to the NLC

Use the issue, frozen NLC, diff, and repository-visible validation summary.
Judge what the patch actually changes under the NLC preconditions. Trace each
required postcondition to a concrete effect, including retained data,
constraints, side effects, and output. Passing a syntax check or eliminating
the reported exception alone does not establish those effects.

When an EC rejected the patch, make the NLC judgment independently. An EC
witness may be over-constrained; its failure is not evidence that the patch
violates the NLC. Do not infer conformance solely because the patch addresses
an EC witness either. If the patch filters, deletes, or skips a required
object, identify where the object is preserved or recreated. Mark `VIOLATES`
when the supplied diff clearly loses a required behavior and `INCONCLUSIVE`
when its fate cannot be established from the supplied material. Mark
`CONFORMS` only when every required clause has concrete support.

List the satisfied and missing clauses with the supporting code effect.
Return the host's exact structured verdict schema. Do not use gold patches,
hidden tests, or evaluator outcomes.
