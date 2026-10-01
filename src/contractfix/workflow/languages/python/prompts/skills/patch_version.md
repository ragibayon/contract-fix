# Version-matched patch validity

Implement the frozen obligation in the allowed repair location. Before using a
new symbol or import, verify its definition and import path in the checked-out
repository version. Do not infer availability from current upstream docs or a
different release. Keep the patch narrow and make its behavior observable by
the selected EC or NLC. If execution reports an import or setup error, repair
that error before interpreting EC failure as a behavioral mismatch.
When adding an argument or keyword to a call, inspect the receiving signature
and every required forwarding step. When converting a container or dtype,
inspect the resulting value and dtype; a conversion method can retain the
representation that caused the bug.
For parsers, names, options, and formatted values, inspect distinct accepted
input forms and neighboring branches before treating one EC example as full
coverage. Check that the patch preserves existing valid forms as well as the
reported failing form. Use a repository example or focused test when available.

When the fix changes a general boundary such as index limits, shape handling,
or a shared conversion, inspect nearby operations for the same defect. Repair
a sibling only when it has the same invariant and stays within authorized
paths. Passing the frozen EC confirms its witness, not every related behavior
or regression test.
In the one EC-feedback refinement, use the observed violation to locate the
failing behavior, then compare the requested revision with the frozen NLC and
issue trigger. Do not introduce a behavior that contradicts the NLC or breaks
an ordinary valid case merely to satisfy one witness assertion. If no
issue-grounded revision is available, report that limit through the host's
schema; the host records the EC conflict and decides whether a separately
labeled ordinary-check fallback is eligible.
Before changing a conditional or adding a fallback, check the existing meanings
of other accepted inputs and provably known branches. Do not merge distinct
falsey inputs (for example `False` and an empty options dict), or run a new
setter on the old default path unless its side effects are known to be neutral.
Keep a fallback for unknown cases from swallowing cases the code can still
resolve exactly.

Return the exact patch schema requested by the host. The independently frozen
EC remains unchanged during patch work.
