# Natural-language contract review

First, enumerate each independent requested behavior from the complete issue
evidence, including later paragraphs and examples. For each, check that the
NLC states both its trigger and caller-visible outcome. Put every absent row
in `missing_required_clauses`; an accurate clause for one example does not
cover another input class merely because both use the same API. Then review
the supplied NLC clauses for evidential support. Do not infer that the author
covered the issue from a shared evidence citation: compare the actual clause
text against each requested behavior. When a selected operation cannot
observe a requested behavior, report the localization gap instead of treating
the narrower NLC as complete.

Treat each supplied clause identifier and clause string as immutable input.
Copy both byte-for-byte into the corresponding structured review entry; put
semantic judgments only in verdict, evidence references, program facts, and
reason fields. Return one review per supplied clause in the required schema.

Judge whether the issue evidence supports the stated behavior at the selected
contracted operation. If a clause is too broad, unsupported, or attached to the
wrong operation, classify it accordingly and explain why; do not silently
rewrite its text. Keep missing requirements separate from unsupported added
requirements. Check that the obligation includes every reported trigger and
every required normal output, side effect, or exception supported by the issue.
Compare the issue's concrete call and final observation with the NLC's callable,
input domain, and postcondition. If the NLC names an internal helper instead,
require a traced path from the concrete call and reject any blanket helper rule
that is stronger than the observed issue behavior. A valid-looking symbolic
result at an intermediate step is not itself the reported outcome.
For a final-output defect, inspect the path after the selected operation. Flag
an NLC that requires a nonempty or exact *intermediate* return merely because
the final output is broken. A downstream formatter or wrapper may correctly
normalize an empty value or other sentinel. Require the supported final
observation, or identify a localization gap if it cannot be observed, and mark
the unsupported intermediate requirement as added meaning.
Use source code to verify API mechanics, not to invent the
desired behavior. If the operation is not connected to the reported behavior,
identify that localization problem explicitly.
Reject an exact output form inferred from an illustrative example when the
issue permits equivalent forms or discusses a boundary that changes the form.
Do not reject an otherwise testable qualitative outcome merely because the
issue leaves implementation details unspecified.
Before accepting, compare the proposed postcondition with each distinct
behavior the issue explicitly requests. Put an omitted variant in
`missing_required_clauses` even when the included clause is true and supported.
For example, a request for order-independent equality is not covered by a
witness that compares one identical ordering.
