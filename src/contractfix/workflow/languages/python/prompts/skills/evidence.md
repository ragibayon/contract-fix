Use the issue as change evidence and documentation to justify the current obligation.
Recover preservation obligations only when the stage explicitly requests them.
Implementation observations, caller restrictions and examples alone do not establish a public requirement.
Retain exact evidence provenance. Missing or contradictory evidence remains unknown.
When the host supplies canonical evidence spans, select only their opaque `span_id`; the host owns
the exact quotation bytes. In an explicitly legacy quote-based stage, copy one contiguous verbatim
span from the selected source. Never insert ellipses, join distant excerpts, or paraphrase a quote.

For a reported failure sequence, first identify the caller-visible operation that fails,
the triggering input or state, and the supported expected behavior. Keep setup facts
needed to reproduce that operation, but do not turn an incidental implementation
detail into a requirement. If the issue states a behavior without spelling out a
particular representation, express the behavior at that level; do not guess an exact
header, field set, string, or ordering. Check that each proposed clause is supported
by the cited evidence plus any explicit program facts, and use the stage's supported
abstention only for meaning that remains genuinely unsupported.
