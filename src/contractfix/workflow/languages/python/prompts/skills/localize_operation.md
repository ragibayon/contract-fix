# Localize the behavior boundary

Choose the edit site and contracted operation separately. Trace the issue's
input through the checked-out version: construction, parser or dispatcher,
configuration, target call, and caller-visible observation. A matching symbol
name is insufficient. Match input syntax and type to the actual route (for
example, the parser selected by a docstring style). Check optional arguments
and configuration that activate the reported mechanism; a default call may
skip it.
If the issue includes a runnable call, keep that call as the observation
anchor. When selecting a lower-level helper, verify that the shown public call
reaches it with the stated arguments and that its return affects the reported
outcome. Do not transfer an issue requirement to a helper merely because the
helper is on the stack; retain the public call and input domain in the
localization record.
If the selected callable returns a value that a later formatter, serializer,
field, or wrapper transforms, follow that handoff before fixing the contract
boundary. An empty value, `None`, or another sentinel can be valid internally
even when the final user-visible output must be nonempty. Select a reachable
public call that observes the completed output, while still recording the
internal edit site and the path between them. Do not infer an internal return
value requirement solely from a broken final display.
For an error message or exception, find the layer that creates the
caller-visible message. A lower-level function may only raise a generic
exception while a collector, dispatcher, or wrapper adds the requested hint.
Choose a contracted operation and witness path that can reach that layer.
If the repair site is an internal method, inspect its callers and state set
before invocation. Prefer a supplied public entrypoint that initializes that
state as the contracted boundary. If only the internal method is visible,
request its caller or initialization context before selecting it.

When the issue names a producer type or returned value but the visible failure
occurs in a consumer, inspect the producer method and the handoff between them.
If the producer is absent from supplied context, request a targeted search for
that class or method before choosing the edit site. A consumer stack frame alone
does not locate the defect.

The contracted operation must be an existing callable the witness can reach.
When the requested behavior is implemented by a **new** method absent from the
buggy version, select an existing constructor or public factory that creates
the subject, provided the witness can then perform the requested action and
observe its result. Do not claim that the constructor itself implements the
new behavior. If no supplied callable can establish such a witness, request
targeted class or caller context before abstaining.

Before finalizing, state the path in concrete terms: issue input → selected
callable → activated mechanism → observable result. Select only supplied
location IDs, preserve the host schema, and never use gold or hidden tests.
