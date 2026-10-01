# Build the executable witness

Use the frozen NLC and checked-out repository version. Construct input that
satisfies both the issue trigger and ordinary API preconditions. Type and shape
are insufficient when an API expects categorical values, registered objects,
or a parser dialect. Prefer a version-matched repository example or fixture to
guessed data. Fakes must preserve the interface and control flow on this path.
When a test helper supplies its own configuration, output directory, or object
name, verify those values before asserting a path. Prefer a public caller-visible
result over an intermediate representation or incidental fixture filename.
For a defect involving a launched script, process entrypoint, or loaded main
module, create the representative entrypoint and invoke the public launch path.
Check that the watched or imported file is that entrypoint, not the contract
runner's own `__file__` or `__main__` module. A fixture that merely reaches an
internal function is insufficient when its process state differs from the issue.
When the reported defect is in the completed display or serialized output,
invoke the normal public path and assert that completed result. A selected
internal operation may return an empty or sentinel value that a later layer
legitimately converts. Do not assert that the internal return is nonempty or
has the final form unless the issue or version-matched API requires it there.
Record that the selected operation was reached, then observe the downstream
result with the frozen NLC's supported level of precision.
For a framework directive or plugin with required environment state, prefer a
version-matched repository fixture or public end-to-end entrypoint over a
hand-constructed object. Read the previous candidate's first failing event
before retrying. Treat the exception type plus failing operation as the setup
failure signature: a retry that preserves both has not repaired the witness.
For an import error, inspect the checked-out package exports or issue import
and change the import route. For a missing internal-state attribute, find the
public caller that initializes it rather than calling the internal method
again. For a missing ORM reverse relation, register the models through the
version-matched app setup or use a repository fixture before building the
query. Change the failing setup step, then retest the operation and assertion.
Verify fixture preconditions before the issue assertion: register the subject
or use a real configured helper, then check that the expected collection or
object exists. A missing subject must not masquerade as the reported bug.

Trace the branch from the declared operation to the reported mechanism. Set
every controlling **call argument** and relevant configuration value; merely
configuring a database or service does not enable a check that the call skips
by default. For parser issues, instantiate the parser selected by the input
syntax, even when the observed method is inherited from a shared base class.
For a newly requested method, invoke an existing constructor, then test the
observable behavior on the resulting object.
For Python operator protocols, invoke the public operator syntax rather than
calling a dunder directly: a dunder may correctly return `NotImplemented`
while Python raises the user-visible error after dispatch. Likewise, test an
error hint through the public collector or wrapper that creates it, not only
the lower-level exception source.
An issue traceback can name an outer wrapper or later consumer rather than
the selected contracted operation. Preserve the grounded normal postcondition:
invoke the real path and assert the required result or normal completion at
the layer that observes it. Do not fabricate the quoted exception inside a
fixture or demand that a producer raise an exception created by its caller.
If the NLC forbids only one exception, do not expand it to every exception;
if it requires normal completion, capture unexpected operation errors and
make that requirement observable through a runtime assertion.

Assert only the frozen outcome. Keep setup outside broad exception handling.
Preserve the issue trigger's condition and quantifier: if a rule applies only
after a duplicate, collision, or other threshold, assert it for triggered
items and preserve ordinary items separately. Do not turn a conditional
obligation into a requirement for every item in the collection.
When the behavior accepts several input forms sharing one rule, exercise
distinct branch classes supported by the issue and checked-out code, such as
short and long options or stable and prerelease versions. Keep every asserted
outcome tied to the frozen NLC; do not invent a new requirement.
For symbolic or structured outputs, assert the requested behavior over its
stated input domain. Do not require exact expression or representation equality
when the issue gives an illustrative form and also identifies a boundary case.
For a formatting or representation defect, a type or nonempty-string check
only proves termination. Assert the output feature the issue requests, using
an exact form only when issue evidence or a version-matched repository
convention supports it. Check the buggy output before choosing that assertion;
it must fail on the reported defect without demanding unsupported formatting.
Scope a forbidden text feature to the structural position where it is wrong.
Before using a broad substring ban, construct an issue-consistent output in
which the same characters appear in another legitimate position. If the ban
rejects that output, assert the relevant structure instead. For a semantic
error demonstrated by later evaluation, compare the resulting behavior on the
issue's input rather than forbidding one intermediate expression spelling.
Decide whether the issue specifies the entire serialized result or a feature
inside a surrounding representation. If existing printer behavior adds a
wrapper, prefix, or display container, an illustrative inner form does not
justify requiring that form at character zero. Assert the requested inner
feature and any independently supported outer convention separately. A simple
substring check may also be too weak if unrelated text could contain the
feature: retain the relevant structure and input components in the assertion.
Before writing that assertion, inspect any supplied `existing_test` evidence.
List the input components that the operation must retain, including nested
content, powers, signs, grouping, and ordering when they matter to the issue.
For each required component, point to the assertion that observes it; a
`str`/nonempty check has no such mapping. Use a base-test representation only
for the same input class and behavior; when the new combination has no exact
precedent, assert the supported structural features without inventing exact
spacing or punctuation. For example, when formatting a powered expression
already containing an inner power, check that both powers survive and are
distinguishable in the result. Do not use an expected string from gold or a
hidden benchmark test.
Before submission, ask whether a degenerate implementation that merely avoids
the exception, returns a constant, or drops one input component would pass the
EC. If so, strengthen the caller-visible assertion using issue or base-test
evidence. Keep separate issue examples and branch classes in the witness;
one branch failing before the next executes does not demonstrate coverage of
the later branch.
If normal completion is required, check the output as well as absence of the
reported failure. Inspect the buggy execution receipt: reject a witness when
its assertion fires because of a different exception, invalid fixture, wrong
route, or a path that never entered the reported mechanism. A buggy-side failure
does not validate an assumed output location; confirm that the witness would
observe the requested behavior if the implementation were correct. Return the host's
exact source-lines schema. Gold is used only after the EC freezes.
