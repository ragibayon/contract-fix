# Review NLC-to-EC alignment

Map each frozen precondition to setup and every requested outcome to an
assertion or observed exception. Verify that the input is valid for the
version-matched API beyond type and shape. Follow the actual parser,
dispatcher, or check branch: inspect call arguments and configuration that
control it. An entered wrapper or configured service is not enough when the
reported inner mechanism is skipped.
Check that the fixture contains the subject it later asserts about and that
fakes supply the interface used on both buggy and corrected paths. For a
reported operator error or diagnostic hint, verify that the witness invokes
the public dispatch layer that produces it; a direct dunder or inner call can
have different, valid semantics.
For an entrypoint, autoreloader, or process-state defect, identify the exact
file/module the witness presents as the subject. Reject use of the contract
runner's own `__file__` or `__main__` when the issue concerns a separately
launched program. Reaching the selected function does not establish that the
fixture reproduced the issue's state.
If the issue quotes an exception from an outer wrapper or later consumer,
check the selected operation and caller-visible assertion separately. A
different exception name at the selected operation does not by itself make
the witness wrong when the NLC requires normal completion. Verify that the
assertion tests the frozen outcome and the observed failure comes from the
issue path, not missing setup, an incompatible fake, or an unrelated call.
Do not accept a broad no-exception assertion when the NLC forbids only one
specific exception.
Compare each assertion's scope to the NLC: preserve conditions such as
"only after a duplicate" and do not require the outcome for ordinary items.
Flag an assertion that strengthens a conditional rule into a blanket one.
For an output-format NLC, reject a witness that checks only successful return,
type, or nonemptiness when the buggy output could satisfy those checks. Identify
the issue-backed output feature or repository convention that the assertion
actually verifies. Keep separate issue examples distinguishable in the review:
one branch failing before later branches run does not establish the later
branches' behavior.
Also check the opposite error: a witness can reject a correct implementation
by demanding the final output form from an internal call. Trace the selected
operation's return through any formatter, field, serializer, or wrapper. If a
downstream layer may validly normalize an empty or sentinel value, reject an
assertion that forbids that intermediate value without issue or API support.
The witness should reach the selected operation and test the completed
caller-visible result.
Make the coverage audit explicit: for each NLC outcome, name its input trigger,
the expected caller-visible feature, and the source line in the EC that would
fail if that feature were absent. Compare with any supplied base `existing_test`
for the same behavior. A representation witness that only checks no exception,
`isinstance(result, str)`, or `bool(result)` has no content coverage. Reject it
when the issue requires preserved content; specify the missing component rather
than asking vaguely for a stronger assertion. Do not demand an exact string
that neither the issue nor the checked-out repository establishes.
For a printer or serializer, classify the asserted string as a full result,
an inner semantic form, or a repository-defined wrapper. Flag a `startswith`,
exact-equality, or fixed-position assertion when the evidence requires only
the inner form and the repository may legitimately wrap it. Also reject a bare
substring assertion if it could pass after losing the required input structure.
Name the specific issue evidence or base convention that supports the chosen
assertion boundary.
For a negative string assertion, check whether the banned characters can occur
legitimately elsewhere in a correct output. Require the witness to isolate the
wrong structural role, rather than banning a token globally. For a semantic
defect, compare the assertion's call and observed outcome with the issue's
concrete example; flag a rule about an internal representation when the issue
only establishes a later behavioral mismatch.

Read the buggy execution trace to identify the first failure's cause and
origin. Reject an assertion caused by unrelated exceptions, invalid domain
data, missing fake-interface members, or wrong parser routes, even if the
declared operation was reached. If the witness catches an exception and
asserts none occurred, inspect the caught exception before treating the
assertion as issue discrimination. Distinguish missing NLC meaning from extra
assertions; do not infer success from a merely completed harness.

Return the host's exact structured review fields. Buggy-side failure is only
pre-gold evidence; independent replay of the byte-identical frozen EC decides
gold acceptance later.
