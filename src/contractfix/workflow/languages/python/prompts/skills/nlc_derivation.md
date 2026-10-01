# Natural-language obligation derivation

Before writing the obligation, make a private coverage table with one row for
each distinct caller-visible request in the issue: the triggering input or
state, the expected result or failure behavior, and the issue evidence span.
Read the entire issue, including later paragraphs, examples, and requests
introduced by words such as "also". Compare the proposed precondition and
postcondition with every row. If one operation handles several input classes,
describe the alternatives and their respective outcomes in the same
obligation; do not select the easiest example and silently drop another
requested behavior. If the selected operation cannot observe a required row,
identify the localization gap rather than presenting a partial contract as
complete. Do not add unrelated wishlist behavior to fill the table.

Treat the issue's request and examples as direct evidence of desired behavior.
Repository docs and source establish invocation mechanics, but unrelated docs
are not required to restate a clear issue request. Include the reported trigger
and every requested observable outcome: return value, displayed content, side
effect, or exception behavior. State exact formatting only when the issue or
relevant documentation requires it.
When the issue demonstrates a wrong result after several operations, state the
postcondition at the operation where that result is observed. Do not turn an
intermediate symbolic form, helper return, or suggested repair technique into
a prohibition over every input. Preserve the example's assumptions and the
specific observation that distinguishes the wrong and desired behavior.
When the issue shows a broken final display or serialized result, state the
required final behavior at the caller-visible boundary. Trace any intermediate
return through downstream formatting before requiring that intermediate to
change. If a later layer can legitimately turn an empty or sentinel value into
the correct final output, do not add a clause that forbids that value at the
earlier operation. Name the selected operation's connection to the final
observation without prescribing an unsupported repair location.
When an issue states an observable goal but leaves implementation choices open,
state that goal at the supported level instead of abstaining. Do not demand an
exact exception subtype, metadata key, or output spacing unless the evidence
specifies it. Abstain only when no testable outcome can be stated from the issue
and checked-out API.
For a conflict between multiple items, specify the observable absence of the
conflict and the trigger that creates it. Do not require every item to receive
the same repair attribute when changing one side would satisfy the request.
If the issue explicitly contrasts cases, include the distinction that makes a
partial fix insufficient, such as order-independent comparison versus equality
for one fixed order. Do not reduce the contract to an easier subcase merely
because that subcase admits a short witness.

Connect each clause to a witness that reaches the selected existing operation.
For a method the issue asks to add, an existing constructor can create the
subject and the witness can then observe the new behavior on that object.
State that later caller-visible observation explicitly; do not claim the
constructor already implements the missing method. Do not abstain solely
because the method is absent from buggy code. If no selected operation can
establish the subject or observation, identify the localization gap and use
the host's abstention mechanism.

Do not turn a patch technique or hidden-test guess into an obligation.
Treat an example expression as behavior evidence, not necessarily an exact
representation requirement. If the issue calls out a boundary such as zero,
state the valid-domain result and boundary behavior separately.
When `existing_test` evidence is supplied, compare its input class and output
convention with the issue's new trigger. Use it to identify preserved behavior
such as nested content, grouping, sign, or power, and cite its evidence ID for
that inference. A base test can itself encode the bug; issue requirements take
priority. If the new combination has no exact precedent, state the structural
outcome supported by the issue and tests without claiming exact formatting.
Preserve the host's required structured fields and evidence IDs.
