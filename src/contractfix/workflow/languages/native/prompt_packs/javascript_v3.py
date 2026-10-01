"""JavaScript EC examples must reproduce the issue on the buggy base."""

from __future__ import annotations

from dataclasses import replace

from .javascript_v2 import PACK as PREVIOUS


PACK = replace(
    PREVIOUS,
    version="native-javascript/3",
    ec=(
        PREVIOUS.ec + " Start with the issue's concrete input and observable "
        "failure. The witness must print CONTRACTFIX_ASSERTION_FAIL on the buggy base "
        "for a real postcondition violation; an example that already passes "
        "the buggy base is uninformative. If previous_ec_feedback reports "
        "BUGGY_BASE_SATISFIED, change the input or API path to reproduce the "
        "same issue obligation; do not invert the pass/fail assertion. "
        "For TypeScript packages, import their compiled package entry after "
        "the repository build rather than requiring raw .ts source."
    ),
)
