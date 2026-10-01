"""JavaScript development revision: ground witness imports in its child folder."""

from __future__ import annotations

from dataclasses import replace

from .javascript_v1 import PACK as PREVIOUS


PACK = replace(
    PREVIOUS,
    version="native-javascript/2",
    ec=(
        "Write a self-contained Node CommonJS witness saved at "
        ".contractfix_witness/contractfix_witness.cjs under the repository root. "
        "A root module index.js must therefore be loaded with require('../index'), "
        "and lib/foo.js with require('../lib/foo'); require('./index') points to "
        "the witness folder and will fail. Import and invoke the real repository "
        "API; never copy its implementation. Await Promise results in an async "
        "main when needed. A local 127.0.0.1 server is available, while external "
        "network access is disabled. Print CONTRACTFIX_OPERATION_REACHED after "
        "invoking the operation, then exactly one of CONTRACTFIX_ASSERTION_PASS "
        "or CONTRACTFIX_ASSERTION_FAIL according to the accepted postcondition. "
        "For expected throws or rejections, catch and assert them."
    ),
)
