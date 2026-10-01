"""Run only the supplied, trusted EC example. No LLM or benchmark input is used."""

import json
import sys
from pathlib import Path
from contractfix.contracts.core import seal, admit_pair
from contractfix.contracts.runner import run
from contractfix.utils.artifacts import RunArtifacts


def smoke(output_dir, examples=None):
    ex = Path(examples) if examples else Path(__file__).resolve().parents[3] / "examples/ec"
    if not (ex / "clauses.json").is_file():
        raise ValueError("source checkout examples are required; use --examples")
    a = RunArtifacts(output_dir, {"purpose": "trusted_synthetic_ec_smoke", "example": str(ex.resolve())})
    bundle = seal(json.loads((ex / "clauses.json").read_text()))
    a.save("bundle.json", bundle)
    reports = {}
    for name, patch in (("base", None), ("good", ex / "good.patch"), ("bad", ex / "bad.patch")):
        reports[name] = run(ex / "toy_repo", bundle, [sys.executable, "witness.py"], patch=patch)
        a.save(name + ".json", reports[name])
        a.events.emit("ec_execution", case=name, summary=reports[name]["summary"])
    gates = {k: admit_pair(bundle, reports["base"], reports[k]) for k in ("good", "bad")}
    passed = (
        reports["base"]["summary"]["disposition"] == "REJECT"
        and gates["good"]["decision"] == "ADMIT_OBSERVED"
        and gates["bad"]["decision"] == "REJECT"
    )
    result = {
        "passed": passed,
        "gates": gates,
        "assurance": "synthetic runtime smoke, not live model or SWE-bench validation",
    }
    a.save("report.json", result)
    return result
