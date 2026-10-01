"""Curated finite-observation evaluation. Not a proof of natural-language equivalence."""

from __future__ import annotations
import json
from pathlib import Path
from contractfix.contracts.expressions import Expression
from contractfix.agents.schema.contracts import ExecutableClause
from contractfix.llm.structured import generate_structured
from contractfix.llm.prompt_renderer import render_prompt_from_path
from contractfix.utils.artifacts import RunArtifacts, digest


def judge_clause(case, proposal):
    """Valid/invalid rows are in-domain behaviors; guard exclusion never counts as rejection."""
    score = {
        "well_formed": False,
        "safe_on_cases": False,
        "valid_accepted": 0,
        "invalid_rejected": 0,
        "valid_total": len(case["valid"]),
        "invalid_total": len(case["invalid"]),
        "excluded_required_inputs": 0,
        "checker_errors": 0,
    }
    try:
        p = ExecutableClause.model_validate(proposal)
        guard = Expression(p.when, set(case["parameters"]), entry=True)
        ensure = Expression(p.ensure, set(case["parameters"]))
        score["well_formed"] = True
    except Exception as exc:
        score.update(passed=False, error_type=type(exc).__name__)
        return score
    for label in ("valid", "invalid"):
        for obs in case[label]:
            try:
                entry = {name: obs["old"][name] for name in case["parameters"]}
                applies = guard.evaluate({**entry, "old": obs["old"]})
                if not applies:
                    score["excluded_required_inputs"] += 1
                    continue
                passed = ensure.evaluate(obs)
                if label == "valid" and passed:
                    score["valid_accepted"] += 1
                if label == "invalid" and not passed:
                    score["invalid_rejected"] += 1
            except Exception:
                score["checker_errors"] += 1
    score["safe_on_cases"] = score["checker_errors"] == 0
    score["passed"] = (
        score["safe_on_cases"]
        and not score["excluded_required_inputs"]
        and score["valid_accepted"] == score["valid_total"]
        and score["invalid_rejected"] == score["invalid_total"]
    )
    return score


def evaluate_contracts(settings, cases_path, output_dir):
    cases = [json.loads(x) for x in Path(cases_path).read_text().splitlines() if x.strip()]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("nonempty unique cases required")
    a = RunArtifacts(
        output_dir,
        {
            "purpose": "contract_development_microevaluation",
            "settings": settings.public_dict(),
            "cases_sha256": digest(cases),
            "denominator": len(cases),
        },
    )
    rows = []
    for case in cases:
        # Hidden valid/invalid observations and reference clauses NEVER enter the model prompt.
        public = {k: case[k] for k in ("id", "requirement", "parameters")}
        prompt = render_prompt_from_path(
            Path(__file__).parents[1] / "agents/prompts/microevaluation.j2", packet=public
        )
        a.save(case["id"] + "-prompt.json", public)
        try:
            parsed, metrics = generate_structured(prompt, ExecutableClause, settings, events=a.events)
            row = {
                "id": case["id"],
                "proposal": parsed.model_dump(),
                "metrics": metrics,
                "quality": judge_clause(case, parsed.model_dump()),
            }
        except Exception as exc:
            row = {"id": case["id"], "quality": {"passed": False}, "error_type": type(exc).__name__}
        rows.append(row)
        a.save(case["id"] + ".json", row)
    result = {
        "denominator": len(cases),
        "passed": sum(x["quality"]["passed"] for x in rows),
        "pass_fraction": sum(x["quality"]["passed"] for x in rows) / len(cases),
        "cases": rows,
        "assurance": "curated finite-observation fidelity only; not universal correctness or a SWE-bench score",
    }
    a.save("report.json", result)
    return result
