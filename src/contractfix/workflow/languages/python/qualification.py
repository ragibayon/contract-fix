"""Current Python contract-qualification workflow."""

from __future__ import annotations

from pathlib import Path
from collections import Counter
import json
import re
from time import perf_counter

from contractfix.contracts.core import digest
from contractfix.contracts.execution import Executor
from contractfix.llm.structured import OutputBudgetError, ReasoningOnlyOutputBudgetError
from contractfix.utils.artifacts import RunArtifacts
from contractfix.utils.logger import logger
from ...accounting import model_accounting
from ...documents import retrieve_documents, retrieve_existing_tests
from ...evidence import canonical_spans, semantic_location, semantic_obligation
from .equivalence import compare_assertions_with_z3
from ...errors import EvidenceGroundingError, LocalizationExhausted, WorkflowError
from .models import (
    EXECUTABLE_CONTRACT_BATCH_LIMIT,
    ExecutableContractCandidates,
    ExecutableContractConformanceBatch,
    FinalLocalization,
    Localization,
    RepairObligationReview,
    RepairObligations,
    CanonicalRepairObligations,
    Task,
    WorkflowSettings,
)
from .executable_contracts import (
    check_base_import,
    equivalent_contract_sources,
    evaluate_executable_contract,
    intended_violation,
    validate_executable_contract,
)
from ...reporting import write_failure_report, write_report
from .repository import (
    RepositoryIndex,
    evidence_packet,
    repository_execution_examples,
    snapshot,
)
from ...stages import Stages


NON_RECOVERABLE_EC_OWNERS = frozenset({
    "RUNTIME_MONITOR", "EXECUTOR", "NLC_OR_WITNESS", "EXECUTABLE_CONTRACT_ENVIRONMENT"
})


def _attach(stages: Stages, artifacts: RunArtifacts) -> None:
    if hasattr(stages, "artifacts"):
        stages.artifacts = artifacts
    prompts = getattr(stages, "prompts", None)
    if prompts is not None:
        artifacts.save("prompt_assets.json", prompts.assets)


def _candidate_label(index: int) -> str:
    """Return stable spreadsheet-style labels: A..Z, AA..AZ, BA..."""
    if index < 0:
        raise ValueError("candidate index must be nonnegative")
    value = index + 1
    label = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def _candidate_batch_offer(
    initial_remaining: int, additional_remaining: int
) -> tuple[int, bool]:
    """Return a provider-safe batch ceiling and whether it uses initial budget."""
    if initial_remaining > 0:
        return min(EXECUTABLE_CONTRACT_BATCH_LIMIT, initial_remaining), True
    return min(1, additional_remaining), False


def _conformance_protocol() -> dict:
    """Describe trusted semantics that must not be inferred from Python exit behavior."""
    return {
        "semantic_sources": [
            "contract witness and contracted-operation invocation establish the precondition",
            "explicit runtime contract assertions establish postconditions",
            "exception branches establish semantics only when they feed a runtime contract assertion",
        ],
        "operational_only": [
            "uncaught unrelated exceptions are evaluation errors or wrong failure reasons",
            "import, setup, timeout, instrumentation, and runtime-monitor failures are evaluation errors",
            "normal harness completion alone is not an implicit normal postcondition",
        ],
        "clause_rule": (
            "Every alleged added meaning must be a substantive clause present in the "
            "reviewer's own back-translation."
        ),
    }


def _index_conformance_reviews(reviews, expected_ids: set[str]) -> dict:
    """Require each requested receipt while safely ignoring surplus model IDs."""
    counts = Counter(review.candidate_id for review in reviews)
    missing = sorted(candidate_id for candidate_id in expected_ids if counts[candidate_id] == 0)
    duplicated = sorted(candidate_id for candidate_id in expected_ids if counts[candidate_id] > 1)
    if missing or duplicated:
        raise WorkflowError(
            "conformance reviewer must return exactly one review for each supplied candidate; "
            f"missing={missing!r}, duplicated={duplicated!r}"
        )
    surplus = sorted(candidate_id for candidate_id in counts if candidate_id not in expected_ids)
    if surplus:
        logger.warning(
            "[conformance] ignored surplus review IDs not present in the request: "
            + ", ".join(surplus)
        )
    return {
        review.candidate_id: review
        for review in reviews
        if review.candidate_id in expected_ids
    }


def _nlc_review_units(obligation: dict) -> list[dict[str, str]]:
    """Expose the current NLC components without introducing a new authoring schema."""
    units = [
        {"clause_id": f"PRE{position}", "clause_text": clause["condition"]}
        for position, clause in enumerate(obligation.get("precondition_clauses") or [], start=1)
    ]
    if not units:
        units.append({"clause_id": "PRE1", "clause_text": obligation["precondition"]})
    if obligation.get("normal_postcondition"):
        units.append(
            {
                "clause_id": "POST_NORMAL",
                "clause_text": obligation["normal_postcondition"],
            }
        )
    if obligation.get("exceptional_postcondition"):
        units.append(
            {
                "clause_id": "POST_EXCEPTION",
                "clause_text": obligation["exceptional_postcondition"],
            }
        )
    return units


def _nlc_review_receipt(
    review: RepairObligationReview,
    units: list[dict[str, str]],
    allowed_evidence_ids: set[str],
    *,
    round_number: int,
) -> dict:
    """Validate complete clause coverage and derive categorical review metrics."""
    expected = {item["clause_id"]: item["clause_text"] for item in units}
    indexed = {item.clause_id: item for item in review.reviews}
    if len(indexed) != len(review.reviews) or set(indexed) != set(expected):
        raise WorkflowError(
            "NLC reviewer must return exactly one review for each supplied clause"
        )
    for clause_id, item in indexed.items():
        if item.clause_text != expected[clause_id]:
            raise WorkflowError("NLC reviewer changed the supplied clause text")
        unknown = set(item.supporting_evidence_ids) - allowed_evidence_ids
        if unknown:
            raise WorkflowError(
                "NLC reviewer cited unknown evidence identifiers: "
                + ", ".join(sorted(unknown))
            )
    counts = {
        verdict: sum(item.verdict == verdict for item in review.reviews)
        for verdict in ("DIRECT", "DERIVED_VALID", "UNSUPPORTED", "CONTRADICTED")
    }
    total = len(review.reviews)
    return {
        "round": round_number,
        "accepted": review.accepted(),
        "justification_coverage": (
            counts["DIRECT"] + counts["DERIVED_VALID"]
        ) / total,
        "counts": counts,
        "missing_required_clauses": review.missing_required_clauses,
        "reviews": [item.model_dump() for item in review.reviews],
    }


def _localize(
    index: RepositoryIndex,
    task: Task,
    settings: WorkflowSettings,
    stages: Stages,
    override: dict | None,
    artifacts: RunArtifacts | None = None,
) -> Localization:
    if override is None:
        candidates = index.search(task.problem_statement, settings.retrieval_limit)
        initial_chars = 32000 if settings.context_strategy == "causal_slice" else settings.excerpt_chars
        packets = {loc.id: loc.packet(initial_chars) for loc in candidates}
        context_feedback = None
        round_number = 0
        invalid_selection_attempts = 0
        while True:
            round_number += 1
            packet = {
                "task": task.model_packet(),
                "locations": list(packets.values()),
                "localization_round": round_number,
                "max_localization_rounds": settings.max_localization_rounds,
                "final_localization_round": (settings.max_localization_rounds is not None
                                             and round_number >= settings.max_localization_rounds),
                "include_rationales": settings.include_rationales,
            }
            if context_feedback:
                packet["context_feedback"] = context_feedback
            final_round = packet["final_localization_round"]
            if final_round:
                packet["selection_only"] = True
                packet["context_feedback"] = (
                    "No further context can be retrieved. Select repair locations and a "
                    "function or method as the contracted operation from supplied IDs "
                    "when evidence supports "
                    "both, or explicitly abstain."
                )
            try:
                result = stages.generate(
                    "localize", FinalLocalization if final_round else Localization, packet
                )
            except OutputBudgetError as exc:
                logger.warning(
                    "[localize] recoverable output-limit failure; retrying once "
                    f"at the configured reasoning policy and completion ceiling | {exc}"
                )
                if artifacts:
                    artifacts.events.emit(
                        "localization_output_retry",
                        stage="localize",
                        round=round_number,
                        retry=1,
                        maximum=1,
                        error_type=type(exc).__name__,
                    )
                result = stages.generate(
                    "localize", FinalLocalization if final_round else Localization, packet
                )
            if final_round:
                if result.abstain:
                    if artifacts:
                        artifacts.save(f"localization/round_{round_number:02d}.json", {
                            "round": round_number,
                            "request_status": "ABSTAINED_FINAL_ROUND",
                            "available_location_ids": sorted(packets),
                            "rationale": result.rationale,
                        })
                    break
                result = Localization(
                    repair_locations=result.repair_locations,
                    contracted_operation_locations=result.contracted_operation_locations,
                    rationale=result.rationale,
                    needs_more_context=False,
                    context_requests=[],
                )
            available = set(packets)
            if not result.needs_more_context:
                invalid_ids = sorted(
                    set(result.repair_locations + result.contracted_operation_locations)
                    - available
                )
                invalid_operations = sorted(
                    item for item in result.contracted_operation_locations
                    if item in available and index.locations[item].kind != "function"
                )
                if invalid_ids or invalid_operations:
                    invalid_selection_attempts += 1
                    context_feedback = " ".join(filter(None, (
                        f"Selected location IDs {invalid_ids!r} were not supplied."
                        if invalid_ids else "",
                        f"Contracted-operation IDs {invalid_operations!r} cannot be instrumented. "
                        "Choose a function or method, not a class or assignment."
                        if invalid_operations else "",
                        "Choose repair IDs from supplied locations and contracted-operation "
                        "IDs from supplied functions or methods.",
                    )))
                    if artifacts:
                        artifacts.save(f"localization/round_{round_number:02d}.json", {
                            "round": round_number,
                            "request_status": "REJECTED_INVALID_SELECTION",
                            "invalid_location_ids": invalid_ids,
                            "invalid_contracted_operation_ids": invalid_operations,
                            "available_location_ids": sorted(available),
                        })
                        artifacts.events.emit(
                            "localization_selection_rejected",
                            stage="localize",
                            round=round_number,
                            invalid_location_ids=invalid_ids,
                            invalid_contracted_operation_ids=invalid_operations,
                        )
                    logger.warning(
                        f"[localize] rejected invalid location IDs {invalid_ids!r} "
                        f"and non-callable operation IDs {invalid_operations!r}; "
                        "requesting corrected selection"
                    )
                    if invalid_selection_attempts >= 2 or (
                        settings.max_localization_rounds is not None
                        and round_number >= settings.max_localization_rounds
                    ):
                        break
                    continue
                if artifacts:
                    artifacts.save(
                        "localization/selection.json",
                        {
                            "rounds_used": round_number,
                            "available_location_ids": sorted(available),
                            "selection": result.model_dump(),
                            "assurance": "llm_selected_from_host_supplied_repository_context",
                        },
                    )
                    artifacts.events.emit(
                        "localization_finalized",
                        stage="localize",
                        round=round_number,
                        repair=result.repair_locations,
                        contracted_operation=result.contracted_operation_locations,
                    )
                return result
            if settings.context_strategy == "compact":
                raise LocalizationExhausted(
                    "compact localization does not retrieve additional model-requested context",
                    rounds=round_number,
                )
            if (
                settings.max_localization_rounds is not None
                and round_number >= settings.max_localization_rounds
            ):
                if artifacts:
                    artifacts.save(f"localization/round_{round_number:02d}.json", {
                        "round": round_number,
                        "requests": [item.model_dump() for item in result.context_requests],
                        "request_status": "NOT_EXECUTED_ROUND_BUDGET",
                        "available_location_ids": sorted(available),
                    })
                break
            logger.table(
                f"Localization context requests · round {round_number}",
                ["Kind", "Target", "Why"],
                [[request.kind, request.target, request.rationale] for request in result.context_requests],
            )
            try:
                additions, expanded = index.requested_context(
                    result.context_requests,
                    available,
                    settings.localization_context_additions,
                )
            except ValueError as exc:
                context_feedback = str(exc)
                if artifacts:
                    artifacts.save(
                        f"localization/round_{round_number:02d}.json",
                        {
                            "round": round_number,
                            "requests": [item.model_dump() for item in result.context_requests],
                            "request_status": "REJECTED",
                            "detail": str(exc),
                            "available_location_ids": sorted(available),
                        },
                    )
                logger.warning(
                    f"[localize] context round {round_number}/"
                    f"{settings.max_localization_rounds or 'unbounded'} "
                    f"request rejected | {exc}"
                )
                if (
                    settings.max_localization_rounds is not None
                    and round_number >= settings.max_localization_rounds
                ):
                    break
                continue
            for ident in expanded:
                previous_chars = len(packets[ident]["source"])
                expanded_chars = max(
                    previous_chars, min(64000, max(initial_chars * 2, previous_chars * 2))
                )
                packets[ident] = index.locations[ident].packet(expanded_chars)
            for location in additions:
                packets[location.id] = location.packet(initial_chars)
            context_feedback = (
                f"Host supplied {len(additions)} additional location(s) and expanded "
                f"{len(expanded)} excerpt(s)."
            )
            if artifacts:
                receipt = {
                    "round": round_number,
                    "requests": [item.model_dump() for item in result.context_requests],
                    "request_status": "FULFILLED",
                    "expanded_location_ids": sorted(expanded),
                    "added_locations": [
                        {"id": item.id, "file": item.file, "symbol": item.symbol} for item in additions
                    ],
                    "available_location_ids": sorted(packets),
                }
                artifacts.save(f"localization/round_{round_number:02d}.json", receipt)
                artifacts.events.emit(
                    "localization_context_fulfilled",
                    stage="localize",
                    round=round_number,
                    expanded=sorted(expanded),
                    added=[item.id for item in additions],
                    total_locations=len(packets),
                )
            if additions:
                logger.table(
                    f"Localization context returned · round {round_number}",
                    ["ID", "File", "Symbol"],
                    [[item.id, item.file, item.symbol] for item in additions],
                )
            logger.info(
                f"[localize] context round {round_number}/"
                f"{settings.max_localization_rounds or 'unbounded'} "
                f"expanded={len(expanded)} added={len(additions)} total={len(packets)}"
            )
            if (
                settings.max_localization_rounds is not None
                and round_number >= settings.max_localization_rounds
            ):
                break
        raise LocalizationExhausted(
            "localizer did not select repair and contracted-operation locations within "
            "the context-round budget",
            rounds=round_number,
        )
    else:

        def ids(targets: list[dict]) -> list[str]:
            selected = []
            for target in targets:
                matches = [loc.id for loc in index.locations.values() if loc.target() == target]
                if len(matches) != 1:
                    raise WorkflowError("external localization target is missing or ambiguous")
                selected.append(matches[0])
            return selected

        operation_targets = override.get(
            "contracted_operation_targets", override.get("observation_targets", [])
        )
        result = Localization(
            repair_locations=ids(override["repair_targets"]),
            contracted_operation_locations=ids(operation_targets),
            rationale=override.get("rationale", "external localizer"),
            needs_more_context=False,
            context_requests=[],
        )
        available = set(index.locations)
    if set(result.repair_locations + result.contracted_operation_locations) - available:
        raise WorkflowError("localizer invented an identifier outside its supplied candidates")
    if any(
        index.locations[item].kind != "function"
        for item in result.contracted_operation_locations
    ):
        raise WorkflowError("contracted operation must be an instrumentable function or method")
    return result


def discover(
    task: Task,
    repo: Path,
    output: Path,
    executor: Executor,
    stages: Stages,
    settings: WorkflowSettings,
    localization_override: dict | None = None,
) -> dict:
    """No gold/test_patch/evaluator fields are accepted or read by this function."""
    started = perf_counter()
    if output.resolve().is_relative_to(repo.resolve()):
        raise ValueError("run output must be outside the target repository")
    artifacts = RunArtifacts(
        output,
        {
            "purpose": "contract_discovery",
            "task_id": task.instance_id,
            "task": task.model_packet(),
            "generator": stages.identity,
            "workflow": settings.model_dump(),
        },
    )
    _attach(stages, artifacts)
    logger.attach(artifacts.root)
    logger.banner(
        "ContractFix · Contract qualification",
        f"**Task:** `{task.instance_id}`  \n**Artifacts:** `{artifacts.root}`",
    )
    try:
        status = _discover_python(task, repo, artifacts, executor, stages, settings, localization_override)
        from .guidance import seal_guidance
        guidance = seal_guidance(artifacts.root, status)
        if guidance:
            status["guidance_sha256"] = guidance["sha256"]
        status["total_elapsed_seconds"] = perf_counter() - started
        status["model_accounting"] = model_accounting(stages)
        artifacts.save("summary.json", status)
        artifacts.save("status.json", status)
        logger.success(f"[discovery] {task.instance_id} finished | status={status['status']}")
        return status
    except Exception as exc:
        failure = {
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "detail": str(exc)[:1200],
            "total_elapsed_seconds": perf_counter() - started,
            "model_accounting": model_accounting(stages),
        }
        artifacts.save("status.json", failure)
        write_failure_report(artifacts.root, task.instance_id, failure)
        logger.error(f"[discovery] {task.instance_id} failed | {type(exc).__name__}: {exc}")
        # Only attributable proposal/provider failures may degrade to APR. A host
        # programming error or corrupted base must remain an error, not a fallback.
        recoverable = isinstance(exc, EvidenceGroundingError) or type(exc).__module__.startswith(
            ("contractfix.llm", "pydantic", "httpx", "openai", "ollama")
        )
        if settings.repair_enabled and recoverable:
            from .guidance import seal_guidance
            guidance = seal_guidance(artifacts.root, failure)
            if guidance:
                failure["guidance_sha256"] = guidance["sha256"]
                failure["apr_fallback_available"] = True
                artifacts.save("status.json", failure)
                return failure
        raise


def _ground_repair(
    obligation: dict,
    evidence: dict,
    allowed_contracted_operations: set[str],
    *,
    spans: dict,
) -> dict:
    """Resolve the current canonical-span obligation into immutable evidence bytes."""
    if obligation["contracted_operation_id"] not in allowed_contracted_operations:
        raise EvidenceGroundingError(
            "UNAPPROVED_CONTRACTED_OPERATION",
            "contracted_operation_id must exactly match one allowed contracted operation",
            contracted_operation_id=obligation["contracted_operation_id"],
        )
    grounded = []
    for position, support in enumerate(obligation["evidence"]):
        span_id = support.get("span_id")
        span = spans.get(span_id)
        if not span:
            raise EvidenceGroundingError(
                "UNKNOWN_EVIDENCE_SPAN",
                "span_id must exactly match one host-supplied canonical evidence span",
                evidence_id=str(span_id),
                evidence_position=str(position),
            )
        record = evidence.get(span["evidence_id"])
        if not record or record["kind"] not in {
            "issue", "docstring", "existing_test", "repository_document"
        }:
            raise EvidenceGroundingError(
                "NON_NORMATIVE_EVIDENCE",
                "buggy source alone cannot establish intended behavior",
                evidence_id=span["evidence_id"],
                evidence_position=str(position),
            )
        grounded.append(
            {
                "span_id": span_id,
                "id": span["evidence_id"],
                "supports": support["supports"],
                "quote": span["text"],
                "quote_sha256": span["sha256"],
            }
        )
    obligation["evidence"] = grounded
    clauses = obligation.get("precondition_clauses") or []
    if not clauses:
        raise EvidenceGroundingError(
            "MISSING_PRECONDITION_CLAUSE_MAP",
            "contract-first-v3 requires atomic precondition clauses with explicit span support",
        )
    derived_precondition = " AND ".join(item["condition"].strip() for item in clauses)

    def normalize(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip().casefold()

    if normalize(derived_precondition) != normalize(obligation["precondition"]):
        raise EvidenceGroundingError(
            "PRECONDITION_CLAUSE_TEXT_MISMATCH",
            "precondition must exactly equal precondition_clauses joined by ' AND '",
            derived_precondition=derived_precondition,
        )
    selected = {
        item["span_id"] for item in grounded if item["supports"] == "precondition"
    }
    used: set[str] = set()
    resolved = []
    for position, clause in enumerate(clauses):
        clause_spans = set(clause["span_ids"])
        unavailable = clause_spans - selected
        if unavailable:
            raise EvidenceGroundingError(
                "UNSELECTED_PRECONDITION_CLAUSE_SPAN",
                "each precondition clause may reference only selected precondition spans",
                precondition_clause_position=str(position),
                span_ids=sorted(unavailable),
            )
        used.update(clause_spans)
        resolved.append(
            {
                **clause,
                "support": [
                    {
                        "span_id": span_id,
                        "id": spans[span_id]["evidence_id"],
                        "quote": spans[span_id]["text"],
                        "quote_sha256": spans[span_id]["sha256"],
                    }
                    for span_id in clause["span_ids"]
                ],
            }
        )
    if selected - used:
        raise EvidenceGroundingError(
            "UNMAPPED_PRECONDITION_EVIDENCE",
            "every selected precondition span must support a precondition clause",
            span_ids=sorted(selected - used),
        )
    issue_spans = {
        item["span_id"]
        for item in grounded
        if item["supports"] == "precondition" and evidence[item["id"]]["kind"] == "issue"
    }
    if any(record.get("kind") == "issue" for record in evidence.values()) and not issue_spans:
        raise EvidenceGroundingError(
            "MISSING_ISSUE_PRECONDITION_EVIDENCE",
            "an issue-derived REPAIR contract must ground at least one precondition "
            "clause in canonical issue evidence; generic API documentation alone "
            "does not establish the reported contract witness",
        )
    obligation["precondition_clauses"] = resolved
    return obligation
def _derive_violation_criterion(obligation: dict, evidence: dict | None = None) -> dict:
    """Derive dynamic violation recognition in trusted host code."""
    import re

    normal_assertion_fallback = bool(obligation.get("normal_postcondition")) and not bool(
        obligation.get("exceptional_postcondition")
    )
    precondition = " ".join(
        item["quote"] for item in obligation["evidence"] if item["supports"] == "precondition"
    )
    required = " ".join(
        value
        for value in (
            obligation.get("normal_postcondition"),
            obligation.get("exceptional_postcondition"),
        )
        if value
    )
    issue_text = " ".join(
        item.get("text", "") for item in (evidence or {}).values() if item.get("kind") == "issue"
    )
    # Only unmistakably reported failures become exception-specific oracles.
    # Desired-exception requirements such as "must raise ValueError" remain
    # assertion oracles and are checked through bidirectional conformance.
    reported_context = re.search(
        r"(?:traceback|crash(?:es|ed)?|reported|unexpected|terminate[sd]?).{0,160}?"
        r"\b([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))\b"
        r"|\b([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))\s*:",
        precondition,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if reported_context:
        exception_type = next(group for group in reported_context.groups() if group)
        return {
            "kind": "reported_exception",
            "exception_type": exception_type,
            "normal_assertion_fallback": normal_assertion_fallback,
            "description": "The contracted operation raises the exception reported by precondition evidence.",
        }
    reported_issue = re.search(
        r"(?:traceback|crash(?:es|ed)?|reported|unexpected|terminate[sd]?).{0,160}?"
        r"\b([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))\b"
        r"|\b([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))\s*:",
        issue_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if (
        reported_issue
        and obligation.get("normal_postcondition")
        and not obligation.get("exceptional_postcondition")
    ):
        exception_type = next(group for group in reported_issue.groups() if group)
        return {
            "kind": "reported_exception",
            "exception_type": exception_type,
            "normal_assertion_fallback": normal_assertion_fallback,
            "description": (
                "The contracted operation raises the exception reported by host-owned "
                "issue evidence instead of satisfying the grounded normal postcondition."
            ),
        }
    # Canonical precondition spans describe the contract witness and need not repeat
    # the issue's traceback.  A direct target exception is nevertheless a trusted
    # violation when the grounded requirement explicitly forbids that exception
    # and the host-owned issue evidence reports the same exception type.
    forbidden = re.findall(
        r"(?:without|rather\s+than|instead\s+of|must\s+not|should\s+not|does\s+not|not)"
        r".{0,80}?\b([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))\b",
        required,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for exception_type in forbidden:
        if re.search(rf"\b{re.escape(exception_type)}\b", issue_text):
            return {
                "kind": "reported_exception",
                "exception_type": exception_type,
                "normal_assertion_fallback": normal_assertion_fallback,
                "description": (
                    "The contracted operation raises an exception that the "
                    "grounded requirement forbids and host-owned issue evidence reports."
                ),
            }
    return {
        "kind": "assertion",
        "exception_type": None,
        "description": "A semantically conformant runtime contract assertion is violated after the contracted operation is exercised.",
    }


def _candidate_failure(execution: dict, violation_criterion: dict) -> list[str]:
    failures = []
    intended = intended_violation(execution, violation_criterion)
    if execution["infrastructure_error"]:
        value = str(execution["infrastructure_error"])
        failures.append("RESOURCE_LIMIT" if "timeout" in value.lower() else value)
    if not execution["evaluation_harness_entered"]:
        failures.append("SETUP_FAILURE")
    if not execution["contracted_operation_reached"]:
        failures.append("CONTRACTED_OPERATION_UNREACHED")
    if not execution["runtime_assertions_exercised"] and not intended:
        failures.append("CONTRACT_ASSERTIONS_UNEXERCISED")
    if (
        execution.get("outcome") == "SATISFIED"
        and not execution.get("all_runtime_assertions_exercised", False)
    ):
        failures.append("CONTRACT_ASSERTIONS_UNEXERCISED")
    if not intended:
        failures.append("WRONG_FAILURE_REASON")
    return list(dict.fromkeys(failures))


def _candidate_gates(
    execution: dict,
    violation_criterion: dict,
    execution_failures: list[str],
    conformance_accepted: bool,
    *,
    repeated: bool,
) -> dict:
    """Return categorical qualification gates; no weighted score is authoritative."""
    intended = intended_violation(execution, violation_criterion)
    checks = {
        "no_infrastructure_error": execution.get("infrastructure_error") is None,
        "evaluation_harness_entered": bool(execution.get("evaluation_harness_entered")),
        "contracted_operation_reached": bool(execution.get("contracted_operation_reached")),
        "oracle_exercised": bool(execution.get("runtime_assertions_exercised")) or intended,
        "intended_bug_violation": intended,
    }
    if repeated:
        checks["deterministic"] = "NONDETERMINISTIC" not in execution_failures
    checks["semantic_conformance"] = conformance_accepted
    return {
        "checks": checks,
        "passed": not execution_failures and conformance_accepted,
    }


def _reviewable_candidate(row: dict) -> bool:
    """Separate execution health from discrimination; SATISFIED is never sufficient to qualify."""
    failures = set(row.get("execution_failures") or [])
    execution = row.get("execution") or {}
    return not failures or (
        failures == {"WRONG_FAILURE_REASON"}
        and execution.get("outcome") == "SATISFIED"
        and execution.get("evaluation_harness_entered")
        and execution.get("contracted_operation_reached")
        and execution.get("all_runtime_assertions_exercised")
        and not execution.get("infrastructure_error")
        and execution.get("command_exit") == 0
    )


def _contract_recovery_feedback(row: dict) -> dict:
    """Return failure-directed evidence for another synthesis attempt."""
    execution = row["execution"]
    conformance = row.get("conformance") or {}
    owner = _candidate_failure_owner(row)
    raised = [
        event
        for event in execution.get("events", [])
        if event.get("event") == "contracted_operation_raised"
    ]
    violation_criterion = row.get("violation_criterion") or {}
    if execution.get("infrastructure_error") == "BASE_IMPORT_FAILURE":
        base_import = execution.get("base_import_check") or {}
        detail = ": ".join(
            value
            for value in (
                str(base_import.get("exception_type") or "").strip(),
                str(base_import.get("detail") or "").strip(),
            )
            if value
        )
        action = (
            "The unmodified target cannot import in the pinned task runtime "
            f"({detail or 'see base_import_check'}). Do not mask this with an executable-contract "
            "shim or retry EC synthesis. Preserve the import-failure receipt and continue to "
            "patch generation from the available issue/NLC guidance; only a real patch can "
            "address a repository-owned compatibility defect."
        )
    elif owner == "NLC_OR_WITNESS":
        action = (
            "The executed witness satisfies the contract on the buggy base, and the self-review "
            "found no missing or added NLC meaning. Stop EC regeneration. This is observed "
            "non-discrimination, not proof that the NLC is wrong. Inspect the witness, selected "
            "operation, environment, and evidence in a separate diagnosis; do not strengthen "
            "assertions or rewrite the obligation merely to obtain a failure."
        )
    elif set(row.get("failures") or []) and set(row.get("failures") or []) <= {
        "CONFORMANCE_MISSING_MEANING",
        "CONFORMANCE_SEMANTIC_OVERREACH",
    }:
        action = (
            "Keep the evidence-backed witness and contracted operation fixed. Revise only the "
            f"runtime contract meaning: add the missing clauses {conformance.get('missing_meaning', [])!r} "
            f"and remove the unsupported clauses {conformance.get('added_meaning', [])!r}."
        )
    elif not execution.get("contracted_operation_reached"):
        action = (
            "Repair witness construction and invocation so the declared contracted operation is "
            "actually called. Do not change the grounded obligation."
        )
    elif not execution.get("runtime_assertions_exercised") and raised:
        observed = ", ".join(
            f"{event.get('exception_type')}: {event.get('detail', '')}".rstrip(": ")
            for event in raised
        )
        action = (
            "The contracted operation was reached but raised before any runtime assertion executed "
            f"({observed}). If this is the evidence-backed forbidden outcome, capture exactly that "
            "outcome and route it to a minimal runtime assertion; otherwise repair the witness or "
            "invocation that caused it. Keep the grounded obligation unchanged."
        )
    elif not execution.get("runtime_assertions_exercised"):
        action = (
            "Make at least one postcondition assertion reachable after the contracted operation. "
            "Do not change the grounded obligation or add unsupported behavior."
        )
    elif (
        "WRONG_FAILURE_REASON" in set(row.get("failures") or [])
        and violation_criterion.get("kind") == "reported_exception"
        and not violation_criterion.get("normal_assertion_fallback")
    ):
        expected = violation_criterion.get("exception_type")
        action = (
            "The assertions failed, but they did not expose the evidence-backed buggy event. "
            f"The contracted operation must encounter the reported {expected} at the frozen "
            "witness boundary and the harness must catch that exact operation exception and "
            "convert it into a descriptive runtime assertion failure. Do not force failure by "
            "asserting that a normally serialized tag must be omitted. If public construction "
            "normalizes the value, report the witness mismatch. Use a controlled test double "
            "only when admissible evidence establishes that dependency behavior, never just "
            "to manufacture the required exception. Do not modify the target implementation."
        )
    elif (
        "WRONG_FAILURE_REASON" in set(row.get("failures") or [])
        and violation_criterion.get("normal_assertion_fallback")
    ):
        action = (
            "The witness has not yet demonstrated a grounded normal-postcondition violation. "
            "The issue's reported exception may be created by an outer caller rather than the "
            "selected operation. Exercise the actual issue path and assert its required "
            "caller-visible result; capture an exception only if that result forbids it. "
            "Do not manufacture the reported exception or count setup failures as the bug."
        )
    elif (
        execution.get("outcome") == "SATISFIED"
        and "WRONG_FAILURE_REASON" in set(row.get("failures") or [])
    ):
        action = (
            "The buggy checkout satisfied the checks, while self-review identified a semantic "
            "translation defect. Use the missing/added-meaning findings below, not mere lack "
            "of failure, to correct the executable contract "
            "with the same grounded precondition and operation, but assert every material part "
            "of the frozen postcondition using caller-visible observations. Do not merely assert "
            "a broad return type or another condition already true on the buggy checkout."
        )
    else:
        action = (
            "Repair only the executable contract harness using the concrete execution and semantic "
            "evidence below; keep the grounded obligation unchanged."
        )
    return {
        "candidate_id": row["id"],
        "failure_owner": owner,
        "failures": row["failures"],
        "execution": {
            "outcome": execution.get("outcome"),
            "contracted_operation_reached": execution.get("contracted_operation_reached"),
            "runtime_assertions_exercised": execution.get("runtime_assertions_exercised"),
            "assertion_execution": execution.get("assertion_execution", {}),
            "infrastructure_error": execution.get("infrastructure_error"),
            "events": execution.get("events", [])[-12:],
            "execution_trace": execution.get("execution_trace", []),
            "command_log_tail": execution.get("command_log_tail", "")[-2000:],
            "base_import_check": execution.get("base_import_check"),
        },
        "conformance": {
            "back_translation": conformance.get("back_translation", ""),
            "missing_meaning": conformance.get("missing_meaning", []),
            "added_meaning": conformance.get("added_meaning", []),
            "reason": conformance.get("reason", ""),
        },
        "gates": row.get("gates", {}),
        "required_action": action,
    }


def _candidate_failure_owner(row: dict) -> str:
    """Attribute one candidate failure before deciding whether an LLM retry can help."""
    execution = row.get("execution") or {}
    events = execution.get("events") or []
    if execution.get("infrastructure_error") == "BASE_IMPORT_FAILURE":
        return "EXECUTABLE_CONTRACT_ENVIRONMENT"
    if execution.get("infrastructure_error") == "CONTRACT_POLICY_VIOLATION":
        return "EXECUTABLE_CONTRACT"
    if execution.get("infrastructure_error"):
        return "EXECUTOR"
    if any(event.get("event") == "runtime_monitor_error" for event in events):
        return "RUNTIME_MONITOR"
    if any(
        event.get("event") == "contracted_operation_raised"
        and event.get("exception_type") == "builtins.TypeError"
        and "descriptor" in str(event.get("detail", "")).lower()
        for event in events
    ) or any(
        event.get("event") == "contracted_operation_raised"
        and "classmethod' object is not callable" in str(event.get("detail", ""))
        for event in events
    ):
        return "RUNTIME_MONITOR"
    failures = set(row.get("failures") or [])
    conformance = row.get("conformance") or {}
    if (
        failures == {"WRONG_FAILURE_REASON"}
        and execution.get("outcome") == "SATISFIED"
        and execution.get("contracted_operation_reached")
        and execution.get("all_runtime_assertions_exercised")
        and row.get("conformance_status") == "QUALIFIED"
        and not conformance.get("missing_meaning")
        and not conformance.get("added_meaning")
    ):
        return "NLC_OR_WITNESS"
    if failures and failures <= {
        "CONFORMANCE_MISSING_MEANING",
        "CONFORMANCE_SEMANTIC_OVERREACH",
    }:
        return "EXECUTABLE_CONTRACT"
    return "EXECUTABLE_CONTRACT"


def _contract_failure_signature(row: dict) -> tuple:
    """Retain concrete evidence so different failures are not collapsed into one label set."""
    execution = row.get("execution") or {}
    conformance = row.get("conformance") or {}
    events = tuple(
        (
            event.get("event"),
            event.get("target"),
            event.get("exception_type"),
            event.get("detail"),
        )
        for event in execution.get("events", [])
        if event.get("event")
        in {
            "contracted_operation_raised",
            "evaluation_harness_error",
            "runtime_monitor_error",
            "contract_violation",
        }
    )
    return (
        _candidate_failure_owner(row),
        tuple(sorted(row.get("failures") or [])),
        events,
        tuple(conformance.get("missing_meaning") or []),
        tuple(conformance.get("added_meaning") or []),
    )


def _normalized_back_translation(value: str) -> str:
    """Normalize reviewer prose only for conservative cross-batch duplicate hints."""
    return " ".join(re.findall(r"[a-z0-9_]+", value.lower()))


def _semantic_class_count(rows: list[dict], duplicate_pairs: list[dict]) -> int:
    """Count equivalence components among qualified candidates."""
    identifiers = {row["id"] for row in rows}
    parent = {identifier: identifier for identifier in identifiers}

    def find(identifier: str) -> str:
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for pair in duplicate_pairs:
        left, right = pair.get("candidate"), pair.get("duplicate_of")
        if left in identifiers and right in identifiers:
            union(left, right)
    return len({find(identifier) for identifier in identifiers})


def _contract_failure_owner(rows: list[dict]) -> str:
    execution_labels = {
        "SETUP_FAILURE",
        "CONTRACTED_OPERATION_UNREACHED",
        "CONTRACT_ASSERTIONS_UNEXERCISED",
        "WRONG_FAILURE_REASON",
        "NONDETERMINISTIC",
        "RESOURCE_LIMIT",
        "REPOSITORY_MUTATION",
    }
    owners = {_candidate_failure_owner(row) for row in rows}
    if len(owners) == 1:
        return next(iter(owners))
    failures = {failure for row in rows for failure in row.get("failures", [])}
    if failures and failures <= {
        "CONFORMANCE_MISSING_MEANING",
        "CONFORMANCE_SEMANTIC_OVERREACH",
    }:
        return "EXECUTABLE_CONTRACT"
    if failures & execution_labels:
        return "EXECUTABLE_CONTRACT_SYNTHESIS_OR_WITNESS"
    return "MIXED_QUALIFICATION"


def _observed_contracted_operation_exceptions(execution: dict) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(event.get("exception_type"))
            for event in execution.get("events", [])
            if event.get("event") == "contracted_operation_raised"
        )
    )


def _discover_python(
    task: Task,
    repo: Path,
    artifacts: RunArtifacts,
    executor: Executor,
    stages: Stages,
    settings: WorkflowSettings,
    localization_override: dict | None,
) -> dict:
    """Gold-blind executable-contract discovery and pre-gold qualification."""
    root = artifacts.root / "base"
    repo_sha = snapshot(repo, task, root)
    artifacts.save("base_identity.json", {"repo_sha256": repo_sha, "base_commit": task.base_commit})
    executor_identity = executor.identity()
    artifacts.save("executor.json", executor_identity)
    index = RepositoryIndex(root)
    artifacts.save(
        "repository_parse_diagnostics.json",
        {
            "failures": index.parse_failures,
            "warnings": index.parse_warnings,
        },
    )
    try:
        localization = _localize(index, task, settings, stages, localization_override, artifacts)
    except LocalizationExhausted as exc:
        status = {
            "status": "NO_QUALIFIED_REPAIR_CONTRACT",
            "reason": "LOCALIZATION_INSUFFICIENT_CONTEXT",
            "detail": exc.detail,
            "localization_rounds": exc.rounds,
            "instance_id": task.instance_id,
        }
        artifacts.save("summary.json", status)
        artifacts.save("status.json", status)
        write_failure_report(artifacts.root, task.instance_id, status)
        logger.panel("Localization abstained", exc.detail, style="yellow")
        return status
    contracted_operations = [
        index.locations[item] for item in dict.fromkeys(localization.contracted_operation_locations)
    ]
    selected = [
        index.locations[item]
        for item in dict.fromkeys(localization.repair_locations + localization.contracted_operation_locations)
    ]
    logger.table(
        "Localization",
        ["Role", "File", "Symbol"],
        [
            ["REPAIR", index.locations[item].file, index.locations[item].symbol]
            for item in localization.repair_locations
        ]
        + [
            ["CONTRACTED OPERATION", index.locations[item].file, index.locations[item].symbol]
            for item in localization.contracted_operation_locations
        ],
    )
    artifacts.events.emit(
        "localization_selected",
        stage="localize",
        repair=localization.repair_locations,
        contracted_operation=localization.contracted_operation_locations,
    )
    artifacts.save("localization/final.json", {
        "localization": localization.model_dump(),
        "localized_edit_paths": sorted({index.locations[item].file for item in localization.repair_locations}),
        "locations": [item.packet(2400) for item in selected],
    })
    if settings.repair_enabled and settings.repair_variant == "no-contract":
        return {"status": "LOCALIZED", "instance_id": task.instance_id,
                "reason": "NO_CONTRACT_ABLATION", "contract_pipeline_skipped": True}
    evidence = evidence_packet(task, selected)
    evidence.update(
        retrieve_documents(
            root, task.problem_statement, [item.symbol for item in selected], base_commit=task.base_commit
        )
    )
    evidence.update(
        retrieve_existing_tests(
            root, task.problem_statement, [item.symbol for item in selected],
            base_commit=task.base_commit,
            source_files=[item.file for item in selected],
        )
    )
    normative = {
        key: dict(value)
        for key, value in evidence.items()
        if value["kind"] in {"issue", "docstring", "existing_test", "repository_document"}
        and value["text"].strip()
    }
    artifacts.save("evidence/evidence.json", evidence)
    spans = canonical_spans(normative)
    artifacts.save("evidence/spans.json", spans)
    obligation_packet = {
        "task": task.model_packet(),
        "contracted_operation_locations": [
            item.packet(settings.excerpt_chars) for item in contracted_operations
        ],
        "allowed_contracted_operation_ids": [item.id for item in contracted_operations],
        "evidence_mode": settings.evidence_mode,
        "include_rationales": settings.include_rationales,
    }
    if settings.evidence_mode == "canonical_spans":
        obligation_packet.update(
            allowed_span_ids=list(spans),
            evidence_spans=spans,
        )
    else:
        obligation_packet.update(
            allowed_evidence_ids=list(normative),
            evidence=normative,
        )
    grounding_failures = []
    previous_proposal = None
    obligation = None
    last_grounded_obligation = None
    nlc_review_history: list[dict] = []
    refinement_failures: list[dict] = []
    obligation_generation_attempts = 0
    for attempt in range(1, settings.max_obligation_attempts + 1):
        obligation_generation_attempts = attempt
        response_schema = (
            CanonicalRepairObligations
            if settings.evidence_mode == "canonical_spans"
            else RepairObligations
        )
        natural = stages.generate(
            "repair_obligations", response_schema, obligation_packet
        )
        if not natural.obligations:
            if last_grounded_obligation is not None:
                refinement_failures.append(
                    {
                        "attempt": attempt,
                        "code": "REFINEMENT_ABSTAINED",
                        "detail": natural.unsupported_reason,
                    }
                )
                obligation = last_grounded_obligation
                break
            status = {
                "status": "NO_QUALIFIED_REPAIR_CONTRACT",
                "reason": "INSUFFICIENT_EVIDENCE",
                "detail": natural.unsupported_reason,
                "instance_id": task.instance_id,
            }
            artifacts.save("summary.json", status)
            artifacts.save("status.json", status)
            write_failure_report(artifacts.root, task.instance_id, status)
            logger.panel("Abstained", natural.unsupported_reason, style="yellow")
            return status
        proposal = natural.obligations[0].model_dump()
        candidate = {"id": "C1", **proposal}
        try:
            if settings.include_rationales and not candidate["rationale"].strip():
                raise EvidenceGroundingError(
                    "MISSING_EVIDENCE_RATIONALE",
                    "contract-first profiles require a rationale that maps the precondition "
                    "and normal or exceptional postcondition to their supporting context",
                )
            _ground_repair(
                candidate,
                evidence,
                set(localization.contracted_operation_locations),
                spans=spans,
            )
            obligation = candidate
            last_grounded_obligation = candidate
        except EvidenceGroundingError as exc:
            duplicate = proposal == previous_proposal
            failure = {"attempt": attempt, **exc.feedback(), "duplicate_proposal": duplicate}
            grounding_failures.append(failure)
            artifacts.events.emit(
                "obligation_grounding_rejected",
                stage="repair_obligations",
                **failure,
            )
            logger.warning(
                f"[repair_obligations] attempt {attempt}/{settings.max_obligation_attempts} "
                f"rejected | {exc.code} | {exc.detail}"
            )
            if last_grounded_obligation is not None:
                refinement_failures.append(failure)
            if duplicate:
                obligation = last_grounded_obligation
                break
            previous_proposal = proposal
            obligation_packet = {
                **obligation_packet,
                "qualification_feedback": {
                    **exc.feedback(),
                    "instruction": (
                        "Correct only the rejected grounding defect. Preserve supported semantics. "
                        + (
                            "Explain what the precondition and postcondition evidence each establish."
                            if exc.code == "MISSING_EVIDENCE_RATIONALE"
                            else (
                                "Split the precondition into atomic precondition_clauses. Set "
                                "precondition to those clause strings joined in order by ' AND '. "
                                "For every clause, select all and only its supporting canonical "
                                "span_ids, and also include those span_ids as precondition evidence."
                                if exc.code
                                in {
                                    "MISSING_PRECONDITION_CLAUSE_MAP",
                                    "PRECONDITION_CLAUSE_MAP_REQUIRES_CANONICAL_SPANS",
                                    "PRECONDITION_CLAUSE_TEXT_MISMATCH",
                                    "UNSELECTED_PRECONDITION_CLAUSE_SPAN",
                                    "UNMAPPED_PRECONDITION_EVIDENCE",
                                    "MISSING_ISSUE_PRECONDITION_EVIDENCE",
                                }
                                else (
                                    "Select only a supplied span_id; the host owns its canonical text."
                                    if settings.evidence_mode == "canonical_spans"
                                    else (
                                        "Every quote must be one contiguous verbatim span under its evidence id. "
                                        "Never insert '...' or combine distant lines."
                                    )
                                )
                            )
                        )
                    ),
                },
            }
            continue

        units = _nlc_review_units(obligation)
        allowed_review_evidence = set(
            spans if settings.evidence_mode == "canonical_spans" else normative
        )
        review_packet = {
            "task": task.model_packet(),
            "review_round": len(nlc_review_history) + 1,
            "obligation": obligation,
            "review_units": units,
            "evidence_mode": settings.evidence_mode,
            "contracted_operation_locations": obligation_packet[
                "contracted_operation_locations"
            ],
        }
        if settings.evidence_mode == "canonical_spans":
            review_packet["evidence_spans"] = spans
        else:
            review_packet["evidence"] = normative
        if nlc_review_history:
            review_packet["prior_review"] = nlc_review_history[-1]
        try:
            reviewed = stages.generate(
                "repair_obligation_review", RepairObligationReview, review_packet
            )
        except ReasoningOnlyOutputBudgetError:
            # Retry once within the selected profile, without silently changing
            # the model, reasoning policy, completion ceiling, or evidence.
            retry_packet = {
                **review_packet,
                "reasoning_only_budget_retry": 1,
                "retry_instruction": (
                    "Return the required structured clause review now. Keep the same evidence, "
                    "do not add discussion, and preserve the configured completion ceiling."
                ),
            }
            logger.warning(
                "[repair_obligation_review] reasoning-only output exhausted the configured ceiling; "
                "retrying once with the same reasoning policy and budget"
            )
            reviewed = stages.generate(
                "repair_obligation_review", RepairObligationReview, retry_packet
            )
        try:
            receipt = _nlc_review_receipt(
                reviewed,
                units,
                allowed_review_evidence,
                round_number=len(nlc_review_history) + 1,
            )
        except WorkflowError as exc:
            # A reviewer transport/provenance mistake (for example citing E1
            # instead of canonical span E1.S001) should receive one bounded
            # correction, not terminate the entire repair task. The correction
            # cannot change the NLC or evidence set.
            logger.warning(
                "[repair_obligation_review] deterministic review validation failed; "
                "retrying once with exact clause/evidence identifiers"
            )
            correction_packet = {
                **review_packet,
                "deterministic_review_correction": {
                    "error": str(exc)[:1200],
                    "allowed_evidence_ids": sorted(allowed_review_evidence),
                    "required_review_units": units,
                    "instruction": (
                        "Return exactly one review per supplied clause, preserve each "
                        "clause_text byte-for-byte, and cite only allowed_evidence_ids. "
                        "Do not change the NLC or infer new evidence."
                    ),
                },
            }
            reviewed = stages.generate(
                "repair_obligation_review", RepairObligationReview, correction_packet
            )
            receipt = _nlc_review_receipt(
                reviewed,
                units,
                allowed_review_evidence,
                round_number=len(nlc_review_history) + 1,
            )
        nlc_review_history.append(receipt)
        artifacts.save(f"nlc_review/review_{len(nlc_review_history):02d}.json", receipt)
        artifacts.events.emit(
            "nlc_review_completed",
            stage="repair_obligation_review",
            round=receipt["round"],
            accepted=receipt["accepted"],
            justification_coverage=receipt["justification_coverage"],
            counts=receipt["counts"],
        )
        logger.table(
            f"NLC justification review · round {receipt['round']}",
            ["Clause", "Verdict", "Reason"],
            [
                [item.clause_id, item.verdict, item.reason]
                for item in reviewed.reviews
            ],
        )
        if reviewed.missing_required_clauses:
            logger.panel(
                f"NLC completeness gaps · round {receipt['round']}",
                "\n".join(
                    f"- {clause}" for clause in reviewed.missing_required_clauses
                ),
                style="yellow",
            )
        if receipt["accepted"]:
            break
        if attempt >= settings.max_obligation_attempts:
            break
        previous_proposal = proposal
        obligation_packet = {
            **obligation_packet,
            "qualification_feedback": {
                "code": "NLC_REVIEW_REJECTED",
                "review_round": receipt["round"],
                "reviews": receipt["reviews"],
                "missing_required_clauses": receipt["missing_required_clauses"],
                "instruction": (
                    "Add every evidence-required clause listed in missing_required_clauses. "
                    "Also refine only existing clauses classified UNSUPPORTED or CONTRADICTED. "
                    "Preserve DIRECT and DERIVED_VALID meaning, use only the supplied "
                    "evidence and program context, and do not invent a preferred repair outcome."
                ),
            },
        }
    if obligation is None:
        status = {
            "status": "NO_QUALIFIED_REPAIR_CONTRACT",
            "reason": "INVALID_NLC",
            "detail": "No obligation passed deterministic evidence grounding within the bounded attempts.",
            "instance_id": task.instance_id,
            "grounding_failures": grounding_failures,
        }
        artifacts.save("summary.json", status)
        artifacts.save("status.json", status)
        write_failure_report(artifacts.root, task.instance_id, status)
        logger.panel(
            "Obligation rejected",
            f"No grounded obligation after {len(grounding_failures)} attempt(s). "
            "See `status.json` and model responses.",
            style="yellow",
        )
        return status
    final_nlc_review = nlc_review_history[-1]
    if final_nlc_review["accepted"]:
        nlc_review_outcome = (
            "ACCEPTED_FIRST_REVIEW"
            if len(nlc_review_history) == 1
            else "ACCEPTED_AFTER_REFINEMENT"
        )
    elif settings.nlc_review_mode == "strict":
        status = {
            "status": "NO_QUALIFIED_REPAIR_CONTRACT",
            "reason": "NLC_REVIEW_REJECTED",
            "detail": "The final bounded NLC review contains unsupported or contradicted clauses.",
            "instance_id": task.instance_id,
            "nlc_review_mode": settings.nlc_review_mode,
            "nlc_review_outcome": "NLC_REVIEW_REJECTED_STRICT",
            "nlc_reviews": nlc_review_history,
            "refinement_failures": refinement_failures,
        }
        artifacts.save("summary.json", status)
        artifacts.save("status.json", status)
        write_failure_report(artifacts.root, task.instance_id, status)
        return status
    else:
        nlc_review_outcome = "NLC_REVIEW_EXHAUSTED_ADVISORY"
    nlc_review_receipt = {
        "mode": settings.nlc_review_mode,
        "outcome": nlc_review_outcome,
        "review_count": len(nlc_review_history),
        "refinement_count": max(0, obligation_generation_attempts - 1),
        "accepted": final_nlc_review["accepted"],
        "enforced": settings.nlc_review_mode == "strict",
        "final": final_nlc_review,
        "history": nlc_review_history,
        "refinement_failures": refinement_failures,
    }
    obligation["nlc_review"] = nlc_review_receipt
    obligation["violation_criterion"] = _derive_violation_criterion(obligation, evidence)
    artifacts.save("evidence/grounded_nlc.json", obligation)
    if settings.repair_enabled and settings.repair_variant == "context":
        return {"status": "NLC_READY", "instance_id": task.instance_id,
                "nlc_review": nlc_review_receipt, "ec_pipeline_skipped": True}
    artifacts.events.emit(
        "obligation_grounded",
        stage="repair_obligations",
        obligation_id="C1",
        evidence_ids=[item["id"] for item in obligation["evidence"]],
    )
    logger.panel(
        "Grounded REPAIR · C1",
        f"**Precondition:** {obligation['precondition']}\n\n"
        f"**Normal postcondition:** {obligation.get('normal_postcondition') or 'not specified'}\n\n"
        f"**Exceptional postcondition:** {obligation.get('exceptional_postcondition') or 'not specified'}\n\n"
        + "\n".join(f"- `{item['id']}` · {item['supports']}" for item in obligation["evidence"]),
        style="green",
    )
    location = index.locations[obligation["contracted_operation_id"]]
    execution_context_query = "\n".join(
        filter(
            None,
            [
                task.problem_statement,
                obligation["precondition"],
                obligation.get("normal_postcondition"),
                obligation.get("exceptional_postcondition"),
            ],
        )
    )
    execution_examples = repository_execution_examples(
        root,
        execution_context_query,
        location.symbol,
    )
    repository_call_context = [
        semantic_location(item.packet(6000))
        for item in index.search(execution_context_query, limit=8)
        if item.id != location.id
    ][:4]
    artifacts.save(
        "executable_contracts/repository_execution_examples.json",
        execution_examples,
    )
    artifacts.save(
        "executable_contracts/repository_call_context.json",
        repository_call_context,
    )
    logger.table(
        "EC repository execution context",
        ["File", "Symbol", "Line", "Purpose"],
        [
            [
                item["file"],
                item["symbol"],
                item["start"],
                "witness/API usage only",
            ]
            for item in execution_examples
        ],
    )
    logger.table(
        "EC repository call/source context",
        ["File", "Symbol", "Line", "Purpose"],
        [
            [
                item["file"],
                item["symbol"],
                item["start"],
                "imports/call construction only",
            ]
            for item in repository_call_context
        ],
    )
    candidate_rows = []
    base_import_receipt = check_base_import(root, location.target(), executor)
    artifacts.save("execution/base_import_check.json", base_import_receipt)
    if base_import_receipt.get("status") == "IMPORT_FAILED":
        logger.warning(
            "[contracts] unmodified target import is blocked in the pinned task image; "
            "EC synthesis skipped rather than masking the environment with a witness shim"
        )
        status = {
            "status": "NO_QUALIFIED_REPAIR_CONTRACT",
            "instance_id": task.instance_id,
            "failure_owner": "EXECUTABLE_CONTRACT_ENVIRONMENT",
            "reason": "BASE_IMPORT_FAILURE",
            "base_import_check": base_import_receipt,
            "contract_synthesis_batches": 0,
            "recovery": {
                "initial_candidate_cap": settings.contract_candidates,
                "additional_candidate_budget": settings.additional_candidate_budget,
                "max_synthesis_batches": settings.max_contract_attempts,
                "extra_conformance_recovery_used": False,
                "same_failure_branch_stopped": False,
                "stopped_before_synthesis": True,
            },
            "nlc_review": nlc_review_receipt,
            "candidates": [],
        }
        artifacts.save("summary.json", status)
        artifacts.save("status.json", status)
        write_failure_report(artifacts.root, task.instance_id, status)
        return status
    conformance_input = semantic_obligation(obligation)
    qualified = []
    duplicate_ids = set()
    duplicate_pairs: list[dict] = []
    contract_feedback = []
    contract_synthesis_batches = 0
    initial_candidates_remaining = settings.contract_candidates
    additional_candidates_remaining = settings.additional_candidate_budget
    previous_failure_signature = None
    repeated_failure_branch_stopped = False
    for synthesis_batch in range(1, settings.max_contract_attempts + 1):
        if initial_candidates_remaining <= 0 and additional_candidates_remaining <= 0:
            break
        contract_synthesis_batches = synthesis_batch
        candidate_limit, consuming_initial_budget = _candidate_batch_offer(
            initial_candidates_remaining, additional_candidates_remaining
        )
        synthesis_packet = {
            "task": task.model_packet(),
            "obligation": conformance_input,
            "contracted_operation": semantic_location(location.packet(32000)),
            "repository_execution_examples": execution_examples,
            "repository_call_context": repository_call_context,
            # This is deliberately a ceiling, not a requested cardinality. The
            # schema requires at least one candidate and lets the authoring
            # model stop as soon as it has a complete realization.
            "candidate_limit": candidate_limit,
            "synthesis_batch": synthesis_batch,
            "max_synthesis_batches": settings.max_contract_attempts,
            "remaining_initial_candidate_budget": initial_candidates_remaining,
            "remaining_additional_candidate_budget": additional_candidates_remaining,
            "include_rationales": settings.include_rationales,
        }
        if contract_feedback:
            synthesis_packet["prior_candidate_feedback"] = contract_feedback
        authored = stages.generate(
            "executable_contract_synthesis",
            ExecutableContractCandidates,
            synthesis_packet,
        )
        syntactic_duplicate_ids = set()
        syntactic_duplicate_pairs = []
        smt_duplicate_pairs = []
        authored_candidates = authored.candidates[:candidate_limit]
        logger.info(
            f"[contracts] batch {synthesis_batch} returned {len(authored_candidates)} "
            f"candidate(s) (candidate ceiling={candidate_limit}, "
            f"cumulative before batch={len(candidate_rows)})"
        )
        for position, candidate in enumerate(authored_candidates):
            for earlier in authored_candidates[:position]:
                if equivalent_contract_sources(earlier.source, candidate.source):
                    syntactic_duplicate_ids.add(candidate.id)
                    syntactic_duplicate_pairs.append((candidate.id, earlier.id))
                elif settings.equivalence_mode == "hybrid_smt":
                    smt = compare_assertions_with_z3(earlier.source, candidate.source)
                    if smt["status"] == "PROVED_EQUIVALENT":
                        syntactic_duplicate_ids.add(candidate.id)
                        smt_duplicate_pairs.append(
                            (candidate.id, earlier.id, smt["reason"])
                        )
        batch_rows = []
        local_to_global = {}
        for candidate in authored_candidates:
            global_id = _candidate_label(len(candidate_rows) + len(batch_rows))
            local_to_global[candidate.id] = global_id
            try:
                validate_executable_contract(candidate.source)
            except (SyntaxError, ValueError) as exc:
                validation_error = f"{type(exc).__name__}: {exc}"[:1200]
            else:
                validation_error = None
            for earlier in candidate_rows:
                if equivalent_contract_sources(earlier["source"], candidate.source):
                    duplicate_ids.update({global_id, earlier["id"]})
                    duplicate_pairs.append(
                        {
                            "kind": "syntactic",
                            "candidate": global_id,
                            "duplicate_of": earlier["id"],
                        }
                    )
            artifacts.save(
                f"executable_contracts/candidate_{global_id.lower()}.json",
                {**candidate.model_dump(), "id": global_id, "batch": synthesis_batch},
            )
            path = artifacts.root / "executable_contracts" / f"candidate_{global_id.lower()}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(candidate.source, encoding="utf-8")
            logger.code(
                f"Executable contract candidate {global_id} · batch {synthesis_batch}",
                candidate.source,
            )
            if validation_error:
                execution = {
                    "outcome": "ERROR",
                    "evaluation_harness_entered": False,
                    "contracted_operation_reached": False,
                    "runtime_assertions_exercised": False,
                    "all_runtime_assertions_exercised": False,
                    "infrastructure_error": "CONTRACT_POLICY_VIOLATION",
                    "events": [],
                    "command_exit": None,
                    "command_log_tail": "",
                    "execution_trace": [f"HOST VALIDATION rejected: {validation_error}"],
                    "validation_error": validation_error,
                }
                artifacts.save(
                    f"execution/candidate_{global_id.lower()}_buggy.json", execution
                )
                failures = _candidate_failure(
                    execution, obligation["violation_criterion"]
                )
                batch_rows.append({
                    "id": global_id,
                    "local_id": candidate.id,
                    "batch": synthesis_batch,
                    "source": candidate.source,
                    "rationale": candidate.rationale,
                    "execution": execution,
                    "repeat_execution": None,
                    "violation_kind": None,
                    "violation_criterion": obligation["violation_criterion"],
                    "execution_failures": list(failures),
                    "failures": list(failures),
                })
                logger.warning(
                    f"[contracts] candidate {global_id} rejected before execution: "
                    f"{validation_error}"
                )
                continue
            execution = evaluate_executable_contract(
                root, candidate.source, location.target(include_accessor=True), executor
            )
            harness_errors = [
                event for event in execution.get("events", [])
                if event.get("event") == "evaluation_harness_error"
            ]
            if (
                harness_errors and not execution.get("contracted_operation_reached")
                and not execution.get("infrastructure_error")
            ):
                if base_import_receipt is None:
                    base_import_receipt = check_base_import(root, location.target(), executor)
                    artifacts.save("execution/base_import_check.json", base_import_receipt)
                execution["base_import_check"] = base_import_receipt
                if base_import_receipt.get("status") == "IMPORT_FAILED" and any(
                    event.get("exception_type") == base_import_receipt.get("exception_type")
                    and event.get("detail") == base_import_receipt.get("detail")
                    for event in harness_errors
                ):
                    execution["infrastructure_error"] = "BASE_IMPORT_FAILURE"
                    execution.setdefault("execution_trace", []).append(
                        "BASE IMPORT BLOCKED: the same failure occurs without the candidate or monitor"
                    )
                    logger.warning(
                        "[contracts] uninstrumented target-module import reproduces the failure; "
                        "passing the exact pinned-environment failure to EC recovery"
                    )
            artifacts.save(f"execution/candidate_{global_id.lower()}_buggy.json", execution)
            logger.panel(
                f"Executable contract execution · candidate {global_id}",
                "\n".join(execution.get("execution_trace", [])),
                style="yellow" if execution.get("outcome") != "SATISFIED" else "green",
            )
            artifacts.events.emit(
                "executable_contract_evaluated",
                stage="contract_evaluation",
                candidate_id=global_id,
                synthesis_batch=synthesis_batch,
                outcome=execution["outcome"],
                contracted_operation_reached=execution["contracted_operation_reached"],
                runtime_assertions_exercised=execution["runtime_assertions_exercised"],
            )
            failures = _candidate_failure(execution, obligation["violation_criterion"])
            violation_kind = (
                "EXCEPTIONAL_POSTCONDITION_VIOLATION"
                if intended_violation(execution, obligation["violation_criterion"])
                and not execution["runtime_assertions_exercised"]
                else ("RUNTIME_CONTRACT_ASSERTION_VIOLATION" if execution["outcome"] == "VIOLATED" else None)
            )
            repeated = None
            if settings.repeat_base and not execution.get("infrastructure_error"):
                repeated = evaluate_executable_contract(
                    root, candidate.source, location.target(include_accessor=True), executor
                )
                artifacts.save(
                    f"execution/candidate_{global_id.lower()}_buggy_repeat.json",
                    repeated,
                )
                stable_fields = (
                    "evaluation_harness_entered",
                    "contracted_operation_reached",
                    "runtime_assertions_exercised",
                    "outcome",
                    "infrastructure_error",
                )
                if any(execution.get(field) != repeated.get(field) for field in stable_fields):
                    failures.append("NONDETERMINISTIC")
                if intended_violation(execution, obligation["violation_criterion"]) != intended_violation(
                    repeated, obligation["violation_criterion"]
                ) or _observed_contracted_operation_exceptions(
                    execution
                ) != _observed_contracted_operation_exceptions(repeated):
                    failures.append("NONDETERMINISTIC")
            batch_rows.append(
                {
                    "id": global_id,
                    "local_id": candidate.id,
                    "batch": synthesis_batch,
                    "source": candidate.source,
                    "rationale": candidate.rationale,
                    "execution": execution,
                    "repeat_execution": repeated,
                    "violation_kind": violation_kind,
                    "violation_criterion": obligation["violation_criterion"],
                    "execution_failures": list(failures),
                    "failures": list(failures),
                }
            )
        duplicate_pairs.extend(
            {
                "kind": "syntactic",
                "candidate": local_to_global[candidate],
                "duplicate_of": local_to_global[duplicate_of],
            }
            for candidate, duplicate_of in syntactic_duplicate_pairs
        )
        duplicate_pairs.extend(
            {
                "kind": "smt_proved",
                "candidate": local_to_global[candidate],
                "duplicate_of": local_to_global[duplicate_of],
                "reason": reason,
            }
            for candidate, duplicate_of, reason in smt_duplicate_pairs
        )
        review_candidates = [
            {
                "id": item["local_id"],
                "source": item["source"],
                "execution": {
                    "outcome": item["execution"].get("outcome"),
                    "contracted_operation_reached": item["execution"].get(
                        "contracted_operation_reached"
                    ),
                    "runtime_assertions_exercised": item["execution"].get(
                        "runtime_assertions_exercised"
                    ),
                    "all_runtime_assertions_exercised": item["execution"].get(
                        "all_runtime_assertions_exercised"
                    ),
                    "assertion_execution": item["execution"].get(
                        "assertion_execution", {}
                    ),
                    "infrastructure_error": item["execution"].get(
                        "infrastructure_error"
                    ),
                    "command_exit": item["execution"].get("command_exit"),
                    "events": item["execution"].get("events", []),
                    "execution_trace": item["execution"].get(
                        "execution_trace", []
                    ),
                    "command_log_tail": item["execution"].get(
                        "command_log_tail", ""
                    )[-4000:],
                },
                "execution_artifact": (
                    f"execution/candidate_{item['id'].lower()}_buggy.json"
                ),
            }
            for item in batch_rows
            if _reviewable_candidate(item)
        ]
        review_packet = {
            "review_mode": "initial",
            "synthesis_batch": synthesis_batch,
            "obligation": {
                key: value
                for key, value in conformance_input.items()
                if key != "dynamic_violation_criterion"
            },
            "trusted_runtime_semantics": _conformance_protocol(),
            "candidates": review_candidates,
        }
        if not review_candidates:
            review_packet["skipped_reason"] = (
                "All candidates failed deterministic execution qualification; "
                "semantic conformance was not invoked."
            )
        artifacts.save(
            f"conformance/input_batch_{synthesis_batch:02d}.json",
            review_packet,
        )
        review_history = {}
        review_by_id = {}
        if review_candidates:
            logger.conformance_input(
                precondition=review_packet["obligation"]["precondition"],
                normal_postcondition=review_packet["obligation"].get(
                    "normal_postcondition"
                ),
                exceptional_postcondition=review_packet["obligation"].get(
                    "exceptional_postcondition"
                ),
                support=review_packet["obligation"].get("support", []),
                candidate_ids=[
                    local_to_global[item["id"]] for item in review_candidates
                ],
            )
            logger.conformance_input_receipt(review_packet)
            reviews = stages.generate(
                "executable_contract_conformance",
                ExecutableContractConformanceBatch,
                review_packet,
            )
            expected_review_ids = {item["id"] for item in review_candidates}
            initial_review_by_id = _index_conformance_reviews(
                reviews.reviews, expected_review_ids
            )
            review_history = {
                candidate_id: [review.model_dump()]
                for candidate_id, review in initial_review_by_id.items()
            }
            review_by_id = dict(initial_review_by_id)
            for review in initial_review_by_id.values():
                logger.conformance_review(
                    local_to_global[review.candidate_id],
                    phase="initial",
                    back_translation=review.back_translation,
                    missing_meaning=review.missing_meaning,
                    added_meaning=review.added_meaning,
                    semantic_duplicates=[
                        local_to_global.get(candidate_id, candidate_id)
                        for candidate_id in review.semantically_duplicates
                    ],
                    reason=review.reason,
                )
        else:
            logger.warning(
                f"[conformance] batch {synthesis_batch} skipped: all candidates "
                "failed deterministic execution qualification"
            )

        # A negative LLM judgment is a fallible semantic signal, not an oracle.
        # Give only rejected candidates a bounded, failure-directed adjudication
        # with the initial receipt and trusted host semantics made explicit.
        rejected_ids = {
            candidate_id
            for candidate_id, review in review_by_id.items()
            if not review.accepted()
        }
        for adjudication_round in range(1, settings.conformance_adjudication_attempts + 1):
            if not rejected_ids:
                break
            adjudication_candidates = [
                item for item in review_candidates if item["id"] in rejected_ids
            ]
            adjudication = stages.generate(
                "executable_contract_conformance",
                ExecutableContractConformanceBatch,
                {
                    **review_packet,
                    "review_mode": "adjudication",
                    "adjudication_round": adjudication_round,
                    "candidates": adjudication_candidates,
                    "prior_reviews": [
                        review_by_id[item["id"]].model_dump()
                        for item in adjudication_candidates
                    ],
                },
            )
            adjudicated = _index_conformance_reviews(
                adjudication.reviews, rejected_ids
            )
            for candidate_id, review in adjudicated.items():
                review_history[candidate_id].append(review.model_dump())
                review_by_id[candidate_id] = review
                logger.conformance_review(
                    local_to_global[candidate_id],
                    phase=f"adjudication {adjudication_round}",
                    back_translation=review.back_translation,
                    missing_meaning=review.missing_meaning,
                    added_meaning=review.added_meaning,
                    semantic_duplicates=[
                        local_to_global.get(duplicate_id, duplicate_id)
                        for duplicate_id in review.semantically_duplicates
                    ],
                    reason=review.reason,
                )
            rejected_ids = {
                candidate_id
                for candidate_id in rejected_ids
                if not review_by_id[candidate_id].accepted()
            }
        local_duplicate_ids = set(syntactic_duplicate_ids)
        for review in review_by_id.values():
            for duplicate_of in review.semantically_duplicates:
                if duplicate_of == review.candidate_id or duplicate_of not in review_by_id:
                    logger.warning(
                        "[conformance] ignored invalid semantic-duplicate reference: "
                        f"{review.candidate_id} -> {duplicate_of}"
                    )
                    continue
                local_duplicate_ids.update({review.candidate_id, duplicate_of})
                duplicate_pairs.append(
                    {
                        "kind": "semantic",
                        "candidate": local_to_global[review.candidate_id],
                        "duplicate_of": local_to_global[duplicate_of],
                    }
                )
        batch_qualified = []
        for row in batch_rows:
            review = review_by_id.get(row["local_id"])
            if review is None:
                row["back_translation"] = ""
                row["conformance"] = None
                row["conformance_reviews"] = []
                row["duplicate_candidate"] = row["id"] in duplicate_ids
                row["conformance_status"] = "NOT_RUN_RUNTIME_FAILURE"
                row["gates"] = _candidate_gates(
                    row["execution"],
                    obligation["violation_criterion"],
                    row["execution_failures"],
                    False,
                    repeated=row["repeat_execution"] is not None,
                )
                artifacts.save(
                    f"conformance/candidate_{row['id'].lower()}.json",
                    {
                        "candidate_id": row["local_id"],
                        "status": "NOT_RUN_RUNTIME_FAILURE",
                        "reason": (
                            "Deterministic execution qualification failed before "
                            "semantic conformance review."
                        ),
                        "execution_failures": row["execution_failures"],
                        "execution": row["execution"],
                        "execution_artifact": (
                            f"execution/candidate_{row['id'].lower()}_buggy.json"
                        ),
                    },
                )
                continue
            row["back_translation"] = review.back_translation
            row["conformance"] = review.model_dump()
            row["conformance_reviews"] = review_history[row["local_id"]]
            normalized_back_translation = _normalized_back_translation(
                review.back_translation
            )
            for earlier in candidate_rows:
                if (
                    normalized_back_translation
                    and normalized_back_translation
                    == _normalized_back_translation(earlier.get("back_translation", ""))
                ):
                    duplicate_ids.update({row["id"], earlier["id"]})
                    duplicate_pairs.append(
                        {
                            "kind": "semantic_back_translation",
                            "candidate": row["id"],
                            "duplicate_of": earlier["id"],
                        }
                    )
            if review.missing_meaning:
                row["failures"].append("CONFORMANCE_MISSING_MEANING")
            if review.added_meaning:
                row["failures"].append("CONFORMANCE_SEMANTIC_OVERREACH")
            if row["local_id"] in local_duplicate_ids or row["id"] in duplicate_ids:
                duplicate_ids.add(row["id"])
                row["duplicate_candidate"] = True
            else:
                row["duplicate_candidate"] = False
            row["conformance_status"] = "QUALIFIED" if review.accepted() else "REJECTED"
            row["gates"] = _candidate_gates(
                row["execution"],
                obligation["violation_criterion"],
                row["execution_failures"],
                review.accepted(),
                repeated=row["repeat_execution"] is not None,
            )
            if row["gates"]["passed"]:
                batch_qualified.append(row)
            artifacts.save(
                f"conformance/candidate_{row['id'].lower()}.json",
                {
                    **review.model_dump(),
                    "history": review_history[row["local_id"]],
                    "review_count": len(review_history[row["local_id"]]),
                    "execution": row["execution"],
                    "execution_artifact": (
                        f"execution/candidate_{row['id'].lower()}_buggy.json"
                    ),
                },
            )
        remaining_after_batch = (
            initial_candidates_remaining
            - (candidate_limit if consuming_initial_budget else 0)
            + additional_candidates_remaining
            - (0 if consuming_initial_budget else candidate_limit)
        )
        logger.table(
            f"Contract qualification · batch {synthesis_batch}",
            ["Candidate", "Runtime", "Conformance", "Decision", "Failures"],
            [
                [
                    row["id"],
                    "PASS" if not row["execution_failures"] else "FAIL",
                    (
                        "SKIPPED"
                        if row["conformance_status"] == "NOT_RUN_RUNTIME_FAILURE"
                        else (
                            "PASS"
                            if row["conformance_status"] == "QUALIFIED"
                            else "FAIL"
                        )
                    ),
                    (
                        "QUALIFIED"
                        if row in batch_qualified
                        else (
                            "BLOCKED"
                            if _candidate_failure_owner(row) in NON_RECOVERABLE_EC_OWNERS
                            else (
                                "RETRY"
                                if remaining_after_batch > 0
                                and synthesis_batch < settings.max_contract_attempts
                                else "REJECTED"
                            )
                        )
                    ),
                    ", ".join(row["failures"]) or "none",
                ]
                for row in batch_rows
            ],
        )
        candidate_rows.extend(batch_rows)
        if consuming_initial_budget:
            # The offered ceiling consumes budget even when the model
            # deliberately returns fewer useful candidates. This preserves the
            # protocol rule that candidate_limit is a maximum, not a quota.
            initial_candidates_remaining -= candidate_limit
        else:
            additional_candidates_remaining -= candidate_limit
        remaining_candidate_budget = (
            initial_candidates_remaining + additional_candidates_remaining
        )
        if batch_qualified:
            qualified.extend(batch_qualified)
            semantic_classes = _semantic_class_count(qualified, duplicate_pairs)
            if (
                settings.candidate_policy == "stop_after_qualified_batch"
                or semantic_classes >= settings.target_semantic_classes
                or remaining_candidate_budget <= 0
                or synthesis_batch >= settings.max_contract_attempts
            ):
                break
            contract_feedback = [
                {
                    "candidate_id": ",".join(row["id"] for row in batch_qualified),
                    "failure_owner": "DIVERSITY_SEARCH",
                    "failures": [],
                    "execution": {},
                    "conformance": {},
                    "gates": {},
                    "required_action": (
                        "The existing executable contracts qualify but occupy only "
                        f"{semantic_classes} semantic class(es). Produce another independently "
                        "constructed, evidence-equivalent realization if one is useful. Do not "
                        "strengthen or reinterpret the frozen obligation merely to create diversity."
                    ),
                }
            ]
            logger.contract_recovery(
                synthesis_batch=synthesis_batch,
                feedback=contract_feedback,
                remaining_budget=remaining_candidate_budget,
            )
            continue
        contract_feedback = [_contract_recovery_feedback(row) for row in batch_rows]
        feedback_by_id = {item["candidate_id"]: item for item in contract_feedback}
        for row in batch_rows:
            row["failure_owner"] = feedback_by_id[row["id"]]["failure_owner"]
        batch_owners = sorted({item["failure_owner"] for item in contract_feedback})
        logger.contract_recovery(
            synthesis_batch=synthesis_batch,
            feedback=contract_feedback,
            remaining_budget=remaining_candidate_budget,
        )
        if any(owner in NON_RECOVERABLE_EC_OWNERS for owner in batch_owners):
            logger.warning(
                f"[contracts] batch {synthesis_batch} failure belongs to "
                f"{', '.join(batch_owners)}; model synthesis retry suppressed"
            )
            break
        failure_signature = tuple(sorted(
            (_contract_failure_signature(row) for row in batch_rows), key=repr
        ))
        if failure_signature == previous_failure_signature:
            repeated_failure_branch_stopped = True
            logger.warning(
                f"[contracts] batch {synthesis_batch} repeated the prior concrete failure "
                "signature (owner, runtime evidence, and semantic findings); "
                "stopping this no-progress branch without escalating reasoning"
            )
            artifacts.events.emit(
                "contract_recovery_stopped", stage="executable_contract_synthesis",
                reason="REPEATED_FAILURE_NO_PROGRESS", synthesis_batch=synthesis_batch,
            )
            break
        previous_failure_signature = failure_signature
        if remaining_candidate_budget > 0 and synthesis_batch < settings.max_contract_attempts:
            logger.warning(
                f"[contracts] batch {synthesis_batch} produced no qualified candidate; "
                f"requesting another failure-directed batch of at most "
                f"{min(EXECUTABLE_CONTRACT_BATCH_LIMIT, remaining_candidate_budget)} "
                "candidate(s) within the configured budget"
            )
    if not qualified:
        status = {
            "status": "NO_QUALIFIED_REPAIR_CONTRACT",
            "instance_id": task.instance_id,
            "failure_owner": _contract_failure_owner(candidate_rows),
            "contract_synthesis_batches": contract_synthesis_batches,
            "recovery": {
                "initial_candidate_cap": settings.contract_candidates,
                "additional_candidate_budget": settings.additional_candidate_budget,
                "max_synthesis_batches": settings.max_contract_attempts,
                "extra_conformance_recovery_used": False,
                "same_failure_branch_stopped": repeated_failure_branch_stopped,
            },
            "nlc_review": nlc_review_receipt,
            "candidates": [
                {
                    "id": row["id"],
                    "failure_owner": row.get("failure_owner", _candidate_failure_owner(row)),
                    "failures": row["failures"],
                    "gates": row.get("gates"),
                }
                for row in candidate_rows
            ],
        }
        artifacts.save("summary.json", status)
        artifacts.save("status.json", status)
        write_failure_report(artifacts.root, task.instance_id, status)
        logger.table(
            "Qualification",
            [
                "Candidate",
                "Observed",
                "Operation",
                "Assertions",
                "Runtime gate",
                "Conformance gate",
                "Owner",
                "Failures",
            ],
            [
                [
                    row["id"],
                    row["violation_kind"] or row["execution"]["outcome"],
                    row["execution"]["contracted_operation_reached"],
                    row["execution"]["runtime_assertions_exercised"],
                    "PASS" if not row["execution_failures"] else "FAIL",
                    (
                        "SKIPPED"
                        if row["conformance_status"] == "NOT_RUN_RUNTIME_FAILURE"
                        else (
                            "PASS"
                            if row["conformance_status"] == "QUALIFIED"
                            else "FAIL"
                        )
                    ),
                    row.get("failure_owner", _candidate_failure_owner(row)),
                    ", ".join(row["failures"]),
                ]
                for row in candidate_rows
            ],
        )
        return status
    primary = min(qualified, key=lambda row: row["id"])
    for row in candidate_rows:
        row["selection_role"] = "PRIMARY" if row is primary else "ALTERNATE_DIAGNOSTIC"
    logger.table(
        "Qualification",
        [
            "Candidate",
            "Observed",
            "Operation",
            "Assertions",
            "Runtime gate",
            "Conformance gate",
            "Conformance",
            "Role",
        ],
        [
            [
                row["id"],
                row["violation_kind"] or row["execution"]["outcome"],
                row["execution"]["contracted_operation_reached"],
                row["execution"]["runtime_assertions_exercised"],
                "PASS" if not row["execution_failures"] else "FAIL",
                (
                    "SKIPPED"
                    if row["conformance_status"] == "NOT_RUN_RUNTIME_FAILURE"
                    else (
                        "PASS"
                        if row["conformance_status"] == "QUALIFIED"
                        else "FAIL"
                    )
                ),
                row["conformance_status"],
                row["selection_role"],
            ]
            for row in candidate_rows
        ],
    )
    accounting = model_accounting(stages)
    selection_receipt = artifacts.root / "localization" / "selection.json"
    localization_rounds = (
        json.loads(selection_receipt.read_text(encoding="utf-8"))["rounds_used"]
        if selection_receipt.exists()
        else 1
    )
    body = {
        "version": "contractfix-python/3",
        "task": task.model_dump(),
        "repo_sha256": repo_sha,
        "executor": executor_identity,
        "generator": stages.identity,
        "workflow": settings.model_dump(),
        "model_accounting": accounting,
        "localization": localization.model_dump(),
        "contracted_operation": location.target(include_accessor=True),
        "allowed_edit_paths": sorted({index.locations[item].file for item in localization.repair_locations}),
        "obligation": obligation,
        "candidates": candidate_rows,
        "primary_candidate_id": primary["id"],
        "qualification": {
            "state": "PRE_GOLD_QUALIFIED",
            "gold_consulted": False,
            "nlc_review": nlc_review_receipt,
            "nlc_review_ec_disagreement": not final_nlc_review["accepted"],
            "disagreement_label": (
                "NLC_REVIEW_EC_DISAGREEMENT"
                if not final_nlc_review["accepted"]
                else None
            ),
            "gate_policy": "categorical_runtime_and_semantic_conformance",
            "duplicate_candidates": bool(duplicate_ids),
            "candidate_diversity": {
                "duplicate_candidate_ids": sorted(duplicate_ids),
                "duplicate_pairs": duplicate_pairs,
            },
            "localization_rounds": localization_rounds,
            "contract_synthesis_batches": contract_synthesis_batches,
            "candidates_generated": len(candidate_rows),
        },
    }
    frozen = {**body, "sha256": digest(body)}
    artifacts.save("frozen.json", frozen)
    artifacts.events.emit(
        "contract_frozen", stage="freeze", sha256=frozen["sha256"], primary_candidate_id=primary["id"]
    )
    artifacts.save("frozen/contract.json", frozen)
    for row in candidate_rows:
        name = (
            "primary_executable_contract.py"
            if row is primary
            else f"alternate_executable_contract_{row['id'].lower()}.py"
        )
        (artifacts.root / "frozen" / name).write_text(row["source"], encoding="utf-8")
    status = {
        "status": "FROZEN",
        "qualification": "PRE_GOLD_QUALIFIED",
        "qualified_for_enforcement": True,
        "instance_id": task.instance_id,
        "primary_candidate_id": primary["id"],
        "nlc_review": nlc_review_receipt,
        "nlc_review_outcome": nlc_review_receipt["outcome"],
        "nlc_review_ec_disagreement": not final_nlc_review["accepted"],
        "disagreement_label": (
            "NLC_REVIEW_EC_DISAGREEMENT"
            if not final_nlc_review["accepted"]
            else None
        ),
        "frozen_sha256": frozen["sha256"],
        "model_accounting": accounting,
        "localization_rounds": localization_rounds,
        "contract_synthesis_batches": contract_synthesis_batches,
        "candidates_generated": len(candidate_rows),
    }
    artifacts.save("summary.json", status)
    artifacts.save("status.json", status)
    write_report(artifacts.root, frozen)
    logger.panel(
        "Frozen",
        f"**Primary:** candidate {primary['id']}\n\n**SHA-256:** `{frozen['sha256']}`\n\n"
        f"Report: `{artifacts.root / 'report.md'}`",
        style="green",
    )
    return status


def repair(
    frozen_dir: Path,
    output: Path,
    executor: Executor,
    stages: Stages,
    mode: str = "enforce",
) -> dict:
    """Compatibility export: repair now consumes sealed guidance, not only ECs."""
    from .repair import repair as generate_and_select
    return generate_and_select(frozen_dir, output, executor, stages, mode)
