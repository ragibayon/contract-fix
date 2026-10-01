"""Small stage-specific schemas; model proposals never constitute validation receipts."""

from __future__ import annotations

from typing import Literal
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from ...task import Task  # noqa: F401 - compatibility export for Python callers


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContextRequest(StrictModel):
    kind: Literal["expand", "callers", "callees", "search"]
    target: str = Field(min_length=1, max_length=200)
    rationale: str = Field(
        default="",
        max_length=500,
        description=(
            "Optional explanation of the unresolved localization decision and why this "
            "request can resolve it. Omit unless the workflow requests rationales."
        ),
    )


class Localization(StrictModel):
    # Keep both localization modes explicit in the provider-visible schema.
    # Defaults made these fields optional to tool providers even though the
    # model validator below requires a complete context-request or final state.
    repair_locations: list[str] = Field(max_length=8)
    contracted_operation_locations: list[str] = Field(
        max_length=4,
        validation_alias=AliasChoices("contracted_operation_locations", "observation_locations"),
    )
    rationale: str = Field(
        default="",
        max_length=2000,
        description=(
            "Optional evidence-linked explanation: state the reported invocation/failure, "
            "identify the supplied location IDs and visible source facts supporting each "
            "repair/contracted-operation choice, and preserve any remaining uncertainty. "
            "Omit unless the workflow requests rationales."
        ),
    )
    needs_more_context: bool
    context_requests: list[ContextRequest] = Field(max_length=6)

    @model_validator(mode="after")
    def context_request_or_final_locations(self):
        if self.needs_more_context:
            if not self.context_requests:
                raise ValueError("needs_more_context requires context_requests")
            if self.repair_locations or self.contracted_operation_locations:
                raise ValueError("do not select locations while requesting more context")
        elif not self.repair_locations or not self.contracted_operation_locations:
            raise ValueError("final localization requires repair and contracted-operation locations")
        elif self.context_requests:
            raise ValueError("final localization must not include context_requests")
        return self


class FinalLocalization(StrictModel):
    """Choose from supplied context at the last round or explicitly abstain."""

    repair_locations: list[str] = Field(max_length=8)
    contracted_operation_locations: list[str] = Field(max_length=4)
    abstain: bool
    rationale: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def selection_or_abstention(self):
        if self.abstain:
            if self.repair_locations or self.contracted_operation_locations:
                raise ValueError("abstention must not select locations")
        elif not self.repair_locations or not self.contracted_operation_locations:
            raise ValueError("final selection requires repair and callable operation locations")
        return self


class EvidenceSupport(StrictModel):
    # ``span_id`` is the contract-first interface. ``id``/``quote`` remain
    # readable for legacy prompt packs and historical artifacts.
    span_id: str | None = None
    id: str | None = None
    supports: Literal["precondition", "normal_postcondition", "exceptional_postcondition"]
    quote: str | None = Field(default=None, min_length=8, max_length=4000)

    @model_validator(mode="after")
    def require_one_provenance_form(self):
        canonical = bool(self.span_id)
        legacy = bool(self.id and self.quote)
        if canonical == legacy:
            raise ValueError("select exactly one of span_id or legacy id+quote")
        return self

    @field_validator("supports", mode="before")
    @classmethod
    def accept_pre_terminology_roles(cls, value: str) -> str:
        return {
            "trigger": "precondition",
            "expected_behavior": "normal_postcondition",
        }.get(value, value)


class PreconditionClause(StrictModel):
    """One atomic precondition clause with explicit canonical provenance."""

    condition: str = Field(min_length=1, max_length=500)
    span_ids: list[str] = Field(min_length=1, max_length=6)

    @field_validator("span_ids")
    @classmethod
    def unique_span_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("precondition-clause span_ids must be unique")
        return value


class ViolationCriterion(StrictModel):
    """Host-derived criterion for recognizing a dynamic contract violation."""

    kind: Literal["reported_exception", "assertion"]
    exception_type: str | None = None
    description: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def exception_requires_type(self):
        if self.kind == "reported_exception" and not self.exception_type:
            raise ValueError("reported_exception requires exception_type")
        return self


class RepairObligation(StrictModel):
    role: Literal["REPAIR"] = "REPAIR"
    contracted_operation_id: str = Field(
        validation_alias=AliasChoices("contracted_operation_id", "observation_id")
    )
    precondition: str = Field(
        min_length=1,
        max_length=1000,
        validation_alias=AliasChoices("precondition", "trigger"),
    )
    normal_postcondition: str | None = Field(
        default=None,
        min_length=1,
        max_length=1500,
        validation_alias=AliasChoices("normal_postcondition", "must_hold"),
        description=(
            "Required behavior on successful return only; describe result/state and "
            "put required or prohibited exceptions in exceptional_postcondition."
        ),
    )
    exceptional_postcondition: str | None = Field(
        default=None,
        min_length=1,
        max_length=1500,
        description=(
            "Required or prohibited exceptional termination, or null when evidence "
            "supports no exception-specific contract."
        ),
    )
    evidence: list[EvidenceSupport] = Field(min_length=2, max_length=8)
    precondition_clauses: list[PreconditionClause] = Field(
        default_factory=list,
        max_length=8,
        validation_alias=AliasChoices("precondition_clauses", "trigger_conditions"),
        description=(
            "Atomic precondition clauses in order. Under contract-first-v3, precondition "
            "must equal these clauses joined by ' AND ', and every clause must select "
            "the canonical precondition-evidence spans that support it."
        ),
    )
    rationale: str = Field(
        default="",
        max_length=1800,
        description=(
            "Optional explanation of which supplied context establishes the precondition and evidence "
            "establishes the normal postcondition, and why together they support exactly "
            "this minimal caller-visible contract. Omit unless the workflow requests rationales."
        ),
    )
    repair_hypothesis: str = Field(default="", max_length=1500)

    @model_validator(mode="after")
    def require_both_evidence_roles(self):
        roles = {item.supports for item in self.evidence}
        postcondition_roles = {
            "normal_postcondition",
            "exceptional_postcondition",
        }
        if "precondition" not in roles or not roles.intersection(postcondition_roles):
            raise ValueError(
                "REPAIR obligation requires precondition evidence and at least one normal "
                "or exceptional postcondition evidence item"
            )
        if not self.normal_postcondition and not self.exceptional_postcondition:
            raise ValueError("REPAIR obligation requires a normal or exceptional postcondition")
        return self


class RepairObligations(StrictModel):
    obligations: list[RepairObligation] = Field(default_factory=list, max_length=1)
    unsupported_reason: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def require_explicit_abstention(self):
        if not self.obligations and not self.unsupported_reason.strip():
            raise ValueError("empty obligations require an unsupported_reason")
        return self


class CanonicalEvidenceSupport(StrictModel):
    """Unambiguous model-facing provenance for canonical-span profiles."""

    span_id: str = Field(
        min_length=1,
        max_length=100,
        description="One exact opaque identifier from the supplied evidence_spans map.",
    )
    supports: Literal["precondition", "normal_postcondition", "exceptional_postcondition"] = Field(
        description=("The single contract component supported by the canonical span.")
    )


class CanonicalRepairObligation(RepairObligation):
    """Active obligation schema that does not advertise legacy quote fields."""

    # These nullable fields are required in the model-facing schema.  Leaving
    # them optional caused tool-calling models to omit both keys even when
    # their reasoning selected a postcondition; the after-validator could
    # diagnose the omission but the provider kept reproducing the same shape.
    # Requiring explicit null in the provider-facing JSON schema makes the
    # normal/exceptional choice unambiguous without changing accepted semantics.
    normal_postcondition: str | None = Field(
        default=None,
        min_length=1,
        max_length=1500,
        description=(
            "Required behavior on successful return only; use null when no "
            "normal-return requirement is supported. Put required or prohibited "
            "exceptions in exceptional_postcondition."
        ),
    )
    exceptional_postcondition: str | None = Field(
        default=None,
        min_length=1,
        max_length=1500,
        description=(
            "Required or prohibited exceptional termination; use null when "
            "evidence supports no exception-specific contract."
        ),
    )
    evidence: list[CanonicalEvidenceSupport] = Field(min_length=2, max_length=8)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """Require explicit nullable postconditions in the provider-facing schema.

        Runtime validation retains the legacy default for stored artifacts and
        non-model callers. The live tool contract is stricter so providers must
        make the normal-versus-exceptional choice explicit.
        """
        schema = super().__get_pydantic_json_schema__(core_schema, handler)
        required = schema.setdefault("required", [])
        for field in ("normal_postcondition", "exceptional_postcondition"):
            if field not in required:
                required.append(field)
        return schema


class CanonicalRepairObligations(StrictModel):
    obligations: list[CanonicalRepairObligation] = Field(default_factory=list, max_length=1)
    unsupported_reason: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def require_explicit_abstention(self):
        if not self.obligations and not self.unsupported_reason.strip():
            raise ValueError("empty obligations require an unsupported_reason")
        return self


class RepairObligationClauseReview(StrictModel):
    """One evidence-to-NLC justification judgment without a confidence score."""

    clause_id: str = Field(pattern=r"^(PRE[1-9][0-9]*|POST_NORMAL|POST_EXCEPTION)$")
    clause_text: str = Field(min_length=1, max_length=1500)
    verdict: Literal["DIRECT", "DERIVED_VALID", "UNSUPPORTED", "CONTRADICTED"]
    supporting_evidence_ids: list[str] = Field(default_factory=list, max_length=8)
    program_facts: list[str] = Field(
        max_length=8,
        description=(
            "Concrete supplied program facts permitting the verdict. Always include this field: "
            "use [] for DIRECT, UNSUPPORTED, or CONTRADICTED when no program fact is needed; "
            "DERIVED_VALID requires one or more facts."
        ),
    )
    reason: str = Field(min_length=1, max_length=1800)

    @model_validator(mode="after")
    def require_auditable_justification(self):
        if self.verdict == "DIRECT" and not self.supporting_evidence_ids:
            raise ValueError("DIRECT requires at least one supporting evidence identifier")
        if self.verdict == "DERIVED_VALID":
            if not self.supporting_evidence_ids:
                raise ValueError("DERIVED_VALID requires supporting evidence identifiers")
            if not self.program_facts:
                raise ValueError("DERIVED_VALID requires the program facts permitting the inference")
        return self


class RepairObligationReview(StrictModel):
    reviews: list[RepairObligationClauseReview] = Field(min_length=1, max_length=10)
    missing_required_clauses: list[str] = Field(
        max_length=8,
        description=(
            "Material precondition or postcondition clauses required by admissible evidence "
            "but absent from the proposed NLC. Return an empty list only after checking "
            "coverage of the complete reported witness and required behavior."
        ),
    )

    def accepted(self) -> bool:
        return not self.missing_required_clauses and all(
            item.verdict in {"DIRECT", "DERIVED_VALID"} for item in self.reviews
        )


class ExecutableContractCandidate(StrictModel):
    id: str = Field(pattern=r"^[A-Z]$", max_length=1)
    source_lines: list[str] = Field(
        min_length=1,
        max_length=100,
        description=(
            "Complete standalone Python source as one physical line per array element. "
            "Do not use newline escape sequences inside an element. The joined source "
            "must define exactly one zero-argument contractfix_contract function with "
            "at least one runtime assert statement."
        ),
    )
    rationale: str = Field(
        default="",
        max_length=1200,
        description=(
            "Optionally map the grounded precondition to the contract witness and the normal or "
            "exceptional postcondition to each runtime contract assertion. Cite concrete "
            "repository symbols or evidence text, "
            "not opaque host IDs. Omit unless the workflow requests rationales."
        ),
    )

    @field_validator("source_lines")
    @classmethod
    def require_valid_executable_contract(cls, value: list[str]) -> list[str]:
        # Reject malformed generated Python inside structured generation so the
        # ordinary bounded schema-correction path can repair it before execution.
        from .executable_contracts import validate_executable_contract

        if any("\n" in line or "\r" in line for line in value):
            raise ValueError("source_lines entries must each contain one physical line")
        source = "\n".join(value) + "\n"
        if len(source) > 24000:
            raise ValueError("executable contract source exceeds 24000 characters")
        try:
            validate_executable_contract(source)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"invalid executable contract source: {exc}") from exc
        return value

    @property
    def source(self) -> str:
        """Materialize host-owned Python bytes from transport-safe physical lines."""
        return "\n".join(self.source_lines) + "\n"


EXECUTABLE_CONTRACT_BATCH_LIMIT = 4


class ExecutableContractCandidates(StrictModel):
    # Keep each provider-facing tool schema deliberately small. Larger search
    # budgets are expressed as recovery batches by WorkflowSettings instead of
    # making one deeply nested structured response harder for providers to
    # validate.
    candidates: list[ExecutableContractCandidate] = Field(
        min_length=1, max_length=EXECUTABLE_CONTRACT_BATCH_LIMIT
    )

    @model_validator(mode="after")
    def require_unique_ordered_ids(self):
        expected = [chr(ord("A") + index) for index in range(len(self.candidates))]
        if [item.id for item in self.candidates] != expected:
            raise ValueError("executable contract candidate IDs must be consecutive starting at A")
        return self


class ExecutableContractConformance(StrictModel):
    candidate_id: str = Field(pattern=r"^[A-Z]$", max_length=1)
    back_translation: str = Field(min_length=1, max_length=2000)
    missing_meaning: list[str] = Field(default_factory=list, max_length=8)
    added_meaning: list[str] = Field(default_factory=list, max_length=8)
    semantically_duplicates: list[str] = Field(default_factory=list, max_length=4)
    reason: str = Field(
        min_length=1,
        max_length=1500,
        description=(
            "Evidence-linked verdict explaining how the contract witness establishes the "
            "precondition and how each runtime contract assertion maps to, omits, or "
            "strengthens the grounded postcondition. Do not rely on opaque IDs or repair hypotheses."
        ),
    )

    def accepted(self) -> bool:
        return not self.missing_meaning and not self.added_meaning


class ExecutableContractConformanceBatch(StrictModel):
    reviews: list[ExecutableContractConformance] = Field(min_length=1, max_length=4)


class Clause(StrictModel):
    when: str = Field(min_length=1, max_length=2000)
    ensure: str = Field(min_length=1, max_length=2000)


class Edit(StrictModel):
    file: str
    old: str = Field(min_length=1, max_length=16000)
    new: str = Field(max_length=16000)


class WorkflowSettings(StrictModel):
    """Settings for the only active Python workflow and prompt contract."""

    # Campaign ceiling, not a required count. The workflow chunks this total
    # into provider-safe structured-response batches, and the authoring model
    # may return fewer useful candidates than any offered batch ceiling.
    contract_candidates: int = Field(default=4, ge=1)
    additional_candidate_budget: int = Field(default=0, ge=0)
    candidate_policy: Literal["stop_after_qualified_batch", "seek_semantic_diversity"] = (
        "stop_after_qualified_batch"
    )
    target_semantic_classes: int = Field(default=2, ge=1, le=4)
    equivalence_mode: Literal["observational_semantic", "hybrid_smt"] = "observational_semantic"
    max_obligation_attempts: int = Field(default=3, ge=1)
    # Bound the initial EC synthesis plus failure-directed recovery batches.
    max_contract_attempts: int = Field(default=2, ge=1)
    # Optional second-pass audit of a rejected semantic review. Disabled by
    # default because execution evidence remains the primary qualification
    # signal and each pass is a paid model call.
    conformance_adjudication_attempts: int = Field(default=0, ge=0, le=3)
    max_localization_rounds: int | None = Field(default=3, ge=1, le=50)
    localization_context_additions: int = Field(default=6, ge=1, le=12)
    context_strategy: Literal["compact", "adaptive", "causal_slice"] = "adaptive"
    evidence_mode: Literal["canonical_spans"] = "canonical_spans"
    prompt_profile: Literal["contract-first-v3"] = "contract-first-v3"
    example_profile: Literal["qualified-successes-v1"] = "qualified-successes-v1"
    stage_reasoning_tokens: dict[str, int] = Field(default_factory=dict)
    # Runtime materialization of the selected experiment policy. Authored
    # workflow files leave these empty; configs/policies is the source of truth.
    stage_reasoning_policy: dict[
        str, Literal["off", "low", "medium", "high"]
    ] = Field(
        default_factory=dict
    )
    stage_completion_limits: dict[str, int] = Field(default_factory=dict)
    stage_reasoning_caps: dict[str, int] = Field(default_factory=dict)
    stage_answer_tokens: dict[str, int] = Field(default_factory=dict)
    nlc_review_mode: Literal["advisory", "strict"] = "advisory"
    retrieval_limit: int = Field(default=12, ge=2, le=30)
    excerpt_chars: int = Field(default=700, ge=200, le=2400)
    stage_output_tokens: int = Field(default=2048, ge=256, le=8192)
    include_rationales: bool = False
    preservation: bool = False
    repeat_base: bool = True
    regression_commands: list[list[str]] = Field(default_factory=list)
    repository_check_mode: Literal["none", "configured", "auto_visible"] = "none"
    repository_validation_scope: Literal[
        "related_visible_tests", "related_and_package", "repository_suite"
    ] = "related_visible_tests"
    repository_test_file_limit: int = Field(default=3, ge=1, le=10)
    repository_broader_test_file_limit: int = Field(default=8, ge=1, le=30)
    repository_validation_failure_policy: Literal["ignore", "abstain"] = "ignore"

    # Repair settings for the selected workflow.
    repair_enabled: bool = False
    repair_variant: Literal["no-contract", "context", "enforce"] = "enforce"
    patch_nlc_review_enabled: bool = True
    require_ec_for_patch: bool = False
    max_patch_attempts: int = Field(default=3, ge=1, le=10)
    max_patch_model_calls: int | None = Field(default=None, ge=1, le=100)
    # Deterministic edit-realization retries for one semantic repair candidate.
    # These do not consume max_patch_attempts until materialization succeeds
    # or this bounded correction budget is exhausted.
    max_patch_materialization_attempts: int = Field(default=3, ge=1, le=5)
    max_patch_context_rounds: int = Field(default=3, ge=0, le=10)
    patch_context_additions: int = Field(default=6, ge=1, le=6)
    patch_context_chars: int = Field(default=32000, ge=4000, le=64000)
    allow_edit_scope_expansion: bool = True
    patch_candidate_policy: Literal[
        "first_eligible", "budgeted", "budgeted_ec_retry"
    ] = "first_eligible"
    ec_conflict_policy: Literal[
        "strict_abstain", "fallback_to_nlc_context", "fallback_after_nlc_review"
    ] = "strict_abstain"
    submit_best_attempt: bool = False
    max_ec_refinement_attempts: int = Field(default=0, ge=0, le=1)
    patch_missing_checks: Literal["abstain", "syntax_only"] = "syntax_only"
    patch_syntax_scope: Literal["changed_files", "touched_packages"] = "changed_files"

    @field_validator("regression_commands")
    @classmethod
    def validate_regression_commands(cls, value: list[list[str]]) -> list[list[str]]:
        for command in value:
            if not command or any(not item or "\x00" in item for item in command):
                raise ValueError("repository check commands must contain nonempty argv strings")
            joined = " ".join(command).lower()
            if any(
                token in joined
                for token in (
                    "fail_to_pass",
                    "pass_to_pass",
                    "test_patch",
                    "gold.patch",
                    "gold_patch",
                )
            ):
                raise ValueError("evaluator-only artifacts cannot be used for repair selection")
        return value

    @model_validator(mode="after")
    def validate_stage_controls(self):
        if self.ec_conflict_policy in {"fallback_to_nlc_context", "fallback_after_nlc_review"}:
            if self.repair_variant != "enforce":
                raise ValueError("EC-conflict fallback requires enforce repair variant")
        if self.patch_candidate_policy == "budgeted_ec_retry":
            if self.max_ec_refinement_attempts != 1:
                raise ValueError("budgeted_ec_retry requires one bounded EC refinement attempt")
            if self.repair_variant != "enforce":
                raise ValueError("budgeted_ec_retry requires enforce repair variant")
        elif self.max_ec_refinement_attempts:
            raise ValueError("EC refinement attempts require budgeted_ec_retry")
        policy_keys = set(self.stage_reasoning_policy)
        limit_keys = set(self.stage_completion_limits)
        if policy_keys != limit_keys:
            raise ValueError("stage reasoning policy and completion limits must define the same stages")
        if policy_keys and self.stage_reasoning_tokens:
            raise ValueError("stage reasoning policy cannot be combined with legacy stage reasoning tokens")
        if set(self.stage_reasoning_caps) - policy_keys:
            raise ValueError("reasoning caps require matching stage policies")
        if any(type(value) is not int or value < 1 for value in self.stage_reasoning_caps.values()):
            raise ValueError("reasoning caps must be positive integers")
        if set(self.stage_answer_tokens) - {"patch", "localize", "repair_obligations", "repair_obligation_review", "executable_contract_synthesis", "executable_contract_conformance"}:
            raise ValueError("unknown stage answer token key")
        if any(type(value) is not int or not 128 <= value <= 8192 for value in self.stage_answer_tokens.values()):
            raise ValueError("invalid stage answer allowance")
        if (
            self.candidate_policy == "seek_semantic_diversity"
            and self.target_semantic_classes < 2
        ):
            raise ValueError(
                "seek_semantic_diversity requires target_semantic_classes >= 2"
            )
        if self.repository_check_mode == "configured" and not self.regression_commands:
            raise ValueError("configured repository checks require regression_commands")
        if self.repository_check_mode == "auto_visible" and self.regression_commands:
            raise ValueError("auto_visible repository checks cannot also define regression_commands")
        if self.repository_check_mode == "none" and self.regression_commands:
            # Existing custom workflows used the command list as the opt-in.
            self.repository_check_mode = "configured"
        return self

    @field_validator("stage_reasoning_tokens")
    @classmethod
    def valid_stage_reasoning_tokens(cls, value: dict[str, int]) -> dict[str, int]:
        allowed = {
            "patch",
            "localize",
            "repair_obligations",
            "repair_obligation_review",
            "executable_contract_synthesis",
            "executable_contract_conformance",
        }
        if set(value) - allowed:
            raise ValueError("unknown stage reasoning token key")
        if any(type(tokens) is not int or not 1 <= tokens <= 65536 for tokens in value.values()):
            raise ValueError("stage reasoning tokens must be integers from 1 to 65536")
        return value

    @field_validator("stage_reasoning_policy")
    @classmethod
    def valid_stage_reasoning_policy(cls, value: dict[str, str]) -> dict[str, str]:
        allowed = {
            "patch",
            "localize",
            "repair_obligations",
            "repair_obligation_review",
            "executable_contract_synthesis",
            "executable_contract_synthesis_recovery",
            "executable_contract_conformance",
            "executable_contract_conformance_adjudication",
        }
        if set(value) - allowed:
            raise ValueError("unknown stage reasoning policy key")
        return value

    @field_validator("stage_completion_limits")
    @classmethod
    def valid_stage_completion_limits(cls, value: dict[str, int]) -> dict[str, int]:
        allowed = {
            "patch",
            "localize",
            "repair_obligations",
            "repair_obligation_review",
            "executable_contract_synthesis",
            "executable_contract_synthesis_recovery",
            "executable_contract_conformance",
            "executable_contract_conformance_adjudication",
        }
        if set(value) - allowed:
            raise ValueError("unknown stage completion limit key")
        if any(type(tokens) is not int or not 256 <= tokens <= 65536 for tokens in value.values()):
            raise ValueError("stage completion limits must be integers from 256 to 65536")
        return value
