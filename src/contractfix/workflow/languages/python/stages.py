"""Python prompt stages; deterministic host code owns transitions and admission."""
from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from typing import Protocol, TypeVar

from pydantic import BaseModel

from contractfix.config import LLMSettings
from contractfix.utils.artifacts import redact
from contractfix.utils.logger import logger
from .promptbook import PromptBook

T = TypeVar("T", bound=BaseModel)


def _rationale(response: BaseModel) -> str:
    """Extract concise, authoritative explanation fields from a stage answer."""
    value = response.model_dump()
    if isinstance(value.get("rationale"), str):
        return value["rationale"].strip()
    if isinstance(value.get("reason"), str):
        return value["reason"].strip()
    rows = []
    for key in ("obligations", "candidates", "reviews"):
        for item in value.get(key, []):
            reason = item.get("rationale") or item.get("reason")
            identity = item.get("id") or item.get("candidate_id")
            if reason:
                rows.append(f"- **{identity}:** {reason}" if identity else f"- {reason}")
    if rows:
        return "\n".join(rows)
    unsupported = value.get("unsupported_reason")
    return str(unsupported).strip() if unsupported else ""


def _thinking_status(thinking: str, usage: dict) -> str:
    """Distinguish hidden reasoning from a provider-reported zero-token call."""
    if thinking:
        return "TEXT_CAPTURED"
    tokens = usage.get("reasoning_tokens")
    if tokens is not None and not usage.get("missing_reasoning_token_calls", 0):
        return "ZERO_TOKENS_REPORTED" if tokens == 0 else "TOKENS_WITHOUT_TEXT"
    return "NOT_REPORTED"


class Stages(Protocol):
    @property
    def identity(self) -> dict: ...
    def generate(self, stage: str, schema: type[T], packet: dict) -> T: ...


class LangChainStages:
    """A bounded staged agent using one frozen prompt/skill/memory release."""

    def __init__(self, settings: LLMSettings, prompts: Path | None = None,
                 *, prompt_profile: str = "contract-first-v3",
        example_profile: str = "qualified-successes-v1",
                 stage_reasoning_tokens: dict[str, int] | None = None,
                 stage_reasoning_policy: dict[str, str] | None = None,
                 stage_completion_limits: dict[str, int] | None = None,
                 stage_reasoning_caps: dict[str, int] | None = None,
                 stage_answer_tokens: dict[str, int] | None = None):
        self.settings = settings
        self.prompts = PromptBook(
            prompts,
            profile=prompt_profile,
            example_profile=example_profile,
        )
        self.artifacts = None
        self.calls = 0
        self.call_sequence_offset = 0
        self.cumulative_sdk_calls = 0
        self.cumulative_cost_usd = 0.0
        self.missing_cost_calls = 0
        self.cost_assurances: set[str] = set()
        self.cumulative_input_tokens = 0
        self.cumulative_output_tokens = 0
        self.cumulative_answer_tokens = 0
        self.missing_answer_token_calls = 0
        self.cumulative_reasoning_tokens = 0
        self.missing_reasoning_token_calls = 0
        self.cumulative_elapsed_seconds = 0.0
        self.last_finish_reason = None
        self.stage_reasoning_tokens = dict(stage_reasoning_tokens or {})
        self.stage_reasoning_policy = dict(stage_reasoning_policy or {})
        self.stage_completion_limits = dict(stage_completion_limits or {})
        self.stage_reasoning_caps = dict(stage_reasoning_caps or {})
        self.stage_answer_tokens = dict(stage_answer_tokens or {})
        self.stage_reasoning_policy.setdefault("patch", "low")
        self.stage_completion_limits.setdefault("patch", 8192)
        self.model_artifacts: list[dict] = []
        self.stage_accounting: dict[str, dict] = {}

    @property
    def identity(self) -> dict:
        return {"kind": "live_langchain", "settings": self.settings.public_dict(),
                "prompt_profile": self.prompts.profile,
                "example_profile": self.prompts.example_profile,
                "stage_reasoning_policy": self.stage_reasoning_policy,
                "stage_completion_limits": self.stage_completion_limits,
                "prompts_sha256": self.prompts.sha256,
                "prompt_policy": self.prompts.policy}

    @staticmethod
    def _policy_stage(stage: str, packet: dict) -> str:
        if stage == "patch_nlc_conformance":
            # Reuse the patch inference budget; this semantic gate is part of repair,
            # not a separate model-selection hyperparameter.
            return "patch"
        if stage == "executable_contract_synthesis" and packet.get("synthesis_batch", 1) > 1:
            return "executable_contract_synthesis_recovery"
        if (
            stage == "executable_contract_conformance"
            and packet.get("review_mode") == "adjudication"
        ):
            return "executable_contract_conformance_adjudication"
        return stage

    def _account_usage(
        self,
        usage: dict,
        *,
        policy_stage: str,
        thinking_enabled: bool,
    ) -> tuple[dict, dict, int | None, str | None]:
        self.cumulative_sdk_calls += int(usage.get("call_count") or 0)
        call_cost = usage.get("reported_cost_usd")
        self.missing_cost_calls += usage.get(
            "missing_cost_calls", 0 if call_cost is not None else 1
        )
        if call_cost is not None:
            self.cumulative_cost_usd += call_cost
        if usage.get("cost_assurance"):
            self.cost_assurances.add(str(usage["cost_assurance"]))
        token_usage = dict(usage.get("token_usage") or {})
        thinking_tokens = usage.get("reasoning_tokens")
        self.missing_reasoning_token_calls += usage.get(
            "missing_reasoning_token_calls",
            0 if thinking_tokens is not None else 1,
        )
        calls = usage.get("calls") or []
        finish_reason = calls[-1].get("finish_reason") if calls else None
        self.last_finish_reason = finish_reason
        self.cumulative_input_tokens += token_usage.get("input_tokens") or 0
        self.cumulative_output_tokens += token_usage.get("output_tokens") or 0
        self.cumulative_reasoning_tokens += thinking_tokens or 0
        answer_tokens = token_usage.get("answer_tokens")
        output_tokens = token_usage.get("output_tokens")
        if answer_tokens is None and output_tokens is not None:
            if thinking_tokens is not None:
                answer_tokens = output_tokens - thinking_tokens if output_tokens >= thinking_tokens else None
                if answer_tokens is None:
                    token_usage["accounting_warning"] = "reasoning_tokens_exceed_output_tokens"
            elif not thinking_enabled:
                answer_tokens = output_tokens
        token_usage["answer_tokens"] = answer_tokens
        usage["token_usage"] = token_usage
        if answer_tokens is None:
            self.missing_answer_token_calls += 1
        else:
            self.cumulative_answer_tokens += answer_tokens
        stage_totals = self.stage_accounting.setdefault(
            policy_stage,
            {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "answer_tokens": 0,
                "missing_reasoning_token_calls": 0,
                "missing_answer_token_calls": 0,
                "finish_reasons": [],
            },
        )
        # Policy metadata may be recorded before the first usage event. Merge
        # accounting counters defensively so a failed or retried call cannot
        # turn an observability receipt into a workflow failure.
        for key, default in {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "answer_tokens": 0,
            "missing_reasoning_token_calls": 0,
            "missing_answer_token_calls": 0,
            "finish_reasons": [],
        }.items():
            stage_totals.setdefault(key, default.copy() if isinstance(default, list) else default)
        stage_totals["calls"] += int(usage.get("call_count") or 0)
        stage_totals["input_tokens"] += token_usage.get("input_tokens") or 0
        stage_totals["output_tokens"] += output_tokens or 0
        stage_totals["reasoning_tokens"] += thinking_tokens or 0
        stage_totals["answer_tokens"] += answer_tokens or 0
        stage_totals["missing_reasoning_token_calls"] += usage.get(
            "missing_reasoning_token_calls", 0 if thinking_tokens is not None else 1
        )
        stage_totals["missing_answer_token_calls"] += int(answer_tokens is None)
        stage_totals["finish_reasons"].append(finish_reason)
        accounting = {
            "call_reported_cost_usd": call_cost,
            "process_cumulative_reported_cost_usd": self.cumulative_cost_usd,
            "process_missing_cost_calls": self.missing_cost_calls,
            "assurance": (
                next(iter(self.cost_assurances))
                if len(self.cost_assurances) == 1
                else "mixed_or_unreported"
            ),
        }
        return accounting, token_usage, thinking_tokens, finish_reason

    def _metric_suffix(
        self,
        token_usage: dict,
        thinking_tokens: int | None,
        finish_reason: str | None,
        call_cost: float | None,
    ) -> str:
        def shown(value: object) -> str:
            return "unreported" if value is None else str(value)

        cost = "unreported" if call_cost is None else f"${call_cost:.8f}"
        return (
            f"input={shown(token_usage.get('input_tokens'))}, "
            f"output={shown(token_usage.get('output_tokens'))}, "
            f"answer={shown(token_usage.get('answer_tokens'))}, "
            f"thinking={shown(thinking_tokens)} | finish={shown(finish_reason)} | "
            f"cost={cost} | cumulative=${self.cumulative_cost_usd:.8f}"
        )

    def generate(self, stage: str, schema: type[T], packet: dict) -> T:
        from contractfix.llm.structured import generate_structured
        self.calls += 1
        call = self.calls + self.call_sequence_offset
        request = self.prompts.assemble(stage, packet)
        answer = self.stage_answer_tokens.get(stage, self.prompts.policy["stages"][stage]["answer_tokens"])
        policy_stage = self._policy_stage(stage, packet)
        reasoning_receipt = None
        if policy_stage in self.stage_reasoning_policy:
            from contractfix.llm.reasoning import resolve_stage_reasoning

            completion_limit = self.stage_completion_limits.get(
                policy_stage,
                self.stage_completion_limits.get(stage, self.settings.max_tokens),
            )
            resolution = resolve_stage_reasoning(
                self.settings,
                intent=self.stage_reasoning_policy[policy_stage],
                answer_tokens=answer,
                completion_limit=completion_limit,
                reasoning_max_tokens=self.stage_reasoning_caps.get(policy_stage),
            )
            settings = resolution.settings
            reasoning_receipt = {**resolution.receipt, "policy_stage": policy_stage}
        else:
            settings = self.settings.for_stage(answer, self.stage_reasoning_tokens.get(stage))
        capability_report = None
        if settings.provider == "openrouter":
            from contractfix.evaluation.preflight import openrouter_capability_preflight

            capability_report = openrouter_capability_preflight(settings)
            if reasoning_receipt is not None:
                reasoning_receipt["provider_assurance"] = (
                    "live_endpoint_metadata_parameter_names; "
                    "nested_values_enforced_by_request"
                )
        from contractfix.llm.reasoning import request_controls_for
        request.update(
            request_controls=request_controls_for(settings),
            schema=schema.model_json_schema(),
            answer_allowance=answer,
            requested_completion_limit=settings.max_tokens,
            reasoning_allowance=settings.reasoning_reserve_tokens,
            reasoning_policy=reasoning_receipt,
            provider_capability_preflight=capability_report,
            model_configuration=settings.public_dict(),
        )
        if settings.provider == "openrouter" and settings.structured_method == "tool":
            from contractfix.llm.tool_schema import openrouter_tool

            request["transport_tool"] = openrouter_tool(schema, settings.model)
        prompt = request["prompt"]
        system = request["system"]
        request_path = f"model/call_{call:03d}_{stage}_request.json"
        response_path = f"model/call_{call:03d}_{stage}_response.json"
        if self.artifacts:
            if capability_report is not None:
                self.artifacts.save(
                    f"provider/openrouter_capabilities_{policy_stage}.json",
                    capability_report,
                )
            self.artifacts.save(request_path, request)
            self.model_artifacts.append(
                {
                    "call": call,
                    "stage": stage,
                    "status": "started",
                    "request": request_path,
                    "request_bytes": len(
                        json.dumps(request, ensure_ascii=False).encode()
                    ),
                    "response": None,
                    "thinking": None,
                    "rationale": None,
                }
            )
            self.artifacts.save("model/artifacts.json", self.model_artifacts)
        policy_summary = ""
        if reasoning_receipt:
            policy_summary = (
                f" | reasoning={reasoning_receipt['intent']}→{reasoning_receipt['resolved']}"
                f" ({reasoning_receipt['mapping_assurance']})"
                f" completion={reasoning_receipt['completion_limit']}"
            )
        logger.info(f"[model {call:03d}] {stage} started{policy_summary}")
        if os.getenv("CONTRACTFIX_SHOW_MODEL_IO", "summary").lower() == "full":
            logger.debug(
                f"[model {call:03d}] {stage} rendered prompt\n"
                f"SYSTEM:\n{system}\n\nUSER:\n{prompt}"
            )
        started = perf_counter()
        try:
            with logger.activity(
                f"[model {call:03d}] {stage}",
                detail="model/provider generation in progress",
            ):
                response, usage = generate_structured(
                    prompt,
                    schema,
                    settings,
                    system_prompt=system,
                    correction_text=self.prompts.texts["schema_retry"],
                    demonstration_messages=request["demonstrations"],
                    events=self.artifacts.events if self.artifacts else None,
                )
        except Exception as exc:
            from contractfix.llm.structured import _status_code

            elapsed = perf_counter() - started
            self.cumulative_elapsed_seconds += elapsed
            failure_usage = dict(getattr(exc, "metrics", {}) or {})
            failure_thinking = str(failure_usage.pop("thinking", "") or "").strip()
            suffix = ""
            failure_path = f"model/call_{call:03d}_{stage}_failure.json"
            error_details = {
                "error_type": type(exc).__name__, "error": str(exc)[:16000],
                "http_status": _status_code(exc),
                "provider_error_body": json.loads(json.dumps(
                    getattr(exc, "body", None), default=str,
                )),
            }
            if failure_usage:
                accounting, token_usage, thinking_tokens, finish_reason = (
                    self._account_usage(
                        failure_usage,
                        policy_stage=policy_stage,
                        thinking_enabled=settings.thinking,
                    )
                )
                suffix = " | " + self._metric_suffix(
                    token_usage,
                    thinking_tokens,
                    finish_reason,
                    accounting["call_reported_cost_usd"],
                )
                if failure_thinking:
                    thinking_path = f"model/call_{call:03d}_{stage}_thinking.md"
                    safe_thinking = str(redact(failure_thinking))
                    if self.artifacts:
                        (self.artifacts.root / thinking_path).write_text(
                            safe_thinking + "\n", encoding="utf-8"
                        )
                    logger.panel(
                        f"Model {call:03d} thinking · {stage}",
                        safe_thinking,
                        style="magenta",
                    )
                if self.artifacts:
                    self.artifacts.save(
                        failure_path,
                        {
                            **error_details,
                            "metrics": failure_usage,
                            "accounting": accounting,
                            "thinking_artifact": thinking_path
                            if failure_thinking
                            else None,
                        },
                    )
                    self.artifacts.save(
                        "model/costs.json",
                        {**accounting, "stage_calls_attempted": call, "currency": "USD"},
                    )
            if self.artifacts:
                if not failure_usage:
                    self.artifacts.save(failure_path, {
                        **error_details,
                        "metrics": None, "accounting": None,
                    })
                self.model_artifacts[-1]["status"] = "failed"
                self.model_artifacts[-1]["error_type"] = type(exc).__name__
                self.model_artifacts[-1]["failure"] = failure_path
                self.model_artifacts[-1]["thinking"] = (
                    thinking_path if failure_thinking else None
                )
                self.artifacts.save("model/artifacts.json", self.model_artifacts)
            logger.error(
                f"[model {call:03d}] {stage} failed after "
                f"{elapsed:.2f}s{suffix} | {type(exc).__name__}: {exc}"
            )
            raise
        elapsed = perf_counter() - started
        self.cumulative_elapsed_seconds += elapsed
        thinking = str(usage.pop("thinking", "") or "").strip()
        thinking_status = _thinking_status(thinking, usage)
        thinking_path = f"model/call_{call:03d}_{stage}_thinking.md"
        if thinking:
            safe_thinking = str(redact(thinking))
            if self.artifacts:
                destination = self.artifacts.root / thinking_path
                destination.write_text(safe_thinking + "\n", encoding="utf-8")
            logger.panel(f"Model {call:03d} thinking · {stage}", safe_thinking,
                         style="magenta")
        else:
            detail = {
                "ZERO_TOKENS_REPORTED": "provider reported 0 reasoning tokens; no thinking text",
                "TOKENS_WITHOUT_TEXT": (
                    f"provider reported {usage['reasoning_tokens']} reasoning tokens "
                    "but did not expose thinking text"
                ),
                "NOT_REPORTED": "provider did not report thinking text or token count",
            }[thinking_status]
            logger.muted(f"[model {call:03d}] {stage} {detail}")
        # Conformance has a dedicated clause-level renderer in the controller;
        # avoid printing the same review reasons twice as an undifferentiated block.
        rationale = "" if stage == "executable_contract_conformance" else _rationale(response)
        rationale_path = f"model/call_{call:03d}_{stage}_rationale.md"
        if rationale:
            safe_rationale = str(redact(rationale))
            if self.artifacts:
                destination = self.artifacts.root / rationale_path
                destination.write_text(safe_rationale + "\n", encoding="utf-8")
            logger.panel(f"Model {call:03d} rationale · {stage}", safe_rationale,
                         style="cyan")
        accounting, token_usage, thinking_tokens, finish_reason = self._account_usage(
            usage,
            policy_stage=policy_stage,
            thinking_enabled=settings.thinking,
        )
        self.stage_accounting[policy_stage].update(
            {
                "requested_effort": (
                    reasoning_receipt.get("intent")
                    if reasoning_receipt
                    else settings.reasoning_effort
                ),
                "resolved_effort": (
                    reasoning_receipt.get("resolved")
                    if reasoning_receipt
                    else settings.reasoning_effort
                ),
                "completion_limit": settings.max_tokens,
            }
        )
        if self.artifacts:
            self.artifacts.save(response_path,
                                 {"proposal": response.model_dump(), "metrics": usage,
                                 "accounting": accounting,
                                 "thinking_status": thinking_status,
                                 "thinking_artifact": thinking_path if thinking else None,
                                 "rationale_artifact": rationale_path if rationale else None})
            self.artifacts.save("model/costs.json", {
                **accounting,
                "completed_stage_calls": call,
                "currency": "USD",
            })
            self.model_artifacts[-1].update(
                {
                    "status": "completed",
                    "response": response_path,
                    "thinking": thinking_path if thinking else None,
                    "rationale": rationale_path if rationale else None,
                }
            )
            self.artifacts.save("model/artifacts.json", self.model_artifacts)
        if (thinking_tokens is not None and settings.reasoning_max_tokens is not None
                and thinking_tokens > settings.reasoning_max_tokens):
            logger.warning(
                f"[model {call:03d}] provider reported {thinking_tokens} thinking tokens, "
                f"above requested cap {settings.reasoning_max_tokens}"
            )
        logger.success(
            f"[model {call:03d}] {stage} finished in {elapsed:.2f}s | "
            + self._metric_suffix(
                token_usage,
                thinking_tokens,
                finish_reason,
                accounting["call_reported_cost_usd"],
            )
        )
        if os.getenv("CONTRACTFIX_SHOW_MODEL_IO", "summary").lower() == "full":
            logger.debug(
                f"[model {call:03d}] {stage} parsed response\n"
                f"{json.dumps(response.model_dump(), indent=2, ensure_ascii=False)}"
            )
        return response
