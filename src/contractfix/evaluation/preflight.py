"""Explicitly opted-in provider checks. No provider/model switching, downloading, or server launch."""

from __future__ import annotations
import importlib.metadata as metadata
import os
import platform
import secrets
from typing import Any
from contractfix.config import LLMSettings
from contractfix.utils.artifacts import RunArtifacts
from contractfix.llm.callbacks import LLMRunMetrics

DISTRIBUTIONS = (
    "langchain",
    "langchain-openai",
    "pydantic",
    "jinja2",
    "httpx",
    "python-dotenv",
    "rich",
)

_OPENROUTER_CAPABILITY_CACHE: dict[tuple[object, ...], dict[str, Any]] = {}


def _required_openrouter_parameters(settings: LLMSettings) -> set[str]:
    from contractfix.llm.reasoning import request_controls_for

    controls = request_controls_for(settings)
    required = {"max_tokens"}
    if not controls.get("omit_sampling", False):
        required.update({"temperature", "top_p"})
        if settings.top_k is not None:
            required.add("top_k")
    if settings.structured_method == "tool":
        required.update({"tools", "tool_choice"})
    elif settings.structured_method in {"json_schema", "provider"}:
        required.add("response_format")
    # The OpenRouter adapter always sends an explicit reasoning object,
    # including enabled=false for stage policies that disable reasoning.
    required.add("reasoning")
    if settings.thinking and settings.reasoning_effort:
        required.add("reasoning_effort")
    if settings.seed is not None:
        required.add("seed")
    return required


def openrouter_capability_preflight(settings: LLMSettings) -> dict[str, Any]:
    """Verify a zero-generation request envelope against live endpoint metadata."""
    if settings.provider != "openrouter":
        return {"checked": False, "reason": "provider is not OpenRouter"}
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is not set")
    required = _required_openrouter_parameters(settings)
    cache_key = (
        settings.base_url,
        settings.model,
        settings.provider_route,
        tuple(sorted(required)),
        settings.context_window,
        settings.max_tokens,
    )
    cached = _OPENROUTER_CAPABILITY_CACHE.get(cache_key)
    if cached is not None:
        return {**cached, "cache_hit": True}

    import httpx

    url = f"{settings.base_url.rstrip('/')}/models/{settings.model}/endpoints"
    with httpx.Client(timeout=settings.timeout_seconds) as client:
        response = client.get(url, headers={"Authorization": "Bearer " + key})
        response.raise_for_status()
    payload = response.json().get("data", {})
    if payload.get("id") != settings.model:
        raise ValueError("OpenRouter endpoint metadata returned a different model ID")
    endpoints = payload.get("endpoints", [])
    if not isinstance(endpoints, list) or not endpoints:
        raise ValueError("OpenRouter returned no endpoints for the configured model")

    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    requested_route = (settings.provider_route or "").casefold()
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        provider = str(endpoint.get("provider_name") or endpoint.get("name") or "unknown")
        supported = {str(item) for item in endpoint.get("supported_parameters", []) if isinstance(item, str)}
        missing = sorted(required - supported)
        context = endpoint.get("context_length")
        completion = endpoint.get("max_completion_tokens")
        reasons = []
        if requested_route and requested_route not in provider.casefold():
            reasons.append("provider_route")
        if missing:
            reasons.append("parameters:" + ",".join(missing))
        if isinstance(context, int) and context < settings.context_window:
            reasons.append("context_window")
        if isinstance(completion, int) and completion < settings.max_tokens:
            reasons.append("completion_limit")
        row = {
            "provider": provider,
            "context_length": context,
            "max_completion_tokens": completion,
            "supported_parameters": sorted(supported),
        }
        if reasons:
            rejected.append({**row, "reasons": reasons})
        else:
            eligible.append(row)
    if not eligible:
        missing_summary = sorted({reason for endpoint in rejected for reason in endpoint.get("reasons", [])})
        raise ValueError(
            "OPENROUTER_CAPABILITY_PREFLIGHT_FAILED: no endpoint satisfies "
            f"parameters={sorted(required)}, context={settings.context_window}, "
            f"completion={settings.max_tokens}; reasons={missing_summary}"
        )
    report = {
        "checked": True,
        "cache_hit": False,
        "model": settings.model,
        "requested_provider_route": settings.provider_route,
        "required_parameters": sorted(required),
        "requested_context_window": settings.context_window,
        "requested_completion_limit": settings.max_tokens,
        "eligible_endpoint_count": len(eligible),
        "eligible_endpoints": eligible,
        "rejected_endpoint_count": len(rejected),
        "schema_acceptance_verified": False,
        "assurance": (
            "live endpoint metadata only; require_parameters checks parameter support, "
            "not acceptance of the actual structured schema or prompt"
        ),
    }
    _OPENROUTER_CAPABILITY_CACHE[cache_key] = report
    return report


def workflow_capability_preflight(settings: LLMSettings, workflow: Any) -> dict[str, Any]:
    """Resolve and inspect every configured workflow stage without inference.

    The endpoint query only reads OpenRouter metadata. It never constructs an
    SDK client, invokes a model, or consumes completion tokens.
    """
    from contractfix.llm.reasoning import resolve_stage_reasoning
    from contractfix.workflow.languages.python.promptbook import PromptBook

    if settings.provider != "openrouter":
        return {
            "checked": False,
            "zero_inference": True,
            "reason": "workflow capability preflight currently supports OpenRouter only",
            "provider": settings.provider,
        }

    policy = dict(workflow.stage_reasoning_policy)
    limits = dict(workflow.stage_completion_limits)
    if not policy:
        return {
            "checked": False,
            "zero_inference": True,
            "reason": "workflow has no stage reasoning policy",
        }

    prompt_stages = {
        "executable_contract_synthesis_recovery": "executable_contract_synthesis",
        "executable_contract_conformance_adjudication": "executable_contract_conformance",
    }
    book = PromptBook(
        profile=workflow.prompt_profile,
        example_profile=workflow.example_profile,
    )
    stage_reports = []
    for policy_stage, intent in policy.items():
        prompt_stage = prompt_stages.get(policy_stage, policy_stage)
        answer_tokens = workflow.stage_answer_tokens.get(
            prompt_stage, book.policy["stages"][prompt_stage]["answer_tokens"]
        )
        resolution = resolve_stage_reasoning(
            settings,
            intent=intent,
            answer_tokens=answer_tokens,
            completion_limit=limits[policy_stage],
            reasoning_max_tokens=workflow.stage_reasoning_caps.get(policy_stage),
        )
        capabilities = openrouter_capability_preflight(resolution.settings)
        receipt = dict(resolution.receipt)
        if resolution.settings.provider == "openrouter":
            receipt["provider_assurance"] = (
                "live_endpoint_metadata_parameter_names; nested_values_enforced_by_request"
            )
        stage_reports.append(
            {
                "policy_stage": policy_stage,
                "prompt_stage": prompt_stage,
                "model_configuration": resolution.settings.public_dict(),
                "reasoning_policy": receipt,
                "capabilities": capabilities,
            }
        )
    return {
        "checked": True,
        "zero_inference": True,
        "provider": settings.provider,
        "model": settings.model,
        "stages": stage_reports,
    }


def ollama_structured_readiness(settings: LLMSettings) -> dict[str, Any]:
    """Check the selected local model and the structured response path.

    Unlike OpenRouter's automatic preflight, this intentionally performs one local
    inference. It cannot incur a provider API charge and records returned usage.
    """
    if settings.provider != "ollama":
        return {"checked": False, "passed": False, "reason": "provider is not Ollama"}

    from pydantic import BaseModel, Field

    from contractfix.llm.structured import generate_structured

    class ReadinessResponse(BaseModel):
        status: str = Field(description="Return exactly ready")
        value: int = Field(description="Return exactly 7")

    deployment = deployment_metadata(settings)
    parsed, accounting = generate_structured(
        "Return status ready and value 7. Follow the supplied schema exactly.",
        ReadinessResponse,
        settings,
        system_prompt="Return only the requested structured response.",
    )
    passed = parsed.status == "ready" and parsed.value == 7
    return {
        "checked": True,
        "passed": passed,
        "zero_paid_inference": True,
        "local_inference_performed": True,
        "provider": "ollama",
        "model": settings.model,
        "deployment": deployment,
        "checks": {
            "model_available": True,
            "structured_output": passed,
            "token_usage_reported": bool(accounting.get("token_usage")),
        },
        "metrics": accounting,
    }


def doctor():
    versions = {}
    for name in DISTRIBUTIONS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "packages": versions,
        "missing_packages": [n for n, v in versions.items() if v is None],
        "sdk_available": all(versions.values()),
        "network_contacted": False,
        "assurance": "installed versions only; use uv pip check and live provider preflight",
    }


def deployment_metadata(settings: LLMSettings):
    import httpx

    base = settings.base_url.rstrip("/")
    with httpx.Client(timeout=settings.timeout_seconds) as client:
        if settings.provider == "ollama":
            tags = client.get(base + "/api/tags")
            tags.raise_for_status()
            names = tags.json().get("models", [])
            item = next(
                (x for x in names if x.get("name") == settings.model or x.get("model") == settings.model),
                None,
            )
            if item is None:
                raise ValueError("Configured model not installed; no pull or substitution was attempted")
            show = client.post(base + "/api/show", json={"model": settings.model})
            show.raise_for_status()
            data = show.json()
            return {
                "requested_model": settings.model,
                "digest": item.get("digest"),
                "details": data.get("details"),
                "capabilities": data.get("capabilities"),
                "model_info": data.get("model_info"),
                "parameters": data.get("parameters"),
                "actual_context_window_verified": False,
                "note": "model metadata is not proof of the runtime allocation or identical hosted weights",
            }
        headers = {}
        if settings.provider == "openrouter":
            key = os.getenv("OPENROUTER_API_KEY")
            if not key:
                raise ValueError("OPENROUTER_API_KEY is not set")
            headers = {"Authorization": "Bearer " + key}
        resp = client.get(base + "/models", headers=headers)
        resp.raise_for_status()
        item = next((x for x in resp.json().get("data", []) if x.get("id") == settings.model), None)
        if item is None:
            raise ValueError("Exact configured model ID was not returned by /models")
        return {
            "requested_model": settings.model,
            "model_listing": item,
            "requested_provider_route": settings.provider_route,
            "weights_digest": None,
            "provider_route_pinned": bool(settings.provider_route),
            "note": "listing and requested route are not proof of the endpoint actually serving each call",
        }


def provider_check(settings: LLMSettings, output_dir, *, cases=3):
    """A paid/network smoke check, not a claim about repair quality or reliable faithfulness."""
    if not 1 <= cases <= 20:
        raise ValueError("provider cases must be 1..20")
    from contractfix.llm.llm_client import get_llm
    from contractfix.llm.structured import generate_structured
    from contractfix.agents.schema.contracts import Echo
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage, ToolMessage

    a = RunArtifacts(
        output_dir, {"purpose": "provider_preflight", "settings": settings.public_dict(), "cases": cases}
    )
    records = []
    metrics = LLMRunMetrics(a.events, "provider_preflight")
    try:
        deployment = deployment_metadata(settings)
        a.save("deployment.json", deployment)
        model = get_llm(settings=settings)
        for index in range(cases):
            nonce = "cf_" + secrets.token_hex(8)
            answer = model.invoke(
                [HumanMessage(content="Return exactly this text and nothing else: " + nonce)],
                config={"callbacks": [metrics]},
            )
            ok = isinstance(answer.content, str) and answer.content.strip() == nonce
            records.append({"case": index, "check": "text_echo", "pass": ok})
            if not ok:
                raise ValueError("text echo did not match")
            parsed, usage = generate_structured(
                "Return an object whose value is exactly " + nonce,
                Echo,
                settings,
                model=model,
                events=a.events,
            )
            records.append(
                {"case": index, "check": "structured_echo", "pass": parsed.value == nonce, "metrics": usage}
            )
            if parsed.value != nonce:
                raise ValueError("structured value did not match")

            @tool
            def inspect_value(value: str) -> str:
                """Submit the requested value and receive an opaque host-generated receipt."""
                return "receipt_" + secrets.token_hex(10)

            bound = model.bind_tools([inspect_value])
            messages = [
                HumanMessage(
                    content="Call inspect_value exactly once with value='"
                    + nonce
                    + "'. After receiving its result, repeat that result exactly, without prose."
                )
            ]
            call = bound.invoke(messages, config={"callbacks": [metrics]})
            calls = call.tool_calls
            valid = (
                len(calls) == 1
                and calls[0].get("name") == "inspect_value"
                and calls[0].get("args") == {"value": nonce}
            )
            records.append({"case": index, "check": "tool_arguments", "pass": valid})
            if not valid:
                raise ValueError("tool-call name, count, or arguments did not match")
            receipt = inspect_value.invoke(calls[0]["args"])
            messages.extend([call, ToolMessage(content=receipt, tool_call_id=calls[0]["id"])])
            final = model.invoke(messages, config={"callbacks": [metrics]})
            valid = isinstance(final.content, str) and final.content.strip() == receipt
            records.append({"case": index, "check": "tool_observation_consumed", "pass": valid})
            if not valid:
                raise ValueError("tool receipt was not consumed correctly")
        report = {
            "passed": True,
            "checks": records,
            "metrics": metrics.as_dict(),
            "configuration_sha256": a.manifest["sha256"],
            "hosted_campaign_route_ready": settings.provider != "openrouter" or bool(settings.provider_route),
            "assurance": "transport/schema/tool smoke only; Deep Agents, semantic quality and baseline adapters require separate tests",
        }
    except Exception as exc:
        report = {
            "passed": False,
            "checks": records,
            "metrics": metrics.as_dict(),
            "error_type": type(exc).__name__,
            "detail": str(exc)[:800],
        }
    a.save("report.json", report)
    a.events.emit("preflight_finished", passed=report["passed"])
    return report
