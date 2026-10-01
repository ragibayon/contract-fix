"""Resolve provider-neutral stage reasoning intent into request settings."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from importlib.resources import files
import json
from typing import Literal

from contractfix.config import LLMSettings

ReasoningIntent = Literal["off", "low", "medium", "high"]


@dataclass(frozen=True)
class ReasoningResolution:
    settings: LLMSettings
    receipt: dict[str, object]


def capability_manifest() -> dict:
    resource = files("contractfix.llm").joinpath("reasoning_capabilities.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def capability_for(settings: LLMSettings) -> dict:
    matches = [
        profile
        for profile in capability_manifest()["profiles"]
        if profile["provider"] == settings.provider
        and fnmatchcase(settings.model, profile["model"])
    ]
    if not matches:
        raise ValueError(
            "REASONING_CAPABILITY_UNKNOWN: no verified profile for "
            f"provider={settings.provider} model={settings.model}"
        )
    matches.sort(key=lambda item: len(item["model"].replace("*", "")), reverse=True)
    return matches[0]


def request_controls_for(settings: LLMSettings) -> dict[str, object]:
    """Return verified controls, retaining the normal envelope when unmapped.

    Workflow stages call :func:`capability_for` separately and remain fail-closed
    for unknown reasoning profiles. This helper is also used by compatibility
    client construction, where historical callers may use an unmapped model.
    """
    matches = [
        profile
        for profile in capability_manifest()["profiles"]
        if profile["provider"] == settings.provider
        and fnmatchcase(settings.model, profile["model"])
    ]
    if not matches:
        return {}
    matches.sort(key=lambda item: len(item["model"].replace("*", "")), reverse=True)
    return dict(matches[0].get("request_controls", {}))


def resolve_stage_reasoning(
    base: LLMSettings,
    *,
    intent: ReasoningIntent,
    answer_tokens: int,
    completion_limit: int,
    reasoning_max_tokens: int | None = None,
) -> ReasoningResolution:
    """Resolve one semantic intent without silently changing models or meaning."""
    if completion_limit < answer_tokens:
        raise ValueError("stage completion limit cannot be below its answer allowance")
    profile = capability_for(base)
    if reasoning_max_tokens is not None:
        if base.provider != "openrouter" or intent == "off":
            raise ValueError("numeric reasoning caps require enabled OpenRouter reasoning")
        if not 1 <= reasoning_max_tokens <= completion_limit - answer_tokens:
            raise ValueError("reasoning cap must leave the requested answer allowance")
        settings = replace(
            base, thinking=True, reasoning_effort=None,
            reasoning_max_tokens=reasoning_max_tokens,
            reasoning_reserve_tokens=reasoning_max_tokens,
            max_tokens=completion_limit,
        )
        return ReasoningResolution(settings=settings, receipt={
            "schema_version": capability_manifest()["schema_version"],
            "provider": base.provider, "model": base.model, "intent": intent,
            "control": "max_tokens", "resolved": f"max_tokens:{reasoning_max_tokens}",
            "mapping_assurance": "provider_requested_not_live_enforcement_verified",
            "provider_assurance": "static_profile_not_live_verified",
            "completion_limit": completion_limit, "answer_allowance": answer_tokens,
            "accounting": profile["accounting"], "profile_pattern": profile["model"],
            "supported_efforts": list(profile["efforts"]),
        })
    control = profile["control"]
    mapping_assurance = "exact"

    effort_map = profile.get("normalized_efforts", {})
    if intent == "off" and profile["can_disable"]:
        settings = replace(
            base,
            thinking=False,
            reasoning_effort=None,
            reasoning_max_tokens=None,
            reasoning_reserve_tokens=0,
            max_tokens=completion_limit,
        )
        resolved = "disabled"
    elif intent == "off" and control == "effort":
        effort = profile.get("off_fallback")
        if effort not in profile["efforts"]:
            raise ValueError(
                "REASONING_POLICY_UNSUPPORTED: model cannot disable reasoning and "
                "declares no valid off fallback"
            )
        settings = replace(
            base,
            thinking=True,
            reasoning_effort=effort,
            reasoning_max_tokens=None,
            reasoning_reserve_tokens=min(
                base.reasoning_reserve_tokens, completion_limit - answer_tokens
            ),
            max_tokens=completion_limit,
        )
        resolved = f"effort:{effort}"
        mapping_assurance = "mandatory_reasoning_fallback"
    elif control == "effort":
        effort = effort_map.get(intent)
        if effort not in profile["efforts"]:
            raise ValueError(
                "REASONING_POLICY_UNSUPPORTED: "
                f"normalized effort {intent!r} has no mapping for provider={base.provider} "
                f"model={base.model}; supported={profile['efforts']}"
            )
        settings = replace(
            base,
            thinking=True,
            reasoning_effort=effort,
            reasoning_max_tokens=None,
            reasoning_reserve_tokens=min(base.reasoning_reserve_tokens, completion_limit - answer_tokens),
            max_tokens=completion_limit,
        )
        resolved = f"effort:{effort}"
        if effort != intent:
            mapping_assurance = "effort_approximation"
    elif control == "boolean":
        settings = replace(
            base,
            thinking=True,
            reasoning_effort=None,
            reasoning_max_tokens=None,
            reasoning_reserve_tokens=min(base.reasoning_reserve_tokens, completion_limit - answer_tokens),
            max_tokens=completion_limit,
        )
        resolved = "enabled"
        mapping_assurance = "boolean_approximation"
    else:
        raise ValueError(
            f"REASONING_POLICY_UNSUPPORTED: control {control!r} cannot resolve {intent!r}"
        )

    return ReasoningResolution(
        settings=settings,
        receipt={
            "schema_version": capability_manifest()["schema_version"],
            "provider": base.provider,
            "model": base.model,
            "intent": intent,
            "control": control,
            "resolved": resolved,
            # This is precise about ContractFix's translation only.  The
            # provider may still reject a nested reasoning value at request
            # time; OpenRouter's live metadata preflight is attached later.
            "mapping_assurance": mapping_assurance,
            "provider_assurance": "static_profile_not_live_verified",
            "completion_limit": completion_limit,
            "answer_allowance": answer_tokens,
            "accounting": profile["accounting"],
            "profile_pattern": profile["model"],
            "supported_efforts": list(profile["efforts"]),
        },
    )
