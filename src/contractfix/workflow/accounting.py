"""Provider-neutral model-call accounting for workflow terminal states."""

from __future__ import annotations

from .stages import Stages


def model_accounting(stages: Stages) -> dict:
    """Return one consistent cumulative accounting packet for any terminal state."""
    settings = getattr(stages, "settings", None)
    configuration = settings.public_dict() if settings is not None else None
    model_artifacts = getattr(stages, "model_artifacts", [])
    completed_artifacts = [
        item for item in model_artifacts if item.get("status") == "completed"
    ]
    thinking_artifacts = [item for item in completed_artifacts if item.get("thinking")]
    return {
        "calls": getattr(stages, "cumulative_sdk_calls", getattr(stages, "calls", None)),
        "stage_calls": getattr(stages, "calls", None),
        "input_tokens": getattr(stages, "cumulative_input_tokens", None),
        "output_tokens": getattr(stages, "cumulative_output_tokens", None),
        "answer_tokens": (
            None
            if getattr(stages, "missing_answer_token_calls", 0)
            else getattr(stages, "cumulative_answer_tokens", None)
        ),
        "missing_answer_token_calls": getattr(stages, "missing_answer_token_calls", None),
        "reasoning_tokens": (
            None
            if getattr(stages, "missing_reasoning_token_calls", 0)
            else getattr(stages, "cumulative_reasoning_tokens", None)
        ),
        "missing_reasoning_token_calls": getattr(stages, "missing_reasoning_token_calls", None),
        "thinking_text_calls": len(thinking_artifacts) if model_artifacts else None,
        "thinking_text_unavailable_calls": (
            len(completed_artifacts) - len(thinking_artifacts) if model_artifacts else None
        ),
        "thinking_text_assurance": "provider_exposed_text_only",
        "model_elapsed_seconds": getattr(stages, "cumulative_elapsed_seconds", None),
        "last_finish_reason": getattr(stages, "last_finish_reason", None),
        "by_stage": getattr(stages, "stage_accounting", None),
        "provider_reported_cost_usd": getattr(stages, "cumulative_cost_usd", None),
        "missing_cost_calls": getattr(stages, "missing_cost_calls", None),
        "cost_assurance": (
            next(iter(getattr(stages, "cost_assurances", set())))
            if len(getattr(stages, "cost_assurances", set())) == 1
            else "mixed_or_unreported"
        ),
        "configuration": configuration,
    }
