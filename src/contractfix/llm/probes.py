"""Small, explicitly paid OpenRouter reasoning-cap probes."""

from __future__ import annotations

import os
from typing import Any

import httpx

from contractfix.config import LLMSettings


def probe_reasoning_cap(
    settings: LLMSettings, cap: int, *, answer_tokens: int = 2048
) -> dict[str, Any]:
    """Send one minimal request and retain only credential-safe request/response facts."""
    if settings.provider != "openrouter":
        raise ValueError("reasoning-cap probes require OpenRouter")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is not set")
    if answer_tokens < 1:
        raise ValueError("probe answer allowance must be positive")
    completion = cap + answer_tokens
    body: dict[str, Any] = {
        "model": settings.model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": completion,
        "reasoning": {"max_tokens": cap},
        "provider": {"require_parameters": True},
    }
    if settings.provider_route:
        body["provider"]["order"] = [settings.provider_route]
    try:
        with httpx.Client(timeout=settings.wall_timeout_seconds) as client:
            response = client.post(
                settings.base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + key},
                json=body,
            )
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return {"request": body, "error_type": type(exc).__name__, "accepted": False}
    choice = (payload.get("choices") or [{}])[0]
    error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    usage = payload.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens")
    output = usage.get("completion_tokens")
    answer = output - reasoning if isinstance(output, int) and isinstance(reasoning, int) else None
    return {
        "request": body,
        "http_status": response.status_code,
        "accepted": response.is_success and bool(payload.get("choices")),
        "routing_provider": payload.get("provider"),
        "response_model": payload.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "reasoning_tokens": reasoning,
        "answer_tokens": answer,
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": output,
        "reported_cost_usd": usage.get("cost"),
        "error_code": error.get("code"),
        "error_type": error.get("type"),
    }
