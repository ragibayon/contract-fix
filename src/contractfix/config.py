"""Explicit, validated settings with no import-time environment mutation."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, replace
from typing import Literal, Mapping
from urllib.parse import urlsplit

Provider = Literal["openai", "ollama", "anthropic", "openrouter", "minsky"]

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 768
# SDK retries can obscure request accounting and replay model calls.
DEFAULT_MAX_RETRIES = 0
DEFAULT_RATE_LIMIT_RETRIES = 3
DEFAULT_RATE_LIMIT_BASE_SECONDS = 2.0
DEFAULT_RATE_LIMIT_MAX_SECONDS = 60.0

MINSKY_API_BASE_URL = "http://127.0.0.1:8000/v1"
MINSKY_MODEL = "qwen3.8-27b"
MINSKY_MODEL_ROOT = "Qwen/Qwen3.8-27B-FP8"
MINSKY_MAX_MODEL_LEN = 65536
MINSKY_TEMPERATURE = 0.0

DEFAULT_MODELS: dict[Provider, str] = {
    "openai": "gpt-4o",
    "ollama": "qwen3.8:27b",
    "anthropic": "claude-sonnet-4-6",
    "openrouter": "qwen/qwen3.8-27b",
    "minsky": MINSKY_MODEL,
}

# Completion limits, not total context-window sizes.
STAGE_MAX_OUTPUT_TOKEN_DEFAULTS = {
    "REQUIREMENTS_MAX_OUTPUT_TOKENS": 768,
    "SPEC_ALIGNMENT_MAX_OUTPUT_TOKENS": 768,
    "CONTRACT_MAX_OUTPUT_TOKENS": 768,
    "CONTRACT_VALIDATION_MAX_OUTPUT_TOKENS": 512,
    "TEST_PLANNER_MAX_OUTPUT_TOKENS": 768,
    "CODING_MAX_OUTPUT_TOKENS": 2048,
}


def _bool(value: str) -> bool:
    normalized = value.lower()
    if normalized not in {"true", "false", "1", "0"}:
        raise ValueError(f"Expected true/false, received {value!r}")
    return normalized in {"true", "1"}


@dataclass(frozen=True)
class LLMSettings:
    """A validated, provider-neutral model configuration."""

    provider: Provider = "ollama"
    model: str = DEFAULT_MODELS["ollama"]
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = DEFAULT_TEMPERATURE
    seed: int | None = None
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    context_window: int = 8192
    top_p: float = 1.0
    top_k: int = 20
    thinking: bool = False
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    # Provider-requested thinking cap. Support remains model/provider specific.
    reasoning_max_tokens: int | None = None
    # Accounting allowance within total completion tokens, not a partition.
    reasoning_reserve_tokens: int = 0
    timeout_seconds: float = 180.0
    # Total elapsed deadline for one provider invocation. Unlike an HTTP read
    # timeout, continuous token streaming cannot extend this deadline.
    wall_timeout_seconds: float = 180.0
    # Explicit, observable retries for provider HTTP 429 responses. SDK-level
    # retries remain disabled because they hide attempts from our accounting.
    rate_limit_retries: int = DEFAULT_RATE_LIMIT_RETRIES
    rate_limit_base_seconds: float = DEFAULT_RATE_LIMIT_BASE_SECONDS
    rate_limit_max_seconds: float = DEFAULT_RATE_LIMIT_MAX_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    provider_route: str | None = None
    structured_method: Literal["tool", "provider", "json_schema"] = "tool"
    # This is a byte-size guard, not a model tokenizer.
    prompt_max_bytes: int = 20000

    def __post_init__(self) -> None:
        if self.reasoning_effort not in {None, "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("unknown reasoning effort")
        if self.reasoning_effort and not self.thinking:
            raise ValueError("reasoning effort requires thinking=true")
        if self.reasoning_max_tokens is not None and not self.thinking:
            raise ValueError("reasoning token cap requires thinking=true")
        if self.reasoning_max_tokens is not None and self.provider != "openrouter":
            raise ValueError("exact reasoning token cap is only mapped by the OpenRouter adapter")
        if self.reasoning_effort and self.reasoning_max_tokens is not None:
            raise ValueError("choose reasoning effort or an exact reasoning token cap, not both")
        if self.reasoning_effort and self.provider not in {"openrouter", "ollama"}:
            raise ValueError("this provider adapter does not support explicit reasoning effort")
        if self.reasoning_reserve_tokens < 0 or (self.reasoning_reserve_tokens and not self.thinking):
            raise ValueError("reasoning allowance requires thinking=true and a nonnegative value")
        if self.reasoning_max_tokens is not None and not 1 <= self.reasoning_max_tokens < self.max_tokens:
            raise ValueError("reasoning token cap must be positive and below the completion limit")
        numeric_values = (
            self.temperature,
            self.top_p,
            self.timeout_seconds,
            self.wall_timeout_seconds,
            self.rate_limit_base_seconds,
            self.rate_limit_max_seconds,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("numeric settings must be finite")
        if self.provider not in DEFAULT_MODELS:
            raise ValueError(f"Unsupported provider: {self.provider}")
        if not self.model.strip():
            raise ValueError("model must be explicit")
        url = urlsplit(self.base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
        ):
            raise ValueError("base_url must be an HTTP(S) endpoint without credentials or query")
        if not 0 <= self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError("temperature/top_p outside supported range")
        if self.top_k < 1 or self.max_retries < 0:
            raise ValueError("invalid top_k or retry count")
        if not 0 <= self.rate_limit_retries <= 10:
            raise ValueError("rate_limit_retries must be between zero and ten")
        if self.rate_limit_base_seconds <= 0 or self.rate_limit_max_seconds <= 0:
            raise ValueError("rate-limit backoff values must be positive")
        if self.rate_limit_base_seconds > self.rate_limit_max_seconds:
            raise ValueError("rate-limit base backoff must not exceed its maximum")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be nonnegative when provided")
        if not 1 <= self.max_tokens < self.context_window:
            raise ValueError("require 0 < output limit < context window")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout must be positive")
        if self.wall_timeout_seconds <= 0:
            raise ValueError("wall-clock timeout must be positive")
        if self.prompt_max_bytes < 256:
            raise ValueError("prompt size budget too small")
        if self.structured_method not in {"tool", "provider", "json_schema"}:
            raise ValueError("unknown structured method")
        if self.structured_method == "json_schema" and self.provider != "ollama":
            raise ValueError("json_schema is the native ChatOllama path; use provider or tool")

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        **overrides: object,
    ) -> "LLMSettings":
        """Build settings without loading or changing a dotenv file implicitly."""
        values_from_env = os.environ if env is None else env
        requested_provider = overrides.pop("provider", None)
        provider = requested_provider or values_from_env.get("LLM_PROVIDER", "ollama")
        if provider not in DEFAULT_MODELS:
            raise ValueError(f"Unsupported provider: {provider}")
        urls = {
            "openai": "https://api.openai.com/v1",
            "ollama": "http://127.0.0.1:11434",
            "anthropic": "https://api.anthropic.com",
            "openrouter": "https://openrouter.ai/api/v1",
            "minsky": MINSKY_API_BASE_URL,
        }
        provider_name = str(provider).upper()
        values: dict[str, object] = {
            "provider": provider,
            "model": values_from_env.get("LLM_MODEL")
            or values_from_env.get(f"{provider_name}_MODEL")
            or DEFAULT_MODELS[provider],
            "base_url": values_from_env.get(f"{provider_name}_BASE_URL", urls[provider]),
            "temperature": float(values_from_env.get("LLM_TEMPERATURE", str(DEFAULT_TEMPERATURE))),
            "seed": (int(values_from_env["LLM_SEED"]) if values_from_env.get("LLM_SEED") else None),
            "max_tokens": int(values_from_env.get("LLM_MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS))),
            "context_window": int(
                values_from_env.get(
                    "LLM_CONTEXT_WINDOW",
                    str(MINSKY_MAX_MODEL_LEN if provider == "minsky" else 8192),
                )
            ),
            "top_p": float(values_from_env.get("LLM_TOP_P", "1")),
            "top_k": int(values_from_env.get("LLM_TOP_K", "20")),
            "thinking": _bool(values_from_env.get("LLM_THINKING", "false")),
            "reasoning_effort": values_from_env.get("LLM_REASONING_EFFORT") or None,
            "reasoning_max_tokens": (
                int(values_from_env["LLM_REASONING_MAX_TOKENS"])
                if values_from_env.get("LLM_REASONING_MAX_TOKENS")
                else None
            ),
            "reasoning_reserve_tokens": int(values_from_env.get("LLM_REASONING_RESERVE_TOKENS", "0")),
            "timeout_seconds": float(values_from_env.get("LLM_TIMEOUT_SECONDS", "180")),
            "wall_timeout_seconds": float(
                values_from_env.get(
                    "LLM_WALL_TIMEOUT_SECONDS",
                    values_from_env.get("LLM_TIMEOUT_SECONDS", "180"),
                )
            ),
            "rate_limit_retries": int(
                values_from_env.get(
                    "LLM_RATE_LIMIT_RETRIES",
                    str(DEFAULT_RATE_LIMIT_RETRIES),
                )
            ),
            "rate_limit_base_seconds": float(
                values_from_env.get(
                    "LLM_RATE_LIMIT_BASE_SECONDS",
                    str(DEFAULT_RATE_LIMIT_BASE_SECONDS),
                )
            ),
            "rate_limit_max_seconds": float(
                values_from_env.get(
                    "LLM_RATE_LIMIT_MAX_SECONDS",
                    str(DEFAULT_RATE_LIMIT_MAX_SECONDS),
                )
            ),
            "max_retries": int(values_from_env.get("LLM_MAX_RETRIES", str(DEFAULT_MAX_RETRIES))),
            "provider_route": (
                values_from_env.get("OPENROUTER_PROVIDER") or None if provider == "openrouter" else None
            ),
            "structured_method": values_from_env.get(
                "LLM_STRUCTURED_METHOD",
                "json_schema" if provider == "ollama" else "tool",
            ),
            "prompt_max_bytes": int(values_from_env.get("LLM_PROMPT_MAX_BYTES", "20000")),
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)  # type: ignore[arg-type]

    def public_dict(self) -> dict[str, object]:
        """Return non-secret settings with explicit assurance qualifiers."""
        return {
            **asdict(self),
            "seed_assurance": (
                "requested_not_determinism_guaranteed" if self.seed is not None else "not_requested"
            ),
            "context_window_assurance": "configured_not_live_verified",
            "reasoning_effort_assurance": (
                "requested_not_endpoint_capability_verified"
                if self.reasoning_effort
                else "not_requested"
            ),
            "reasoning_token_cap_assurance": (
                "provider_requested_not_live_enforcement_verified"
                if self.reasoning_max_tokens is not None
                else "not_requested"
            ),
            "reasoning_allowance_assurance": "budget_allowance_not_enforced_partition",
        }

    def for_output(self, tokens: int) -> "LLMSettings":
        """Return a copy with a stage-specific output limit."""
        return replace(self, max_tokens=tokens)

    def for_stage(self, answer_tokens: int, reasoning_tokens: int | None = None) -> "LLMSettings":
        """Reserve completion space for thinking without cutting the answer."""
        allowance = self.reasoning_reserve_tokens
        stage = self
        if reasoning_tokens is not None and self.thinking and self.reasoning_effort is None:
            # A model/profile-specific exact cap is an endpoint capability and
            # a cost boundary. A workflow stage may request less, but must not
            # silently raise that configured cap.
            allowance = min(
                reasoning_tokens,
                self.reasoning_max_tokens or reasoning_tokens,
            )
            if self.provider == "openrouter":
                stage = replace(
                    self,
                    reasoning_effort=None,
                    reasoning_max_tokens=allowance,
                    reasoning_reserve_tokens=allowance,
                )
        requested = answer_tokens + allowance
        if self.thinking and requested > self.max_tokens:
            raise ValueError(
                f"stage needs {requested} completion tokens including reasoning "
                f"allowance; configured limit is {self.max_tokens}"
            )
        return replace(stage, max_tokens=min(requested, self.max_tokens))
