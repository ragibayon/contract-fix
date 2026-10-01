"""Construct explicit LangChain clients without provider/model fallback."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from contractfix.config import LLMSettings, STAGE_MAX_OUTPUT_TOKEN_DEFAULTS
from contractfix.llm.reasoning import request_controls_for


def _provider_mapping(value: Any) -> dict[str, Any]:
    """Return an SDK response object as a mapping, retaining Pydantic extras.

    OpenRouter adds provider-specific fields such as ``reasoning`` and
    ``completion_tokens_details``.  OpenAI's typed models preserve those as
    Pydantic extras, but callers must not assume they are normal model fields.
    """
    if isinstance(value, Mapping):
        return dict(value)
    try:
        dumped = value.model_dump(warnings=False)
    except (AttributeError, TypeError, ValueError):
        return {}
    if not isinstance(dumped, dict):
        return {}
    extras = getattr(value, "model_extra", None)
    if isinstance(extras, Mapping):
        dumped.update(extras)
    return dumped


def _openrouter_reasoning(message: Any) -> str | None:
    """Extract provider-exposed reasoning, excluding encrypted trace data."""
    raw = _provider_mapping(message)
    reasoning = raw.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    details = raw.get("reasoning_details", [])
    if not isinstance(details, list):
        return None
    text = "\n\n".join(
        item_text.strip()
        for item in details
        if isinstance(item, Mapping)
        and item.get("type") == "reasoning.text"
        and isinstance((item_text := item.get("text")), str)
        and item_text.strip()
    )
    return text or None


def client_kwargs(settings: LLMSettings) -> dict[str, Any]:
    """Translate validated settings to the selected provider's SDK arguments."""
    if settings.provider == "ollama":
        ollama: dict[str, Any] = {
            "model": settings.model,
            "base_url": settings.base_url,
            "temperature": settings.temperature,
            "num_predict": settings.max_tokens,
            "num_ctx": settings.context_window,
            "top_p": settings.top_p,
            "top_k": settings.top_k,
            "repeat_penalty": 1.0,
            "reasoning": settings.reasoning_effort or settings.thinking,
            "client_kwargs": {"timeout": settings.timeout_seconds},
        }
        if settings.seed is not None:
            ollama["seed"] = settings.seed
        return ollama

    common: dict[str, Any] = {
        "model": settings.model,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "max_retries": settings.max_retries,
        "timeout": settings.timeout_seconds,
    }
    if settings.provider == "anthropic":
        return common
    if settings.provider == "openai":
        request = {**common, "top_p": settings.top_p}
        if settings.seed is not None:
            request["seed"] = settings.seed
        return request

    if settings.provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        routing: dict[str, Any] = {
            # Automatic routing should be able to move away from an upstream
            # provider that is temporarily saturated. A pinned provider is an
            # explicit reproducibility constraint and must fail closed.
            "allow_fallbacks": settings.provider_route is None,
            "require_parameters": True,
        }
        if settings.provider_route:
            routing["only"] = [settings.provider_route]
        reasoning: dict[str, Any] = {
            "enabled": settings.thinking,
            "exclude": False,
        }
        if settings.reasoning_max_tokens is not None:
            reasoning["max_tokens"] = settings.reasoning_max_tokens
        elif settings.reasoning_effort:
            reasoning["effort"] = settings.reasoning_effort
        controls = request_controls_for(settings)
        request = {
            "model": settings.model,
            "max_tokens": settings.max_tokens,
            "max_retries": settings.max_retries,
            "timeout": settings.timeout_seconds,
            "api_key": api_key,
            "base_url": settings.base_url,
            "extra_body": {
                "provider": routing,
                "reasoning": reasoning,
                # OpenRouter returns provider-billed token and USD accounting in
                # the response when explicitly requested.
                "usage": {"include": True},
            },
            "default_headers": {
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", ""),
                "X-OpenRouter-Title": os.getenv("OPENROUTER_SITE_NAME", "contractfix"),
            },
        }
        if not controls.get("omit_sampling", False):
            request["temperature"] = settings.temperature
            request["top_p"] = settings.top_p
            request["extra_body"]["top_k"] = settings.top_k
        if settings.seed is not None:
            request["seed"] = settings.seed
        return request

    request = {
        **common,
        "api_key": os.getenv("MINSKY_API_KEY", "EMPTY"),
        "base_url": settings.base_url,
        "top_p": settings.top_p,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": settings.thinking},
            "top_k": settings.top_k,
            "repetition_penalty": 1.0,
        },
    }
    if settings.seed is not None:
        request["seed"] = settings.seed
    return request


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_retries: int | None = None,
    max_tokens_env: str | None = None,
    *,
    settings: LLMSettings | None = None,
):
    """Create one model client from explicit settings or compatible overrides."""
    if settings is not None and any(
        value is not None for value in (provider, model, temperature, max_tokens, max_retries)
    ):
        raise ValueError("pass settings or individual model overrides, not both")

    if settings is None:
        if max_tokens is None and max_tokens_env:
            default = STAGE_MAX_OUTPUT_TOKEN_DEFAULTS.get(max_tokens_env, 768)
            max_tokens = int(os.getenv(max_tokens_env, str(default)))
        settings = LLMSettings.from_env(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    kwargs = client_kwargs(settings)
    try:
        if settings.provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(**kwargs)
        if settings.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(**kwargs)

        from langchain_openai import ChatOpenAI

        if settings.provider == "openrouter":

            class OpenRouterChatOpenAI(ChatOpenAI):
                """Preserve OpenRouter reasoning text dropped by the base adapter."""

                def bind_tools(
                    self,
                    tools,
                    *,
                    tool_choice=None,
                    strict=None,
                    parallel_tool_calls=None,
                    response_format=None,
                    **kwargs,
                ):
                    """Do not advertise redundant parallelism for one forced tool.

                    LangChain's structured-output helper binds exactly one schema
                    tool and injects ``parallel_tool_calls=False``. OpenRouter then
                    treats that framework-added field as a required provider
                    capability when ``require_parameters`` is enabled. The field is
                    redundant when one tool is forced, and several otherwise
                    compatible endpoints do not advertise it.
                    """
                    options = dict(kwargs)
                    if settings.model.startswith("google/gemini-"):
                        from contractfix.llm.tool_schema import openrouter_tool

                        tools = [openrouter_tool(tool, settings.model) for tool in tools]
                    if parallel_tool_calls is not False:
                        options["parallel_tool_calls"] = parallel_tool_calls
                    return super().bind_tools(
                        tools,
                        tool_choice=tool_choice,
                        strict=strict,
                        response_format=response_format,
                        **options,
                    )

                def _create_chat_result(self, response, generation_info=None):
                    result = super()._create_chat_result(response, generation_info)
                    payload = _provider_mapping(response)
                    choices = payload.get("choices", [])
                    if not isinstance(choices, list):
                        choices = []
                    usage = _provider_mapping(payload.get("usage"))
                    for choice, generation in zip(choices, result.generations):
                        raw_choice = _provider_mapping(choice)
                        reasoning = _openrouter_reasoning(raw_choice.get("message"))
                        if reasoning:
                            generation.message.additional_kwargs["reasoning_content"] = reasoning
                        # Keep only accounting fields. In particular, do not put
                        # OpenRouter's encrypted reasoning detail into artifacts.
                        if usage:
                            generation.message.response_metadata["usage"] = usage
                    return result

            return OpenRouterChatOpenAI(**kwargs)

        return ChatOpenAI(**kwargs)
    except ImportError as exc:
        raise RuntimeError("Install the selected provider dependencies with uv sync") from exc


def get_llm_settings() -> dict[str, object]:
    """Return the effective non-secret model configuration."""
    return LLMSettings.from_env().public_dict()


def ask_llm(prompt: str, provider: str | None = None) -> str:
    """Invoke one plain-text prompt using the selected provider."""
    response = get_llm(provider=provider).invoke(prompt)
    if not isinstance(response.content, str):
        raise TypeError("Expected a text response")
    return response.content
