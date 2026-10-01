"""Explicit structured extraction with bounded, artifact-aware schema correction."""

from __future__ import annotations

from contextlib import contextmanager
import json
import hashlib
import random
from pathlib import Path
import re
import signal
import threading
import time
from typing import TypeVar

from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel, ValidationError

from contractfix.config import LLMSettings
from contractfix.llm.callbacks import LLMRunMetrics, message_metadata, provider_thinking_text
from contractfix.llm.llm_client import get_llm
from contractfix.utils.logger import logger

T = TypeVar("T", bound=BaseModel)


class StructuredGenerationError(RuntimeError):
    """No schema-valid proposal was returned within the attempt budget."""

    def __init__(self, message: str, *, metrics: dict | None = None):
        super().__init__(message)
        self.metrics = metrics or {}


class OutputBudgetError(StructuredGenerationError):
    """The provider stopped at its output limit; not a contract or schema failure."""


class ReasoningOnlyOutputBudgetError(OutputBudgetError):
    """The completion limit was consumed by reasoning without a deliverable."""


class SchemaResponseError(ValueError):
    """The provider returned no schema-valid final response."""


class ProviderRateLimitError(StructuredGenerationError):
    """Observable provider rate-limit retries were exhausted."""


class ProviderWallClockTimeout(StructuredGenerationError):
    """One provider invocation exceeded its total elapsed-time deadline."""


@contextmanager
def _wall_clock_deadline(seconds: float):
    """Interrupt a synchronous provider call after a true POSIX wall deadline."""
    if threading.current_thread() is not threading.main_thread():
        raise ProviderWallClockTimeout(
            "hard provider deadline requires invocation from the main thread"
        )

    def expired(_signum, _frame):
        raise ProviderWallClockTimeout(
            f"provider call exceeded {seconds:g}s wall-clock deadline"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_delay, previous_interval = signal.setitimer(signal.ITIMER_REAL, 0)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_delay > 0:
            remaining = max(0.000001, previous_delay - (time.monotonic() - started))
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_interval)


def _exception_chain(exc: BaseException):
    """Yield one bounded exception chain without depending on a provider SDK."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(seen) < 8:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _status_code(exc: BaseException) -> int | None:
    """Extract an HTTP status from common provider-neutral exception shapes."""
    for current in _exception_chain(exc):
        direct = getattr(current, "status_code", None)
        if isinstance(direct, int):
            return direct
        response = getattr(current, "response", None)
        response_code = getattr(response, "status_code", None)
        if isinstance(response_code, int):
            return response_code
        match = re.search(r"(?:error code|status(?:_code)?|code)['\" :=]+(\d{3})", str(current), re.I)
        if match:
            return int(match.group(1))
    return None


def _is_rate_limit(exc: BaseException) -> bool:
    """Recognize a gateway 504 only when it records an upstream HTTP 429."""
    status = _status_code(exc)
    if status == 429:
        return True
    if status != 504:
        return False
    for current in _exception_chain(exc):
        for body in (getattr(current, "body", None), getattr(current, "message", None)):
            if isinstance(body, dict):
                error = body.get("error")
                container = error if isinstance(error, dict) else body
                metadata = container.get("metadata")
                if isinstance(metadata, dict):
                    previous = metadata.get("previous_errors")
                    if isinstance(previous, list) and any(
                        isinstance(item, dict) and item.get("code") == 429
                        for item in previous
                    ):
                        return True
        if re.search(r"previous_errors.{0,400}['\"]code['\"]\s*:\s*429\b", str(current), re.S):
            return True
    return False


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Honor Retry-After when an SDK exposes response headers."""
    for current in _exception_chain(exc):
        response = getattr(current, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            continue
        try:
            value = headers.get("retry-after") or headers.get("Retry-After")
            delay = float(value)
        except (AttributeError, TypeError, ValueError):
            continue
        if delay >= 0:
            return delay
    return None


def _rate_limit_delay(settings: LLMSettings, retry_number: int, exc: BaseException) -> float:
    requested = _retry_after_seconds(exc)
    if requested is not None:
        return min(requested, settings.rate_limit_max_seconds)
    ceiling = min(
        settings.rate_limit_max_seconds,
        settings.rate_limit_base_seconds * (2 ** (retry_number - 1)),
    )
    # Equal jitter retains exponential growth while avoiding synchronized
    # retries when several workers share the same upstream provider pool.
    return random.uniform(ceiling / 2, ceiling)


def _decode_top_level_object(value: object) -> object:
    """Decode bounded transport wrappers without changing payload meaning.

    Open-weight tool adapters sometimes return an object as a JSON string, or
    place that string in a singleton list.  Decode only those unambiguous
    transport shapes.  Arrays of objects, prose, Markdown, multiple values and
    malformed JSON remain invalid.
    """
    current = value
    for _ in range(3):
        if isinstance(current, dict):
            return current
        if isinstance(current, list):
            if len(current) != 1 or not isinstance(current[0], str):
                return _MISSING
            current = current[0]
            continue
        if not isinstance(current, str):
            return _MISSING
        try:
            current = json.loads(current)
        except json.JSONDecodeError:
            return _MISSING
    return current if isinstance(current, dict) else _MISSING


def validate_output(schema: type[T], value: object) -> T:
    if isinstance(value, schema):
        return value
    normalized = _decode_top_level_object(value)
    return schema.model_validate(value if normalized is _MISSING else normalized)


def structured_runnable(model: object, schema: type[T], settings: LLMSettings, system_prompt: str = ""):
    """No side-effecting repository tools participate in schema correction."""
    if settings.structured_method == "json_schema":
        return model.with_structured_output(schema, method="json_schema", include_raw=True), "native"
    if settings.structured_method == "tool":
        # The outer generate_structured loop owns all schema correction. A
        # create_agent tool loop can silently invoke the provider again after a
        # length-limited or malformed response, hiding cost and repeating the
        # same failure. Bind the schema directly so one schema attempt is
        # exactly one observable SDK call for every provider adapter.
        return (
            model.with_structured_output(
                schema,
                method="function_calling",
                include_raw=True,
            ),
            "native",
        )
    from langchain.agents import create_agent
    from langchain.agents.structured_output import ProviderStrategy

    strategy = ProviderStrategy(schema)
    return create_agent(
        model=model, tools=[], system_prompt=system_prompt or None, response_format=strategy
    ), "agent"


def _get(value: object, key: str, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _last_answer(result: dict | None, kind: str) -> object | None:
    if result is None:
        return None
    if kind == "native":
        return result.get("raw")
    messages = result.get("messages", [])
    for message in reversed(messages):
        if _get(message, "type") == "ai" or _get(message, "role") == "assistant":
            return message
    return None


def _check_output_limit(message: object | None) -> None:
    metadata = _get(message, "response_metadata", {}) or {}
    reason = metadata.get("finish_reason") or metadata.get("done_reason")
    if reason in {"length", "max_tokens", "max_output_tokens"}:
        visible = _proposal_text(None, message).strip()
        thinking = provider_thinking_text(message).strip()
        reasoning_tokens = message_metadata(message).get("reasoning_tokens")
        reasoning_only = isinstance(reasoning_tokens, int) and reasoning_tokens > 0
        if not visible and (thinking or reasoning_only):
            raise ReasoningOnlyOutputBudgetError(
                "provider output limit exhausted by reasoning without a usable response"
            )
        raise OutputBudgetError("provider output limit exhausted; inspect partial output before changing the allowance")


def _proposal_text(value: object, message: object | None) -> str:
    """Extract final answer/tool arguments, not provider reasoning channels."""
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if value is not None:
        return value if isinstance(value, str) else json.dumps(value, default=str)
    tool_calls = _get(message, "tool_calls", []) or []
    if tool_calls:
        return json.dumps([_get(call, "args", {}) for call in tool_calls], default=str)
    invalid_tool_calls = _get(message, "invalid_tool_calls", []) or []
    if invalid_tool_calls:
        return json.dumps(
            [_get(call, "args", "") for call in invalid_tool_calls], default=str
        )
    content = _get(message, "content", "")
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return content if isinstance(content, str) else ""


_MISSING = object()


def _native_tool_arguments(message: object | None) -> object:
    """Recover one unambiguous native tool payload without repairing its meaning.

    Some Ollama/LangChain combinations expose valid tool arguments as a JSON
    string while also reporting a root-level Pydantic ``model_type`` error.
    ContractFix owns that transport normalization. Multiple calls, missing
    arguments, and malformed JSON remain schema failures.
    """
    tool_calls = _get(message, "tool_calls", []) or []
    if len(tool_calls) != 1:
        return _MISSING
    arguments = _get(tool_calls[0], "args", _MISSING)
    return _decode_top_level_object(arguments)


def _native_invalid_tool_arguments(
    message: object | None, expected_name: str
) -> object:
    """Recover one strictly decodable object from an invalid tool-call envelope.

    OpenRouter models can occasionally JSON-encode the complete argument object
    twice. LangChain decodes the outer layer to a string, rejects that string as
    non-dict tool arguments, and retains the original call only in
    ``invalid_tool_calls``. Decode at most two JSON layers and leave every other
    malformed, ambiguous, or non-object payload rejected. The requested
    Pydantic schema remains the final authority.
    """
    if _get(message, "tool_calls", []) or []:
        return _MISSING
    calls = _get(message, "invalid_tool_calls", []) or []
    if len(calls) != 1:
        return _MISSING
    if _get(calls[0], "name", "") != expected_name:
        return _MISSING
    return _decode_top_level_object(_get(calls[0], "args", _MISSING))


def _native_json_content(message: object | None) -> object:
    """Recover one bare JSON object emitted in the assistant content channel.

    Some OpenRouter models, including GLM-5.3, can honor the requested schema
    but emit the object as ordinary assistant content instead of making the
    forced schema tool call.  Accept only an exact JSON object when no tool
    calls are present.  Markdown fences, prose, arrays, and malformed JSON
    remain schema failures; tool-bearing responses use the tool path above.
    """
    if (_get(message, "tool_calls", []) or []) or (
        _get(message, "invalid_tool_calls", []) or []
    ):
        return _MISSING
    content = _get(message, "content", "")
    if not isinstance(content, str):
        return _MISSING
    return _decode_top_level_object(content)


def _invalid_payload_reason(value: object) -> str:
    """Describe one rejected transport shape for the model's next attempt."""
    if value is _MISSING or value is None or value == "":
        return "the response contained no structured payload"
    if isinstance(value, list):
        return "the response used a top-level array; the schema requires one top-level object"
    if isinstance(value, dict):
        return "the decoded object failed the supplied schema"
    if not isinstance(value, str):
        return (
            f"the top-level value had type {type(value).__name__}; "
            "the schema requires an object"
        )
    stripped = value.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        return "the response used a Markdown code fence; return the object without Markdown"
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        return (
            "the response was prose or malformed JSON "
            f"(decode error at character {error.pos}); return one valid object"
        )
    if isinstance(decoded, list):
        return "the response decoded to a top-level array; the schema requires one object"
    if isinstance(decoded, str):
        return (
            "the response remained a JSON-encoded string after bounded decoding; "
            "return the object directly"
        )
    return _invalid_payload_reason(decoded)


def _native_transport_rejection(message: object | None, expected_name: str) -> str:
    """Return an actionable diagnosis for a native structured-output failure."""
    tool_calls = _get(message, "tool_calls", []) or []
    if len(tool_calls) > 1:
        return (
            f"received {len(tool_calls)} tool calls; submit exactly one "
            f"{expected_name} payload"
        )
    if len(tool_calls) == 1:
        name = _get(tool_calls[0], "name", "")
        if name and name != expected_name:
            return f"used structured tool {name!r}; expected {expected_name!r}"
        return _invalid_payload_reason(_get(tool_calls[0], "args", _MISSING))

    invalid_calls = _get(message, "invalid_tool_calls", []) or []
    if len(invalid_calls) > 1:
        return (
            f"received {len(invalid_calls)} invalid tool calls; submit exactly one "
            f"{expected_name} payload"
        )
    if len(invalid_calls) == 1:
        name = _get(invalid_calls[0], "name", "")
        if name and name != expected_name:
            return f"used structured tool {name!r}; expected {expected_name!r}"
        return _invalid_payload_reason(
            _get(invalid_calls[0], "args", _MISSING)
        )

    return _invalid_payload_reason(_get(message, "content", _MISSING))


def _prompt_size(system: str, messages: list[dict], schema: type[BaseModel], kind: str) -> int:
    # Conservatively include serialized roles and JSON schema. Provider chat templates
    # and framework additions still need endpoint calibration; this is not a tokenizer.
    data = {
        "system": system if kind != "native" else "",
        "messages": messages,
        "schema": schema.model_json_schema(),
    }
    return len(json.dumps(data, ensure_ascii=False).encode("utf-8"))


def generate_structured(
    prompt: str,
    schema: type[T],
    settings: LLMSettings,
    *,
    model: object | None = None,
    events: object | None = None,
    max_attempts: int = 3,
    system_prompt: str = "",
    correction_template: Path | None = None,
    correction_text: str | None = None,
    demonstration_messages: list[dict] | None = None,
) -> tuple[T, dict]:
    """Retry schema failures only; preserve the failed proposal and fixed task.

    A retry restarts the side-effect-free extraction stage with the same examples,
    original task and one correction packet. It does not rerun a repository agent.
    """
    if not 1 <= max_attempts <= 3:
        raise ValueError("schema attempts must be between one and three")
    examples = demonstration_messages or []
    for index, message in enumerate(examples):
        if set(message) != {"role", "content"} or message["role"] != (
            "user" if index % 2 == 0 else "assistant"
        ):
            raise ValueError("demonstrations must be alternating user/assistant text pairs")
    if len(examples) % 2:
        raise ValueError("incomplete demonstration pair")
    initial = (
        ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        + examples
        + [{"role": "user", "content": prompt}]
    )
    initial_size = _prompt_size("", initial, schema, "native")
    if initial_size > settings.prompt_max_bytes:
        raise ValueError(
            "prompt byte budget exceeded: "
            f"serialized={initial_size}, limit={settings.prompt_max_bytes}; "
            "increase LLM_PROMPT_MAX_BYTES or reduce supplied context"
        )
    selected_model = model or get_llm(settings=settings)
    runnable, kind = structured_runnable(selected_model, schema, settings, system_prompt)
    if kind != "native" and system_prompt:
        initial = initial[1:]
    messages = list(initial)
    metrics = LLMRunMetrics(events, stage="structured")
    failures, requests, rejections, rate_limit_retries = [], [], [], []
    allowed_errors = {
        "StructuredOutputValidationError",
        "MultipleStructuredOutputsError",
        "OutputParserException",
    }
    for attempt in range(1, max_attempts + 1):
        size = _prompt_size(system_prompt, messages, schema, kind)
        if size > settings.prompt_max_bytes:
            raise ValueError(
                "schema correction exceeds complete prompt byte budget: "
                f"serialized={size}, limit={settings.prompt_max_bytes}"
            )
        requests.append({"attempt": attempt, "serialized_input_bytes": size})
        if events:
            events.emit(
                "schema_attempt",
                stage="structured",
                attempt=attempt,
                strategy=settings.structured_method,
                serialized_input_bytes=size,
            )
        result, value, answer = None, None, None
        transport_recovery = None
        try:
            rate_limit_attempt = 0
            while True:
                try:
                    with _wall_clock_deadline(settings.wall_timeout_seconds):
                        result = runnable.invoke(
                            messages if kind == "native" else {"messages": messages},
                            config={"callbacks": [metrics], "recursion_limit": 4},
                        )
                    break
                except Exception as exc:
                    if not _is_rate_limit(exc):
                        raise
                    if rate_limit_attempt >= settings.rate_limit_retries:
                        if events:
                            events.emit(
                                "provider_rate_limit_exhausted",
                                stage="structured",
                                schema_attempt=attempt,
                                retries=rate_limit_attempt,
                            )
                        raise ProviderRateLimitError(
                            "provider rate-limit retry budget exhausted after "
                            f"{rate_limit_attempt} retries",
                            metrics={
                                "schema_attempts": attempt,
                                "failures": failures,
                                "requests": requests,
                                "rejections": rejections,
                                "rate_limit_retries": rate_limit_retries,
                                **metrics.as_dict(),
                            },
                        ) from exc
                    rate_limit_attempt += 1
                    delay = _rate_limit_delay(settings, rate_limit_attempt, exc)
                    retry = {
                        "schema_attempt": attempt,
                        "retry": rate_limit_attempt,
                        "maximum": settings.rate_limit_retries,
                        "delay_seconds": round(delay, 3),
                    }
                    rate_limit_retries.append(retry)
                    if events:
                        events.emit("provider_rate_limit_retry", stage="structured", **retry)
                    logger.warning(
                        "[provider] rate limit; retrying the same model call "
                        f"in {delay:.2f}s ({rate_limit_attempt}/"
                        f"{settings.rate_limit_retries})"
                    )
                    time.sleep(delay)
            answer = _last_answer(result, kind)
            _check_output_limit(answer)
            value = result.get("parsed" if kind == "native" else "structured_response")
            if kind == "native" and result.get("parsing_error"):
                error = result["parsing_error"]
                recovered = _native_tool_arguments(answer)
                recovery_channel = "normalized_tool_arguments"
                if recovered is _MISSING and settings.provider == "openrouter":
                    recovered = _native_invalid_tool_arguments(
                        answer, schema.__name__
                    )
                    recovery_channel = "invalid_tool_arguments"
                if recovered is _MISSING and settings.provider == "openrouter":
                    recovered = _native_json_content(answer)
                    recovery_channel = "assistant_json_content"
                if recovered is _MISSING:
                    if isinstance(error, ValidationError):
                        raise error
                    raise SchemaResponseError(
                        _native_transport_rejection(answer, schema.__name__)
                    )
                value = recovered
                transport_recovery = recovery_channel
                if events:
                    events.emit(
                        "structured_transport_recovered",
                        stage="structured",
                        attempt=attempt,
                        channel=recovery_channel,
                    )
            metric_values = metrics.as_dict()
            if transport_recovery is not None:
                metric_values["transport_recovery"] = transport_recovery
            final_thinking = provider_thinking_text(answer)
            captured = str(metric_values.get("thinking") or "").strip()
            if final_thinking and final_thinking not in captured:
                captured = "\n\n".join(value for value in (captured, final_thinking) if value)
            metric_values["thinking"] = captured
            return validate_output(schema, value), {
                "schema_attempts": attempt,
                "failures": failures,
                "requests": requests,
                "rejections": rejections,
                "rate_limit_retries": rate_limit_retries,
                **metric_values,
            }
        except Exception as exc:
            if isinstance(exc, (OutputBudgetError, ProviderWallClockTimeout)):
                # Retain the non-reasoning response for offline quality/format diagnosis.
                # It is never accepted, executed, or automatically retried.
                partial = _proposal_text(value, answer) if answer is not None else ""
                encoded = partial.encode("utf-8")
                partial_receipt = {
                    "text": encoded[:16000].decode("utf-8", errors="ignore"),
                    "original_bytes": len(encoded),
                    "truncated": len(encoded) > 16000,
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "usable_artifact": False,
                }
                exc.metrics = {
                    "partial_output": partial_receipt,
                    "schema_attempts": attempt,
                    "failures": failures,
                    "requests": requests,
                    "rejections": rejections,
                    "rate_limit_retries": rate_limit_retries,
                    **metrics.as_dict(),
                }
                raise
            if (
                not isinstance(exc, (ValidationError, SchemaResponseError))
                and type(exc).__name__ not in allowed_errors
            ):
                # A provider failure can follow a completed reasoning-only
                # response in this stage. Preserve every completed call receipt.
                if not getattr(exc, "metrics", None):
                    exc.metrics = metrics.as_dict()
                raise
            if answer is None:
                answer = getattr(exc, "ai_message", None)
            _check_output_limit(answer)
            failures.append(type(exc).__name__)
            detail = (
                str(exc.errors(include_input=False, include_url=False))[:1200]
                if isinstance(exc, ValidationError)
                else str(exc)[:1200] or type(exc).__name__
            )
            previous = _proposal_text(value, answer)
            # Keep one bounded failed artifact, not an ever-growing conversation.
            excerpt = previous[:6000]
            if events:
                events.emit(
                    "schema_rejected",
                    stage="structured",
                    attempt=attempt,
                    error_type=type(exc).__name__,
                    error_detail=detail,
                    previous_answer=excerpt,
                    previous_answer_truncated=len(previous) > len(excerpt),
                )
            rejections.append(
                {
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "error_detail": detail,
                    "previous_answer": excerpt,
                    "previous_answer_truncated": len(previous) > len(excerpt),
                }
            )
            if attempt == max_attempts:
                break
            text = correction_text
            if text is None:
                path = correction_template or (Path(__file__).parents[1] / "agents/prompts/schema_retry.j2")
                text = path.read_text(encoding="utf-8")
            correction = (
                Environment(undefined=StrictUndefined, autoescape=False)
                .from_string(text)
                .render(detail=detail, previous_output=excerpt)
            )
            messages = list(initial) + [{"role": "user", "content": correction}]
    last_detail = rejections[-1]["error_detail"] if rejections else "no validation detail"
    raise StructuredGenerationError(
        f"No valid {schema.__name__} after {max_attempts} attempts: {failures}; "
        f"last rejection: {last_detail}",
        metrics={
            "schema_attempts": max_attempts,
            "failures": failures,
            "requests": requests,
            "rejections": rejections,
            "rate_limit_retries": rate_limit_retries,
            **metrics.as_dict(),
        },
    )
