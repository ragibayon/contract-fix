"""Per-call accounting that never retries or counts callback data twice."""

from __future__ import annotations

import threading
import time
import math
from typing import Any, Mapping
from uuid import uuid4

from contractfix.utils.artifacts import redact
from contractfix.utils.logger import logger

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    # Accounting stays unit-testable offline; client construction still needs the SDK.
    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (ValueError, TypeError, OverflowError):
        return None
    if parsed < 0 or isinstance(value, float) and value != parsed:
        return None
    return parsed


def _cost(value: Any) -> float | None:
    """Accept a finite, non-negative provider-reported USD cost."""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def normalize_token_usage(raw: Any) -> dict[str, int | None] | None:
    """Normalize OpenAI, LangChain, and Ollama usage field names."""
    if not isinstance(raw, Mapping):
        return None
    for input_name, output_name, total_name in (
        ("input_tokens", "output_tokens", "total_tokens"),
        ("prompt_tokens", "completion_tokens", "total_tokens"),
        ("prompt_eval_count", "eval_count", "total_tokens"),
    ):
        values = {
            "input_tokens": _integer(raw.get(input_name)),
            "output_tokens": _integer(raw.get(output_name)),
            "total_tokens": _integer(raw.get(total_name)),
        }
        if all(value is None for value in values.values()):
            continue
        if (
            values["total_tokens"] is None
            and values["input_tokens"] is not None
            and values["output_tokens"] is not None
        ):
            values["total_tokens"] = values["input_tokens"] + values["output_tokens"]
        return values
    return None


def normalize_reasoning_tokens(raw: Any) -> int | None:
    """Extract provider/SDK variants of the thinking-token counter."""
    if not isinstance(raw, Mapping):
        return None
    for key in ("reasoning_tokens", "reasoning"):
        direct = _integer(raw.get(key))
        if direct is not None:
            return direct
    for key in ("completion_tokens_details", "output_tokens_details", "output_token_details"):
        details = raw.get(key)
        if isinstance(details, Mapping):
            for detail_key in ("reasoning_tokens", "reasoning"):
                value = _integer(details.get(detail_key))
                if value is not None:
                    return value
    return None


def provider_thinking_text(message: Any) -> str:
    """Extract only provider-exposed reasoning, never ordinary answer text."""
    if message is None:
        return ""
    additional = getattr(message, "additional_kwargs", {}) or {}
    metadata = getattr(message, "response_metadata", {}) or {}
    for container in (additional, metadata):
        if not isinstance(container, Mapping):
            continue
        for key in ("reasoning_content", "reasoning", "thinking"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    content = getattr(message, "content", "")
    if isinstance(content, list):
        blocks = [
            str(block.get("text", "")).strip()
            for block in content
            if isinstance(block, Mapping) and block.get("type") in {"reasoning", "thinking"}
        ]
        return "\n\n".join(block for block in blocks if block)
    return ""


def message_metadata(message: Any, llm_output: Any = None) -> dict[str, Any]:
    """Extract safe usage and response metadata from provider objects."""
    metadata = getattr(message, "response_metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    output = llm_output if isinstance(llm_output, Mapping) else {}
    candidates = (
        getattr(message, "usage_metadata", None),
        metadata.get("token_usage"),
        metadata.get("usage"),
        metadata,
        output.get("token_usage"),
        output.get("usage"),
        output,
    )
    usage = next(
        (
            normalized
            for candidate in candidates
            if (normalized := normalize_token_usage(candidate)) is not None
        ),
        None,
    )
    cost_candidates = [metadata.get("cost"), output.get("cost")]
    for container in (
        getattr(message, "usage_metadata", None),
        metadata.get("usage"),
        metadata.get("token_usage"),
        output.get("usage"),
        output.get("token_usage"),
    ):
        if isinstance(container, Mapping):
            cost_candidates.extend(
                [
                    container.get("cost"),
                    container.get("total_cost"),
                    container.get("reported_cost_usd"),
                ]
            )
    cost = next(
        (parsed for value in cost_candidates if (parsed := _cost(value)) is not None),
        None,
    )
    provider = (
        metadata.get("model_provider")
        or metadata.get("provider")
        or output.get("model_provider")
        or output.get("provider")
    )
    cost_assurance = "provider_reported_not_estimated"
    if cost is None and str(provider).lower() == "ollama":
        # A local Ollama response has no provider-billed USD amount. Record the
        # API charge as zero while keeping the provenance distinct from a
        # provider-reported price. Hardware/electricity cost is out of scope.
        cost = 0.0
        cost_assurance = "local_inference_zero_api_charge"
    reasoning_tokens = next(
        (value for candidate in candidates if (value := normalize_reasoning_tokens(candidate)) is not None),
        None,
    )
    return redact(
        {
            "usage": usage,
            "reasoning_tokens": reasoning_tokens,
            "reported_cost_usd": cost,
            "cost_assurance": cost_assurance,
            "model_name": metadata.get("model_name") or metadata.get("model") or output.get("model_name"),
            "provider_reported": metadata.get("provider"),
            "thinking": provider_thinking_text(message) or None,
            "finish_reason": metadata.get("finish_reason") or metadata.get("done_reason"),
            "response_id": getattr(message, "id", None),
        }
    )


class LLMRunMetrics(BaseCallbackHandler):
    """Record each logical SDK call by LangChain run ID."""

    def __init__(self, events: Any = None, stage: str = "llm") -> None:
        self.events = events
        self.stage = stage
        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.started: float | None = None
        self.ended: float | None = None

    def _start(self, run_id: Any = None) -> str:
        key = str(run_id or uuid4())
        with self._lock:
            now = time.perf_counter()
            if self.started is None:
                self.started = now
            self.records.setdefault(key, {"run_id": key, "started": now, "status": "running"})
        return key

    def on_chat_model_start(
        self, serialized: Any, messages: Any, *, run_id: Any = None, **kwargs: Any
    ) -> None:
        key = self._start(run_id)
        if self.events:
            self.events.emit("llm_started", stage=self.stage, call_id=key)

    def on_llm_start(self, serialized: Any, prompts: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        self.on_chat_model_start(serialized, prompts, run_id=run_id, **kwargs)

    def _finish(
        self,
        run_id: Any,
        metadata: Mapping[str, Any],
        status: str,
        error: Exception | None = None,
    ) -> None:
        with self._lock:
            if run_id is None:
                running = [key for key, row in self.records.items() if row["status"] == "running"]
                key = running[0] if len(running) == 1 else str(uuid4())
            else:
                key = str(run_id)
            if key not in self.records:
                self._start(key)
            row = self.records[key]
            if row["status"] != "running":
                return
            self.ended = time.perf_counter()
            row.update(
                metadata,
                status=status,
                latency_seconds=self.ended - row["started"],
            )
            if error is not None:
                row["error_type"] = type(error).__name__
            if self.events:
                self.events.emit(
                    "llm_finished",
                    stage=self.stage,
                    **{key: value for key, value in row.items() if key != "started"},
                )

    def on_llm_end(self, response: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        try:
            message = response.generations[0][0].message
        except (AttributeError, IndexError, TypeError):
            message = None
        metadata = message_metadata(message, getattr(response, "llm_output", None))
        self._finish(run_id, metadata, "completed")
        with self._lock:
            row = self.records.get(str(run_id), {})
        usage = metadata.get("usage") or {}
        summary = (
            f"[provider] SDK call finished | duration={row.get('latency_seconds', 'unreported')}s "
            f"| finish={metadata.get('finish_reason') or 'unreported'} "
            f"| input={usage.get('input_tokens', 'unreported')} "
            f"| output={usage.get('output_tokens', 'unreported')} "
            f"| thinking={metadata.get('reasoning_tokens', 'unreported')} "
            f"| cost={metadata.get('reported_cost_usd', 'unreported')}"
        )
        if metadata.get("finish_reason") in {"length", "max_tokens", "max_output_tokens"}:
            logger.warning(summary + " | output limit exhausted")
        else:
            logger.debug(summary)

    def on_llm_error(self, error: Exception, *, run_id: Any = None, **kwargs: Any) -> None:
        self._finish(run_id, {"usage": None}, "error", error)
        logger.debug(f"[provider] SDK call failed | {type(error).__name__}: {error}")

    def capture_from_message(self, message: Any, *, run_id: Any = None) -> None:
        """Fallback for transports that do not invoke callbacks."""
        key = str(run_id or getattr(message, "id", None) or uuid4())
        self._finish(key, message_metadata(message), "completed")

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            rows = [
                {key: value for key, value in row.items() if key != "started"}
                for row in self.records.values()
            ]
        totals: dict[str, int | None] = {}
        missing: dict[str, int] = {}
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            values = [(row.get("usage") or {}).get(field) for row in rows]
            known = [value for value in values if value is not None]
            totals[field] = sum(known) if known else None
            missing[field] = len(values) - len(known)
        # Failed provider calls have no completed response cost receipt.
        # Keep them visible without calling them completed calls with missing cost.
        completed = [row for row in rows if row["status"] == "completed"]
        costs = [row.get("reported_cost_usd") for row in completed]
        known_costs = [cost for cost in costs if cost is not None]
        cost_assurances = {
            row.get("cost_assurance")
            for row in rows
            if row.get("reported_cost_usd") is not None and row.get("cost_assurance")
        }
        reasoning = [row.get("reasoning_tokens") for row in rows]
        known_reasoning = [value for value in reasoning if value is not None]
        thinking = [row.get("thinking") for row in rows if row.get("thinking")]
        return {
            "call_count": len(rows),
            "call_count_unit": "logical_sdk_calls",
            "hidden_http_retry_count": None,
            "completed_calls": sum(row["status"] == "completed" for row in rows),
            "error_calls": sum(row["status"] == "error" for row in rows),
            "unpriced_error_calls": sum(
                row["status"] == "error" and row.get("reported_cost_usd") is None
                for row in rows
            ),
            "latency_seconds": (
                None if self.started is None or self.ended is None else self.ended - self.started
            ),
            "token_usage": totals,
            "missing_usage_calls": missing,
            "reported_cost_usd": sum(known_costs) if known_costs else None,
            "reasoning_tokens": sum(known_reasoning) if known_reasoning else None,
            "thinking": "\n\n".join(
                f"SDK call {index}:\n{value}" for index, value in enumerate(thinking, start=1)
            ),
            "missing_reasoning_token_calls": len(reasoning) - len(known_reasoning),
            "missing_cost_calls": len(costs) - len(known_costs),
            "cost_assurance": (
                next(iter(cost_assurances))
                if len(cost_assurances) == 1
                else "mixed_or_unreported"
            ),
            "calls": rows,
        }


def invoke_with_metrics(
    runnable: Any,
    payload: Any,
    metrics: LLMRunMetrics | None = None,
    *,
    config: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[Any, LLMRunMetrics]:
    """Invoke exactly once; retries belong at a safe leaf transport."""
    metrics = metrics or LLMRunMetrics()
    invoke_config = dict(config or {})
    callbacks = list(invoke_config.get("callbacks", []))
    if metrics not in callbacks:
        callbacks.append(metrics)
    invoke_config["callbacks"] = callbacks
    response = runnable.invoke(payload, config=invoke_config, **kwargs)
    return response, metrics
