"""Host-authored Python runtime monitor used only during contract evaluation."""

# Keep this runtime-only module compatible with Python 3.6 task images.

import functools
import inspect
import json
import os
from pathlib import Path


def _emit(event: str, **data: object) -> None:
    root = Path(os.environ["CONTRACTFIX_EC_EVENTS"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"executable-contract.{os.getpid()}.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, **data}, default=str) + "\n")


def observe(target: str):
    """Record target entry and termination without trusting generated contract state."""

    def decorate(function):
        # Instrumentation normally wraps the body before other decorators.
        # Also preserve descriptors when this helper is used directly.
        if isinstance(function, classmethod):
            return classmethod(decorate(function.__func__))
        if isinstance(function, staticmethod):
            return staticmethod(decorate(function.__func__))
        if isinstance(function, property):
            getter = decorate(function.fget) if function.fget is not None else None
            return property(getter, function.fset, function.fdel, function.__doc__)

        if inspect.iscoroutinefunction(function):
            @functools.wraps(function)
            async def wrapped_async(*args, **kwargs):
                _emit("contracted_operation_reached", target=target)
                try:
                    value = await function(*args, **kwargs)
                except BaseException as exc:
                    _emit(
                        "contracted_operation_raised",
                        target=target,
                        exception_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
                        detail=str(exc),
                    )
                    raise
                _emit("contracted_operation_returned", target=target)
                return value

            return wrapped_async

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            _emit("contracted_operation_reached", target=target)
            try:
                value = function(*args, **kwargs)
            except BaseException as exc:
                _emit(
                    "contracted_operation_raised",
                    target=target,
                    exception_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
                    detail=str(exc),
                )
                raise
            _emit("contracted_operation_returned", target=target)
            return value

        return wrapped

    return decorate
