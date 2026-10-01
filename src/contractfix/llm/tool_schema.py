"""Provider transport schemas; application validation always keeps all bounds."""

from copy import deepcopy
from typing import Any


def openrouter_tool(tool: Any, model: str) -> dict[str, Any]:
    from langchain_core.utils.function_calling import convert_to_openai_tool

    converted = convert_to_openai_tool(tool)
    return gemini_tool_schema(converted) if model.startswith("google/gemini-") else converted


def gemini_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Reduce grammar complexity without changing field names, types, or enums.

    Nested bounded arrays (12 edits times two 400-line arrays) can overwhelm
    Gemini's schema compiler. Bounds remain authoritative in Pydantic; transmit
    them as descriptions instead of provider grammar constraints. This is a
    compatibility mitigation, not proof that every upstream endpoint accepts it.
    """
    result = deepcopy(tool)
    bounds = {"minItems", "maxItems", "minLength", "maxLength", "minimum",
              "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}

    def visit(node: dict) -> None:
        constraints = {key: node.pop(key) for key in sorted(bounds) if key in node}
        node.pop("default", None)
        if constraints:
            note = "; ".join(f"{key}={value}" for key, value in constraints.items())
            node["description"] = (node.get("description", "") +
                                   f" Host validates: {note}.").strip()
        for key in ("properties", "$defs", "definitions"):
            for child in node.get(key, {}).values():
                visit(child)
        for key in ("items", "additionalProperties"):
            if isinstance(node.get(key), dict):
                visit(node[key])
        for key in ("anyOf", "allOf", "oneOf", "prefixItems"):
            for child in node.get(key, []):
                visit(child)

    parameters = result.get("function", {}).get("parameters")
    if isinstance(parameters, dict):
        visit(parameters)
    return result
