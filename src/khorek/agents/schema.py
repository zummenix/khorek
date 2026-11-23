"""Shared helpers for terminal tool JSON schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from khorek.llm import Tool


def _inline_refs(node: Any, definitions: dict[str, Any]) -> Any:
    """Inline JSON-schema ``$ref``s.

    Pydantic nests named models under ``$defs`` and refers to them with
    ``$ref``. Small local models (the prototype target) often mishandle that.
    Inlining keeps the exact field/enum constraints in a flat schema the model
    can follow.
    """
    if isinstance(node, dict):
        if "$ref" in node:
            reference = str(node["$ref"])
            name = reference.rsplit("/", 1)[-1]
            resolved = dict(definitions.get(name, {}))
            resolved.update({key: value for key, value in node.items() if key != "$ref"})
            return _inline_refs(resolved, definitions)
        return {
            key: _inline_refs(value, definitions) for key, value in node.items() if key != "$defs"
        }
    if isinstance(node, list):
        return [_inline_refs(item, definitions) for item in node]
    return node


def flat_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return ``model_json_schema()`` with every ``$ref`` inlined."""
    schema = model.model_json_schema()
    definitions = schema.get("$defs", {})
    if not isinstance(definitions, dict):
        definitions = {}
    flattened = _inline_refs(schema, definitions)
    assert isinstance(flattened, dict)
    return flattened


def terminal_tool(name: str, description: str, model: type[BaseModel]) -> Tool:
    """Build a terminal tool with a flat Pydantic-generated parameter schema."""
    return Tool(
        name=name,
        description=description,
        parameters=flat_schema(model),
        handler=None,
    )
