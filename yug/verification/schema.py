"""SchemaVerifier: deterministic structural validation of arbitrary
data against an explicitly supplied schema. No LLM, no embeddings, no
semantic interpretation -- pure structural comparison, same category
of check as ``yug.tools.validation.validate_parameters`` but for
general nested data rather than specifically a tool call's parameters
(deliberately NOT reusing that function directly: it is
tool-call-shaped -- keyed off ``ToolDefinition.parameters`` -- and
retrofitting it to arbitrary nested schemas would either weaken it or
require modifying yug.tools, both against this milestone's rules).

Minimal schema format supported (a small, closed subset -- not a full
JSON Schema implementation, which would be speculative infrastructure
this milestone does not need):

    {"type": "string" | "integer" | "number" | "boolean"}
    {"type": "object", "properties": {name: <schema>, ...}, "required": [name, ...]}

Objects reject any field not listed in "properties" (no
"additionalProperties" escape hatch -- kept simple deliberately).
"properties"/"required" are optional on an object schema (default to
no properties / nothing required).
"""

from __future__ import annotations

from typing import Any

from yug.verification.errors import SchemaVerificationError
from yug.verification.types import VerificationResult

_SCALAR_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
}
_VALID_TYPES = frozenset(_SCALAR_TYPE_CHECKS) | {"object"}


def _validate_schema_shape(schema: Any, path: str = "$") -> None:
    if not isinstance(schema, dict):
        raise SchemaVerificationError(f"schema at {path} must be a dict, got {type(schema).__name__}")

    schema_type = schema.get("type")
    if schema_type not in _VALID_TYPES:
        raise SchemaVerificationError(
            f"schema at {path}: 'type' must be one of {sorted(_VALID_TYPES)}, got {schema_type!r}"
        )

    if schema_type == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise SchemaVerificationError(f"schema at {path}: 'properties' must be a dict")
        for name, sub_schema in properties.items():
            _validate_schema_shape(sub_schema, path=f"{path}.{name}")

        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(name, str) for name in required):
            raise SchemaVerificationError(f"schema at {path}: 'required' must be a list of strings")


def _check(data: Any, schema: dict, path: str) -> list[str]:
    schema_type = schema["type"]

    if schema_type == "object":
        if not isinstance(data, dict):
            return [f"{path}: expected object, got {type(data).__name__}"]

        properties: dict = schema.get("properties", {})
        required: list = schema.get("required", [])

        reasons = [f"{path}.{name}: missing required field" for name in required if name not in data]

        unexpected = set(data) - set(properties)
        if unexpected:
            reasons.append(f"{path}: unexpected field(s) {sorted(unexpected)}")

        for name, sub_schema in properties.items():
            if name in data:
                reasons.extend(_check(data[name], sub_schema, f"{path}.{name}"))
        return reasons

    if not _SCALAR_TYPE_CHECKS[schema_type](data):
        return [f"{path}: expected {schema_type}, got {type(data).__name__}"]
    return []


def verify_schema(data: Any, schema: dict) -> VerificationResult:
    """Check ``data`` against ``schema``.

    Raises:
        SchemaVerificationError: ``schema`` itself is malformed (see
            module docstring for the supported format). NOT raised for
            valid ``data`` that simply doesn't match a valid schema --
            that produces ``VerificationResult(passed=False)``.
    """
    _validate_schema_shape(schema)

    reasons = _check(data, schema, path="$")
    if reasons:
        return VerificationResult(passed=False, verifier="schema", reason="; ".join(reasons))
    return VerificationResult(passed=True, verifier="schema")
