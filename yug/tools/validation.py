"""Parameter validation: checked BEFORE a tool's handler ever runs.
Rejects missing required parameters, wrong types, and unexpected
parameter names -- never lets a malformed call reach tool code."""

from __future__ import annotations

from typing import Any

from yug.tools.errors import ToolValidationError
from yug.tools.types import ToolDefinition


def validate_parameters(tool: ToolDefinition, parameters: dict[str, Any]) -> dict[str, Any]:
    """Validate ``parameters`` against ``tool.parameters``.

    Returns:
        A new dict containing exactly the tool's declared parameters
        (required ones present; optional ones present only if the
        caller supplied them) -- safe to pass as ``**kwargs`` to
        ``tool.handler``.

    Raises:
        ToolValidationError: a required parameter is missing, a
            parameter's value does not match its declared type, or
            ``parameters`` contains a key not declared by the tool.
    """
    if not isinstance(parameters, dict):
        raise ToolValidationError(f"parameters must be a dict, got {type(parameters).__name__}")

    declared = {spec.name: spec for spec in tool.parameters}

    unexpected = set(parameters) - set(declared)
    if unexpected:
        raise ToolValidationError(
            f"Unexpected parameter(s) for tool {tool.name!r}: {sorted(unexpected)}"
        )

    validated: dict[str, Any] = {}
    for spec in tool.parameters:
        if spec.name not in parameters:
            if spec.required:
                raise ToolValidationError(
                    f"Missing required parameter {spec.name!r} for tool {tool.name!r}"
                )
            continue

        value = parameters[spec.name]
        if not spec.matches(value):
            raise ToolValidationError(
                f"Parameter {spec.name!r} for tool {tool.name!r} must be of type "
                f"{spec.type!r}, got {type(value).__name__} ({value!r})"
            )
        validated[spec.name] = value

    return validated
