"""Shared types for the tool subsystem.

No competing representation exists elsewhere in the repository --
``yug.brain.types`` covers brain-facing types (``Message``,
``GenerationResult``), which are deliberately not reused here (a tool
call is not a brain generation call; keeping the two independent is
what lets tools be swapped or the brain be swapped without coupling
them together).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# The parameter type names a ToolDefinition's schema may use. Kept as
# a small closed set (JSON-Schema-like, but not a full JSON Schema
# implementation -- that would be a speculative abstraction Phase 3
# does not need).
_VALID_PARAMETER_TYPES = frozenset({"string", "integer", "number", "boolean"})

_PYTHON_TYPES_BY_NAME: dict[str, tuple[type, ...]] = {
    "string": (str,),
    # bool is an int subclass in Python; excluded explicitly so a
    # boolean value is never silently accepted where an integer/number
    # was requested (and vice versa -- see ParameterSpec.matches).
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


@dataclass(frozen=True)
class ParameterSpec:
    """One parameter in a :class:`ToolDefinition`'s schema."""

    name: str
    type: str
    required: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if self.type not in _VALID_PARAMETER_TYPES:
            raise ValueError(
                f"ParameterSpec.type must be one of {sorted(_VALID_PARAMETER_TYPES)}, got {self.type!r}"
            )

    def matches(self, value: Any) -> bool:
        """Whether ``value`` is an acceptable Python value for this
        parameter's declared type."""
        allowed = _PYTHON_TYPES_BY_NAME[self.type]
        if self.type != "boolean" and isinstance(value, bool):
            return False
        return isinstance(value, allowed)


@dataclass(frozen=True)
class ToolDefinition:
    """Metadata and execution entry point for one tool.

    ``handler`` is called with validated keyword arguments (already
    checked against ``parameters`` by
    :func:`yug.tools.validation.validate_parameters`) and must return a
    plain value (the tool's ``output``) or raise a
    :class:`~yug.tools.errors.ToolError` subclass on failure. It must
    not itself construct a :class:`ToolResult` -- that is the
    executor's job, keeping "what a tool computes" separate from "how
    a call outcome is reported."
    """

    name: str
    description: str
    parameters: tuple[ParameterSpec, ...]
    handler: Callable[..., Any]
    requires_permission: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ToolDefinition.name must be a non-empty string")
        if not self.requires_permission:
            object.__setattr__(self, "requires_permission", self.name)


@dataclass(frozen=True)
class ToolResult:
    """The outcome of one tool execution, as returned by
    :meth:`yug.tools.executor.ToolExecutor.execute`. Never raises --
    callers (eventually the agent runtime) inspect ``success`` rather
    than catching exceptions for ordinary tool failures."""

    success: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
