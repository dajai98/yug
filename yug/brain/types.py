"""Types shared by :class:`BrainInterface` and its adapters.

Scope note: ``ToolDefinition`` is included here only because it appears
in the frozen ``BrainInterface.generate()`` signature (Directive,
"BRAIN INTERFACE" section). It is a minimal structural placeholder --
enough for a brain adapter to pass tool schemas through to a model that
supports them. The real tool system (``ToolRegistry``, ``ToolExecutor``,
permission integration, execution) is Phase 3 scope and is NOT
implemented here; no adapter in Phase 1 actually invokes a tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class Message:
    """A single turn in a conversation passed to the brain.

    ``role`` is intentionally a plain string (not an enum) in Phase 1:
    different backends use different role vocabularies (e.g. "system",
    "user", "assistant", "tool"), and normalizing that is an adapter
    concern, not a core-type concern.
    """

    role: str
    content: str


@dataclass(frozen=True)
class ToolDefinition:
    """Minimal tool schema placeholder (see module docstring).

    Not exercised by any Phase 1 adapter; exists so the frozen
    ``generate()`` signature type-checks. The full schema (JSON Schema
    parameter validation, execution boundary, permissions) is defined
    in Phase 3's ``yug/tools/schemas.py``.
    """

    name: str
    description: str
    parameters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    """Normalized output of :meth:`BrainInterface.generate`.

    Every adapter must translate its backend-specific response into
    this shape, so callers never depend on a backend's native response
    format (architectural invariant #1).
    """

    content: str
    finish_reason: str
    model_id: str
    tool_calls: tuple | None = None
    usage: dict | None = None  # e.g. {"prompt_tokens": .., "completion_tokens": ..}


@dataclass(frozen=True)
class Embedding:
    """A single embedding vector produced by :meth:`BrainInterface.embed`."""

    values: tuple[float, ...]
    model_id: str

    @property
    def dim(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class ModelCapabilities:
    """Model metadata. ``parameter_count`` is informational ONLY.

    Architectural invariant: no code anywhere in YUG may branch on
    ``parameter_count`` (Directive, "MODEL CAPABILITIES": "NEVER use
    parameter_count as runtime logic."). It exists purely for human
    reference / logging.
    """

    model_id: str
    context_length: int
    supports_tools: bool
    supports_vision: bool
    embedding_dim: int | None
    parameter_count: int | None = None
    quantization: str | None = None
    recommended_temperature_range: tuple[float, float] = (0.0, 1.0)
    known_limitations: tuple[str, ...] = ()


class HealthState(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class HealthStatus:
    """Result of :meth:`BrainInterface.health_check`.

    Must reflect genuinely observed state. An adapter must never report
    HEALTHY without having actually checked; "not yet checked" is
    DEGRADED or UNAVAILABLE with an explanatory ``detail``, never a
    guess.
    """

    state: HealthState
    detail: str
    model_loaded: bool
