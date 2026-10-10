"""Shared types used across YUG subsystems.

Scope note: per the "no speculative abstractions" and "do not implement
future milestone functionality" rules in the Chief Implementation
Directive (Sections 24, 26), this module holds only the types Phase 0
code actually consumes. Brain-, memory-, tool-, agent-, and
verification-related types (Message, ToolDefinition, Plan, AgentState,
GenerationResult, etc.) belong to their owning subsystem modules
(e.g. ``yug/brain/interface.py``, ``yug/tools/schemas.py``,
``yug/agent/state_machine.py``) and will be added there when those
phases are implemented, not declared here in advance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VersionInfo:
    """Identifies the running YUG build.

    Used by the CLI ``--version`` command and by anything (e.g. audit
    logs, later health checks) that needs to report which YUG version
    produced an output.
    """

    version: str
    milestone: str

    def __str__(self) -> str:
        return f"yug {self.version} ({self.milestone})"
