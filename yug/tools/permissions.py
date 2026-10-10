"""Minimal Phase 3 permission boundary.

No permission/security subsystem exists elsewhere in the repository
(confirmed by inspection before implementing this -- there is no
``yug/security/`` or autonomy-level concept anywhere). Per Directive
Section 12 ("For Phase 3, maintain the existing autonomy model. Do not
implement higher autonomy levels merely because they appear in future
architecture documents"), this is deliberately the smallest possible
gate: an explicit allow-list of enabled tool names, driven by
:class:`yug.config.schema.ToolsConfig`. A real autonomy-level /
approval-workflow permission system is future-phase scope, not
invented here.
"""

from __future__ import annotations


class PermissionChecker:
    """Explicitly constructed, not a global -- mirrors
    :class:`yug.memory.working_memory.WorkingMemory`'s "no hidden
    state" convention."""

    def __init__(self, enabled_tools: frozenset[str]) -> None:
        self._enabled_tools = enabled_tools

    def is_allowed(self, tool_name: str) -> bool:
        return tool_name in self._enabled_tools
