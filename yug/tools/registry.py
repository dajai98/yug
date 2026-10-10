"""Tool registry: name -> :class:`~yug.tools.types.ToolDefinition`
lookup. Registration and execution are deliberately separate concerns
-- the registry never calls a tool's handler."""

from __future__ import annotations

from yug.tools.errors import ToolAlreadyRegisteredError, ToolNotFoundError
from yug.tools.types import ToolDefinition


class ToolRegistry:
    """An explicitly-constructed registry. No hidden module-level
    global: two instances never share state unless a caller
    deliberately shares tool objects between them."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Raises:
        ToolAlreadyRegisteredError: a tool with this name is already
            registered. Re-registering the exact same object is not a
            special case -- it is rejected the same as any duplicate,
            so accidental double-registration is always visible.
        """
        if tool.name in self._tools:
            raise ToolAlreadyRegisteredError(f"Tool already registered: {tool.name!r}")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Raises:
        ToolNotFoundError: no tool with this name is registered.
        """
        if name not in self._tools:
            raise ToolNotFoundError(f"Cannot unregister unknown tool: {name!r}")
        del self._tools[name]

    def get(self, name: str) -> ToolDefinition:
        """Raises:
        ToolNotFoundError: no tool with this name is registered.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(f"Unknown tool: {name!r}") from None

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[str]:
        """Registered tool names, sorted for deterministic ordering."""
        return sorted(self._tools)
