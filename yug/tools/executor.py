"""Tool executor: the ONLY sanctioned path to running a tool.

Pipeline, in order, matching Directive Section 13 exactly:

    registry lookup -> schema validation -> permission check -> execute
    -> structured ToolResult

A denied or invalid call never reaches the tool's handler. No
exception escapes :meth:`ToolExecutor.execute` as the normal API --
every outcome, success or failure, is a :class:`ToolResult`.

Synchronous, not ``async def``, deliberately: nothing else in this
repository uses asyncio (confirmed by inspection before writing this),
and none of the four Phase 3 tools has a genuine I/O-concurrency need
that would justify introducing it now -- python_exec's timeout is
handled via ``subprocess.run(timeout=...)``, not an event loop. This
follows Directive Rule 1 ("determine exact naming and API
conventions... do not recreate existing functionality") over the
prompt's own illustrative ``await executor.execute(...)`` example,
which is not itself a requirement.
"""

from __future__ import annotations

from typing import Any

from yug.tools.errors import ToolError, ToolNotFoundError
from yug.tools.permissions import PermissionChecker
from yug.tools.registry import ToolRegistry
from yug.tools.types import ToolResult
from yug.tools.validation import validate_parameters
from yug.utils.logging import get_logger

_logger = get_logger(__name__)


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, permission_checker: PermissionChecker) -> None:
        self._registry = registry
        self._permission_checker = permission_checker

    def execute(self, tool_name: str, parameters: dict[str, Any] | None = None) -> ToolResult:
        parameters = parameters or {}

        try:
            tool = self._registry.get(tool_name)
        except ToolNotFoundError as exc:
            _logger.info("tool_call_unknown_tool", extra={"tool_name": tool_name})
            return ToolResult(success=False, error=str(exc))

        if not self._permission_checker.is_allowed(tool_name):
            _logger.info("tool_call_permission_denied", extra={"tool_name": tool_name})
            return ToolResult(success=False, error=f"Permission denied for tool: {tool_name!r}")

        try:
            validated = validate_parameters(tool, parameters)
        except ToolError as exc:
            _logger.info("tool_call_validation_failed", extra={"tool_name": tool_name})
            return ToolResult(success=False, error=str(exc))

        try:
            output = tool.handler(**validated)
        except ToolError as exc:
            _logger.info("tool_call_failed", extra={"tool_name": tool_name})
            return ToolResult(success=False, error=str(exc))
        except Exception as exc:  # noqa: BLE001 -- deliberate: last-resort safety
            # net so an unexpected bug in a tool handler surfaces as a
            # structured failure, not an exception escaping the
            # executor's documented API. The exception type is
            # preserved in the message so it is not silently swallowed.
            _logger.info("tool_call_unexpected_error", extra={"tool_name": tool_name})
            return ToolResult(success=False, error=f"{type(exc).__name__}: {exc}")

        _logger.info("tool_call_succeeded", extra={"tool_name": tool_name})
        return ToolResult(success=True, output=output)
