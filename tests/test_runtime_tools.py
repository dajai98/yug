from __future__ import annotations

from yug.brain.adapters.mock import MockBrainAdapter
from yug.memory.working_memory import WorkingMemory
from yug.runtime.core import YugCore
from yug.tools.builtin import build_default_tools
from yug.tools.executor import ToolExecutor
from yug.tools.permissions import PermissionChecker
from yug.tools.registry import ToolRegistry
from yug.tools.types import ToolDefinition


def _executor(tmp_path, enabled):
    registry = ToolRegistry()
    for tool in build_default_tools(
        file_root=str(tmp_path),
        file_read_max_bytes=1_000_000,
        file_search_max_results=100,
        file_search_max_file_bytes=1_000_000,
        python_exec_timeout_seconds=2.0,
    ):
        registry.register(tool)
    return ToolExecutor(registry, PermissionChecker(enabled_tools=frozenset(enabled)))


def test_explicit_tool_call_reaches_the_executor_and_succeeds(tmp_path):
    core = YugCore(MockBrainAdapter(), WorkingMemory(), _executor(tmp_path, {"calculator"}))

    response = core.respond("calc please", tool_name="calculator", tool_parameters={"expression": "2 + 2"})

    assert response.tool_result.success is True
    assert response.tool_result.output == 4
    assert response.success is True  # brain call succeeds independently of the tool outcome


def test_denied_tool_call_remains_denied(tmp_path):
    core = YugCore(MockBrainAdapter(), WorkingMemory(), _executor(tmp_path, enabled=set()))

    response = core.respond("calc please", tool_name="calculator", tool_parameters={"expression": "2 + 2"})

    assert response.tool_result.success is False
    assert "Permission denied" in response.tool_result.error


def test_invalid_tool_parameters_remain_invalid(tmp_path):
    core = YugCore(MockBrainAdapter(), WorkingMemory(), _executor(tmp_path, {"calculator"}))

    response = core.respond("calc please", tool_name="calculator", tool_parameters={"wrong_param": "x"})

    assert response.tool_result.success is False


def test_no_executor_configured_produces_structured_failure_not_exception():
    core = YugCore(MockBrainAdapter(), WorkingMemory())  # executor=None

    response = core.respond("calc please", tool_name="calculator", tool_parameters={"expression": "1+1"})

    assert response.tool_result.success is False
    assert "No ToolExecutor" in response.tool_result.error
    assert response.success is True  # the brain part still worked


def test_no_tool_name_means_no_tool_result(tmp_path):
    core = YugCore(MockBrainAdapter(), WorkingMemory(), _executor(tmp_path, {"calculator"}))

    response = core.respond("just chatting, no tool")

    assert response.tool_result is None


def test_runtime_never_bypasses_the_executor_permission_check():
    """A denied call must never reach the tool's own handler."""
    calls = []
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(name="spy", description="d", parameters=(), handler=lambda: calls.append("ran"))
    )
    executor = ToolExecutor(registry, PermissionChecker(enabled_tools=frozenset()))
    core = YugCore(MockBrainAdapter(), WorkingMemory(), executor)

    core.respond("test", tool_name="spy", tool_parameters={})

    assert calls == []
