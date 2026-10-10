from __future__ import annotations

from yug.tools.builtin import build_default_tools
from yug.tools.executor import ToolExecutor
from yug.tools.permissions import PermissionChecker
from yug.tools.registry import ToolRegistry
from yug.tools.types import ToolDefinition


def _executor_with_default_tools(tmp_path, enabled):
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


def test_allowed_tool_executes_successfully(tmp_path):
    executor = _executor_with_default_tools(tmp_path, enabled={"calculator"})

    result = executor.execute("calculator", {"expression": "2 + 2"})

    assert result.success is True
    assert result.output == 4
    assert result.error is None


def test_tool_not_in_enabled_list_is_denied(tmp_path):
    executor = _executor_with_default_tools(tmp_path, enabled={"calculator"})

    result = executor.execute("python_exec", {"code": "print(1)"})

    assert result.success is False
    assert "Permission denied" in result.error


def test_unknown_tool_returns_structured_failure(tmp_path):
    executor = _executor_with_default_tools(tmp_path, enabled={"calculator"})

    result = executor.execute("does_not_exist", {})

    assert result.success is False
    assert "Unknown tool" in result.error


def test_invalid_parameters_returns_structured_failure(tmp_path):
    executor = _executor_with_default_tools(tmp_path, enabled={"calculator"})

    result = executor.execute("calculator", {"wrong_param_name": "2 + 2"})

    assert result.success is False
    assert result.error is not None


def test_denied_execution_never_reaches_the_tool_handler():
    calls = []
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(name="spy", description="d", parameters=(), handler=lambda: calls.append("ran"))
    )
    executor = ToolExecutor(registry, PermissionChecker(enabled_tools=frozenset()))

    result = executor.execute("spy", {})

    assert result.success is False
    assert calls == []


def test_unexpected_exception_in_handler_becomes_a_result_not_a_raise():
    registry = ToolRegistry()

    def boom():
        raise RuntimeError("boom")

    registry.register(ToolDefinition(name="boom", description="d", parameters=(), handler=boom))
    executor = ToolExecutor(registry, PermissionChecker(enabled_tools=frozenset({"boom"})))

    result = executor.execute("boom", {})  # must not raise

    assert result.success is False
    assert "RuntimeError" in result.error


def test_execute_with_no_parameters_argument_defaults_to_empty_dict():
    registry = ToolRegistry()
    registry.register(ToolDefinition(name="noop", description="d", parameters=(), handler=lambda: "ok"))
    executor = ToolExecutor(registry, PermissionChecker(enabled_tools=frozenset({"noop"})))

    result = executor.execute("noop")

    assert result.success is True
    assert result.output == "ok"
