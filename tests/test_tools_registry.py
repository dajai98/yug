from __future__ import annotations

import pytest

from yug.tools.errors import ToolAlreadyRegisteredError, ToolNotFoundError
from yug.tools.registry import ToolRegistry
from yug.tools.types import ToolDefinition


def _tool(name="x"):
    return ToolDefinition(name=name, description="d", parameters=(), handler=lambda: 1)


def test_register_then_get_returns_the_same_definition():
    registry = ToolRegistry()
    tool = _tool()
    registry.register(tool)

    assert registry.get("x") is tool


def test_has_reflects_registration_state():
    registry = ToolRegistry()
    assert registry.has("x") is False
    registry.register(_tool())
    assert registry.has("x") is True


def test_list_tools_is_sorted():
    registry = ToolRegistry()
    registry.register(_tool("zeta"))
    registry.register(_tool("alpha"))

    assert registry.list_tools() == ["alpha", "zeta"]


def test_duplicate_registration_rejected():
    registry = ToolRegistry()
    registry.register(_tool())

    with pytest.raises(ToolAlreadyRegisteredError):
        registry.register(_tool())


def test_get_unknown_tool_raises():
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError):
        registry.get("nope")


def test_unregister_unknown_tool_raises():
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError):
        registry.unregister("nope")


def test_unregister_removes_the_tool():
    registry = ToolRegistry()
    registry.register(_tool())
    registry.unregister("x")

    assert registry.has("x") is False


def test_two_registry_instances_are_isolated():
    registry_a = ToolRegistry()
    registry_b = ToolRegistry()
    registry_a.register(_tool())

    assert registry_b.has("x") is False
