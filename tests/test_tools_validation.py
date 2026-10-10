from __future__ import annotations

import pytest

from yug.tools.errors import ToolValidationError
from yug.tools.types import ParameterSpec, ToolDefinition
from yug.tools.validation import validate_parameters


def _tool(*specs):
    return ToolDefinition(name="t", description="d", parameters=specs, handler=lambda **kw: kw)


def test_valid_parameters_pass_through():
    tool = _tool(ParameterSpec("a", "integer"))

    assert validate_parameters(tool, {"a": 5}) == {"a": 5}


def test_optional_parameter_may_be_omitted():
    tool = _tool(ParameterSpec("a", "integer", required=False))

    assert validate_parameters(tool, {}) == {}


def test_missing_required_parameter_raises():
    tool = _tool(ParameterSpec("a", "integer", required=True))

    with pytest.raises(ToolValidationError):
        validate_parameters(tool, {})


def test_wrong_type_raises():
    tool = _tool(ParameterSpec("a", "integer"))

    with pytest.raises(ToolValidationError):
        validate_parameters(tool, {"a": "not an int"})


def test_unexpected_parameter_raises():
    tool = _tool(ParameterSpec("a", "integer"))

    with pytest.raises(ToolValidationError):
        validate_parameters(tool, {"a": 1, "b": "unexpected"})


def test_bool_rejected_where_integer_expected():
    """bool is an int subclass in Python -- must not be silently
    accepted as a valid integer parameter."""
    tool = _tool(ParameterSpec("a", "integer"))

    with pytest.raises(ToolValidationError):
        validate_parameters(tool, {"a": True})


def test_number_type_accepts_both_int_and_float():
    tool = _tool(ParameterSpec("a", "number"))

    assert validate_parameters(tool, {"a": 5})["a"] == 5
    assert validate_parameters(tool, {"a": 5.5})["a"] == 5.5


def test_non_dict_parameters_raises():
    tool = _tool(ParameterSpec("a", "integer"))

    with pytest.raises(ToolValidationError):
        validate_parameters(tool, ["not", "a", "dict"])  # type: ignore[arg-type]
