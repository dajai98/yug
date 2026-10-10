from __future__ import annotations

import pytest

from yug.verification.errors import SchemaVerificationError
from yug.verification.schema import verify_schema


def test_valid_data_passes():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
        "required": ["a", "b"],
    }

    result = verify_schema({"a": 1, "b": "x"}, schema)

    assert result.passed is True


def test_missing_required_field_fails():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}

    result = verify_schema({}, schema)

    assert result.passed is False
    assert "missing required" in result.reason


def test_wrong_type_fails():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}

    result = verify_schema({"a": "not an int"}, schema)

    assert result.passed is False


def test_unexpected_field_fails():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}

    result = verify_schema({"a": 1, "extra": 2}, schema)

    assert result.passed is False
    assert "unexpected" in result.reason


def test_nested_object_is_validated_recursively():
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"inner": {"type": "integer"}},
                "required": ["inner"],
            }
        },
        "required": ["outer"],
    }

    assert verify_schema({"outer": {"inner": 5}}, schema).passed is True
    assert verify_schema({"outer": {"inner": "not an int"}}, schema).passed is False
    assert verify_schema({"outer": {}}, schema).passed is False


def test_malformed_schema_raises():
    with pytest.raises(SchemaVerificationError):
        verify_schema({"a": 1}, {"type": "not_a_real_type"})


def test_non_dict_schema_raises():
    with pytest.raises(SchemaVerificationError):
        verify_schema({"a": 1}, "not a schema")  # type: ignore[arg-type]


def test_result_is_deterministic():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}

    assert verify_schema({"a": 1}, schema) == verify_schema({"a": 1}, schema)


def test_bool_rejected_where_integer_expected():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}

    result = verify_schema({"a": True}, schema)

    assert result.passed is False


def test_data_of_wrong_top_level_type_fails():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}

    result = verify_schema("not a dict", schema)

    assert result.passed is False


def test_result_verifier_field_is_schema():
    result = verify_schema({}, {"type": "object"})

    assert result.verifier == "schema"
