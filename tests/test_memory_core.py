"""Tests for yug.memory.working_memory (Phase 2): basic in-process
storage, validation, and instance isolation. Persistence/restart
behavior is covered separately in test_memory_persistence.py.

VERIFICATION STATUS: the logic in this file was executed for real in
the development sandbox (pure stdlib -- no PyTorch/pytest required),
via a standalone script exercising the same behavior these test
functions assert. The pytest versions below have not been run through
pytest itself in that sandbox (pytest is not installed there); they
are written to be run by whoever has pytest installed.
"""

from __future__ import annotations

import pytest

from yug.memory.errors import MemorySerializationError, MemoryValidationError
from yug.memory.working_memory import WorkingMemory


def test_set_then_get_returns_the_stored_value():
    memory = WorkingMemory()
    memory.set("greeting", "hello")

    assert memory.get("greeting") == "hello"


def test_set_overwrites_existing_value():
    memory = WorkingMemory()
    memory.set("count", 1)
    memory.set("count", 2)

    assert memory.get("count") == 2


def test_delete_removes_the_key():
    memory = WorkingMemory()
    memory.set("temp", "value")
    memory.delete("temp")

    assert memory.contains("temp") is False


def test_delete_of_missing_key_is_a_no_op():
    memory = WorkingMemory()

    memory.delete("never-existed")  # must not raise


def test_get_missing_key_raises_key_error():
    memory = WorkingMemory()

    with pytest.raises(KeyError):
        memory.get("missing")


def test_contains_reflects_current_state():
    memory = WorkingMemory()

    assert memory.contains("x") is False
    memory.set("x", 1)
    assert memory.contains("x") is True
    memory.delete("x")
    assert memory.contains("x") is False


@pytest.mark.parametrize("bad_key", [123, None, 3.14, [], {}])
def test_non_string_key_rejected(bad_key):
    memory = WorkingMemory()

    with pytest.raises(MemoryValidationError):
        memory.set(bad_key, "value")


def test_empty_string_key_rejected():
    memory = WorkingMemory()

    with pytest.raises(MemoryValidationError):
        memory.set("", "value")


def test_get_and_delete_and_contains_also_validate_key():
    memory = WorkingMemory()

    with pytest.raises(MemoryValidationError):
        memory.get("")
    with pytest.raises(MemoryValidationError):
        memory.delete(123)
    with pytest.raises(MemoryValidationError):
        memory.contains(None)


def test_non_json_serializable_value_rejected():
    memory = WorkingMemory()

    with pytest.raises(MemorySerializationError):
        memory.set("k", object())


@pytest.mark.parametrize(
    "value",
    ["a string", 42, 3.14, True, None, {"nested": "dict"}, [1, 2, 3]],
)
def test_json_serializable_values_are_accepted(value):
    memory = WorkingMemory()
    memory.set("k", value)

    assert memory.get("k") == value


def test_two_instances_do_not_share_state():
    """No hidden module-level global: instances without a shared
    storage_path must be fully independent."""
    memory_a = WorkingMemory()
    memory_b = WorkingMemory()

    memory_a.set("k", "only in a")

    assert memory_b.contains("k") is False


def test_repeated_get_is_deterministic():
    memory = WorkingMemory()
    memory.set("k", "stable value")

    assert memory.get("k") == memory.get("k") == "stable value"
