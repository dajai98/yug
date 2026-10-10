"""Tests for yug.memory persistence (Phase 2): the restart scenario,
corruption handling, and the on-disk JSON format. See
test_memory_core.py for basic in-process operations.

VERIFICATION STATUS: see test_memory_core.py's module docstring -- the
restart scenario, corruption cases, and format checks below were all
executed for real (as a standalone script, not through pytest) in the
development sandbox before this file was written, using the same
storage_path / discard-instance / new-instance pattern used here.
"""

from __future__ import annotations

import json

import pytest

from yug.memory.errors import MemoryPersistenceError
from yug.memory.persistence import load_memory, save_memory
from yug.memory.working_memory import WorkingMemory


def test_restart_scenario_store_discard_reload_retrieve(tmp_path):
    """The actual Phase 2 exit condition: a fact can be stored, the
    instance discarded, a FRESH instance constructed, and the fact
    retrieved. This must not reuse the original object."""
    storage_path = str(tmp_path / "memory.json")

    memory = WorkingMemory(storage_path=storage_path)
    memory.set("fact", "the sky is blue")
    del memory  # simulate the process/instance ending

    reloaded = WorkingMemory(storage_path=storage_path)  # a NEW instance

    assert reloaded.get("fact") == "the sky is blue"


def test_delete_is_also_persisted_across_restart(tmp_path):
    storage_path = str(tmp_path / "memory.json")

    memory = WorkingMemory(storage_path=storage_path)
    memory.set("temp", "value")
    memory.delete("temp")
    del memory

    reloaded = WorkingMemory(storage_path=storage_path)

    assert reloaded.contains("temp") is False


def test_constructing_with_nonexistent_path_starts_empty_not_an_error(tmp_path):
    storage_path = str(tmp_path / "does_not_exist_yet.json")

    memory = WorkingMemory(storage_path=storage_path)

    assert memory.contains("anything") is False


def test_none_storage_path_never_touches_disk(tmp_path, monkeypatch):
    """storage_path=None (the default) must be purely in-process."""
    monkeypatch.chdir(tmp_path)
    memory = WorkingMemory()  # no storage_path
    memory.set("k", "v")

    assert list(tmp_path.iterdir()) == []


def test_two_instances_with_different_paths_are_isolated(tmp_path):
    path_a = str(tmp_path / "a.json")
    path_b = str(tmp_path / "b.json")

    memory_a = WorkingMemory(storage_path=path_a)
    memory_b = WorkingMemory(storage_path=path_b)
    memory_a.set("k", "only in a")

    assert memory_b.contains("k") is False


def test_on_disk_format_is_versioned_json_with_data_field(tmp_path):
    storage_path = tmp_path / "memory.json"

    memory = WorkingMemory(storage_path=str(storage_path))
    memory.set("fact", "the sky is blue")

    raw = json.loads(storage_path.read_text(encoding="utf-8"))

    assert raw["yug_memory_format_version"] == "1"
    assert raw["data"] == {"fact": "the sky is blue"}


def test_load_memory_missing_file_raises_persistence_error(tmp_path):
    with pytest.raises(MemoryPersistenceError):
        load_memory(tmp_path / "nope.json")


def test_load_memory_malformed_json_raises_persistence_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(MemoryPersistenceError):
        load_memory(path)


def test_load_memory_non_object_json_raises_persistence_error(tmp_path):
    path = tmp_path / "array.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(MemoryPersistenceError):
        load_memory(path)


def test_load_memory_wrong_format_version_raises_persistence_error(tmp_path):
    path = tmp_path / "wrong_version.json"
    path.write_text(json.dumps({"yug_memory_format_version": "999", "data": {}}), encoding="utf-8")

    with pytest.raises(MemoryPersistenceError):
        load_memory(path)


def test_load_memory_missing_format_version_raises_persistence_error(tmp_path):
    path = tmp_path / "no_version.json"
    path.write_text(json.dumps({"data": {}}), encoding="utf-8")

    with pytest.raises(MemoryPersistenceError):
        load_memory(path)


def test_load_memory_missing_data_field_raises_persistence_error(tmp_path):
    path = tmp_path / "no_data.json"
    path.write_text(json.dumps({"yug_memory_format_version": "1"}), encoding="utf-8")

    with pytest.raises(MemoryPersistenceError):
        load_memory(path)


def test_constructing_working_memory_from_corrupt_file_raises(tmp_path):
    """The corruption check must also fire from WorkingMemory's own
    constructor, not just the lower-level load_memory() function."""
    path = tmp_path / "corrupt.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(MemoryPersistenceError):
        WorkingMemory(storage_path=str(path))


def test_save_memory_creates_parent_directories(tmp_path):
    nested_path = tmp_path / "nested" / "dir" / "memory.json"

    save_memory(nested_path, {"k": "v"})

    assert nested_path.exists()
    assert load_memory(nested_path) == {"k": "v"}
