"""Memory persistence: versioned-JSON save/load, matching the pattern
already established by ``yug.native.config.ModelConfig.save/load`` and
``yug.tokenizer.ByteLevelBpeTokenizer.save/load``.

JSON, never pickle (pickle can execute arbitrary code on load -- same
reasoning as those two modules). An explicit
``yug_memory_format_version`` field lets a future format change be
detected and rejected cleanly instead of silently misread.
"""

from __future__ import annotations

import json
from pathlib import Path

from yug.memory.errors import MemoryPersistenceError

_FORMAT_VERSION = "1"


def save_memory(path: str | Path, data: dict) -> None:
    """Serialize ``data`` (the working-memory key/value store) to a
    JSON file at ``path``, creating parent directories if needed.

    Args:
        path: destination file path.
        data: a flat ``dict`` of JSON-serializable values. Not
            validated here -- :class:`~yug.memory.working_memory.WorkingMemory`
            is responsible for rejecting non-serializable values at
            ``set()`` time, before they ever reach this function.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "yug_memory_format_version": _FORMAT_VERSION,
        "data": data,
    }
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_memory(path: str | Path) -> dict:
    """Load a memory store previously written by :func:`save_memory`.

    Returns:
        The stored key/value ``dict``.

    Raises:
        MemoryPersistenceError: the file does not exist, is not valid
            JSON, is not a JSON object, is missing or has an
            unsupported ``yug_memory_format_version``, or its ``data``
            field is missing or not a JSON object.
    """
    file_path = Path(path)
    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise MemoryPersistenceError(f"Memory file not found: {file_path}") from exc

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise MemoryPersistenceError(f"Invalid JSON in memory file {file_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise MemoryPersistenceError(f"Memory file {file_path} must contain a JSON object")

    version = payload.get("yug_memory_format_version")
    if version != _FORMAT_VERSION:
        raise MemoryPersistenceError(
            f"Unsupported memory format version {version!r} in {file_path}; expected {_FORMAT_VERSION!r}."
        )

    data = payload.get("data")
    if not isinstance(data, dict):
        raise MemoryPersistenceError(f"Memory file {file_path} is missing a valid 'data' object")

    return data
