"""Working memory: an explicitly-constructed key/value store, with
optional write-through JSON persistence.

Phase 2 scope only (see ``yug.memory.__init__`` docstring): storage and
retrieval, nothing intelligent. Values must be plain, JSON-serializable
data -- no brain/adapter/model types.
"""

from __future__ import annotations

import json

from yug.memory.errors import MemoryPersistenceError, MemorySerializationError, MemoryValidationError
from yug.memory.persistence import load_memory, save_memory


class WorkingMemory:
    """An explicitly-constructed fact store. No hidden module-level
    global state: two ``WorkingMemory`` instances are always
    independent unless they are explicitly given the same
    ``storage_path``.

    Args:
        storage_path: if given, memory is persisted to this JSON file.
            If the file already exists, its contents are loaded
            immediately (this is what makes the "store a fact, discard
            the instance, construct a new one, retrieve the fact"
            exit condition work -- the fresh instance's constructor
            does the loading). Every :meth:`set` and :meth:`delete`
            call immediately re-persists the full store to this path
            (write-through), so there is no separate save step to
            forget to call. If ``None`` (the default), memory is
            purely in-process and never touches disk.

    Raises:
        MemoryPersistenceError: if ``storage_path`` is given, the file
            exists, and it fails to load (propagated unchanged from
            :func:`~yug.memory.persistence.load_memory` -- see that
            function's docstring for the specific causes).
    """

    def __init__(self, storage_path: str | None = None) -> None:
        self._storage_path = storage_path
        self._data: dict[str, object] = {}
        if storage_path is not None:
            try:
                self._data = load_memory(storage_path)
            except MemoryPersistenceError as exc:
                if "not found" in str(exc):
                    # No file yet is expected on first use, not an error.
                    self._data = {}
                else:
                    raise

    def set(self, key: str, value: object) -> None:
        """Store ``value`` under ``key``, overwriting any existing
        value for that key.

        Raises:
            MemoryValidationError: ``key`` is not a non-empty ``str``.
            MemorySerializationError: ``value`` is not JSON-serializable.
        """
        self._validate_key(key)
        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise MemorySerializationError(
                f"Value for key {key!r} is not JSON-serializable: {exc}"
            ) from exc

        self._data[key] = value
        self._persist()

    def get(self, key: str) -> object:
        """Retrieve the value stored under ``key``.

        Raises:
            MemoryValidationError: ``key`` is not a non-empty ``str``.
            KeyError: no value is stored under ``key``. (Not a
                MemorySubsystemError: a missing key is normal control
                flow, exactly like a missing ``dict`` key -- check
                with :meth:`contains` first if the caller wants to
                avoid the exception.)
        """
        self._validate_key(key)
        return self._data[key]

    def delete(self, key: str) -> None:
        """Remove the value stored under ``key``, if present. Deleting
        a key that does not exist is a no-op, not an error.

        Raises:
            MemoryValidationError: ``key`` is not a non-empty ``str``.
        """
        self._validate_key(key)
        self._data.pop(key, None)
        self._persist()

    def contains(self, key: str) -> bool:
        """Return whether a value is currently stored under ``key``.

        Raises:
            MemoryValidationError: ``key`` is not a non-empty ``str``.
        """
        self._validate_key(key)
        return key in self._data

    @staticmethod
    def _validate_key(key: str) -> None:
        if not isinstance(key, str) or not key:
            raise MemoryValidationError(f"key must be a non-empty str, got {key!r}")

    def _persist(self) -> None:
        if self._storage_path is not None:
            save_memory(self._storage_path, self._data)
