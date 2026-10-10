"""Memory subsystem exceptions.

Follows the same convention as ``yug.brain.errors``,
``yug.tokenizer.errors``, and ``yug.native.errors``: a subsystem-rooted
class under :class:`yug.utils.errors.YugError`, with subclasses only
for failure modes the implementation actually distinguishes and
exercises (no speculative branches).

Named ``MemorySubsystemError`` rather than plain ``MemoryError``: the
latter is a Python builtin (raised on genuine out-of-memory
conditions). Reusing that name here would let
``except MemoryError:`` inside YUG code silently catch this
subsystem's errors instead of real OOM conditions (or vice versa),
which is exactly the kind of silently-swallowed failure this
project's error-handling conventions exist to prevent.
"""

from __future__ import annotations

from yug.utils.errors import YugError


class MemorySubsystemError(YugError):
    """Base class for all yug.memory failures."""


class MemoryValidationError(MemorySubsystemError):
    """Raised for an invalid argument to a :class:`WorkingMemory`
    operation -- e.g. a non-string or empty key."""


class MemorySerializationError(MemorySubsystemError):
    """Raised when a value passed to :meth:`WorkingMemory.set` cannot
    be represented in the persisted format (must be JSON-serializable:
    str, int, float, bool, None, or a dict/list composed of those)."""


class MemoryPersistenceError(MemorySubsystemError):
    """Raised for a problem with the persisted memory *file* itself --
    missing, malformed JSON, not a JSON object, or an unsupported/
    missing format version. Distinct from
    :class:`MemorySerializationError`, which concerns a single value
    at write time rather than the file as a whole."""
