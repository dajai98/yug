"""yug.runtime exceptions.

Deliberately minimal (one class, not five): brain failures already
have their own typed hierarchy (:class:`yug.brain.errors.BrainError`)
and tool failures are represented as a :class:`yug.tools.types.ToolResult`
rather than raised at all -- wrapping either in a new runtime-specific
error type would just obscure an already-useful distinction the caller
needs. :class:`CoreRuntimeError` covers the one failure mode that is
genuinely new at this layer: a malformed call to
:meth:`yug.runtime.core.YugCore.respond` itself, before any brain,
memory, or tool interaction is attempted.
"""

from __future__ import annotations

from yug.utils.errors import YugError


class CoreRuntimeError(YugError):
    """Raised for an invalid call to :meth:`YugCore.respond` -- e.g. an
    empty or non-string request -- caught before any brain/memory/tool
    interaction happens."""
