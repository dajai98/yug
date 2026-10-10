"""Public runtime types.

No separate ``Request`` dataclass: the "request model" is realized as
:meth:`yug.runtime.core.YugCore.respond`'s validated parameters
(matching this milestone's own conceptual target API exactly), not a
standalone object whose only purpose would be wrapping those same
arguments before they're immediately unpacked again. ``Response`` is a
real, separate type because it is genuinely returned to and inspected
by a caller (a future CLI/API layer, per the Response Contract
section) -- that asymmetry is deliberate, not an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass

from yug.tools.types import ToolResult
from yug.verification.types import VerificationResult


@dataclass(frozen=True)
class Response:
    """The outcome of one :meth:`YugCore.respond` call.

    ``success`` reflects whether the brain generation step completed
    (i.e. whether ``content`` is real model/mock output). It is
    independent of ``tool_result.success`` -- a request can produce a
    valid brain response AND a failed tool call in the same
    ``Response`` (both are surfaced, neither is collapsed into the
    other); see :mod:`yug.runtime.core`'s docstring for the reasoning.

    ``verification_result`` follows the exact same independence rule:
    it reflects whether an explicitly-requested verification passed,
    and is orthogonal to both ``success`` and ``tool_result.success``.
    """

    success: bool
    content: str
    request_text: str
    tool_result: ToolResult | None = None
    verification_result: VerificationResult | None = None
    error: str | None = None
    conversation_key: str | None = None
