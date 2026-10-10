"""VerificationEngine: explicit caller-selected verifier dispatch. No
automatic invocation, no brain, no Agent, no tool execution -- a
caller must name a verifier and supply its exact arguments.

Malformed-input exceptions (:class:`~yug.verification.errors.SchemaVerificationError`,
:class:`~yug.verification.errors.MathVerificationError`) are NOT caught
here and turned into a failed result -- they propagate to the caller
unchanged, exactly like ``yug.tools.validation.validate_parameters``
raises rather than returning a sentinel. Swallowing them would blur
the deliberate exception-vs-result distinction this subsystem is built
around (see ``yug.verification.errors`` module docstring).
"""

from __future__ import annotations

from typing import Any

from yug.verification.errors import VerifierNotFoundError
from yug.verification.math import verify_math
from yug.verification.schema import verify_schema
from yug.verification.types import VerificationResult


class VerificationEngine:
    """Explicitly constructed, no hidden global state -- mirrors every
    other subsystem's convention (``WorkingMemory``, ``ToolRegistry``,
    ``PermissionChecker``)."""

    _VERIFIERS = frozenset({"schema", "math"})

    def verify(self, verifier_name: str, **kwargs: Any) -> VerificationResult:
        """Run the named verifier against ``kwargs``.

        Args:
            verifier_name: ``"schema"`` (expects ``data``, ``schema``)
                or ``"math"`` (expects ``expression``, ``expected``).
            **kwargs: forwarded as keyword arguments to the selected
                verifier function.

        Raises:
            VerifierNotFoundError: ``verifier_name`` is not one of the
                verifiers above.
        """
        if verifier_name == "schema":
            return verify_schema(**kwargs)
        if verifier_name == "math":
            return verify_math(**kwargs)
        raise VerifierNotFoundError(
            f"Unknown verifier: {verifier_name!r}. Known verifiers: {sorted(self._VERIFIERS)}"
        )
