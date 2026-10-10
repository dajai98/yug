"""yug.verification exceptions.

Follows the established convention: a subsystem-rooted class under
:class:`yug.utils.errors.YugError`.

Deliberate distinction, stated once here rather than repeated in every
docstring: a FAILED verification (data doesn't match a valid schema;
an arithmetic claim's expected value is wrong) is NOT an exception --
it is a :class:`~yug.verification.types.VerificationResult` with
``passed=False``. These exceptions are reserved for the verifier
INPUT itself being malformed or unsupported (an invalid schema
definition, an unparseable or disallowed math expression) -- a
condition the caller must fix before a meaningful pass/fail judgement
is even possible. This mirrors ``yug.tools.validation``'s own
convention: ``ToolValidationError`` is raised for a malformed call,
while the tool's actual outcome is always a returned result object.
"""

from __future__ import annotations

from yug.utils.errors import YugError


class VerificationError(YugError):
    """Base class for all yug.verification failures."""


class SchemaVerificationError(VerificationError):
    """Raised when a schema passed to
    :func:`~yug.verification.schema.verify_schema` is itself malformed
    (not a dict, an unrecognized 'type', malformed 'properties' or
    'required') -- NOT raised when valid data simply fails to match a
    valid schema (that is a ``VerificationResult(passed=False)``)."""


class MathVerificationError(VerificationError):
    """Raised when a claim passed to
    :func:`~yug.verification.math.verify_math` cannot be evaluated at
    all: not valid Python expression syntax, contains anything outside
    the safe arithmetic allow-list, divides by zero, or ``expected``
    is not a number -- NOT raised when the expression evaluates fine
    but to a different value than ``expected`` (that is a
    ``VerificationResult(passed=False)``)."""


class VerifierNotFoundError(VerificationError):
    """Raised by :meth:`~yug.verification.engine.VerificationEngine.verify`
    for an unrecognized verifier name."""
