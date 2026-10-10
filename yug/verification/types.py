"""Shared verification types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VerificationResult:
    """The outcome of one verification check. Deterministic (same
    inputs always produce an equal result), JSON-serializable by
    construction (every field is a plain str/bool/dict of JSON-safe
    values -- never an exception, never a live object reference), and
    carries a specific reason on failure rather than a bare boolean.

    ``passed=False`` means the check ran successfully and found the
    data invalid/the claim wrong -- it is a normal, expected outcome,
    not an error. See :mod:`yug.verification.errors` for the
    distinction from a raised exception (malformed verifier input).
    """

    passed: bool
    verifier: str
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
