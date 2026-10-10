"""Tokenizer subsystem exceptions.

All inherit from :class:`yug.utils.errors.YugError`, matching the
convention already established by ``yug.brain.errors``.
"""

from __future__ import annotations

from yug.utils.errors import YugError


class TokenizerError(YugError):
    """Base class for all tokenizer-subsystem failures."""


class VocabularyError(TokenizerError):
    """Raised for invalid vocabulary/training parameters.

    E.g. a requested ``vocab_size`` too small to hold the 256 base
    byte tokens plus the requested special tokens.
    """


class DecodeError(TokenizerError):
    """Raised when ``decode()`` is given token IDs it cannot resolve:
    an ID outside the tokenizer's vocabulary, a special-token ID passed
    with ``skip_special_tokens=False``, or a byte sequence that isn't
    valid UTF-8 once reassembled (e.g. a slice that cuts a multi-byte
    character in half)."""


class SerializationError(TokenizerError):
    """Raised when a saved tokenizer file is missing, malformed, the
    wrong version, or fails its identity-hash integrity check on load."""
