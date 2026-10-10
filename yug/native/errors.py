"""Native (model) subsystem exceptions.

Follows the same convention as ``yug.brain.errors`` and
``yug.tokenizer.errors``: a subsystem-rooted class under
:class:`yug.utils.errors.YugError`. Named ``ModelConfig*`` rather than
reusing ``yug.config.schema.ConfigValidationError`` (Phase 0's
application-configuration error) -- these are unrelated concerns: one
validates ``YugConfig`` (logging/paths/brain-backend selection), the
other validates a native model's architecture configuration.
"""

from __future__ import annotations

from yug.utils.errors import YugError


class NativeError(YugError):
    """Base class for all yug.native failures."""


class ModelConfigValidationError(NativeError):
    """Raised when a :class:`~yug.native.config.ModelConfig` has an
    invalid field value or an invalid combination of fields (e.g.
    ``hidden_size`` not divisible by ``num_heads``)."""


class ModelConfigSerializationError(NativeError):
    """Raised when a saved model configuration file is missing,
    malformed, has an unsupported format version, or fails its
    identity-hash integrity check on load."""
