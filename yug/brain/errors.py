"""Brain subsystem exceptions.

All inherit from :class:`yug.utils.errors.YugError` so callers can catch
"any YUG failure" at the root, or narrow to brain-specific failures via
:class:`BrainError`.
"""

from __future__ import annotations

from yug.utils.errors import YugError


class BrainError(YugError):
    """Base class for all brain-subsystem failures."""


class ModelLoadError(BrainError):
    """Raised when a model backend cannot be initialized.

    Covers missing dependencies (e.g. the ``llama-cpp-python`` package
    is not installed) and backend-reported load failures (corrupt file,
    incompatible format, out of memory, etc.).
    """


class ModelNotFoundError(ModelLoadError):
    """Raised when the configured model file does not exist on disk."""


class GenerationError(BrainError):
    """Raised when a loaded model fails during ``generate()``."""


class EmbeddingNotSupportedError(BrainError):
    """Raised when ``embed()`` is called on a model/backend that has no
    embedding capability, rather than silently returning zero vectors
    or fabricated output."""


class BrainUnavailableError(BrainError):
    """Raised when an operation is attempted on a brain that is not
    currently usable (dependency missing, model not loaded, and the
    caller did not request an implicit load)."""
