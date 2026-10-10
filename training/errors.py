"""NB-07: training subsystem exceptions.

Follows the same convention as ``yug.native.errors``,
``yug.tokenizer.errors``, and ``yug.brain.errors``: a subsystem-rooted
class under :class:`yug.utils.errors.YugError`. This module itself has
no PyTorch dependency (unlike most of ``training/``), so it is real,
executable, testable Python in this environment.
"""

from __future__ import annotations

from yug.utils.errors import YugError


class TrainingError(YugError):
    """Base class for all ``training`` package failures."""


class TrainingConfigValidationError(TrainingError):
    """Raised for a structurally invalid :class:`training.config.TrainingConfig`."""


class DatasetError(TrainingError):
    """Raised for a malformed or empty training dataset/corpus."""


class CorpusPrepError(TrainingError):
    """Raised by ``training.corpus_prep`` for a malformed corpus-prep
    input or configuration (not a training-dataset-shape problem,
    which stays :class:`DatasetError` -- kept as its own subclass so a
    caller can distinguish "my raw documents/policy were malformed"
    from "my already-accepted text won't tokenize into a dataset")."""


class RealCorpusAcquisitionError(TrainingError):
    """Raised by ``training.experiments.real_corpus_experiment`` when a
    small, deterministic real-text sample cannot be acquired in the
    current environment. Deliberately its own class, not a reused
    ``DatasetError``/``CorpusPrepError`` -- this is a failure BEFORE
    any document reaches ``corpus_prep`` at all, not a problem with a
    document's shape or a corpus-prep policy."""


class CheckpointError(TrainingError):
    """Raised when a checkpoint cannot be saved or a saved checkpoint
    cannot be safely loaded (missing files, mismatched
    tokenizer/model identity, corrupted metadata)."""
