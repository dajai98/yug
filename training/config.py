"""NB-07: training configuration contract.

VERIFICATION STATUS: not executed -- see ``training/__init__.py``.

Mirrors ``yug.native.config.ModelConfig``'s own conventions
deliberately: frozen dataclass, explicit fields (no implicit
"default architecture"/"default training run" the caller didn't ask
for), and a ``__post_init__`` that calls a standalone ``validate()``
function rather than validating inline -- kept consistent with the
project's existing pattern in that module so a reader who already
knows ``ModelConfig`` recognizes this immediately.
"""

from __future__ import annotations

from dataclasses import dataclass

from training.errors import TrainingConfigValidationError


@dataclass(frozen=True)
class TrainingConfig:
    """Everything a training run needs, with no hidden defaults for
    anything that affects reproducibility.

    ``seed`` is required (not defaulted) deliberately -- an
    accidentally-omitted seed silently producing "whatever the current
    RNG state happens to be" is exactly the kind of non-determinism
    this milestone exists to rule out.
    """

    seed: int
    learning_rate: float
    batch_size: int
    sequence_length: int
    max_steps: int
    eval_interval: int
    checkpoint_interval: int
    grad_clip_norm: float | None = None
    weight_decay: float = 0.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        validate(self)


def validate(config: TrainingConfig) -> None:
    """Raise :class:`TrainingConfigValidationError` for any field
    combination that could not possibly produce a valid training run.

    Deliberately narrow: this checks structural validity (positive
    counts, non-negative rates), not training-quality heuristics (e.g.
    "is this learning rate reasonable for this model size?") -- the
    latter is a judgment call for whoever runs the training, not
    something this function should silently override or block.
    """
    if config.batch_size < 1:
        raise TrainingConfigValidationError(
            f"batch_size must be >= 1, got {config.batch_size}"
        )
    if config.sequence_length < 1:
        raise TrainingConfigValidationError(
            f"sequence_length must be >= 1, got {config.sequence_length}"
        )
    if config.max_steps < 1:
        raise TrainingConfigValidationError(
            f"max_steps must be >= 1, got {config.max_steps}"
        )
    if config.eval_interval < 1:
        raise TrainingConfigValidationError(
            f"eval_interval must be >= 1, got {config.eval_interval}"
        )
    if config.checkpoint_interval < 1:
        raise TrainingConfigValidationError(
            f"checkpoint_interval must be >= 1, got {config.checkpoint_interval}"
        )
    if config.learning_rate <= 0:
        raise TrainingConfigValidationError(
            f"learning_rate must be > 0, got {config.learning_rate}"
        )
    if config.weight_decay < 0:
        raise TrainingConfigValidationError(
            f"weight_decay must be >= 0, got {config.weight_decay}"
        )
    if config.grad_clip_norm is not None and config.grad_clip_norm <= 0:
        raise TrainingConfigValidationError(
            f"grad_clip_norm must be > 0 if set, got {config.grad_clip_norm}"
        )
