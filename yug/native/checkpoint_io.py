"""NB-10: the single, authoritative reader of the *model-state* portion
of a YUG checkpoint's ``state.pt`` file.

Why this module exists (not "why checkpoints exist" -- that remains
``training/checkpoint.py``'s docstring): ``YugNativeAdapter``
(``yug/brain/adapters/yug_native.py``) needs real model weights for
inference, and the only existing code that read ``state.pt`` was
``training.checkpoint.load_checkpoint()`` -- but that lives in
``training/``, and ``yug/`` must never import ``training/`` (the
frozen dependency boundary; see ``ARCHITECTURE/POST_NB07_ROADMAP.md``
and ``ARCHITECTURE/NB10_NATIVE_ADAPTER_PLAN.md`` Section 9, Risk 1,
Option A -- this module IS that approved resolution).

This module knows nothing about training. It deliberately contains:
  - no optimizer restoration
  - no training state
  - no RNG restoration
  - no training configuration handling
  - no training resume logic
  - no optimizer stepping
  - no training abstractions of any kind

It is the single, authoritative interpretation of what "the model
portion of state.pt" means. ``training.checkpoint.load_checkpoint()``
now calls :func:`load_model_state` instead of independently
implementing that same interpretation a second time -- see this
module and that module's docstrings for the shared contract.

Checkpoint format (unchanged; ``training/checkpoint.py`` remains the
authoritative description of the *whole* format -- restated here only
to the extent this module's own contract needs it):

    <dir>/state.pt   torch.save({"model": <state_dict>, ...}, ...)

This module reads and returns ONLY the ``"model"`` entry. Any other
entries in that file (e.g. ``"optimizer"``) are training-internal and
this module does not look at them, require them, or validate their
presence.
"""

from __future__ import annotations

from pathlib import Path

import torch

from yug.native.errors import NativeError


class NativeCheckpointError(NativeError):
    """Raised when a checkpoint's ``state.pt`` file is missing,
    unreadable, or does not contain a ``"model"`` entry.

    Scope note: this covers only the model-state portion of a
    checkpoint (this module's entire responsibility). It says nothing
    about ``model_config.json`` (see
    ``yug.native.errors.ModelConfigSerializationError`` for that) or
    about the ``"optimizer"``/RNG/training-config portions of
    ``state.pt``/``metadata.json``, which remain
    ``training.errors.CheckpointError``'s responsibility.
    """


def load_model_state(
    directory: str | Path,
    *,
    map_location: str = "cpu",
) -> dict:
    """Return the model ``state_dict`` from the checkpoint at
    ``directory``, ready to pass to ``model.load_state_dict(...)``.

    A pure, read-only, inference-safe operation: it does not construct
    a model, does not touch any ``"optimizer"`` entry that may also be
    present in the same file, does not restore RNG state, and does not
    read ``metadata.json`` (that file is plain JSON with zero PyTorch
    dependency -- callers needing it, such as an adapter reporting a
    checkpoint's recorded step, can read it directly).

    Args:
        directory: a directory previously written by
            ``training.checkpoint.save_checkpoint`` (or anything
            producing the same ``state.pt`` layout).
        map_location: passed through to ``torch.load``. Defaults to
            ``"cpu"`` so this never silently requires a GPU to be
            present, matching this project's "do not assume GPU
            availability" convention -- callers that have already
            moved a model to another device are still free to
            ``model.to(device)`` themselves after
            ``load_state_dict(...)``.

    Raises:
        NativeCheckpointError: ``state.pt`` does not exist under
            ``directory``, cannot be read by ``torch.load``, or does
            not contain a ``"model"`` entry.
    """
    directory = Path(directory)
    state_path = directory / "state.pt"

    if not state_path.exists():
        raise NativeCheckpointError(f"Checkpoint state file not found: {state_path}")

    try:
        state = torch.load(state_path, map_location=map_location)
    except Exception as exc:  # noqa: BLE001 - normalize any torch.load failure
        raise NativeCheckpointError(
            f"Failed to read checkpoint state file {state_path}: {exc}"
        ) from exc

    if not isinstance(state, dict) or "model" not in state:
        raise NativeCheckpointError(
            f"Checkpoint state file {state_path} does not contain a 'model' entry"
        )

    return state["model"]
