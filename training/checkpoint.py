"""NB-07: checkpoint save/load/resume.

Optional GradScaler and initialized CUDA RNG state extend version-1 metadata;
legacy checkpoints and FP32 callers remain supported.

Format (three files per checkpoint, all written under one directory):

    <dir>/model_config.json   yug.native.config.ModelConfig.save() output --
                               reused verbatim, not reinvented. Its own
                               identity-hash integrity check already
                               covers this file.
    <dir>/metadata.json       step, TrainingConfig fields, tokenizer
                               identity, and RNG state -- plain JSON,
                               never pickle (matching ModelConfig.save()'s
                               own stated reasoning: pickle can execute
                               arbitrary code on load).
                               ``random.getstate()`` is a nested tuple
                               of plain ints, represented as nested
                               JSON lists. PyTorch's RNG state is a 1-D
                               ``uint8`` tensor, represented as a hex
                               string of its raw bytes. Optional CUDA RNG states
                               use a list of hex strings; GradScaler state uses
                               its JSON-compatible scalar state_dict.
    <dir>/state.pt            model and optimizer *state dicts* -- i.e.
                               flat ``{name: tensor}`` mappings -- saved
                               via ``torch.save``/``torch.load``.

This is deliberately NOT "pickle as an ad hoc object dump": per
PyTorch's own documented recommendation, ``torch.save`` on a
``state_dict()`` (a flat dict of tensors, not an arbitrary live Python
object graph) is the standard mechanism for this, and safetensors is
not a project dependency (adding one merely for this would violate the
parent milestone's "Do not add unnecessary dependencies" rule). Every
field that is NOT a tensor (step, config, identities, RNG state) is
deliberately kept out of ``state.pt`` and lives in the two JSON files
instead, precisely so nothing here relies on unpickling arbitrary
objects.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import torch

from training.config import TrainingConfig
from training.errors import CheckpointError
from yug.native.checkpoint_io import NativeCheckpointError, load_model_state

_METADATA_FORMAT_VERSION = 1


def save_checkpoint(
    directory: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config,  # yug.native.config.ModelConfig
    training_config: TrainingConfig,
    step: int,
    tokenizer_identity: str,
    scaler=None,
) -> None:
    """Write a checkpoint to ``directory`` (created if it doesn't exist).

    Optional ``scaler`` is a caller-owned GradScaler. Its scalar state_dict
    is stored in JSON metadata, not a pickled scaler object. Call after
    scaler.update() at an optimizer-update boundary (no in-flight gradients).
    CUDA RNG states are saved only if CUDA is already initialized; saving
    a CPU checkpoint does not initialize a CUDA context.

    Raises:
        CheckpointError: ``tokenizer_identity`` does not match
            ``model_config.tokenizer_identity`` -- saving a checkpoint
            whose own two identity fields already disagree would only
            defer the inevitable failure to load time, where it would
            be harder to diagnose.
    """
    if tokenizer_identity != model_config.tokenizer_identity:
        raise CheckpointError(
            f"tokenizer_identity {tokenizer_identity!r} does not match "
            f"model_config.tokenizer_identity {model_config.tokenizer_identity!r}"
        )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    model_config.save(directory / "model_config.json")

    torch_rng_bytes = bytes(torch.get_rng_state().tolist())
    metadata = {
        "metadata_format_version": _METADATA_FORMAT_VERSION,
        "step": step,
        "tokenizer_identity": tokenizer_identity,
        "training_config": asdict(training_config),
        "random_state": _tuple_to_list(random.getstate()),
        "torch_rng_state_hex": torch_rng_bytes.hex(),
    }
    if scaler is not None:
        metadata["grad_scaler_state"] = scaler.state_dict()
    if torch.cuda.is_initialized():
        metadata["cuda_rng_states_hex"] = [
            bytes(state.tolist()).hex() for state in torch.cuda.get_rng_state_all()
        ]
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        directory / "state.pt",
    )


def load_checkpoint(
    directory: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    restore_rng_state: bool = True,
    scaler=None,
) -> dict:
    """Load a checkpoint written by :func:`save_checkpoint` into an
    already-constructed ``model``/``optimizer`` pair (in place, via
    ``load_state_dict``) and return its metadata dict (``step``,
    ``training_config`` as a plain dict, ``tokenizer_identity``).

    If supplied, ``scaler`` receives the saved nonempty GradScaler state.
    Legacy checkpoints with no scaler state (or disabled scalers with empty
    state) leave the caller's scaler unchanged. Without a scaler argument,
    saved scaler state is ignored for FP32/inference compatibility. Exact AMP
    continuation requires matching AMP/scaler settings, not legacy fallback.

    ``restore_rng_state`` controls Python, CPU and optional CUDA RNG together;
    it does not control optimizer/scaler restoration. Saved CUDA states require
    the same number/order of visible CUDA devices when RNG restoration is on.
    CPU inference can use restore_rng_state=False. Legacy checkpoints lacking
    CUDA RNG state leave CUDA RNG untouched; they cannot promise stochastic
    CUDA continuation. Returned step metadata and caller-owned step assignment
    are unchanged.

    The caller is responsible for constructing ``model`` from the
    checkpoint's own ``model_config.json`` (via
    ``yug.native.config.ModelConfig.load(directory / "model_config.json")``)
    *before* calling this -- this function does not construct the
    model itself, matching the same "adapter is an integration
    boundary, not another model implementation" separation of concerns
    established elsewhere in this milestone sequence.

    Raises:
        CheckpointError: the metadata or state file is missing, the
            metadata format version is unsupported, ``state.pt`` does
            not contain a ``"model"`` entry (surfaced via
            :class:`yug.native.checkpoint_io.NativeCheckpointError`,
            re-raised here as ``CheckpointError`` so callers of this
            function see a single, training-subsystem-rooted exception
            type as before -- NB-10 changed *how* the model entry is
            read, not what this function raises for it), or
            ``state.pt`` does not contain an ``"optimizer"`` entry.
    """
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    state_path = directory / "state.pt"

    if not metadata_path.exists():
        raise CheckpointError(f"Checkpoint metadata file not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if metadata.get("metadata_format_version") != _METADATA_FORMAT_VERSION:
        raise CheckpointError(
            f"Unsupported checkpoint metadata format version "
            f"{metadata.get('metadata_format_version')!r} in {metadata_path}"
        )

    cuda_states = None
    if restore_rng_state and "cuda_rng_states_hex" in metadata:
        encoded = metadata["cuda_rng_states_hex"]
        if not isinstance(encoded, list) or not encoded:
            raise CheckpointError("Invalid CUDA RNG state list")
        if not torch.cuda.is_available() or torch.cuda.device_count() != len(encoded):
            raise CheckpointError("CUDA RNG restore requires the saved visible CUDA device count")
        try:
            cuda_states = [torch.tensor(list(bytes.fromhex(value)), dtype=torch.uint8)
                           for value in encoded]
            if any(state.numel() == 0 for state in cuda_states):
                raise ValueError("Empty CUDA RNG state")
        except (TypeError, ValueError) as exc:
            raise CheckpointError("Invalid CUDA RNG state encoding") from exc

    # NB-10: the "model" entry is now read via the shared,
    # inference-safe reader under yug/ (yug.native.checkpoint_io) --
    # the single authoritative interpretation of that portion of
    # state.pt, also used by YugNativeAdapter. This function remains
    # the authority on everything else in the file (optimizer state,
    # RNG restoration) -- those are training-only concepts the shared
    # reader deliberately knows nothing about, so they are still read
    # directly here, same as before.
    try:
        model_state = load_model_state(directory, map_location="cpu")
    except NativeCheckpointError as exc:
        raise CheckpointError(str(exc)) from exc
    model.load_state_dict(model_state)

    state = torch.load(state_path, map_location="cpu")
    if "optimizer" not in state:
        raise CheckpointError(f"Checkpoint state file {state_path} is missing an 'optimizer' entry")
    optimizer.load_state_dict(state["optimizer"])

    scaler_state = metadata.get("grad_scaler_state")
    if scaler is not None and scaler_state:
        try:
            scaler.load_state_dict(scaler_state)
        except (TypeError, ValueError, KeyError, RuntimeError) as exc:
            raise CheckpointError("Invalid GradScaler checkpoint state") from exc

    if restore_rng_state:
        random.setstate(_list_to_tuple(metadata["random_state"]))
        torch_rng_bytes = bytes.fromhex(metadata["torch_rng_state_hex"])
        torch.set_rng_state(torch.tensor(list(torch_rng_bytes), dtype=torch.uint8))
        if cuda_states is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except RuntimeError as exc:
                raise CheckpointError("Could not restore CUDA RNG states") from exc

    return {
        "step": metadata["step"],
        "tokenizer_identity": metadata["tokenizer_identity"],
        "training_config": metadata["training_config"],
    }


def _tuple_to_list(value):
    """Recursively convert a (possibly nested) tuple to a (possibly
    nested) list, so it round-trips through ``json.dumps``/``json.loads``
    with :func:`_list_to_tuple` restoring the original tuple structure
    that :func:`random.setstate` requires."""
    if isinstance(value, tuple):
        return [_tuple_to_list(item) for item in value]
    return value


def _list_to_tuple(value):
    """Inverse of :func:`_tuple_to_list`."""
    if isinstance(value, list):
        return tuple(_list_to_tuple(item) for item in value)
    return value
