"""CPU checkpoint evaluation: global target NLL and two reproducibility identities.

The v1 semantic contract preserves existing tokenization, packing and target
shifting. Execution identity describes tracked conditions, not full hardware
attestation. Checkpoint loading remains owned by training.checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from training.checkpoint import load_checkpoint
from training.dataset import encode_corpus, iter_batches, make_sequences
from training.errors import TrainingError
from training.loss import evaluation_nll_count
from yug.native.config import ModelConfig
from yug.native.model import YugTransformer

EVALUATION_SEMANTICS = "yug-causal-lm-eval-v1"
REPORT_SCHEMA = "yug-evaluation-report-v1"
_IDENTITY = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PROVENANCE_KEYS = (
    "corpus_identity", "split_name", "split_method", "split_seed", "split_fraction",
)
_EXECUTION_KEYS = (
    "device", "model_dtype", "torch_version", "torch_num_threads",
    "torch_num_interop_threads", "deterministic_algorithms", "deterministic_warn_only",
    "float32_matmul_precision", "mkldnn_enabled", "autocast_enabled",
    "nll_accumulation_dtype",
)
_WEIGHT_CHUNK_VALUES = 32768


class EvaluationError(TrainingError):
    """Invalid evaluation inputs, metrics or reproducibility metadata.

    Existing checkpoint/config/dataset errors still propagate from their owners.
    """


def canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise EvaluationError(f"Cannot construct evaluation identity: {exc}") from exc


def _hash(payload: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _check_identity(value: str) -> None:
    if not isinstance(value, str) or not _IDENTITY.fullmatch(value):
        raise EvaluationError(f"Invalid content identity: {value!r}")


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise EvaluationError(f"{name} must be a positive integer")


def normalize_provenance(provenance: dict | None) -> dict:
    if provenance is not None and not isinstance(provenance, dict):
        raise EvaluationError("data_provenance must be a mapping or None")
    supplied = {} if provenance is None else provenance
    if set(supplied) - set(_PROVENANCE_KEYS):
        raise EvaluationError("Unknown data_provenance fields")
    result = {key: supplied.get(key) for key in _PROVENANCE_KEYS}
    if result["corpus_identity"] is not None:
        _check_identity(result["corpus_identity"])
    for key in ("split_name", "split_method"):
        if result[key] is not None and (not isinstance(result[key], str) or not result[key]):
            raise EvaluationError(f"{key} must be a nonempty string or None")
    if result["split_seed"] is not None and type(result["split_seed"]) is not int:
        raise EvaluationError("split_seed must be an integer or None")
    fraction = result["split_fraction"]
    if fraction is not None:
        if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction < 1:
            raise EvaluationError("split_fraction must be finite and in (0, 1)")
        result["split_fraction"] = float(fraction)
    return result


def evaluation_data_identity(documents: list[str]) -> str:
    if not isinstance(documents, list) or any(not isinstance(x, str) for x in documents):
        raise EvaluationError("Evaluation documents must be a list of strings")
    return _hash({"schema": "yug-eval-data-v1", "documents": documents})


def model_weights_identity(state: dict[str, torch.Tensor]) -> str:
    """Hash effective values, independent of floating storage dtype/aliasing.

    FP64 conversion and Python/byte temporaries are bounded to 32,768 values.
    A noncontiguous entry may require one contiguous copy in its original dtype.
    struct's '<d' specifies binary64 little endian independently of host endian.
    """
    supported = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    if not state or any(not isinstance(name, str) for name in state):
        raise EvaluationError("Model state must have nonempty named tensor entries")
    entries = []
    for name in sorted(state):
        tensor = state[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.dtype not in supported or tensor.device.type == "meta"):
            raise EvaluationError(f"Unsupported model state dtype/layout: {name}")
        flat = tensor.detach().contiguous().view(-1)
        digest = hashlib.sha256()
        for start in range(0, flat.numel(), _WEIGHT_CHUNK_VALUES):
            chunk = flat[start:start + _WEIGHT_CHUNK_VALUES].to(device="cpu", dtype=torch.float64)
            if not bool(torch.isfinite(chunk).all()):
                raise EvaluationError(f"Non-finite model weights: {name}")
            values = [0.0 if value == 0 else value for value in chunk.tolist()]
            digest.update(struct.pack("<" + "d" * len(values), *values))
        entries.append({"name": name, "shape": list(tensor.shape),
                        "values_sha256": "sha256:" + digest.hexdigest()})
    return _hash({"schema": "yug-eval-weight-values-v1", "tensors": entries})


def semantic_fingerprint(*, model_weights_identity: str, model_identity: str,
                         tokenizer_identity: str, eval_data_identity: str,
                         data_provenance: dict | None, sequence_length: int,
                         evaluation_semantics: str = EVALUATION_SEMANTICS) -> str:
    identities = dict(model_weights_identity=model_weights_identity, model_identity=model_identity,
                      tokenizer_identity=tokenizer_identity, eval_data_identity=eval_data_identity)
    for value in identities.values():
        _check_identity(value)
    _positive_integer(sequence_length, "sequence_length")
    if not isinstance(evaluation_semantics, str) or not evaluation_semantics:
        raise EvaluationError("Missing evaluation semantics version")
    return _hash({"schema": "yug-evaluation-identity-v1",
                  "evaluation_semantics": evaluation_semantics, **identities,
                  "data_provenance": normalize_provenance(data_provenance),
                  "sequence_length": sequence_length})


def execution_fingerprint(evaluation_fingerprint: str, batch_size: int,
                          execution_config: dict) -> str:
    _check_identity(evaluation_fingerprint)
    _positive_integer(batch_size, "batch_size")
    if not isinstance(execution_config, dict) or set(execution_config) != set(_EXECUTION_KEYS):
        raise EvaluationError("Incomplete or unknown execution settings")
    return _hash({"schema": "yug-evaluation-execution-v1",
                  "evaluation_fingerprint": evaluation_fingerprint,
                  "batch_size": batch_size, "execution_config": execution_config})


def _optional_setting(owner, name: str):
    """Null means API absent, never an exception from an available API."""
    if not hasattr(owner, name):
        return None
    value = getattr(owner, name)
    return value() if callable(value) else value


def capture_execution_config(model: torch.nn.Module) -> dict:
    try:
        tensors = list(model.parameters()) + list(model.buffers())
        devices = {str(t.device) for t in tensors}
        dtypes = {str(t.dtype).removeprefix("torch.") for t in tensors if t.is_floating_point()}
        if devices != {"cpu"} or len(dtypes) != 1:
            raise EvaluationError("P0 evaluation requires a CPU model with one floating dtype")
        # Torch 2.0-2.3 expose CPU autocast through a separate getter.
        version = tuple(int(x) for x in str(torch.__version__).split("+")[0].split(".")[:2])
        cpu_autocast = (torch.is_autocast_enabled("cpu") if version >= (2, 4)
                        else _optional_setting(torch, "is_autocast_cpu_enabled"))
        if cpu_autocast is None:
            raise EvaluationError("Cannot establish CPU autocast state")
        autocast = bool(cpu_autocast or torch.is_autocast_enabled())
        if autocast:
            raise EvaluationError("Ambient autocast is unsupported for P0 evaluation")
        mkldnn = getattr(torch.backends, "mkldnn", None)
        return {
            "device": "cpu", "model_dtype": next(iter(dtypes)),
            "torch_version": str(torch.__version__),
            "torch_num_threads": _optional_setting(torch, "get_num_threads"),
            "torch_num_interop_threads": _optional_setting(torch, "get_num_interop_threads"),
            "deterministic_algorithms": _optional_setting(torch, "are_deterministic_algorithms_enabled"),
            "deterministic_warn_only": _optional_setting(torch, "is_deterministic_algorithms_warn_only_enabled"),
            "float32_matmul_precision": _optional_setting(torch, "get_float32_matmul_precision"),
            "mkldnn_enabled": _optional_setting(mkldnn, "enabled"),
            "autocast_enabled": False, "nll_accumulation_dtype": "float64",
        }
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"Cannot capture execution settings: {exc}") from exc


def perplexity_from_loss(loss: float) -> float:
    try:
        result = math.exp(loss)
    except OverflowError as exc:
        raise EvaluationError("Evaluation perplexity overflow") from exc
    if not math.isfinite(loss) or not math.isfinite(result):
        raise EvaluationError("Non-finite evaluation loss/perplexity")
    return result


@dataclass(frozen=True)
class EvaluationReport:
    checkpoint_dir: str
    checkpoint_step: int
    model_identity: str
    model_parameter_count: int
    tokenizer_identity: str
    eval_token_count: int
    eval_sequence_count: int
    sequence_length: int
    batch_size: int
    seed: int
    loss: float
    perplexity: float
    total_nll: float
    encoded_token_count: int
    model_weights_identity: str
    eval_data_identity: str
    evaluation_fingerprint: str
    execution_fingerprint: str
    data_provenance: dict
    execution_config: dict
    schema_version: str = REPORT_SCHEMA
    evaluation_semantics: str = EVALUATION_SEMANTICS

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if self.schema_version != REPORT_SCHEMA or self.evaluation_semantics != EVALUATION_SEMANTICS:
            raise EvaluationError("Unsupported evaluation report/semantics version")
        if self.data_provenance != normalize_provenance(self.data_provenance):
            raise EvaluationError("Report provenance must include explicit null fields")
        for key in ("eval_token_count", "encoded_token_count", "eval_sequence_count", "model_parameter_count",
                    "sequence_length", "batch_size"):
            _positive_integer(getattr(self, key), key)
        if (self.eval_token_count > self.eval_sequence_count * self.sequence_length
                or self.eval_sequence_count != self.encoded_token_count // (self.sequence_length + 1)):
            raise EvaluationError("Inconsistent evaluation counts")
        if not math.isfinite(self.total_nll) or self.total_nll < 0:
            raise EvaluationError("Non-finite or negative total NLL")
        if self.loss != self.total_nll / self.eval_token_count:
            raise EvaluationError("Loss does not match NLL/target count")
        if self.perplexity != perplexity_from_loss(self.loss):
            raise EvaluationError("Perplexity does not match loss")
        semantic = semantic_fingerprint(
            model_weights_identity=self.model_weights_identity, model_identity=self.model_identity,
            tokenizer_identity=self.tokenizer_identity, eval_data_identity=self.eval_data_identity,
            data_provenance=self.data_provenance, sequence_length=self.sequence_length,
            evaluation_semantics=self.evaluation_semantics,
        )
        if semantic != self.evaluation_fingerprint:
            raise EvaluationError("Semantic fingerprint mismatch")
        if execution_fingerprint(semantic, self.batch_size, self.execution_config) != self.execution_fingerprint:
            raise EvaluationError("Execution fingerprint mismatch")

    def to_dict(self) -> dict:
        self.validate()  # catches mutation of nested dictionaries too
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    def to_human_readable(self) -> str:
        return "Evaluation Report\n" + "\n".join(f"  {key} = {value}" for key, value in self.to_dict().items())


def evaluate_checkpoint(checkpoint_dir: str | Path, tokenizer, eval_sentences: list[str], *,
                        sequence_length: int, batch_size: int, seed: int = 0,
                        data_provenance: dict | None = None) -> EvaluationReport:
    """Evaluate without training. Provenance is caller-declared, not verified.

    `seed` remains report-only bookkeeping. Existing checkpoint loaders and
    dataset errors are preserved. Model construction still consumes global RNG;
    this function does not restore checkpoint training RNG or change that API.
    """
    _positive_integer(sequence_length, "sequence_length")
    _positive_integer(batch_size, "batch_size")
    provenance = normalize_provenance(data_provenance)
    data_identity = evaluation_data_identity(eval_sentences)
    checkpoint_dir = Path(checkpoint_dir)
    config = ModelConfig.load(checkpoint_dir / "model_config.json")
    _check_identity(tokenizer.identity)
    if config.tokenizer_identity != tokenizer.identity:
        raise EvaluationError("Checkpoint and supplied tokenizer identities differ")
    if sequence_length > config.context_length:
        raise EvaluationError("sequence_length exceeds model context_length")
    model = YugTransformer(config)
    execution = capture_execution_config(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metadata = load_checkpoint(checkpoint_dir, model=model, optimizer=optimizer, restore_rng_state=False)
    if metadata["tokenizer_identity"] != tokenizer.identity:
        raise EvaluationError("Checkpoint metadata tokenizer identity differs")
    weights_identity = model_weights_identity(model.state_dict())
    semantic = semantic_fingerprint(
        model_weights_identity=weights_identity, model_identity=config.identity,
        tokenizer_identity=tokenizer.identity, eval_data_identity=data_identity,
        data_provenance=provenance, sequence_length=sequence_length,
    )
    execution_id = execution_fingerprint(semantic, batch_size, execution)
    token_ids = encode_corpus(tokenizer, eval_sentences)
    sequences = make_sequences(token_ids, sequence_length)
    batches = iter_batches(sequences, batch_size)
    total_nll = 0.0
    total_valid_targets = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for inputs, targets in batches:
                nll, count = evaluation_nll_count(model(inputs), targets)
                total_nll += nll
                total_valid_targets += count
        if not total_valid_targets:
            raise EvaluationError("Evaluation has zero valid targets")
        if not math.isfinite(total_nll):
            raise EvaluationError("Non-finite total evaluation NLL")
    except TrainingError as exc:
        raise EvaluationError(str(exc)) from exc
    finally:
        model.train(was_training)
    if capture_execution_config(model) != execution:
        raise EvaluationError("Execution settings changed during evaluation")
    loss = total_nll / total_valid_targets
    return EvaluationReport(
        checkpoint_dir=str(checkpoint_dir), checkpoint_step=metadata["step"],
        model_identity=config.identity, model_parameter_count=model.count_parameters(),
        tokenizer_identity=tokenizer.identity, eval_token_count=total_valid_targets,
        eval_sequence_count=len(sequences), sequence_length=sequence_length,
        batch_size=batch_size, seed=seed, loss=loss, perplexity=perplexity_from_loss(loss),
        total_nll=total_nll, encoded_token_count=len(token_ids),
        model_weights_identity=weights_identity, eval_data_identity=data_identity,
        evaluation_fingerprint=semantic, execution_fingerprint=execution_id,
        data_provenance=provenance, execution_config=execution,
    )
