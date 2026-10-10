"""YUG native model configuration.

This module describes a model's architecture as data -- it does NOT
implement any of the architecture (no layers, attention, embeddings,
forward pass; that is NB-04/NB-05). Its job is:

1. Give every architectural knob a named, validated, serializable field
   (:class:`ModelConfig`).
2. Bind a config to a specific tokenizer via ``tokenizer_identity``, so
   a future checkpoint can detect a model/tokenizer mismatch (NB-1x).
3. Estimate parameter count from the config alone, deterministically,
   under explicitly documented architectural assumptions (see
   :func:`estimate_parameter_count`) -- this is arithmetic on the
   config, not a measurement of real weights (none exist yet).

No field here freezes a production hyperparameter choice. Values used
in tests/examples are illustrative, not authoritative -- see
``docs/architecture/NATIVE_BRAIN_ARCHITECTURE.md`` Open Decisions for
what remains genuinely undecided.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from yug.native.errors import ModelConfigSerializationError, ModelConfigValidationError

_CONFIG_FORMAT_VERSION = "1"
_ARCHITECTURE = "yug_decoder_transformer"

_VALID_NORMALIZATION = frozenset({"layernorm", "rmsnorm"})
_VALID_POSITIONAL_ENCODING = frozenset({"rope", "learned", "none"})
_VALID_ACTIVATION = frozenset({"gelu", "relu", "silu", "swiglu"})

# Matches the identity format produced by yug.tokenizer.ByteLevelBpeTokenizer
# ("sha256:" + 64 hex chars). This is a format check only -- verifying that
# the referenced tokenizer actually exists/matches is a future checkpoint-
# loading concern (NB-12), out of scope here.
_IDENTITY_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class ModelConfig:
    """Describes a YUG native model's architecture, without implementing it.

    Every field is independently justified in the accompanying NB-03
    report (table: field / purpose / checkpoint-critical?). Nothing
    here is a hardcoded production choice -- construct via
    :meth:`from_tokenizer` or the constructor directly, always
    supplying real values, never relying on a "default architecture."
    """

    vocab_size: int
    tokenizer_identity: str
    hidden_size: int
    num_layers: int
    num_heads: int
    ffn_size: int
    context_length: int
    dropout: float = 0.0
    normalization: str = "rmsnorm"
    positional_encoding: str = "rope"
    activation: str = "silu"
    bias: bool = False
    tie_embeddings: bool = True
    architecture: str = _ARCHITECTURE

    # -- construction ---------------------------------------------------

    @classmethod
    def from_tokenizer(cls, tokenizer, **architecture_kwargs) -> "ModelConfig":
        """Build a config bound to a real, already-trained tokenizer.

        Pulls ``vocab_size`` and ``tokenizer_identity`` directly from
        the tokenizer instance (must expose ``.vocab_size`` and
        ``.identity``, as ``yug.tokenizer.ByteLevelBpeTokenizer`` does)
        rather than requiring the caller to copy them by hand -- this
        is the mechanism that prevents silently inventing a vocab size
        disconnected from the tokenizer actually used.

        All other architecture fields must be supplied explicitly via
        ``architecture_kwargs`` (e.g. ``hidden_size=256, num_layers=6,
        ...``) -- there is no implicit default architecture.
        """
        return cls(
            vocab_size=tokenizer.vocab_size,
            tokenizer_identity=tokenizer.identity,
            **architecture_kwargs,
        )

    def __post_init__(self) -> None:
        validate(self)

    # -- identity ---------------------------------------------------------

    @property
    def identity(self) -> str:
        """A stable content hash over this config's semantic fields.

        Computed the same way as
        ``yug.tokenizer.ByteLevelBpeTokenizer.identity``: ``"sha256:" +``
        hexdigest of canonical (sorted-key, compact) JSON. Excludes
        nothing except the serialization format version (analogous to
        the tokenizer's own hash excluding
        ``yug_tokenizer_format_version``) -- every architectural field,
        including ``tokenizer_identity``, contributes, since changing
        any of them changes what checkpoint this config is compatible
        with.
        """
        return _compute_identity(self._semantic_payload())

    def _semantic_payload(self) -> dict:
        payload = asdict(self)
        return payload

    # -- serialization ------------------------------------------------

    def to_dict(self) -> dict:
        data = self._semantic_payload()
        data["yug_model_config_format_version"] = _CONFIG_FORMAT_VERSION
        data["identity"] = self.identity
        return data

    def save(self, path: str | Path) -> None:
        """Serialize to a JSON file (never pickle, matching the same
        reasoning as the tokenizer's and the checkpoint proposal's
        rejection of pickle: it can execute arbitrary code on load)."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        """Load a config previously written by :meth:`save`.

        Raises:
            ModelConfigSerializationError: file missing, invalid JSON,
                not an object, wrong/missing format version, missing a
                required field, or its stored identity hash does not
                match a hash recomputed from its own contents.
        """
        file_path = Path(path)
        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ModelConfigSerializationError(f"Model config file not found: {file_path}") from exc

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ModelConfigSerializationError(
                f"Invalid JSON in model config file {file_path}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ModelConfigSerializationError(f"Model config file {file_path} must contain a JSON object")

        version = data.get("yug_model_config_format_version")
        if version != _CONFIG_FORMAT_VERSION:
            raise ModelConfigSerializationError(
                f"Unsupported model config format version {version!r} in {file_path}; "
                f"expected {_CONFIG_FORMAT_VERSION!r}."
            )

        field_names = {f for f in cls.__dataclass_fields__}
        missing = field_names - data.keys()
        if missing:
            raise ModelConfigSerializationError(
                f"Model config file {file_path} is missing required field(s): {sorted(missing)}"
            )

        stored_identity = data.get("identity")
        if stored_identity is None:
            raise ModelConfigSerializationError(
                f"Model config file {file_path} is missing required field: 'identity'"
            )

        try:
            instance = cls(**{name: data[name] for name in field_names})
        except ModelConfigValidationError:
            raise
        except (TypeError, ValueError) as exc:
            raise ModelConfigSerializationError(
                f"Malformed model config field(s) in {file_path}: {exc}"
            ) from exc

        if instance.identity != stored_identity:
            raise ModelConfigSerializationError(
                f"Identity hash mismatch in {file_path}: file claims "
                f"{stored_identity!r} but recomputes to {instance.identity!r}. "
                f"The file may be corrupted or hand-edited."
            )

        return instance


def _compute_identity(payload: dict) -> str:
    import hashlib

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def validate(config: ModelConfig) -> None:
    """Validate a :class:`ModelConfig`'s fields and field combinations.

    Called automatically from ``__post_init__`` -- a :class:`ModelConfig`
    can never exist in an invalid state; this function is also exposed
    directly for callers who want to validate a config assembled some
    other way (e.g. after mutation via ``dataclasses.replace``).

    Raises:
        ModelConfigValidationError: on the first invariant violated.
    """
    if config.architecture != _ARCHITECTURE:
        raise ModelConfigValidationError(
            f"architecture must be {_ARCHITECTURE!r}, got {config.architecture!r}"
        )

    if not isinstance(config.vocab_size, int) or config.vocab_size <= 0:
        raise ModelConfigValidationError(f"vocab_size must be a positive integer, got {config.vocab_size!r}")

    if not isinstance(config.tokenizer_identity, str) or not _IDENTITY_PATTERN.match(config.tokenizer_identity):
        raise ModelConfigValidationError(
            f"tokenizer_identity must look like 'sha256:<64 hex chars>' "
            f"(the identity format produced by ByteLevelBpeTokenizer), "
            f"got {config.tokenizer_identity!r}"
        )

    if not isinstance(config.hidden_size, int) or config.hidden_size <= 0:
        raise ModelConfigValidationError(f"hidden_size must be a positive integer, got {config.hidden_size!r}")

    if not isinstance(config.num_layers, int) or config.num_layers <= 0:
        raise ModelConfigValidationError(f"num_layers must be a positive integer, got {config.num_layers!r}")

    if not isinstance(config.num_heads, int) or config.num_heads <= 0:
        raise ModelConfigValidationError(f"num_heads must be a positive integer, got {config.num_heads!r}")

    if config.hidden_size % config.num_heads != 0:
        raise ModelConfigValidationError(
            f"hidden_size ({config.hidden_size}) must be divisible by "
            f"num_heads ({config.num_heads})"
        )

    if not isinstance(config.ffn_size, int) or config.ffn_size <= 0:
        raise ModelConfigValidationError(f"ffn_size must be a positive integer, got {config.ffn_size!r}")

    if not isinstance(config.context_length, int) or config.context_length <= 0:
        raise ModelConfigValidationError(
            f"context_length must be a positive integer, got {config.context_length!r}"
        )

    if not isinstance(config.dropout, (int, float)) or not (0.0 <= config.dropout < 1.0):
        raise ModelConfigValidationError(f"dropout must be in [0, 1), got {config.dropout!r}")

    if config.normalization not in _VALID_NORMALIZATION:
        raise ModelConfigValidationError(
            f"normalization must be one of {sorted(_VALID_NORMALIZATION)}, got {config.normalization!r}"
        )

    if config.positional_encoding not in _VALID_POSITIONAL_ENCODING:
        raise ModelConfigValidationError(
            f"positional_encoding must be one of {sorted(_VALID_POSITIONAL_ENCODING)}, "
            f"got {config.positional_encoding!r}"
        )

    if config.activation not in _VALID_ACTIVATION:
        raise ModelConfigValidationError(
            f"activation must be one of {sorted(_VALID_ACTIVATION)}, got {config.activation!r}"
        )

    if not isinstance(config.bias, bool):
        raise ModelConfigValidationError(f"bias must be a bool, got {config.bias!r}")

    if not isinstance(config.tie_embeddings, bool):
        raise ModelConfigValidationError(f"tie_embeddings must be a bool, got {config.tie_embeddings!r}")


@dataclass(frozen=True)
class ParameterEstimate:
    """Result of :func:`estimate_parameter_count`.

    ``total`` and every entry in ``breakdown`` are exact arithmetic
    given ``basis`` (the stated architectural assumptions) -- they are
    not a measurement of real weights, since no architecture
    implementation exists yet (NB-04/05). Whether ``basis`` matches
    what NB-04/05 eventually implement is not yet knowable; treat
    ``total`` as [EXACT given basis] / [ESTIMATE relative to the real
    future implementation], not as a verified parameter count.
    """

    total: int
    breakdown: dict
    basis: str


def estimate_parameter_count(config: ModelConfig) -> ParameterEstimate:
    """Compute parameter count from ``config`` alone, deterministically.

    Assumptions (the "basis" -- see :class:`ParameterEstimate`), all
    standard for a pre-norm decoder-only Transformer:

    - Token embedding: ``vocab_size x hidden_size``.
    - Positional embedding: ``context_length x hidden_size`` if
      ``positional_encoding == "learned"``, else 0 (RoPE and "none"
      both add zero learned positional parameters).
    - Per layer, attention: 4 projections (Q, K, V, O), each
      ``hidden_size x hidden_size``, i.e. ``4 * hidden_size**2``
      (+ ``4 * hidden_size`` if ``bias``). Note: this total does NOT
      depend on ``num_heads`` -- splitting the same
      ``hidden_size``-dimensional projections into more or fewer heads
      changes head width, not total parameter count. ``num_heads`` is
      validated for shape compatibility, not used in this formula.
    - Per layer, feed-forward: standard 2-projection FFN
      (``activation`` in gelu/relu/silu) costs
      ``2 * hidden_size * ffn_size`` (+ ``ffn_size + hidden_size`` if
      ``bias``); a gated FFN (``activation == "swiglu"``) costs a third
      projection, ``3 * hidden_size * ffn_size`` (+
      ``2 * ffn_size + hidden_size`` if ``bias``).
    - Per layer, normalization: 2 instances (pre-attention,
      pre-feed-forward). ``rmsnorm`` costs ``hidden_size`` per instance
      (scale only); ``layernorm`` costs ``2 * hidden_size`` per
      instance (scale + shift).
    - One final normalization instance before the output projection.
    - Output projection: 0 if ``tie_embeddings`` (shares the token
      embedding matrix), else ``hidden_size x vocab_size``
      (+ ``vocab_size`` if ``bias``).
    """
    h = config.hidden_size
    v = config.vocab_size
    ffn = config.ffn_size
    L = config.num_layers
    bias = config.bias

    embedding = v * h
    positional = (config.context_length * h) if config.positional_encoding == "learned" else 0

    attn_per_layer = 4 * h * h + (4 * h if bias else 0)

    if config.activation == "swiglu":
        ffn_per_layer = 3 * h * ffn + ((2 * ffn + h) if bias else 0)
    else:
        ffn_per_layer = 2 * h * ffn + ((ffn + h) if bias else 0)

    norm_per_instance = h if config.normalization == "rmsnorm" else 2 * h
    norm_per_layer = 2 * norm_per_instance

    per_layer_total = attn_per_layer + ffn_per_layer + norm_per_layer
    all_layers = L * per_layer_total

    final_norm = norm_per_instance

    if config.tie_embeddings:
        output_projection = 0
    else:
        output_projection = h * v + (v if bias else 0)

    total = embedding + positional + all_layers + final_norm + output_projection

    breakdown = {
        "embedding": embedding,
        "positional": positional,
        "attention_all_layers": L * attn_per_layer,
        "ffn_all_layers": L * ffn_per_layer,
        "normalization_all_layers_plus_final": L * norm_per_layer + final_norm,
        "output_projection": output_projection,
    }

    basis = (
        "Standard pre-norm decoder-only Transformer: 4 hidden x hidden "
        "attention projections per layer (num_heads affects head width, "
        "not parameter count); "
        f"{'gated (swiglu) 3-projection' if config.activation == 'swiglu' else 'standard 2-projection'} "
        "FFN; 2 normalization instances per layer + 1 final; "
        f"{'tied' if config.tie_embeddings else 'untied'} input/output embeddings; "
        f"{'with' if bias else 'without'} linear-layer biases; "
        f"positional encoding = {config.positional_encoding!r} "
        f"({'contributes learned params' if config.positional_encoding == 'learned' else 'contributes zero learned params'})."
    )

    return ParameterEstimate(total=total, breakdown=breakdown, basis=basis)
