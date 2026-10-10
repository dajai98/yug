"""Native YUG adapter for BrainInterface.

NB-10: exposes the native ``YugTransformer`` (NB-04) + deterministic
greedy generation (NB-06) + checkpoint loading (via the shared
``yug.native.checkpoint_io`` reader, also NB-10) through the same
``BrainInterface`` contract :class:`yug.brain.adapters.llama_cpp.LlamaCppAdapter`
satisfies -- so callers depending on ``BrainInterface`` cannot tell
which backend they are talking to (architectural invariant #1: the
model is replaceable).

Design constraints (mirroring ``LlamaCppAdapter``'s own constraints,
restated for this backend):
  - "model loading isolated inside adapter" -- ``torch``,
    ``YugTransformer``, ``yug.native.generation``, and
    ``yug.native.checkpoint_io`` are all imported lazily, inside
    :meth:`_load_model_sync` (or inside :meth:`generate` itself for
    the generation call), never at module scope, so this module can be
    imported -- and :meth:`get_capabilities`/:meth:`health_check`
    called -- in a base install with no ``native`` extra.
    ``yug.native.config`` is the one exception: it is imported at
    module scope because it has zero PyTorch dependency by design (see
    ``pyproject.toml``'s own comment on this) and is needed to answer
    :meth:`get_capabilities` without a load.
  - "configuration-driven model path" -- ``checkpoint_dir`` is a
    constructor argument; nothing here hardcodes a path.
  - "DO NOT fake successful inference" -- generation failures,
    unsupported capabilities (tools, non-empty ``stop_sequences``,
    embeddings), and tokenizer mismatches all raise a specific
    :class:`yug.brain.errors.BrainError` subclass; nothing returns
    placeholder content, and ``temperature`` is never silently honored
    as if sampling were supported.

Reuses, does not duplicate (``ARCHITECTURE/NB10_NATIVE_ADAPTER_PLAN.md``
Section 6):
  - ``yug.native.config.ModelConfig`` / ``estimate_parameter_count``
  - ``yug.native.model.YugTransformer``
  - ``yug.native.generation.generate``          (UNCHANGED -- greedy only)
  - ``yug.native.checkpoint_io.load_model_state`` (NB-10's shared reader)
  - the caller-supplied tokenizer (``yug.tokenizer.bpe.ByteLevelBpeTokenizer``)

No second checkpoint format, no second tokenizer, no sampling logic,
no chat-template infrastructure beyond the minimal, adapter-local
formatting in :meth:`_format_prompt`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from yug.brain.errors import (
    EmbeddingNotSupportedError,
    GenerationError,
    ModelLoadError,
    ModelNotFoundError,
)
from yug.brain.interface import BrainInterface
from yug.brain.types import (
    Embedding,
    GenerationResult,
    HealthState,
    HealthStatus,
    Message,
    ModelCapabilities,
    ToolDefinition,
)
from yug.native.config import ModelConfig, estimate_parameter_count
from yug.native.errors import ModelConfigSerializationError
from yug.tokenizer.errors import DecodeError


class YugNativeAdapter(BrainInterface):
    """``BrainInterface`` implementation backed by the native
    ``YugTransformer``.

    The model is NOT constructed or loaded in ``__init__`` -- only
    ``model_config.json`` is read there (zero-PyTorch, per
    ``yug.native.config``'s own design), so :meth:`get_capabilities`
    can answer immediately without requiring the ``native`` extra to
    be installed. PyTorch, ``YugTransformer``, native generation, and
    the checkpoint-state reader are all imported and constructed
    lazily, on the first call to :meth:`generate`, exactly matching
    ``LlamaCppAdapter``'s own lazy-load contract.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        tokenizer: Any,  # yug.tokenizer.bpe.ByteLevelBpeTokenizer
        model_id: str | None = None,
        device: str = "cpu",
        max_new_tokens_default: int = 256,
    ) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._tokenizer = tokenizer
        self._device = device
        self._max_new_tokens_default = max_new_tokens_default
        self._model: Any | None = None  # set on first successful load

        # ModelConfig has zero PyTorch dependency (yug.native.config,
        # by design), so reading it here neither requires the "native"
        # extra nor counts as "eagerly loading model weights" -- it is
        # deliberately NOT wrapped in a lazy-load path, unlike
        # everything else in this class. If it's missing/corrupt, this
        # does not raise out of __init__ (matching LlamaCppAdapter,
        # whose own __init__ never touches the filesystem either) --
        # the failure surfaces honestly via health_check() and
        # get_capabilities() instead.
        try:
            self._model_config: ModelConfig | None = ModelConfig.load(
                self._checkpoint_dir / "model_config.json"
            )
        except ModelConfigSerializationError:
            self._model_config = None

        self._model_id = model_id or self._checkpoint_dir.name

    # -- lazy loading -----------------------------------------------

    def _load_model_sync(self) -> Any:
        """Blocking model construction + weight load. Runs off the
        event loop via ``asyncio.to_thread`` in :meth:`generate`,
        matching ``LlamaCppAdapter``'s own pattern."""
        # Deferred imports: torch, YugTransformer, and the shared
        # checkpoint-state reader are only pulled in here, so importing
        # this module -- and calling get_capabilities()/health_check()
        # -- never requires the "native" extra to be installed
        # (ARCHITECTURE/NB10_NATIVE_ADAPTER_PLAN.md Section 9, Risk 2).
        try:
            from yug.native.checkpoint_io import (
                NativeCheckpointError,
                load_model_state,
            )
            from yug.native.model import YugTransformer
        except ImportError as exc:
            raise ModelLoadError(
                "PyTorch is not installed. Install it with "
                "`pip install -e '.[native]'` to use YugNativeAdapter."
            ) from exc

        if self._model_config is None:
            raise ModelNotFoundError(
                f"No valid model_config.json found under {self._checkpoint_dir}"
            )

        if self._model_config.tokenizer_identity != self._tokenizer.identity:
            raise ModelLoadError(
                f"Checkpoint's tokenizer_identity "
                f"{self._model_config.tokenizer_identity!r} does not match "
                f"the supplied tokenizer's identity "
                f"{self._tokenizer.identity!r}. Loading would silently "
                f"produce meaningless output."
            )

        model = YugTransformer(self._model_config)

        try:
            model_state = load_model_state(self._checkpoint_dir)
        except NativeCheckpointError as exc:
            raise ModelLoadError(
                f"Failed to load checkpoint state from {self._checkpoint_dir}: {exc}"
            ) from exc

        try:
            model.load_state_dict(model_state)
        except Exception as exc:  # noqa: BLE001 - normalize any shape/key mismatch
            raise ModelLoadError(
                f"Checkpoint state at {self._checkpoint_dir} is incompatible "
                f"with this model configuration: {exc}"
            ) from exc

        model.to(self._device)
        model.eval()
        return model

    async def _ensure_loaded(self) -> Any:
        if self._model is None:
            self._model = await asyncio.to_thread(self._load_model_sync)
        return self._model

    # -- prompt formatting --------------------------------------------

    @staticmethod
    def _format_prompt(messages: list[Message]) -> str:
        """Adapter-local message formatting (``ARCHITECTURE/
        NB10_NATIVE_ADAPTER_PLAN.md`` Sections 4/14) -- deliberately
        not a general chat-template system, and not a
        ``BrainInterface``/tokenizer concern. ``role: content`` per
        line, ending with an ``assistant:`` cue."""
        lines = [f"{m.role}: {m.content}" for m in messages]
        lines.append("assistant:")
        return "\n".join(lines)

    # -- BrainInterface -----------------------------------------------

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop_sequences: list[str] | None = None,
    ) -> GenerationResult:
        if tools:
            # Matches LlamaCppAdapter's exact precedent: be explicit
            # rather than silently ignoring caller intent.
            raise GenerationError("YugNativeAdapter does not support tool calling.")
        if stop_sequences:
            raise GenerationError(
                "YugNativeAdapter does not support stop_sequences: native "
                "generation (NB-06) only accepts a single opaque integer "
                "stop_token_id, not arbitrary strings."
            )
        # `temperature` is intentionally accepted but ignored: native
        # generation (NB-06) is deterministic greedy/argmax only, with
        # no sampling path of any kind. This is a documented, permanent
        # limitation (see get_capabilities()'s known_limitations), not
        # something this method works around.

        import torch

        from yug.native.generation import generate as native_generate

        model = await self._ensure_loaded()

        prompt = self._format_prompt(messages)
        prompt_ids = self._tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self._device)

        # Native generation only understands a single opaque integer
        # stop id (NB-06) -- if this tokenizer happens to define an
        # "eos" special token, reuse it so generation can genuinely
        # stop early; if not, generation simply runs the full
        # max_tokens budget, exactly as it always has.
        stop_token_id = self._tokenizer.special_tokens.get("eos")

        def _run() -> "torch.Tensor":
            try:
                return native_generate(
                    model,
                    input_ids,
                    max_new_tokens=max_tokens,
                    stop_token_id=stop_token_id,
                )
            except Exception as exc:  # noqa: BLE001 - normalize backend failure
                raise GenerationError(f"Generation failed: {exc}") from exc

        output_ids = await asyncio.to_thread(_run)

        new_ids = output_ids[0, input_ids.shape[1] :].tolist()
        # Greedy decoding has no notion of "valid UTF-8 byte boundary"
        # -- it can legitimately land on a token sequence whose
        # underlying bytes do not reconstitute valid UTF-8 (found
        # during NB-08 V2's real-environment run; see
        # training/experiments/nb08_experiment.py's
        # safe_decode_generation() for the same failure mode in that
        # context). ByteLevelBpeTokenizer.decode() correctly and
        # deliberately raises DecodeError for this rather than
        # silently returning corrupted text -- that is correct,
        # intentional tokenizer behavior, not something to work around
        # in the tokenizer. But a raw DecodeError is not a BrainError,
        # so left uncaught it would escape generate() entirely and
        # break BrainInterface's contract (and, in turn,
        # YugCore.respond()'s guarantee of never raising to its
        # caller). Normalize it the same way _run() above normalizes
        # any native_generate() failure.
        try:
            content = self._tokenizer.decode(new_ids)
        except DecodeError as exc:
            raise GenerationError(f"Generation failed: {exc}") from exc

        return GenerationResult(
            content=content,
            finish_reason="stop" if len(new_ids) < max_tokens else "length",
            model_id=self._model_id,
            tool_calls=None,
            usage={
                "prompt_tokens": input_ids.shape[1],
                "completion_tokens": len(new_ids),
            },
        )

    async def embed(self, texts: list[str]) -> list[Embedding]:
        raise EmbeddingNotSupportedError(
            "YugNativeAdapter's native model has no embedding head."
        )

    def get_capabilities(self) -> ModelCapabilities:
        # Deliberately does not require a loaded model to answer
        # (matches LlamaCppAdapter's own contract) -- but DOES require
        # a valid model_config.json, since unlike LlamaCppAdapter's
        # constructor-supplied context_length, every field here is
        # genuinely derived from that file. If it couldn't be read,
        # there is no honest, non-fabricated value to return for
        # context_length/parameter_count; that failure is reported
        # honestly here rather than invented as a placeholder --
        # health_check() is the method whose job is to describe this
        # situation as a state (UNAVAILABLE), not raise.
        if self._model_config is None:
            raise ModelNotFoundError(
                f"No valid model_config.json found under {self._checkpoint_dir}; "
                f"cannot report capabilities."
            )

        if self._model is not None:
            # A real, measured count from the actually-loaded model
            # takes precedence once available.
            parameter_count = self._model.count_parameters()
        else:
            # Config-only estimate -- yug.native.config.estimate_parameter_count
            # is deterministic and documented to match count_parameters()'s
            # own intent once a real model is built from the same config.
            parameter_count = estimate_parameter_count(self._model_config).total

        return ModelCapabilities(
            model_id=self._model_id,
            context_length=self._model_config.context_length,
            supports_tools=False,
            supports_vision=False,
            embedding_dim=None,
            parameter_count=parameter_count,
            quantization=None,
            recommended_temperature_range=(0.0, 0.0),
            known_limitations=(
                "Generation is deterministic greedy (argmax) decoding only "
                "-- temperature is accepted but ignored; no sampling of any "
                "kind is implemented.",
                "stop_sequences are not supported (raises GenerationError if "
                "non-empty) -- native generation only accepts a single "
                "opaque integer stop_token_id, not arbitrary strings.",
                "Tool calling is not implemented (raises GenerationError).",
                "No vision support.",
                "No embedding support (embed() raises EmbeddingNotSupportedError).",
            ),
        )

    def health_check(self) -> HealthStatus:
        if self._model_config is None:
            return HealthStatus(
                state=HealthState.UNAVAILABLE,
                detail=f"No valid model_config.json found under {self._checkpoint_dir}.",
                model_loaded=False,
            )

        try:
            import torch  # noqa: F401
        except ImportError:
            return HealthStatus(
                state=HealthState.UNAVAILABLE,
                detail="PyTorch is not installed.",
                model_loaded=False,
            )

        if not (self._checkpoint_dir / "state.pt").exists():
            return HealthStatus(
                state=HealthState.UNAVAILABLE,
                detail=f"Checkpoint state file not found under {self._checkpoint_dir}.",
                model_loaded=False,
            )

        if self._model is not None:
            return HealthStatus(
                state=HealthState.HEALTHY,
                detail="Model is loaded and was previously usable.",
                model_loaded=True,
            )

        return HealthStatus(
            state=HealthState.DEGRADED,
            detail=(
                "Dependency installed and checkpoint files present, but the "
                "model has not been loaded yet (loaded lazily on first "
                "generate() call)."
            ),
            model_loaded=False,
        )
