"""llama.cpp adapter for BrainInterface.

Design constraints from the Directive:
  - "model loading isolated inside adapter" -- the ``llama_cpp`` package
    is imported lazily (only when actually needed), so this module can
    be imported and its error paths tested even on a machine without
    ``llama-cpp-python`` installed.
  - "configuration-driven model path" -- ``model_path`` is a
    constructor argument, sourced by the caller from ``YugConfig``.
    Nothing here hardcodes a path.
  - "DO NOT fake successful inference" -- if the dependency or the
    model file is missing, every method raises a specific
    :class:`yug.brain.errors.BrainError` subclass; nothing returns
    placeholder content.
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


class LlamaCppAdapter(BrainInterface):
    """BrainInterface implementation backed by ``llama-cpp-python``.

    The model is NOT loaded in ``__init__`` -- construction only
    records configuration. The first call to :meth:`generate` or
    :meth:`embed` triggers a lazy load (via :meth:`_ensure_loaded`),
    matching "model loading isolated inside adapter" without paying the
    (potentially large) load cost for adapters that are only being
    introspected (e.g. ``get_capabilities()``) or health-checked.
    """

    def __init__(
        self,
        model_path: str,
        model_id: str | None = None,
        context_length: int = 4096,
        n_threads: int | None = None,
        embedding: bool = True,
    ) -> None:
        self._model_path = model_path
        self._model_id = model_id or Path(model_path).stem
        self._context_length = context_length
        self._n_threads = n_threads
        self._embedding_enabled = embedding
        self._model: Any | None = None  # set on first successful load

    # -- lazy loading -----------------------------------------------

    def _load_model_sync(self) -> Any:
        """Blocking model load. Runs off the event loop via
        ``asyncio.to_thread`` in the async public methods."""
        if not Path(self._model_path).is_file():
            raise ModelNotFoundError(f"Model file not found: {self._model_path}")

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ModelLoadError(
                "llama-cpp-python is not installed. Install it with "
                "`pip install llama-cpp-python` to use LlamaCppAdapter."
            ) from exc

        try:
            return Llama(
                model_path=self._model_path,
                n_ctx=self._context_length,
                n_threads=self._n_threads,
                embedding=self._embedding_enabled,
            )
        except Exception as exc:  # noqa: BLE001 - normalize any backend failure
            raise ModelLoadError(
                f"Failed to load model at {self._model_path}: {exc}"
            ) from exc

    async def _ensure_loaded(self) -> Any:
        if self._model is None:
            self._model = await asyncio.to_thread(self._load_model_sync)
        return self._model

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
            # Tool-calling is Phase 3 scope; be explicit rather than
            # silently ignoring caller intent.
            raise GenerationError(
                "LlamaCppAdapter does not support tool calling in Phase 1."
            )

        model = await self._ensure_loaded()
        chat_messages = [{"role": m.role, "content": m.content} for m in messages]

        def _run() -> dict:
            try:
                return model.create_chat_completion(
                    messages=chat_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop_sequences,
                )
            except Exception as exc:  # noqa: BLE001 - normalize backend failure
                raise GenerationError(f"Generation failed: {exc}") from exc

        response = await asyncio.to_thread(_run)

        choice = response["choices"][0]
        return GenerationResult(
            content=choice["message"]["content"],
            finish_reason=choice.get("finish_reason", "stop"),
            model_id=self._model_id,
            tool_calls=None,
            usage=response.get("usage"),
        )

    async def embed(self, texts: list[str]) -> list[Embedding]:
        if not self._embedding_enabled:
            raise EmbeddingNotSupportedError(
                f"Adapter for {self._model_id} was constructed with embedding=False."
            )

        model = await self._ensure_loaded()

        def _run() -> list[list[float]]:
            try:
                result = model.embed(texts)
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingNotSupportedError(
                    f"Model {self._model_id} does not support embeddings: {exc}"
                ) from exc
            # llama-cpp-python returns either one vector per text or,
            # for a single text, a single flat vector -- normalize.
            if texts and result and isinstance(result[0], float):
                return [result]
            return result

        raw_vectors = await asyncio.to_thread(_run)
        return [
            Embedding(values=tuple(vector), model_id=self._model_id)
            for vector in raw_vectors
        ]

    def get_capabilities(self) -> ModelCapabilities:
        # Deliberately does not require a loaded model: this must be
        # answerable from configuration alone.
        return ModelCapabilities(
            model_id=self._model_id,
            context_length=self._context_length,
            supports_tools=False,
            supports_vision=False,
            embedding_dim=None,  # unknown until the model is actually loaded
            parameter_count=None,  # not introspected; informational only anyway
            quantization=None,
            recommended_temperature_range=(0.0, 1.5),
            known_limitations=(
                "parameter_count and quantization are not introspected from the "
                "model file in Phase 1.",
                "embedding_dim is unknown until the model is loaded.",
                "Tool calling is not implemented (raises GenerationError).",
            ),
        )

    def health_check(self) -> HealthStatus:
        # Cheap, non-loading check by design: health_check must not
        # force a potentially slow/expensive model load just to answer
        # "are you usable in principle". It reports what it actually
        # observed, never a guess.
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            return HealthStatus(
                state=HealthState.UNAVAILABLE,
                detail="llama-cpp-python is not installed.",
                model_loaded=False,
            )

        if not Path(self._model_path).is_file():
            return HealthStatus(
                state=HealthState.UNAVAILABLE,
                detail=f"Model file not found: {self._model_path}",
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
                "Dependency installed and model file present, but the model "
                "has not been loaded yet (loaded lazily on first generate/embed call)."
            ),
            model_loaded=False,
        )
