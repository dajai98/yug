"""A dependency-free, deterministic BrainInterface implementation.

Purpose (Directive, Phase 1 TESTING section): "Maintain mock/fake
adapter tests for interface behavior." This adapter never claims to
produce real inference -- its content is explicitly and visibly
synthetic (see ``_MOCK_MARKER``) so it can never be mistaken for a real
model's output further up the stack.
"""

from __future__ import annotations

import hashlib

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

_MOCK_MARKER = "[mock-brain]"
_MOCK_MODEL_ID = "mock-brain-v1"
_MOCK_EMBEDDING_DIM = 8


class MockBrainAdapter(BrainInterface):
    """In-memory fake brain. No model loading, no I/O, always available.

    ``generate()`` returns a deterministic, clearly-marked string
    derived from the input so tests can assert on it. ``embed()``
    returns a deterministic vector derived from a hash of the input
    text (NOT a semantically meaningful embedding -- explicitly
    unsuitable for real similarity search).
    """

    def __init__(self, model_id: str = _MOCK_MODEL_ID) -> None:
        self._model_id = model_id

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop_sequences: list[str] | None = None,
    ) -> GenerationResult:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        content = (
            f"{_MOCK_MARKER} received {len(messages)} message(s); "
            f"last user text: {last_user!r}"
        )
        return GenerationResult(
            content=content,
            finish_reason="stop",
            model_id=self._model_id,
            tool_calls=None,
            usage={
                "prompt_tokens": sum(len(m.content) for m in messages),
                "completion_tokens": len(content),
            },
        )

    async def embed(self, texts: list[str]) -> list[Embedding]:
        embeddings = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = tuple(b / 255.0 for b in digest[:_MOCK_EMBEDDING_DIM])
            embeddings.append(Embedding(values=values, model_id=self._model_id))
        return embeddings

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_id=self._model_id,
            context_length=8192,
            supports_tools=False,
            supports_vision=False,
            embedding_dim=_MOCK_EMBEDDING_DIM,
            parameter_count=None,
            quantization=None,
            recommended_temperature_range=(0.0, 1.0),
            known_limitations=(
                "Not a real model: output is a deterministic synthetic marker.",
                "Embeddings are hash-derived, not semantically meaningful.",
            ),
        )

    def health_check(self) -> HealthStatus:
        return HealthStatus(
            state=HealthState.HEALTHY,
            detail="Mock adapter has no external dependency; always available.",
            model_loaded=True,
        )
