"""BrainInterface: the one contract every brain adapter must satisfy.

Architectural invariant #1: the model is replaceable. Nothing in this
module may import or reference a specific inference backend
(llama.cpp, vLLM, transformers, a cloud API, ...). Backend-specific
logic belongs exclusively in ``yug/brain/adapters/*``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from yug.brain.types import (
    Embedding,
    GenerationResult,
    HealthStatus,
    Message,
    ModelCapabilities,
    ToolDefinition,
)


class BrainInterface(ABC):
    """Abstract contract for a replaceable language-model brain."""

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop_sequences: list[str] | None = None,
    ) -> GenerationResult:
        """Generate a response from the model.

        Implementations must raise :class:`yug.brain.errors.BrainError`
        (or a subclass) on failure -- never return a fabricated
        :class:`GenerationResult` for a call that did not actually
        reach the backend.
        """
        raise NotImplementedError

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[Embedding]:
        """Generate embeddings for the given texts.

        Implementations must raise
        :class:`yug.brain.errors.EmbeddingNotSupportedError` if the
        backend/model has no embedding capability, rather than
        returning placeholder vectors.
        """
        raise NotImplementedError

    @abstractmethod
    def get_capabilities(self) -> ModelCapabilities:
        """Return static model metadata (context length, supported
        features, etc.). Must not require a loaded model to answer."""
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> HealthStatus:
        """Report genuinely observed current health.

        Must never report HEALTHY without having checked. See
        :class:`yug.brain.types.HealthStatus` docstring.
        """
        raise NotImplementedError
