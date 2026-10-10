from __future__ import annotations

from yug.brain.errors import (
    BrainError,
    BrainUnavailableError,
    EmbeddingNotSupportedError,
    GenerationError,
    ModelLoadError,
    ModelNotFoundError,
)
from yug.utils.errors import YugError


def test_brain_error_is_a_yug_error():
    assert issubclass(BrainError, YugError)


def test_model_not_found_error_is_a_model_load_error():
    assert issubclass(ModelNotFoundError, ModelLoadError)
    assert issubclass(ModelLoadError, BrainError)


def test_generation_embedding_and_unavailable_errors_are_brain_errors():
    assert issubclass(GenerationError, BrainError)
    assert issubclass(EmbeddingNotSupportedError, BrainError)
    assert issubclass(BrainUnavailableError, BrainError)


def test_brain_error_catchable_as_yug_error():
    caught = False
    try:
        raise ModelNotFoundError("no such file")
    except YugError as exc:
        caught = "no such file" in str(exc)
    assert caught
