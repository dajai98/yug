"""Tests for LlamaCppAdapter.

Test categories (Directive: "Tests must distinguish UNIT TEST,
INTEGRATION TEST, REQUIRES REAL MODEL"):

- UNIT TEST: adapter construction, capability reporting, and file-not-
  found paths. These never touch the ``llama_cpp`` package and pass
  regardless of whether it is installed.
- INTEGRATION TEST (dependency-conditional): confirms the adapter
  fails honestly (ModelLoadError) when asked to load a file that is
  not a valid model, whether or not ``llama-cpp-python`` happens to be
  installed in the current environment.
- REQUIRES REAL MODEL: actually loads a real GGUF file and generates.
  Skipped unless ``YUG_TEST_MODEL_PATH`` is set to an existing file.
  Not runnable in this development sandbox (no network to obtain a
  model); must be run by a developer with a real model file.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os

import pytest

from yug.brain.adapters.llama_cpp import LlamaCppAdapter
from yug.brain.errors import ModelLoadError, ModelNotFoundError
from yug.brain.interface import BrainInterface
from yug.brain.types import HealthState, Message

_LLAMA_CPP_INSTALLED = importlib.util.find_spec("llama_cpp") is not None


# ---- UNIT TESTS (no dependency, no model file required) -------------------


def test_llama_cpp_adapter_satisfies_brain_interface():
    assert isinstance(LlamaCppAdapter(model_path="/nonexistent.gguf"), BrainInterface)


def test_module_importable_without_llama_cpp_installed():
    """The adapter module itself must import cleanly even if
    llama-cpp-python is not installed -- the dependency is imported
    lazily inside _load_model_sync, not at module scope."""
    import yug.brain.adapters.llama_cpp  # noqa: F401 - import success is the assertion


def test_get_capabilities_does_not_require_a_loaded_model():
    adapter = LlamaCppAdapter(model_path="/nonexistent.gguf", context_length=8192)
    caps = adapter.get_capabilities()
    assert caps.context_length == 8192
    assert caps.embedding_dim is None
    assert caps.parameter_count is None  # never fabricated


def test_model_id_defaults_to_filename_stem():
    adapter = LlamaCppAdapter(model_path="/models/tiny-model.gguf")
    assert adapter.get_capabilities().model_id == "tiny-model"


def test_health_check_reports_unavailable_for_missing_file():
    """health_check() checks the dependency before the file (see its
    docstring), so with a missing file AND no llama-cpp-python
    installed, the dependency message is reported first. Either way
    the state must be UNAVAILABLE; the specific detail text depends on
    which precondition actually failed."""
    adapter = LlamaCppAdapter(model_path="/definitely/does/not/exist.gguf")
    status = adapter.health_check()
    assert status.state == HealthState.UNAVAILABLE
    assert status.model_loaded is False
    if _LLAMA_CPP_INSTALLED:
        assert "not found" in status.detail.lower()
    else:
        assert "not installed" in status.detail.lower()


def test_generate_raises_model_not_found_for_missing_file():
    adapter = LlamaCppAdapter(model_path="/definitely/does/not/exist.gguf")
    with pytest.raises(ModelNotFoundError):
        asyncio.run(adapter.generate([Message(role="user", content="hi")]))


def test_generate_rejects_tools_in_phase_1():
    from yug.brain.types import ToolDefinition

    adapter = LlamaCppAdapter(model_path="/definitely/does/not/exist.gguf")
    with pytest.raises(Exception):  # GenerationError, checked for tools before file check
        asyncio.run(
            adapter.generate(
                [Message(role="user", content="hi")],
                tools=[ToolDefinition(name="calc", description="adds")],
            )
        )


# ---- INTEGRATION TEST: dependency-conditional, no real model needed -------


def test_loading_a_non_model_file_fails_honestly(tmp_path):
    """Regardless of whether llama-cpp-python is installed, pointing the
    adapter at a file that is not a real GGUF model must raise
    ModelLoadError -- never fabricate a successful load."""
    fake_model = tmp_path / "not-a-real-model.gguf"
    fake_model.write_bytes(b"this is not a real gguf file")

    adapter = LlamaCppAdapter(model_path=str(fake_model))

    with pytest.raises(ModelLoadError) as exc_info:
        asyncio.run(adapter.generate([Message(role="user", content="hi")]))

    if not _LLAMA_CPP_INSTALLED:
        assert "pip install llama-cpp-python" in str(exc_info.value)


def test_health_check_degraded_when_dependency_and_file_present_but_unloaded(tmp_path):
    fake_model = tmp_path / "not-a-real-model.gguf"
    fake_model.write_bytes(b"this is not a real gguf file")

    adapter = LlamaCppAdapter(model_path=str(fake_model))
    status = adapter.health_check()

    if _LLAMA_CPP_INSTALLED:
        assert status.state == HealthState.DEGRADED
        assert status.model_loaded is False
    else:
        assert status.state == HealthState.UNAVAILABLE
        assert "not installed" in status.detail.lower()


# ---- REQUIRES REAL MODEL: skipped unless explicitly configured ------------

_REAL_MODEL_PATH = os.environ.get("YUG_TEST_MODEL_PATH")


@pytest.mark.skipif(
    not _LLAMA_CPP_INSTALLED,
    reason="REQUIRES REAL MODEL: llama-cpp-python is not installed in this environment.",
)
@pytest.mark.skipif(
    not _REAL_MODEL_PATH,
    reason=(
        "REQUIRES REAL MODEL: set the YUG_TEST_MODEL_PATH environment variable "
        "to a local .gguf file to run this test."
    ),
)
def test_real_model_generates_a_response():
    adapter = LlamaCppAdapter(model_path=_REAL_MODEL_PATH)
    result = asyncio.run(
        adapter.generate([Message(role="user", content="Say hello in one word.")])
    )
    assert isinstance(result.content, str)
    assert len(result.content) > 0

    status = adapter.health_check()
    assert status.state == HealthState.HEALTHY
    assert status.model_loaded is True
