from __future__ import annotations

from yug.brain.adapters.llama_cpp import LlamaCppAdapter
from yug.brain.adapters.mock import MockBrainAdapter
from yug.config.loader import load_config


def test_default_config_can_construct_llama_cpp_adapter():
    """The default (unconfigured) BrainConfig must still be enough to
    construct an adapter -- it just won't be usable for generation
    until model_path is set. This proves the config -> adapter wiring
    works even before a user supplies a real model."""
    config = load_config()
    adapter = LlamaCppAdapter(
        model_path=config.brain.model_path or "/no/model/configured.gguf",
        model_id=config.brain.model_id,
        context_length=config.brain.context_length,
        n_threads=config.brain.n_threads,
    )
    caps = adapter.get_capabilities()
    assert caps.model_id == config.brain.model_id
    assert caps.context_length == config.brain.context_length


def test_mock_backend_selection_constructs_mock_adapter(tmp_path):
    override = tmp_path / "mock.yaml"
    override.write_text("brain:\n  backend: mock\n  model_id: dev-mock\n")

    config = load_config(override)
    assert config.brain.backend == "mock"

    # Composition-root-style selection: real YUGCore wiring (Phase 6)
    # will do this dispatch; here we just prove the pieces connect.
    adapter = MockBrainAdapter(model_id=config.brain.model_id)
    assert adapter.get_capabilities().model_id == "dev-mock"
