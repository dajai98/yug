from __future__ import annotations

import inspect

import pytest

from yug.brain.adapters.llama_cpp import LlamaCppAdapter
from yug.brain.adapters.mock import MockBrainAdapter
from yug.brain.adapters.yug_native import YugNativeAdapter
from yug.brain.interface import BrainInterface
from yug.composition import build_brain, build_core
from yug.config.schema import BrainConfig, MemoryConfig, YugConfig
from yug.native.config import ModelConfig
from yug.runtime.core import YugCore
from yug.tokenizer.bpe import ByteLevelBpeTokenizer
from yug.utils.errors import ConfigValidationError

_TINY_CORPUS = ["the quick brown fox", "jumps over the lazy dog"]


def _tiny_tokenizer_and_config_on_disk(tmp_path):
    """Build a real (tiny) tokenizer + matching ModelConfig on disk --
    same construction pattern tests/test_brain_yug_native_adapter.py
    and tests/test_training_checkpoint.py already use for "tiny"
    fixtures, not a new convention invented for this test file.
    Deliberately does NOT write a state.pt checkpoint file: these tests
    prove construction/wiring only, never real inference (Directive
    Step 4 -- "do not silently substitute one for the other")."""
    tokenizer = ByteLevelBpeTokenizer.train(_TINY_CORPUS, vocab_size=260)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    model_config = ModelConfig.from_tokenizer(
        tokenizer,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_size=16,
        context_length=16,
    )
    model_config.save(checkpoint_dir / "model_config.json")

    return str(checkpoint_dir), str(tokenizer_path)


# -- Test 1: mock dispatch -------------------------------------------------


def test_mock_backend_dispatches_to_mock_brain_adapter():
    brain = build_brain(BrainConfig(backend="mock", model_id="test-mock"))

    assert isinstance(brain, MockBrainAdapter)
    assert isinstance(brain, BrainInterface)
    assert brain.get_capabilities().model_id == "test-mock"


# -- Test 2: llama_cpp dispatch, no real inference -------------------------


def test_llama_cpp_backend_dispatches_to_llama_cpp_adapter_without_loading_a_model():
    brain = build_brain(
        BrainConfig(backend="llama_cpp", model_id="test-llama", model_path=None)
    )

    assert isinstance(brain, LlamaCppAdapter)
    assert isinstance(brain, BrainInterface)
    # Constructing the adapter must not attempt to load anything -- no
    # llama-cpp-python is installed in this environment, so if
    # build_brain() eagerly loaded a model, this test itself would
    # fail here rather than just documenting lazy-load behavior.
    caps = brain.get_capabilities()
    assert caps.model_id == "test-llama"


# -- Test 2b: yug_native dispatch, construction/wiring only, no real
#             checkpoint weights and no PyTorch requirement -----------------


def test_yug_native_backend_dispatches_to_yug_native_adapter_without_loading_a_model(tmp_path):
    checkpoint_dir, tokenizer_path = _tiny_tokenizer_and_config_on_disk(tmp_path)

    brain = build_brain(
        BrainConfig(
            backend="yug_native",
            model_id="test-native",
            checkpoint_dir=checkpoint_dir,
            tokenizer_path=tokenizer_path,
        )
    )

    assert isinstance(brain, YugNativeAdapter)
    assert isinstance(brain, BrainInterface)
    # No state.pt exists in checkpoint_dir -- if build_brain() had
    # eagerly loaded real model weights, this construction-only test
    # would itself fail here rather than just proving wiring/dispatch.
    caps = brain.get_capabilities()
    assert caps.model_id == "test-native"


def test_yug_native_backend_requires_checkpoint_dir():
    with pytest.raises(ConfigValidationError):
        build_brain(
            BrainConfig(backend="yug_native", checkpoint_dir=None, tokenizer_path="/some/tokenizer.json")
        )


def test_yug_native_backend_requires_tokenizer_path(tmp_path):
    with pytest.raises(ConfigValidationError):
        build_brain(
            BrainConfig(backend="yug_native", checkpoint_dir=str(tmp_path), tokenizer_path=None)
        )


def test_yug_native_backend_reports_config_validation_error_for_unloadable_tokenizer(tmp_path):
    with pytest.raises(ConfigValidationError):
        build_brain(
            BrainConfig(
                backend="yug_native",
                checkpoint_dir=str(tmp_path),
                tokenizer_path=str(tmp_path / "does_not_exist.json"),
            )
        )


# -- Test 3: unknown backend fails explicitly ------------------------------


def test_unknown_backend_raises_config_validation_error():
    with pytest.raises(ConfigValidationError):
        build_brain(BrainConfig(backend="not_a_real_backend"))


def test_unknown_backend_does_not_silently_fall_back_to_mock_or_llama_cpp():
    try:
        build_brain(BrainConfig(backend="not_a_real_backend"))
    except ConfigValidationError as exc:
        # The error must name the requested (bad) backend, not quietly
        # substitute a default -- proves this wasn't caught-and-ignored
        # somewhere and silently resolved to a real adapter instead.
        assert "not_a_real_backend" in str(exc)
    else:
        pytest.fail("expected ConfigValidationError, no exception was raised")


# -- Test 4: core integration -----------------------------------------------


def test_build_core_constructs_a_working_yugcore_with_the_mock_brain(tmp_path):
    config = YugConfig(
        brain=BrainConfig(backend="mock", model_id="core-test-mock"),
        memory=MemoryConfig(storage_path=str(tmp_path / "memory.json")),
    )

    core = build_core(config)

    assert isinstance(core, YugCore)
    response = core.respond("hello")
    assert response.success is True
    assert "[mock-brain]" in response.content  # genuinely reached MockBrainAdapter.generate()


def test_build_core_constructs_a_working_yugcore_with_the_yug_native_brain(tmp_path):
    """Construction/wiring only -- proves YugConfig -> build_core ->
    YugCore reaches a real YugNativeAdapter.generate() call, through
    the public YugCore API only (no reaching into private state).

    This does NOT prove real inference works: no state.pt checkpoint
    exists in checkpoint_dir (deliberately -- see
    _tiny_tokenizer_and_config_on_disk's own docstring), so
    YugNativeAdapter.generate() cannot succeed here regardless of
    whether PyTorch is installed in this environment:
      - no torch installed -> generate() raises ModuleNotFoundError
      - torch installed -> _load_model_sync() reaches
        load_model_state() and raises ModelLoadError (no state.pt)
    Either way YugCore.respond()'s own documented contract (brain
    failures are reported via Response.success=False, never raised)
    is what this test actually checks -- the one thing genuinely
    provable without a real trained checkpoint or PyTorch. Real
    generation is covered by tests/test_runtime_yug_native_regression.py
    (torch-gated, unmodified by this task)."""
    checkpoint_dir, tokenizer_path = _tiny_tokenizer_and_config_on_disk(tmp_path)
    config = YugConfig(
        brain=BrainConfig(
            backend="yug_native",
            model_id="core-test-native",
            checkpoint_dir=checkpoint_dir,
            tokenizer_path=tokenizer_path,
        ),
        memory=MemoryConfig(storage_path=str(tmp_path / "memory.json")),
    )

    core = build_core(config)
    assert isinstance(core, YugCore)

    response = core.respond("hello")

    assert response.success is False
    assert response.error  # some BrainError/ImportError description, not fabricated content


# -- Test 5: brain isolation -------------------------------------------------


def test_composition_does_not_import_training():
    """Directive Step 4.7: wiring in yug_native must not introduce a
    yug -> training dependency. yug.composition itself is the only
    file this task touched that could plausibly have grown one (the
    other edits were to yug.config.schema/defaults.yaml, which add
    plain str fields, not imports) -- checked directly rather than
    assumed."""
    import yug.composition as composition_module

    source = inspect.getsource(composition_module)
    assert "training" not in source


def test_yugcore_does_not_import_concrete_adapters():
    """Static inspection: yug.runtime.core must depend only on
    BrainInterface, never on a concrete adapter or the llama_cpp
    package. This is the property the whole composition root exists
    to preserve -- the composition module is allowed to import
    LlamaCppAdapter, YugCore must never be.

    NB-11: extended with a YugNativeAdapter assertion once that third,
    real backend existed (NB-10) -- the existing three assertions
    above are unchanged; this is additive, confirming the same
    property still holds rather than replacing the check with a
    different mechanism."""
    import yug.runtime.core as runtime_core

    source = inspect.getsource(runtime_core)

    assert "LlamaCppAdapter" not in source
    assert "MockBrainAdapter" not in source
    assert "llama_cpp" not in source
    assert "YugNativeAdapter" not in source
    assert "BrainInterface" in source  # sanity: it does depend on the abstraction
