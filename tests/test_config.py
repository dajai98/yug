from __future__ import annotations

import pytest

from yug.config.loader import load_config
from yug.config.schema import BrainConfig, LoggingConfig, MemoryConfig, PathsConfig, YugConfig, validate
from yug.utils.errors import ConfigurationError, ConfigValidationError


def test_load_config_defaults_only():
    config = load_config()

    assert config.logging.level == "INFO"
    assert config.paths.data_dir == "./data"
    assert config.paths.logs_dir == "./data/logs"
    assert config.brain.backend == "llama_cpp"
    assert config.brain.model_id == "unconfigured"
    assert config.brain.model_path is None
    assert config.brain.context_length == 4096
    assert config.memory.storage_path == "./data/memory.json"


def test_load_config_with_memory_override(tmp_path):
    override = tmp_path / "memory_override.yaml"
    override.write_text("memory:\n  storage_path: /tmp/custom_memory.json\n")

    config = load_config(override)

    assert config.memory.storage_path == "/tmp/custom_memory.json"


def test_validate_rejects_empty_memory_storage_path():
    config = YugConfig(memory=MemoryConfig(storage_path=""))

    with pytest.raises(ConfigValidationError):
        validate(config)


def test_load_config_with_brain_override(tmp_path):
    override = tmp_path / "brain_override.yaml"
    override.write_text(
        "brain:\n"
        "  backend: mock\n"
        "  model_id: test-model\n"
        "  model_path: /models/test.gguf\n"
        "  context_length: 2048\n"
    )

    config = load_config(override)

    assert config.brain.backend == "mock"
    assert config.brain.model_id == "test-model"
    assert config.brain.model_path == "/models/test.gguf"
    assert config.brain.context_length == 2048


def test_load_config_with_yug_native_brain_override(tmp_path):
    override = tmp_path / "yug_native_override.yaml"
    override.write_text(
        "brain:\n"
        "  backend: yug_native\n"
        "  model_id: test-native\n"
        "  checkpoint_dir: /models/native-checkpoint\n"
        "  tokenizer_path: /models/tokenizer.json\n"
    )

    config = load_config(override)

    assert config.brain.backend == "yug_native"
    assert config.brain.checkpoint_dir == "/models/native-checkpoint"
    assert config.brain.tokenizer_path == "/models/tokenizer.json"


def test_validate_rejects_invalid_brain_backend():
    config = YugConfig(brain=BrainConfig(backend="not_a_real_backend"))

    with pytest.raises(ConfigValidationError):
        validate(config)


def test_validate_rejects_non_positive_context_length():
    config = YugConfig(brain=BrainConfig(context_length=0))

    with pytest.raises(ConfigValidationError):
        validate(config)


def test_validate_allows_null_model_path():
    config = YugConfig(brain=BrainConfig(model_path=None))
    validate(config)  # should not raise


def test_validate_accepts_yug_native_backend():
    # validate() intentionally does NOT require checkpoint_dir/
    # tokenizer_path here, mirroring the existing precedent that
    # backend="llama_cpp" with model_path=None also passes validate()
    # -- a missing backend-specific field is a construction-time
    # failure (yug.composition._build_yug_native_adapter), not a
    # config-shape failure. See test_composition.py for that check.
    config = YugConfig(brain=BrainConfig(backend="yug_native"))
    validate(config)  # should not raise


def test_validate_rejects_empty_checkpoint_dir():
    config = YugConfig(brain=BrainConfig(backend="yug_native", checkpoint_dir="  "))

    with pytest.raises(ConfigValidationError):
        validate(config)


def test_validate_rejects_empty_tokenizer_path():
    config = YugConfig(brain=BrainConfig(backend="yug_native", tokenizer_path="  "))

    with pytest.raises(ConfigValidationError):
        validate(config)


def test_validate_allows_null_checkpoint_dir_and_tokenizer_path():
    config = YugConfig(brain=BrainConfig(checkpoint_dir=None, tokenizer_path=None))
    validate(config)  # should not raise


def test_load_config_with_user_override(tmp_path):
    override = tmp_path / "override.yaml"
    override.write_text("logging:\n  level: DEBUG\n")

    config = load_config(override)

    assert config.logging.level == "DEBUG"
    # Unspecified fields still come from defaults
    assert config.paths.data_dir == "./data"


def test_load_config_missing_user_file_raises(tmp_path):
    missing = tmp_path / "does-not-exist.yaml"

    with pytest.raises(ConfigurationError):
        load_config(missing)


def test_load_config_unknown_section_raises(tmp_path):
    # NOTE: "brain" was used as the unknown-section fixture in Phase 0.
    # Phase 1 legitimately adds a "brain" config section (see
    # BrainConfig), so it is no longer unknown -- switched to a section
    # name that will never be a real top-level section.
    bad = tmp_path / "bad.yaml"
    bad.write_text("totally_unrecognized_section:\n  foo: bar\n")

    with pytest.raises(ConfigurationError):
        load_config(bad)


def test_load_config_malformed_yaml_raises(tmp_path):
    bad = tmp_path / "malformed.yaml"
    bad.write_text("logging: [unclosed\n")

    with pytest.raises(ConfigurationError):
        load_config(bad)


def test_validate_rejects_invalid_log_level():
    config = YugConfig(logging=LoggingConfig(level="NOT_A_LEVEL"), paths=PathsConfig())

    with pytest.raises(ConfigValidationError):
        validate(config)


def test_validate_rejects_empty_path():
    config = YugConfig(logging=LoggingConfig(), paths=PathsConfig(data_dir="  "))

    with pytest.raises(ConfigValidationError):
        validate(config)


def test_validate_accepts_default_config():
    config = YugConfig()
    validate(config)  # should not raise


def test_default_config_has_expected_tools_section():
    config = YugConfig()

    assert set(config.tools.enabled) == {"calculator", "python_exec", "file_read", "file_search"}
    assert config.tools.file_root == "./data/tool_files"
    assert config.tools.python_exec_timeout_seconds == 5.0


def test_load_config_with_tools_override(tmp_path):
    override = tmp_path / "tools_override.yaml"
    override.write_text(
        "tools:\n"
        "  enabled:\n"
        "    - calculator\n"
        "  file_root: /tmp/custom_tools\n"
        "  python_exec_timeout_seconds: 2.5\n"
    )

    config = load_config(override)

    assert tuple(config.tools.enabled) == ("calculator",)
    assert config.tools.file_root == "/tmp/custom_tools"
    assert config.tools.python_exec_timeout_seconds == 2.5


def test_validate_rejects_empty_tools_file_root():
    from yug.config.schema import ToolsConfig

    config = YugConfig(tools=ToolsConfig(file_root=""))

    with pytest.raises(ConfigValidationError):
        validate(config)


def test_validate_rejects_non_positive_file_read_max_bytes():
    from yug.config.schema import ToolsConfig

    config = YugConfig(tools=ToolsConfig(file_read_max_bytes=0))

    with pytest.raises(ConfigValidationError):
        validate(config)


def test_validate_rejects_non_positive_python_exec_timeout():
    from yug.config.schema import ToolsConfig

    config = YugConfig(tools=ToolsConfig(python_exec_timeout_seconds=0))

    with pytest.raises(ConfigValidationError):
        validate(config)
