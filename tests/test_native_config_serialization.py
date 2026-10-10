from __future__ import annotations

import json

import pytest

from yug.native import ModelConfig, ModelConfigSerializationError, estimate_parameter_count

_ID_A = "sha256:" + "a" * 64
_ID_B = "sha256:" + "b" * 64


def _cfg(**overrides):
    kwargs = dict(
        vocab_size=8192,
        tokenizer_identity=_ID_A,
        hidden_size=256,
        num_layers=6,
        num_heads=8,
        ffn_size=1024,
        context_length=1024,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


# ---- serialization / round trip --------------------------------------------


def test_save_and_load_round_trip_preserves_fields(tmp_path):
    cfg = _cfg(dropout=0.1, activation="swiglu", bias=True, tie_embeddings=False)
    path = tmp_path / "config.json"
    cfg.save(path)

    reloaded = ModelConfig.load(path)

    assert reloaded == cfg  # dataclass equality: all fields equal


def test_save_and_load_round_trip_preserves_identity(tmp_path):
    cfg = _cfg()
    path = tmp_path / "config.json"
    cfg.save(path)
    reloaded = ModelConfig.load(path)
    assert reloaded.identity == cfg.identity


def test_serialization_is_deterministic():
    cfg = _cfg()
    text_a = json.dumps(cfg.to_dict(), sort_keys=True)
    text_b = json.dumps(cfg.to_dict(), sort_keys=True)
    assert text_a == text_b


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(ModelConfigSerializationError):
        ModelConfig.load(tmp_path / "missing.json")


def test_load_malformed_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(ModelConfigSerializationError):
        ModelConfig.load(path)


def test_load_wrong_format_version_raises(tmp_path):
    cfg = _cfg()
    path = tmp_path / "config.json"
    cfg.save(path)
    data = json.loads(path.read_text())
    data["yug_model_config_format_version"] = "999"
    path.write_text(json.dumps(data))

    with pytest.raises(ModelConfigSerializationError):
        ModelConfig.load(path)


def test_load_missing_required_field_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"yug_model_config_format_version": "1"}))
    with pytest.raises(ModelConfigSerializationError):
        ModelConfig.load(path)


def test_load_tampered_field_fails_identity_check(tmp_path):
    cfg = _cfg()
    path = tmp_path / "config.json"
    cfg.save(path)
    data = json.loads(path.read_text())
    # num_layers changed but still a valid value on its own -- this
    # must be caught by the identity-hash check, not incidentally by
    # an unrelated validation rule (contrast with hidden_size, which
    # would also break the hidden_size % num_heads invariant).
    data["num_layers"] = data["num_layers"] + 1
    path.write_text(json.dumps(data))

    with pytest.raises(ModelConfigSerializationError):
        ModelConfig.load(path)


# ---- identity hash ------------------------------------------------------------


def test_identity_is_stable_across_construction_paths():
    cfg_a = _cfg()
    cfg_b = ModelConfig(
        vocab_size=8192,
        tokenizer_identity=_ID_A,
        hidden_size=256,
        num_layers=6,
        num_heads=8,
        ffn_size=1024,
        context_length=1024,
    )
    assert cfg_a.identity == cfg_b.identity


def test_identity_changes_when_semantic_field_changes():
    base = _cfg().identity
    assert _cfg(hidden_size=128).identity != base
    assert _cfg(num_layers=4).identity != base
    assert _cfg(activation="gelu").identity != base
    assert _cfg(tie_embeddings=False).identity != base
    assert _cfg(tokenizer_identity=_ID_B).identity != base


def test_identity_unaffected_by_format_version_field():
    """Format version is a serialization-schema concern, not part of
    the model's semantic identity (same convention as the tokenizer)."""
    cfg = _cfg()
    # to_dict() embeds the format version; identity must not depend on it.
    data_with_identity = cfg.to_dict()
    assert "yug_model_config_format_version" in data_with_identity
    assert data_with_identity["identity"] == cfg.identity


# ---- parameter count ------------------------------------------------------------


def test_parameter_count_is_positive_and_deterministic():
    cfg = _cfg()
    est1 = estimate_parameter_count(cfg)
    est2 = estimate_parameter_count(cfg)
    assert est1.total > 0
    assert est1.total == est2.total
    assert est1.breakdown == est2.breakdown


def test_parameter_count_breakdown_sums_to_total():
    cfg = _cfg()
    est = estimate_parameter_count(cfg)
    assert sum(est.breakdown.values()) == est.total


def test_tied_embeddings_reduces_parameter_count():
    tied = estimate_parameter_count(_cfg(tie_embeddings=True))
    untied = estimate_parameter_count(_cfg(tie_embeddings=False))
    assert tied.total < untied.total
    assert untied.total - tied.total == _cfg().hidden_size * _cfg().vocab_size


def test_swiglu_activation_increases_ffn_parameter_count():
    standard = estimate_parameter_count(_cfg(activation="gelu"))
    gated = estimate_parameter_count(_cfg(activation="swiglu"))
    assert gated.total > standard.total


def test_learned_positional_encoding_adds_parameters():
    rope = estimate_parameter_count(_cfg(positional_encoding="rope"))
    none_pe = estimate_parameter_count(_cfg(positional_encoding="none"))
    learned = estimate_parameter_count(_cfg(positional_encoding="learned"))

    assert rope.breakdown["positional"] == 0
    assert none_pe.breakdown["positional"] == 0
    assert learned.breakdown["positional"] == _cfg().context_length * _cfg().hidden_size
    assert learned.total > rope.total


def test_bias_increases_parameter_count():
    no_bias = estimate_parameter_count(_cfg(bias=False))
    with_bias = estimate_parameter_count(_cfg(bias=True))
    assert with_bias.total > no_bias.total


def test_num_heads_does_not_affect_parameter_count():
    """Documented assumption: head count changes head width, not total
    attention parameter count, for a fixed hidden_size."""
    a = estimate_parameter_count(_cfg(hidden_size=256, num_heads=4))
    b = estimate_parameter_count(_cfg(hidden_size=256, num_heads=8))
    assert a.total == b.total


def test_estimate_basis_documents_key_assumptions():
    est = estimate_parameter_count(_cfg(activation="swiglu", tie_embeddings=False, bias=True))
    assert "gated" in est.basis or "swiglu" in est.basis.lower()
    assert "untied" in est.basis
    assert "with" in est.basis and "bias" in est.basis
