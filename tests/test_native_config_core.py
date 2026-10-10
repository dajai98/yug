from __future__ import annotations

import pytest

from yug.native import ModelConfig, ModelConfigValidationError
from yug.tokenizer import ByteLevelBpeTokenizer

_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64


def _minimal_kwargs(**overrides):
    kwargs = dict(
        vocab_size=300,
        tokenizer_identity=_PLACEHOLDER_IDENTITY,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        ffn_size=256,
        context_length=128,
    )
    kwargs.update(overrides)
    return kwargs


# ---- valid configs --------------------------------------------------------


def test_minimal_valid_configuration_constructs():
    cfg = ModelConfig(**_minimal_kwargs())
    assert cfg.vocab_size == 300
    assert cfg.hidden_size == 64


def test_realistic_configuration_constructs():
    cfg = ModelConfig(
        **_minimal_kwargs(
            vocab_size=8192,
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            ffn_size=2048,
            context_length=1024,
            dropout=0.1,
            normalization="rmsnorm",
            positional_encoding="rope",
            activation="swiglu",
            bias=False,
            tie_embeddings=True,
        )
    )
    assert cfg.activation == "swiglu"
    assert cfg.normalization == "rmsnorm"


def test_defaults_are_sensible():
    cfg = ModelConfig(**_minimal_kwargs())
    assert cfg.dropout == 0.0
    assert cfg.normalization == "rmsnorm"
    assert cfg.positional_encoding == "rope"
    assert cfg.bias is False
    assert cfg.tie_embeddings is True


# ---- tokenizer integration --------------------------------------------------


def test_from_tokenizer_pulls_vocab_size_and_identity():
    tok = ByteLevelBpeTokenizer.train(
        ["the quick brown fox", "hello world"], vocab_size=270, special_tokens=("bos", "eos")
    )
    cfg = ModelConfig.from_tokenizer(
        tok, hidden_size=64, num_layers=2, num_heads=4, ffn_size=256, context_length=128
    )
    assert cfg.vocab_size == tok.vocab_size
    assert cfg.tokenizer_identity == tok.identity


def test_config_bound_to_different_tokenizers_have_different_identity():
    tok_a = ByteLevelBpeTokenizer.train(["aaa bbb ccc"], vocab_size=270)
    tok_b = ByteLevelBpeTokenizer.train(["completely different corpus text"], vocab_size=270)

    cfg_a = ModelConfig.from_tokenizer(tok_a, hidden_size=64, num_layers=2, num_heads=4, ffn_size=256, context_length=128)
    cfg_b = ModelConfig.from_tokenizer(tok_b, hidden_size=64, num_layers=2, num_heads=4, ffn_size=256, context_length=128)

    assert cfg_a.identity != cfg_b.identity


# ---- validation invariants --------------------------------------------------


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("vocab_size", 0),
        ("vocab_size", -1),
        ("hidden_size", 0),
        ("hidden_size", -8),
        ("num_layers", 0),
        ("num_layers", -1),
        ("num_heads", 0),
        ("num_heads", -1),
        ("ffn_size", 0),
        ("ffn_size", -1),
        ("context_length", 0),
        ("context_length", -1),
    ],
)
def test_non_positive_fields_raise(field, bad_value):
    with pytest.raises(ModelConfigValidationError):
        ModelConfig(**_minimal_kwargs(**{field: bad_value}))


def test_hidden_size_not_divisible_by_num_heads_raises():
    with pytest.raises(ModelConfigValidationError):
        ModelConfig(**_minimal_kwargs(hidden_size=65, num_heads=4))


def test_hidden_size_divisible_by_num_heads_succeeds():
    cfg = ModelConfig(**_minimal_kwargs(hidden_size=64, num_heads=4))
    assert cfg.hidden_size % cfg.num_heads == 0


@pytest.mark.parametrize("bad_dropout", [-0.1, 1.0, 1.5, -1.0])
def test_dropout_outside_valid_range_raises(bad_dropout):
    with pytest.raises(ModelConfigValidationError):
        ModelConfig(**_minimal_kwargs(dropout=bad_dropout))


@pytest.mark.parametrize("good_dropout", [0.0, 0.1, 0.5, 0.9999])
def test_dropout_inside_valid_range_succeeds(good_dropout):
    cfg = ModelConfig(**_minimal_kwargs(dropout=good_dropout))
    assert cfg.dropout == good_dropout


def test_unsupported_normalization_raises():
    with pytest.raises(ModelConfigValidationError):
        ModelConfig(**_minimal_kwargs(normalization="batchnorm"))


def test_unsupported_positional_encoding_raises():
    with pytest.raises(ModelConfigValidationError):
        ModelConfig(**_minimal_kwargs(positional_encoding="sinusoidal"))


def test_unsupported_activation_raises():
    with pytest.raises(ModelConfigValidationError):
        ModelConfig(**_minimal_kwargs(activation="tanh"))


def test_invalid_architecture_raises():
    with pytest.raises(ModelConfigValidationError):
        ModelConfig(**_minimal_kwargs(architecture="not_a_real_architecture"))


def test_invalid_tokenizer_identity_format_raises():
    with pytest.raises(ModelConfigValidationError):
        ModelConfig(**_minimal_kwargs(tokenizer_identity="not-a-valid-identity"))


def test_tokenizer_identity_wrong_hash_length_raises():
    with pytest.raises(ModelConfigValidationError):
        ModelConfig(**_minimal_kwargs(tokenizer_identity="sha256:deadbeef"))


def test_bias_must_be_bool():
    with pytest.raises(ModelConfigValidationError):
        ModelConfig(**_minimal_kwargs(bias="yes"))


def test_tie_embeddings_must_be_bool():
    with pytest.raises(ModelConfigValidationError):
        ModelConfig(**_minimal_kwargs(tie_embeddings=1))


# ---- boundary values --------------------------------------------------------


def test_smallest_valid_configuration():
    cfg = ModelConfig(**_minimal_kwargs(vocab_size=1, hidden_size=1, num_layers=1, num_heads=1, ffn_size=1, context_length=1))
    assert cfg.vocab_size == 1


def test_num_heads_equal_to_hidden_size_is_valid():
    cfg = ModelConfig(**_minimal_kwargs(hidden_size=64, num_heads=64))
    assert cfg.num_heads == 64
