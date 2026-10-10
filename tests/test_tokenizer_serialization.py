from __future__ import annotations

import pytest

from yug.tokenizer import ByteLevelBpeTokenizer, SerializationError

_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumps over the lazy dog again and again",
    "नमस्ते संसार, म ठिक छु",
    "emoji test: 😀🎉🚀",
]


def _train() -> ByteLevelBpeTokenizer:
    return ByteLevelBpeTokenizer.train(_CORPUS, vocab_size=300, special_tokens=("bos", "eos"))


# ---- determinism --------------------------------------------------------


def test_training_is_deterministic_for_same_corpus_and_vocab_size():
    tok_a = _train()
    tok_b = _train()
    assert tok_a.identity == tok_b.identity


def test_repeated_encoding_of_same_text_is_identical():
    tok = _train()
    text = "the quick brown fox नमस्ते 😀"
    first = tok.encode(text)
    second = tok.encode(text)
    third = tok.encode(text)
    assert first == second == third


def test_different_vocab_size_produces_different_identity():
    tok_a = ByteLevelBpeTokenizer.train(_CORPUS, vocab_size=280)
    tok_b = ByteLevelBpeTokenizer.train(_CORPUS, vocab_size=300)
    assert tok_a.identity != tok_b.identity


def test_different_corpus_produces_different_identity():
    tok_a = ByteLevelBpeTokenizer.train(_CORPUS, vocab_size=300)
    tok_b = ByteLevelBpeTokenizer.train(["completely different text here"], vocab_size=300)
    assert tok_a.identity != tok_b.identity


# ---- serialization / reload ----------------------------------------------


def test_save_and_load_round_trip_produces_identical_identity(tmp_path):
    tok = _train()
    path = tmp_path / "tokenizer.json"
    tok.save(path)

    reloaded = ByteLevelBpeTokenizer.load(path)

    assert reloaded.identity == tok.identity
    assert reloaded.vocab_size == tok.vocab_size
    assert reloaded.special_tokens == tok.special_tokens


def test_save_and_load_round_trip_produces_identical_encoding(tmp_path):
    tok = _train()
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    reloaded = ByteLevelBpeTokenizer.load(path)

    text = "the quick brown fox नमस्ते 😀 jumps"
    assert reloaded.encode(text) == tok.encode(text)
    assert reloaded.decode(tok.encode(text)) == text


def test_load_missing_file_raises_serialization_error(tmp_path):
    with pytest.raises(SerializationError):
        ByteLevelBpeTokenizer.load(tmp_path / "does-not-exist.json")


def test_load_malformed_json_raises_serialization_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json")
    with pytest.raises(SerializationError):
        ByteLevelBpeTokenizer.load(path)


def test_load_wrong_format_version_raises_serialization_error(tmp_path):
    tok = _train()
    path = tmp_path / "tokenizer.json"
    tok.save(path)

    import json

    data = json.loads(path.read_text())
    data["yug_tokenizer_format_version"] = "999"
    path.write_text(json.dumps(data))

    with pytest.raises(SerializationError):
        ByteLevelBpeTokenizer.load(path)


def test_load_tampered_merges_fails_identity_check(tmp_path):
    """A hand-edited file whose merges no longer match its stored
    identity hash must be rejected, not silently loaded."""
    tok = _train()
    path = tmp_path / "tokenizer.json"
    tok.save(path)

    import json

    data = json.loads(path.read_text())
    if data["merges"]:
        data["merges"][0] = [1, 2]  # tamper with a merge rule
    else:
        data["merges"] = [[1, 2]]
    path.write_text(json.dumps(data))

    with pytest.raises(SerializationError):
        ByteLevelBpeTokenizer.load(path)


def test_load_missing_required_field_raises_serialization_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"yug_tokenizer_format_version": "1", "algorithm": "byte_level_bpe"}')
    with pytest.raises(SerializationError):
        ByteLevelBpeTokenizer.load(path)
