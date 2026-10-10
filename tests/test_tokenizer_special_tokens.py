from __future__ import annotations

import pytest

from yug.tokenizer import ByteLevelBpeTokenizer, DecodeError, TokenizerError, VocabularyError

_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "hello world, hello there",
]


def _train(special_tokens=("bos", "eos"), vocab_size=280):
    return ByteLevelBpeTokenizer.train(_CORPUS, vocab_size=vocab_size, special_tokens=special_tokens)


# ---- special tokens -------------------------------------------------------


def test_special_tokens_have_stable_distinct_ids():
    tok = _train()
    assert tok.special_tokens["bos"] != tok.special_tokens["eos"]
    assert all(isinstance(v, int) for v in tok.special_tokens.values())


def test_add_bos_and_eos_wrap_encoded_sequence():
    tok = _train()
    text = "hello world"
    plain = tok.encode(text)
    wrapped = tok.encode(text, add_bos=True, add_eos=True)

    assert wrapped[0] == tok.special_tokens["bos"]
    assert wrapped[-1] == tok.special_tokens["eos"]
    assert wrapped[1:-1] == plain


def test_decode_skips_special_tokens_by_default():
    tok = _train()
    text = "hello world"
    wrapped = tok.encode(text, add_bos=True, add_eos=True)
    assert tok.decode(wrapped) == text


def test_decode_raises_on_special_token_when_not_skipping():
    tok = _train()
    wrapped = tok.encode("hello", add_bos=True)
    with pytest.raises(DecodeError):
        tok.decode(wrapped, skip_special_tokens=False)


def test_no_unk_token_exists():
    """Architectural decision: byte-level coverage makes <unk>
    structurally unnecessary, so it is never defined."""
    tok = _train()
    assert "unk" not in tok.special_tokens


def test_custom_special_token_set_is_honored():
    tok = ByteLevelBpeTokenizer.train(_CORPUS, vocab_size=280, special_tokens=("start", "stop", "sep"))
    assert set(tok.special_tokens) == {"start", "stop", "sep"}


def test_tokenizer_with_no_special_tokens_is_valid():
    tok = ByteLevelBpeTokenizer.train(_CORPUS, vocab_size=270, special_tokens=())
    text = "hello world"
    assert tok.decode(tok.encode(text)) == text
    assert tok.special_tokens == {}


def test_requesting_bos_on_tokenizer_without_it_raises():
    tok = ByteLevelBpeTokenizer.train(_CORPUS, vocab_size=270, special_tokens=())
    with pytest.raises(TokenizerError):
        tok.encode("hello", add_bos=True)


# ---- vocabulary / training errors -----------------------------------------


def test_vocab_size_too_small_for_base_bytes_raises():
    with pytest.raises(VocabularyError):
        ByteLevelBpeTokenizer.train(_CORPUS, vocab_size=100, special_tokens=())


def test_vocab_size_too_small_for_special_tokens_raises():
    # 256 base bytes + 2 specials = 258 minimum; 257 is one short.
    with pytest.raises(VocabularyError):
        ByteLevelBpeTokenizer.train(_CORPUS, vocab_size=257, special_tokens=("bos", "eos"))


def test_exact_minimum_vocab_size_succeeds_with_zero_merges():
    tok = ByteLevelBpeTokenizer.train(_CORPUS, vocab_size=258, special_tokens=("bos", "eos"))
    assert tok.vocab_size == 258


def test_training_on_empty_corpus_yields_base_vocab_only():
    tok = ByteLevelBpeTokenizer.train([], vocab_size=300, special_tokens=("bos", "eos"))
    assert tok.vocab_size == 258  # 256 base + 2 specials; no merges possible
    assert tok.decode(tok.encode("abc")) == "abc"  # base bytes still work


def test_actual_vocab_size_may_be_smaller_than_requested():
    """Training must not silently pad to the requested vocab_size."""
    tiny_corpus = ["ab"]  # far too little repetition to reach a large vocab
    tok = ByteLevelBpeTokenizer.train(tiny_corpus, vocab_size=1000, special_tokens=("bos", "eos"))
    assert tok.vocab_size < 1000


# ---- decode error paths ----------------------------------------------------


def test_decode_out_of_vocabulary_id_raises():
    tok = _train()
    with pytest.raises(DecodeError):
        tok.decode([tok.vocab_size + 1000])


def test_decode_negative_id_raises():
    tok = _train()
    with pytest.raises(DecodeError):
        tok.decode([-1])


def test_decode_empty_list_returns_empty_string():
    tok = _train()
    assert tok.decode([]) == ""


def test_decode_partial_multibyte_sequence_raises_decode_error():
    """Slicing a multi-byte character's token(s) in half must be a
    clear, typed error -- not silent corruption or a raw stdlib
    traceback."""
    tok = _train()
    ids = tok.encode("न")  # multi-byte Devanagari character
    if len(ids) > 1:
        with pytest.raises(DecodeError):
            tok.decode(ids[:1])
    # If the tokenizer happened to merge this into a single token id,
    # there is nothing to slice -- the guarantee only applies when a
    # character spans more than one id, so we don't assert failure
    # unconditionally.
