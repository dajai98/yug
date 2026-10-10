"""Core encode/decode correctness tests for ByteLevelBpeTokenizer.

Note on scripts tested: "Kirati" is not a single standardized Unicode
script most environments can reliably render/verify test text for, so
it is not hardcoded here. What IS tested and structurally guaranteed
(see module docstring in yug/tokenizer/bpe.py) is that byte-level BPE
represents ANY valid Unicode string as, at minimum, its raw UTF-8
bytes -- Devanagari (Nepali), CJK, Arabic, emoji, or any other script
included below are representative examples of that general guarantee,
not an exhaustive script list.
"""

from __future__ import annotations

import pytest

from yug.tokenizer import ByteLevelBpeTokenizer

# A small multilingual training corpus used across these tests.
_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumps over the lazy dog again and again",
    "hello world, hello there! how are you today?",
    "1234567890 numbers and punctuation: !@#$%^&*()",
    "नमस्ते संसार",  # Nepali/Devanagari: "hello world"
    "म ठिक छु, तपाईं कस्तो हुनुहुन्छ?",  # Nepali sentence
    "你好，世界",  # Chinese: "hello, world"
    "مرحبا بالعالم",  # Arabic: "hello world"
    "emoji test: 😀🎉🚀❤️🔥",
    "combining characters: e\u0301 a\u0300 n\u0303",  # e + combining acute, etc.
]


def _small_tokenizer() -> ByteLevelBpeTokenizer:
    return ByteLevelBpeTokenizer.train(_CORPUS, vocab_size=320, special_tokens=("bos", "eos"))


# ---- basic ------------------------------------------------------------


def test_empty_string_round_trips():
    tok = _small_tokenizer()
    ids = tok.encode("")
    assert ids == []
    assert tok.decode(ids) == ""


def test_ascii_text_round_trips():
    tok = _small_tokenizer()
    text = "the quick brown fox"
    assert tok.decode(tok.encode(text)) == text


def test_numbers_and_punctuation_round_trip():
    tok = _small_tokenizer()
    text = "1234567890 !@#$%^&*()"
    assert tok.decode(tok.encode(text)) == text


def test_whitespace_only_round_trips():
    tok = _small_tokenizer()
    text = "   \t\n  "
    assert tok.decode(tok.encode(text)) == text


def test_punctuation_only_round_trips():
    tok = _small_tokenizer()
    text = "...,,,!!!???"
    assert tok.decode(tok.encode(text)) == text


# ---- unicode / multilingual --------------------------------------------


def test_nepali_devanagari_round_trips():
    tok = _small_tokenizer()
    text = "नमस्ते संसार"
    assert tok.decode(tok.encode(text)) == text


def test_chinese_round_trips():
    tok = _small_tokenizer()
    text = "你好，世界"
    assert tok.decode(tok.encode(text)) == text


def test_arabic_round_trips():
    tok = _small_tokenizer()
    text = "مرحبا بالعالم"
    assert tok.decode(tok.encode(text)) == text


def test_emoji_round_trips():
    tok = _small_tokenizer()
    text = "😀🎉🚀❤️🔥"
    assert tok.decode(tok.encode(text)) == text


def test_combining_characters_round_trip():
    tok = _small_tokenizer()
    text = "e\u0301 a\u0300 n\u0303"
    assert tok.decode(tok.encode(text)) == text


def test_mixed_scripts_in_one_string_round_trip():
    tok = _small_tokenizer()
    text = "hello नमस्ते 你好 😀 مرحبا"
    assert tok.decode(tok.encode(text)) == text


def test_unicode_never_produces_an_unknown_token_error():
    """Structural guarantee: byte-level coverage means encode() must
    never fail on any valid str, even text with no relation to the
    training corpus at all."""
    tok = _small_tokenizer()
    text = "Ελληνικά ру́сский язы́к ελληνικά日本語한국어"
    ids = tok.encode(text)  # must not raise
    assert tok.decode(ids) == text


# ---- byte robustness / long input --------------------------------------


def test_long_repeated_text_round_trips():
    tok = _small_tokenizer()
    text = "abcabc " * 500
    assert tok.decode(tok.encode(text)) == text


def test_long_unicode_sequence_round_trips():
    tok = _small_tokenizer()
    text = "नमस्ते " * 200
    assert tok.decode(tok.encode(text)) == text


def test_single_character_round_trips():
    tok = _small_tokenizer()
    assert tok.decode(tok.encode("a")) == "a"


def test_single_multibyte_character_round_trips():
    tok = _small_tokenizer()
    assert tok.decode(tok.encode("न")) == "न"


# ---- type errors on bad input -------------------------------------------


def test_encode_rejects_non_string_input():
    tok = _small_tokenizer()
    with pytest.raises(TypeError):
        tok.encode(12345)


def test_encode_rejects_none_input():
    tok = _small_tokenizer()
    with pytest.raises(TypeError):
        tok.encode(None)


def test_decode_rejects_non_sequence_input():
    tok = _small_tokenizer()
    with pytest.raises(TypeError):
        tok.decode("not a list of ints")


def test_decode_rejects_non_int_elements():
    tok = _small_tokenizer()
    with pytest.raises(TypeError):
        tok.decode([1, 2, "three"])
