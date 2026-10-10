"""Original fitter/counting oracle: counts, tie order, merges, IDs and identity."""
from collections import Counter
import random

import pytest

from yug.tokenizer.bpe import ByteLevelBpeTokenizer, _get_adjacent_pairs


def original_counts(sequences):
    counts = Counter()
    for seq in sequences:
        for a, b in zip(seq, seq[1:]):
            counts[(a, b)] += 1
    return counts


def original_fit(texts, vocab, names):
    seqs = [list(text.encode("utf-8")) for text in texts]
    merges = []
    for index in range(vocab - 256 - len(names)):
        counts = original_counts(seqs)
        if not counts:
            break
        pair, count = counts.most_common(1)[0]
        if count < 2:
            break
        output = []
        for seq in seqs:
            new, pos = [], 0
            while pos < len(seq):
                if pos + 1 < len(seq) and (seq[pos], seq[pos + 1]) == pair:
                    new.append(256 + index)
                    pos += 2
                else:
                    new.append(seq[pos])
                    pos += 1
            output.append(new)
        seqs = output
        merges.append(pair)
    return ByteLevelBpeTokenizer(merges, {name: 256 + len(merges) + i for i, name in enumerate(names)})


@pytest.mark.parametrize("seed", range(8))
def test_counts_and_first_encounter_order(seed):
    rng = random.Random(seed)
    seqs = [[rng.randrange(7) for _ in range(rng.randrange(200))] for _ in range(20)]
    seqs += [[], [4], [1, 1, 1, 1], [2, 3, 2, 3]]
    actual, expected = _get_adjacent_pairs(seqs), original_counts(seqs)
    assert list(actual.items()) == list(expected.items())
    assert actual.most_common() == expected.most_common()


@pytest.mark.parametrize("vocab", [258, 270, 320, 4096, 8192])
@pytest.mark.parametrize("texts", [[], [""], ["aaaaaaa", "abac", "ac ab", "abab"],
                                    ["नेपाल English", "code\n  x = 3", "🙂é e\u0301", "नेपाल code"]])
def test_fitter_output_and_serialization_identity(texts, vocab, tmp_path):
    names = ("bos", "eos")
    expected = original_fit(texts, vocab, names)
    actual = ByteLevelBpeTokenizer.train(texts, vocab, names)
    assert actual.to_dict() == expected.to_dict()
    assert actual.identity == expected.identity
    for text in texts + ["unseen\nनेपाली🙂", ""]:
        assert actual.encode(text) == expected.encode(text)
        assert actual.encode(text, add_bos=True, add_eos=True) == expected.encode(text, add_bos=True, add_eos=True)
        assert actual.decode(actual.encode(text)) == text
    path = tmp_path / "tokenizer.json"
    actual.save(path)
    assert ByteLevelBpeTokenizer.load(path).identity == expected.identity


@pytest.mark.parametrize("names", [(), ("eos",), ("eos", "bos", "custom")])
def test_special_order_and_randomized_fitting(names):
    rng = random.Random(11)
    texts = ["".join(rng.choice("abc \nनेपाल") for _ in range(90)) for _ in range(6)]
    expected = original_fit(texts, 320, names)
    actual = ByteLevelBpeTokenizer.train(texts, 320, names)
    assert actual.to_dict() == expected.to_dict()
    assert actual.identity == expected.identity

@pytest.mark.parametrize('seed', range(24))
def test_incremental_fit_matches_independent_oracle(seed, tmp_path):
    rng = random.Random(seed + 91)
    alphabet = 'abcz \n' + '\u0928\u0947\u092a\u093e\u0932\U0001f642\u00e9'
    texts = [''.join(rng.choice(alphabet) for _ in range(rng.randrange(0, 180)))
             for _ in range(rng.randrange(1, 12))]
    texts += ['aaaaaaa', 'ababab', '', 'x', 'abcabcabc', 'bcabcabca']
    names = (('eos', 'bos', 'custom') if seed % 3 == 0 else ('bos', 'eos'))
    target = (270, 320, 512, 8192)[seed % 4]
    expected = original_fit(texts, target, names)
    actual = ByteLevelBpeTokenizer.train(texts, target, names)
    assert actual.to_dict() == expected.to_dict()
    assert actual.special_tokens == expected.special_tokens
    assert actual.vocab_size == expected.vocab_size
    assert actual.identity == expected.identity
    actual.save(tmp_path / 'actual.json')
    expected.save(tmp_path / 'original.json')
    assert (tmp_path / 'actual.json').read_bytes() == (tmp_path / 'original.json').read_bytes()
    for text in texts + ['unseen \u0928\u0947\u092a\u093e\u0932\U0001f642']:
        assert actual.encode(text, add_bos=True, add_eos=True) == expected.encode(text, add_bos=True, add_eos=True)


def test_exhaustive_short_binary_document_ties():
    import itertools
    words = [''.join(x) for n in range(5) for x in itertools.product('ab', repeat=n)]
    for first in words:
        for second in words:
            texts = [first, second, '']
            expected = original_fit(texts, 280, ('bos', 'eos'))
            actual = ByteLevelBpeTokenizer.train(texts, 280)
            assert actual.to_dict() == expected.to_dict(), texts


@pytest.mark.parametrize('texts', [
    ['ab', 'cd', 'cd', 'ab'],
    ['cd', 'ab', 'ab', 'cd'],
    ['ababac', 'bcabac', 'bc', 'ac'],
    ['aaaaaaa', 'aaa', 'aa', 'a'],
    ['', 'a', '', 'bc'],
    ['abac', 'ac ab', 'abab', 'bcabcabca'],
])
def test_tie_reordering_and_overlapping_pairs_at_every_prefix(texts):
    for target in range(258, 281):
        actual = ByteLevelBpeTokenizer.train(texts, target)
        expected = original_fit(texts, target, ('bos', 'eos'))
        assert actual.to_dict() == expected.to_dict(), target


@pytest.mark.parametrize('texts', [[], ['abc']])
def test_zero_merge_target_preserves_integer_validation(texts):
    with pytest.raises(TypeError):
        original_fit(texts, 258.0, ('bos', 'eos'))
    with pytest.raises(TypeError):
        ByteLevelBpeTokenizer.train(texts, 258.0)
