"""NB-08 V3: deterministic, combinatorial, multi-template corpus.

VERIFICATION STATUS: the generation logic itself has zero PyTorch
dependency and IS genuinely tested by real execution in this sandbox
(see tests/test_nb08_v3_corpus.py) -- only the downstream
tokenizer/model/training stages remain unverified pending a real
PyTorch environment.

Diagnosis this corpus responds to (see
ARCHITECTURE/NB08_DATA_REGIME_EXPERIMENT_V3.md, Section 3, for the full
evidence): the V1/V2 corpus (`nb08_corpus.CORPUS_SENTENCES`) turned out
to be MORE syntactically varied than expected (37 distinct
function-word skeletons out of 40 sentences) -- but each specific
pattern occurs essentially once. At the word level, only 19 of 44
unique validation words (43%) ever appeared in training at all,
confirmed by actually running `training.split.split_sentences` against
the real corpus. A model cannot learn a reusable pattern, or even
predict a familiar word, from a single exposure -- this is a data-
regime problem distinct from "the corpus has too little text".

Design response: a SMALL, FIXED vocabulary (12 words per slot),
combined deterministically across 6 distinct grammatical templates, so
every vocabulary word appears MANY times across the corpus (in
different templates, different slot positions, different neighboring
words) -- giving both a train/validation split, and the model itself,
many independent chances to see each word and each template pattern,
rather than the ~1 exposure per fact in V1/V2. Combination order is a
pure, seedless, deterministic index-cycling scheme (no randomness
anywhere in generation itself -- reproducibility does not depend on a
seed at this stage; the seed only enters later, at the sentence-level
train/validation split, exactly as in V1/V2).
"""

from __future__ import annotations

# Fixed, small vocabulary -- deliberately reused across every template
# and many sentences, not expanded for its own sake (Directive
# Section 5: "substantially larger... NOT... make the corpus
# intelligent").
SUBJECTS = [
    "cat", "dog", "bird", "fox", "horse", "rabbit",
    "child", "farmer", "teacher", "baker", "fisherman", "squirrel",
]
VERBS_PAST = [
    "sat", "ran", "walked", "jumped", "flew", "swam",
    "climbed", "slept", "played", "worked", "waited", "rested",
]
OBJECTS = [
    "mat", "ball", "book", "fish", "apple", "box",
    "flower", "stone", "leaf", "bread", "cup", "lamp",
]
LOCATIONS = [
    "park", "forest", "river", "garden", "market", "school",
    "village", "mountain", "beach", "field", "bridge", "farm",
]
ADJECTIVES = [
    "small", "large", "quiet", "happy", "old", "young",
    "fast", "gentle", "bright", "dark", "warm", "calm",
]
PREPOSITIONS = ["on", "under", "near", "beside", "across", "behind"]
NUMBER_WORDS = ["two", "three", "four", "five", "six", "seven"]
TIME_PERIODS = ["morning", "evening", "winter", "spring", "summer", "autumn"]

_N = 12  # number of sentences generated per template below


def _cycle(words: list[str], i: int, offset: int = 0) -> str:
    """Deterministic index-cycling: no randomness, fully reproducible
    without a seed. ``offset`` staggers different slots so the same
    index ``i`` does not always pick correspondingly-same-index words
    across slots (which would understate combinatorial variety)."""
    return words[(i + offset) % len(words)]


def _template_1(i: int) -> str:
    # "the cat sat on the mat."
    return (
        f"the {_cycle(SUBJECTS, i)} {_cycle(VERBS_PAST, i)} "
        f"{_cycle(PREPOSITIONS, i, 1)} the {_cycle(OBJECTS, i, 2)}."
    )


def _template_2(i: int) -> str:
    # "a small dog ran through the park."
    return (
        f"a {_cycle(ADJECTIVES, i)} {_cycle(SUBJECTS, i, 3)} "
        f"{_cycle(VERBS_PAST, i, 2)} through the {_cycle(LOCATIONS, i)}."
    )


def _template_3(i: int) -> str:
    # "the fox found a stone in the garden."
    return (
        f"the {_cycle(SUBJECTS, i, 5)} {_cycle(VERBS_PAST, i, 4)} "
        f"a {_cycle(OBJECTS, i, 1)} in the {_cycle(LOCATIONS, i, 2)}."
    )


def _template_4(i: int) -> str:
    # "three rabbits rested near the river." Naive "+s" pluralization
    # (not "foxes"/"children") -- a deliberate simplification: this
    # corpus tests statistical structure learning, not English
    # grammar correctness (Directive Section 5: "The goal is NOT to
    # make the corpus intelligent"), and a consistent, if imperfect,
    # pluralization rule is still fully deterministic and reproducible.
    return (
        f"{_cycle(NUMBER_WORDS, i)} {_cycle(SUBJECTS, i, 7)}s "
        f"{_cycle(VERBS_PAST, i, 6)} near the {_cycle(LOCATIONS, i, 4)}."
    )


def _template_5(i: int) -> str:
    # "the old horse is calm."
    return f"the {_cycle(ADJECTIVES, i, 3)} {_cycle(SUBJECTS, i, 9)} is {_cycle(ADJECTIVES, i, 8)}."


def _template_6(i: int) -> str:
    # "the bread was warm before the evening."
    return (
        f"the {_cycle(OBJECTS, i, 6)} was {_cycle(ADJECTIVES, i, 5)} "
        f"before the {_cycle(TIME_PERIODS, i)}."
    )


_TEMPLATES = [_template_1, _template_2, _template_3, _template_4, _template_5, _template_6]


def _generate_corpus() -> list[str]:
    sentences = [template(i) for template in _TEMPLATES for i in range(_N)]
    # De-duplicate while preserving deterministic order -- the
    # index-cycling scheme above is designed to avoid collisions, but
    # de-duplication is a correctness safeguard, not the primary
    # variety mechanism (verified empirically in
    # tests/test_nb08_v3_corpus.py -- zero duplicates are expected in
    # practice at this _N and these list lengths, and the test asserts
    # that directly, not merely relying on this de-dup step to hide
    # a collision silently).
    seen = set()
    unique_sentences = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)
    return unique_sentences


CORPUS_SENTENCES: list[str] = _generate_corpus()
