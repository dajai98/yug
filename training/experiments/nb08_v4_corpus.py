"""NB-08 V4: a deterministic, multi-scale corpus generator.

VERIFICATION STATUS: the generation logic itself has zero PyTorch
dependency and IS genuinely tested by real execution in this sandbox
(tests/test_nb08_v4_corpus.py) -- only the downstream tokenizer-reuse/
model/training stages remain unverified pending a real PyTorch
environment.

NECESSARY CHANGE FROM V3'S GENERATOR, DECLARED EXPLICITLY (per this
milestone's Section 8 "if something must change, STOP and document"):
V3's ``nb08_v3_corpus.py`` cycles a single shared index ``i`` through
every vocabulary slot. Every slot list has exactly 12 words, and index
-cycling with a fixed offset is periodic with that same period -- so
V3's scheme can NEVER produce more than 12 unique sentences per
template, no matter how large a range it's called over. Reusing V3's
generator unchanged is therefore mathematically incapable of
satisfying V4's actual requirement (a genuine scale ladder past V3's
size). The vocabulary lists and template SENTENCE STRUCTURES are
reused unchanged from V3 (imported, not duplicated); only the
combinatorial enumeration scheme is generalized, from one shared index
to two independent indices ``(i, j)`` per template, driving the two
semantically primary slots directly and the remaining slots via a
simple deterministic mixing function of both indices -- large enough
to enumerate up to 12x12=144 unique combinations per template (x 6
templates = up to 864 total, before deduplication) while remaining
exactly as reproducible (no randomness anywhere) as V3's scheme.

This change affects ONLY the corpus generator's internal combinatorial
scheme -- it does not touch the independent variable under test (data
scale), nor any of the locked experimental controls (model,
tokenizer, optimizer, evaluation methodology, seed).
"""

from __future__ import annotations

from training.experiments.nb08_v3_corpus import (
    ADJECTIVES,
    LOCATIONS,
    NUMBER_WORDS,
    OBJECTS,
    PREPOSITIONS,
    SUBJECTS,
    TIME_PERIODS,
    VERBS_PAST,
)


def _mix(i: int, j: int, a: int, b: int, words: list[str]) -> str:
    """Deterministic, well-mixed index for a slot that isn't one of a
    template's two primary driving indices -- avoids the two primary
    slots' values trivially determining every other slot too."""
    return words[(i * a + j * b) % len(words)]


def _template_1(i: int, j: int) -> str:
    # "the {subject} {verb} {prep} the {object}." -- primary: subject, object
    subject = SUBJECTS[i % len(SUBJECTS)]
    obj = OBJECTS[j % len(OBJECTS)]
    verb = _mix(i, j, 5, 3, VERBS_PAST)
    prep = _mix(i, j, 2, 7, PREPOSITIONS)
    return f"the {subject} {verb} {prep} the {obj}."


def _template_2(i: int, j: int) -> str:
    # "a {adjective} {subject} {verb} through the {location}." -- primary: subject, location
    subject = SUBJECTS[i % len(SUBJECTS)]
    location = LOCATIONS[j % len(LOCATIONS)]
    adjective = _mix(i, j, 4, 6, ADJECTIVES)
    verb = _mix(i, j, 3, 5, VERBS_PAST)
    return f"a {adjective} {subject} {verb} through the {location}."


def _template_3(i: int, j: int) -> str:
    # "the {subject} {verb} a {object} in the {location}." -- primary: object, location
    obj = OBJECTS[i % len(OBJECTS)]
    location = LOCATIONS[j % len(LOCATIONS)]
    subject = _mix(i, j, 6, 2, SUBJECTS)
    verb = _mix(i, j, 4, 3, VERBS_PAST)
    return f"the {subject} {verb} a {obj} in the {location}."


def _template_4(i: int, j: int) -> str:
    # "{number} {subject}s {verb} near the {location}." -- primary: subject, location
    # Naive "+s" pluralization, same deliberate simplification as V3.
    subject = SUBJECTS[i % len(SUBJECTS)]
    location = LOCATIONS[j % len(LOCATIONS)]
    number = _mix(i, j, 5, 4, NUMBER_WORDS)
    verb = _mix(i, j, 2, 6, VERBS_PAST)
    return f"{number} {subject}s {verb} near the {location}."


def _template_5(i: int, j: int) -> str:
    # "the {adjective1} {subject} is {adjective2}." -- primary: subject, adjective2
    subject = SUBJECTS[i % len(SUBJECTS)]
    adjective2 = ADJECTIVES[j % len(ADJECTIVES)]
    adjective1 = _mix(i, j, 7, 5, ADJECTIVES)
    return f"the {adjective1} {subject} is {adjective2}."


def _template_6(i: int, j: int) -> str:
    # "the {object} was {adjective} before the {time_period}." -- primary: object, time_period
    obj = OBJECTS[i % len(OBJECTS)]
    time_period = TIME_PERIODS[j % len(TIME_PERIODS)]
    adjective = _mix(i, j, 3, 7, ADJECTIVES)
    return f"the {obj} was {adjective} before the {time_period}."


_TEMPLATES = [_template_1, _template_2, _template_3, _template_4, _template_5, _template_6]
_MAX_PAIRS_PER_TEMPLATE = 12 * 12  # both driving indices range over 12 words


def generate_corpus(pairs_per_template: int) -> list[str]:
    """Generate a deterministic corpus using the first
    ``pairs_per_template`` (i, j) index pairs, in a fixed nested-loop
    order (``i`` outer, ``j`` inner, both 0-based), for each of the 6
    templates.

    Raises:
        ValueError: ``pairs_per_template`` exceeds the maximum
            possible distinct index pairs (144).
    """
    if not (1 <= pairs_per_template <= _MAX_PAIRS_PER_TEMPLATE):
        raise ValueError(
            f"pairs_per_template must be in [1, {_MAX_PAIRS_PER_TEMPLATE}], "
            f"got {pairs_per_template}"
        )

    pairs = [(i, j) for i in range(12) for j in range(12)][:pairs_per_template]

    sentences = [template(i, j) for template in _TEMPLATES for i, j in pairs]

    seen = set()
    unique_sentences = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)
    return unique_sentences


# The scale ladder (Directive Section 14): Scale 0 is V3's own corpus,
# reused verbatim as the baseline (not regenerated via this module's
# scheme -- it uses V3's original single-index generator, imported
# directly, so Scale 0 is bit-for-bit identical to the already-run V3
# experiment). Scales 1-3 use this module's 2-index scheme at
# increasing pairs_per_template, chosen to give a roughly geometric
# progression while staying within the 144-pair-per-template ceiling:
# 24 (~2x V3's 72 sentences), 60 (~5x), 144 (full combinatorial space,
# ~12x).
SCALE_PAIRS_PER_TEMPLATE = {
    "scale_1": 24,
    "scale_2": 60,
    "scale_3": 144,
}
