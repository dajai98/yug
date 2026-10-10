"""Tests for training.experiments.tokenizer_vocab_scaling.

Tokenizer-only experiment: no PyTorch, no network. Fixtures are tiny,
original, deterministic synthetic text (NOT real Wikimedia data); they
exist only to prove the experiment CODE is correct.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from training.corpus_prep import RawDocument
from training.experiments import tokenizer_vocab_scaling as tvs
from training.experiments.real_corpus_experiment import WIKIMEDIA_LICENSE, compute_tokenizer_stats
from yug.tokenizer.bpe import ByteLevelBpeTokenizer

SIZES = (270, 300, 340, 400)
BASELINE = 270

_WORDS = (
    "river valley ancient trade routes shaped early cities farmers soil manuscripts library "
    "astronomers catalogued faint stars plates musicians rhythms instruments mountain pass "
    "cultures botanists documented species coastal fishing economy engineers irrigation "
    "channels museum curator pottery geologists riverbed council market square visitors"
).split()


def _docs(count: int = 30) -> list[RawDocument]:
    rng = random.Random(7)
    docs = []
    for i in range(count):
        body = " ".join(rng.choice(_WORDS) for _ in range(40))
        docs.append(
            RawDocument(
                document_id=f"fx-{i:03d}",
                text=f"Entry {i}. {body}. Caf\u00e9 \u0928\u0947\u092a\u093e\u0932 note {i}.",
                source_id="s",
                source_name="Fixture",
                license=WIKIMEDIA_LICENSE,
                language="en",
                primary_domain="ref",
            )
        )
    return docs


@pytest.fixture(scope="module")
def result():
    return tvs.run_tokenizer_vocab_scaling(_docs(), vocab_sizes=SIZES, baseline_vocab_size=BASELINE, verify_prefix_up_to=400)


def _by_size(res):
    return {t["requested_vocab_size"]: t for t in res["tokenizers"]}


# --- configuration --------------------------------------------------------


def test_default_sizes_cover_required_range():
    assert tvs.DEFAULT_VOCAB_SIZES == (384, 512, 1024, 2048, 4096, 8192)
    assert tvs.BASELINE_VOCAB_SIZE == 384


@pytest.mark.parametrize(
    "sizes,baseline",
    [([], 270), ([270, 270], 270), ([100, 270], 270), ([300, 400], 270), ([270.5], 270), ([True, 300], 300)],
)
def test_invalid_configuration_rejected(sizes, baseline):
    with pytest.raises(tvs.VocabScalingConfigError):
        tvs.validate_vocab_sizes(sizes, baseline)


def test_invalid_fit_mode_rejected():
    with pytest.raises(tvs.VocabScalingConfigError):
        tvs.run_tokenizer_vocab_scaling(_docs(), vocab_sizes=SIZES, baseline_vocab_size=BASELINE, fit_mode="bogus")


# --- structure / flags ------------------------------------------------------


def test_result_structure_and_flags(result):
    assert result["experiment"] == "tokenizer_vocabulary_scaling_investigation"
    assert result["not_a_milestone"] is True
    assert result["nb13_assigned"] is False
    assert result["trains_transformer"] is False
    assert result["errors"] == []
    assert [t["requested_vocab_size"] for t in result["tokenizers"]] == sorted(SIZES)
    assert result["corpus"]["accepted_content_hash"]
    assert "best" not in json.dumps(result).lower().replace("best_", "x")  # no "best vocab" labelling
    json.dumps(result)  # JSON-compatible


def test_previous_run_comparison_reports_difference_honestly(result):
    cmp = result["corpus"]["previous_run_comparison"]
    assert cmp["accepted_content_hash_matches"] is False  # fixture != real corpus
    assert cmp["train_document_count_matches"] is False


# --- correctness checks -----------------------------------------------------


def test_all_correctness_checks_pass(result):
    for t in result["tokenizers"]:
        assert all(t["checks"].values()), (t["requested_vocab_size"], t["checks"])
        for part in ("train", "validation"):
            assert t[part]["encoding_failures"] == 0
            assert t[part]["decoding_failures"] == 0
            assert t[part]["round_trip_mismatches"] == 0


def test_partition_sizes_and_identities(result):
    idents = [t["tokenizer_identity"] for t in result["tokenizers"]]
    assert len(set(idents)) == len(idents)
    for t in result["tokenizers"]:
        assert t["actual_vocab_size"] <= t["requested_vocab_size"]
        assert t["train"]["document_count"] == result["leakage_checks"]["train_document_count"]
        assert t["validation"]["document_count"] == result["leakage_checks"]["validation_document_count"]


def test_larger_vocab_never_increases_train_tokens(result):
    toks = [t["train"]["total_tokens"] for t in result["tokenizers"]]
    assert toks == sorted(toks, reverse=True)
    assert toks[0] > toks[-1]


# --- determinism ---------------------------------------------------------------


def test_deterministic_result(result):
    again = tvs.run_tokenizer_vocab_scaling(_docs(), vocab_sizes=SIZES, baseline_vocab_size=BASELINE, verify_prefix_up_to=400)
    assert json.dumps(tvs.strip_nondeterministic(result), sort_keys=True) == json.dumps(
        tvs.strip_nondeterministic(again), sort_keys=True
    )
    assert result["determinism_check"]["retrain_identity_matches"] is True


# --- fit modes -----------------------------------------------------------------


def test_prefix_tokenizers_equal_independent_training(result):
    for t in result["tokenizers"]:
        assert t["fit"]["prefix_equals_independent_training"] is True


def test_independent_mode_matches_prefix_mode(result):
    indep = tvs.run_tokenizer_vocab_scaling(
        _docs(), vocab_sizes=SIZES, baseline_vocab_size=BASELINE, fit_mode="independent"
    )
    assert [t["tokenizer_identity"] for t in indep["tokenizers"]] == [t["tokenizer_identity"] for t in result["tokenizers"]]
    assert [t["train"]["total_tokens"] for t in indep["tokenizers"]] == [t["train"]["total_tokens"] for t in result["tokenizers"]]


def test_stopped_short_is_reported_not_padded():
    res = tvs.run_tokenizer_vocab_scaling(
        _docs(), vocab_sizes=(270, 100_000), baseline_vocab_size=270, verify_prefix_up_to=0
    )
    big = _by_size(res)[100_000]
    assert big["stopped_short_of_request"] is True
    assert big["actual_vocab_size"] < 100_000
    assert big["num_merges"] > 0


# --- leakage ---------------------------------------------------------------------


def test_tokenizer_fit_uses_only_train_partition(monkeypatch):
    seen: list[list[str]] = []
    real_train = ByteLevelBpeTokenizer.train.__func__

    def spy(cls, corpus, vocab_size, special_tokens=("bos", "eos")):
        seen.append(list(corpus))
        return real_train(cls, corpus, vocab_size, special_tokens)

    monkeypatch.setattr(ByteLevelBpeTokenizer, "train", classmethod(spy))
    res = tvs.run_tokenizer_vocab_scaling(_docs(), vocab_sizes=SIZES, baseline_vocab_size=BASELINE, verify_prefix_up_to=400)

    from training.corpus_prep import prepare_corpus
    from training.experiments.real_corpus_experiment import REAL_CORPUS_LICENSE_POLICY, SEED, VAL_FRACTION

    prep = prepare_corpus(_docs(), license_policy=REAL_CORPUS_LICENSE_POLICY, val_fraction=VAL_FRACTION, seed=SEED)
    train_texts = {d.text for d in prep.train}
    val_texts = {d.text for d in prep.validation}
    assert seen and train_texts.isdisjoint(val_texts)
    for corpus in seen:
        assert set(corpus) <= train_texts
        assert set(corpus).isdisjoint(val_texts)
    leak = res["leakage_checks"]
    assert leak["document_id_overlap_train_validation"] == 0
    assert leak["content_hash_overlap_train_validation"] == 0
    assert leak["tokenizer_fit_partition"] == "train_only"


# --- statistics ---------------------------------------------------------------------


def test_stats_match_existing_compute_tokenizer_stats():
    tok = ByteLevelBpeTokenizer.train([d.text for d in _docs()[:20]], 300)
    texts = [d.text for d in _docs()[20:]]
    mine, _counts = tvs.measure_partition(tok, texts)
    ref = compute_tokenizer_stats(tok, texts)
    for key in (
        "document_count", "total_characters", "total_words", "total_tokens",
        "tokens_per_word", "tokens_per_character", "encoding_failures", "decoding_failures",
    ):
        assert mine[key] == ref[key], key


def test_measure_partition_counts_are_exact():
    tok = ByteLevelBpeTokenizer.train(["ab ab ab ab", "ab ab"], 260)
    stats, counts = tvs.measure_partition(tok, ["ab ab", "ab"])
    assert stats["total_characters"] == 7
    assert stats["total_words"] == 3
    assert stats["total_tokens"] == sum(counts.values())
    assert stats["tokens_per_character"] == stats["total_tokens"] / 7


def test_measure_partition_records_encode_failure():
    class Broken:
        vocab_size = 258
        special_tokens = {"bos": 256, "eos": 257}

        def encode(self, text):
            if "bad" in text:
                raise RuntimeError("boom")
            return [ord(c) for c in text]

        def decode(self, ids):
            return "".join(chr(i) for i in ids)

    stats, counts = tvs.measure_partition(Broken(), ["ok", "bad", "fine"])
    assert stats["encoding_failures"] == 1
    assert stats["total_tokens"] == 6
    assert stats["round_trip_mismatches"] == 0
    assert stats["document_count"] == 3


def test_measure_partition_detects_round_trip_mismatch_and_special_ids():
    class Lossy:
        vocab_size = 258
        special_tokens = {"bos": 256, "eos": 257}

        def encode(self, text):
            return [65, 256]

        def decode(self, ids):
            return "A"

    stats, _ = tvs.measure_partition(Lossy(), ["xyz"])
    assert stats["round_trip_mismatches"] == 1
    assert stats["ids_out_of_range"] == 1


# --- baseline comparison ---------------------------------------------------------------


def test_comparison_to_baseline_values(result):
    by = _by_size(result)
    base = by[BASELINE]
    for s, t in by.items():
        c = result["comparison_to_baseline"][str(s)]
        exp = (base["train"]["total_tokens"] - t["train"]["total_tokens"]) / base["train"]["total_tokens"] * 100
        assert c["train_token_count_reduction_pct"] == pytest.approx(exp)
        exp_v = (base["validation"]["total_tokens"] - t["validation"]["total_tokens"]) / base["validation"]["total_tokens"] * 100
        assert c["validation_token_count_reduction_pct"] == pytest.approx(exp_v)
        assert set(c) >= {
            "train_tokens_per_word_reduction_pct", "validation_tokens_per_word_reduction_pct",
            "train_tokens_per_character_reduction_pct", "validation_tokens_per_character_reduction_pct",
        }
    assert result["comparison_to_baseline"][str(BASELINE)]["train_token_count_reduction_pct"] == 0.0


def test_pct_reduction_edge_cases():
    assert tvs._pct_reduction(200, 150) == 25.0
    assert tvs._pct_reduction(100, 120) == -20.0
    assert tvs._pct_reduction(0, 5) is None
    assert tvs._pct_reduction(None, 5) is None


# --- utilization -------------------------------------------------------------------------


def test_utilization_handcrafted():
    # universe: 256 bytes + 2 merges = ids 0..257 ; specials 258, 259
    counts = Counter({0: 1, 1: 4, 2: 10, 256: 100, 258: 3})
    u = tvs.compute_utilization(counts, num_merges=2, num_specials=2, rare_thresholds=(5, 10, 100))
    assert u["universe_size_non_special"] == 258
    assert u["tokens_observed"] == 4  # ids 0,1,2,256 ; id 258 is special
    assert u["percent_observed"] == pytest.approx(4 / 258 * 100)
    assert u["tokens_never_observed"] == 254
    assert u["never_observed_base_bytes"] == 253
    assert u["never_observed_merges"] == 1  # id 257
    assert u["special_tokens_emitted"] == 3
    assert u["rare_token_counts"] == {"count_lt_5": 256, "count_lt_10": 256, "count_lt_100": 257}
    s = u["frequency_summary_observed_tokens"]
    assert (s["min"], s["max"]) == (1, 100)
    assert s["mean"] == pytest.approx(115 / 4)
    assert s["top10_token_share"] == 1.0
    assert u["never_observed_first_ids"][:3] == [3, 4, 5]


def test_utilization_entropy_uniform_two_tokens():
    u = tvs.compute_utilization(Counter({0: 5, 1: 5}), num_merges=0, num_specials=2)
    assert u["frequency_summary_observed_tokens"]["unigram_entropy_bits"] == pytest.approx(1.0)


def test_utilization_empty_counts():
    u = tvs.compute_utilization(Counter(), num_merges=1, num_specials=2)
    assert u["tokens_observed"] == 0
    assert u["frequency_summary_observed_tokens"]["median"] is None
    assert u["tokens_never_observed"] == 257


def test_utilization_in_result(result):
    for t in result["tokenizers"]:
        tr = t["utilization"]["train"]
        assert tr["universe_size_non_special"] == t["actual_vocab_size"] - 2
        assert 0 < tr["tokens_observed"] <= tr["universe_size_non_special"]
        assert tr["special_tokens_emitted"] == 0
        assert "observed_in_validation_not_in_train" in t["utilization"]["validation"]


def test_nearest_rank_percentile():
    assert tvs._nearest_rank([1, 2, 3, 4], 0.5) == 2
    assert tvs._nearest_rank([1, 2, 3, 4], 0.99) == 4


# --- failure handling ------------------------------------------------------------------------


def test_per_size_fit_failure_is_recorded_not_fatal(monkeypatch):
    real = tvs._train_independent

    def flaky(texts, size):
        if size == 340:
            raise RuntimeError("synthetic failure")
        return real(texts, size)

    monkeypatch.setattr(tvs, "_train_independent", flaky)
    res = tvs.run_tokenizer_vocab_scaling(
        _docs(), vocab_sizes=SIZES, baseline_vocab_size=BASELINE, fit_mode="independent"
    )
    assert [e["requested_vocab_size"] for e in res["errors"]] == [340]
    assert set(_by_size(res)) == {270, 300, 400}


def test_missing_baseline_result_is_reported(monkeypatch):
    real = tvs._train_independent

    def flaky(texts, size):
        if size == BASELINE:
            raise RuntimeError("no baseline")
        return real(texts, size)

    monkeypatch.setattr(tvs, "_train_independent", flaky)
    res = tvs.run_tokenizer_vocab_scaling(
        _docs(), vocab_sizes=SIZES, baseline_vocab_size=BASELINE, fit_mode="independent"
    )
    assert any(e["stage"] == "comparison" for e in res["errors"])
    assert res["comparison_to_baseline"] == {}


# --- architecture ------------------------------------------------------------------------------


def test_yug_package_does_not_import_training():
    root = Path(__file__).resolve().parent.parent / "yug"
    offenders = []
    for py in root.rglob("*.py"):
        for line in py.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(("import training", "from training")):
                offenders.append(f"{py}: {stripped}")
    assert offenders == []