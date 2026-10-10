"""Tokenizer Vocabulary Scaling Investigation.

NOT a milestone: no NB-13 (or any other milestone) is assigned, and no
roadmap file is touched. This is a tokenizer-only experiment. It trains
NO Transformer.

QUESTION: how does tokenizer vocabulary size change tokenization
efficiency, correctness, determinism and vocabulary utilization on the
SAME fixed real corpus?

Reused, not duplicated:
  - acquisition: ``real_corpus_experiment.acquire_common_pile_wikimedia_sample``
  - corpus preparation / license gate / document-level split:
    ``training.corpus_prep.prepare_corpus`` with the same seed, val
    fraction and license policy as the first real-corpus experiment
  - tokenizer: the existing ``yug.tokenizer.ByteLevelBpeTokenizer``
    (unmodified); train-partition-only fitting goes through
    ``real_corpus_experiment.train_tokenizer_on_train_partition``
  - word-count definition: ``real_corpus_experiment._count_words``

FIT MODES
---------
``ByteLevelBpeTokenizer.train`` is a deterministic greedy algorithm
with first-seen tie-breaking, so the merge list learned for a target of
N tokens is a prefix of the merge list learned for any larger target on
the same ordered corpus. ``fit_mode="prefix"`` (default) exploits that:
train ONCE at the largest requested size, then assemble each smaller
tokenizer from the first K merges via the public constructor (special
tokens are re-assigned ids after the merges, exactly as ``train`` does).
This is a PROPOSAL-level efficiency shortcut, not an assumption: the
runner verifies it by also training independently (from scratch) for
every size <= ``verify_prefix_up_to`` and comparing tokenizer identity,
recording the outcome per size. ``fit_mode="independent"`` trains every
size from scratch and is the authoritative (slow) mode.

Either way, only TRAINING-partition text is ever passed to tokenizer
training. Validation text is used only for measurement afterwards.

METRIC DEFINITIONS (all documented, none invented silently)
-----------------------------------------------------------
- words: whitespace-split count (``_count_words``).
- tokens_per_word / tokens_per_character: total tokens / total words or
  characters over successfully encoded documents (same as
  ``compute_tokenizer_stats``).
- relative reduction vs baseline vocab B for metric m:
  ``(m_B - m_V) / m_B * 100``. Positive = fewer tokens than baseline.
- utilization universe: the non-special ids ``0 .. 256 + num_merges - 1``
  (256 base bytes + learned merges). Special tokens are reported
  separately because ``encode`` never emits them unless asked.
- observed: count > 0 in the encoded partition. never_observed:
  universe id with count 0.
- rare (threshold T): universe ids with count < T, INCLUDING never
  observed ones; thresholds are listed in the config.
- unigram entropy: Shannon entropy in bits of the empirical token
  distribution over the partition.
- ``observed_in_validation_not_in_train`` counts ids seen when encoding
  validation but with zero count when encoding train.

LIMITATIONS: see ``LIMITATIONS`` below; also reported in the result.
"""

from __future__ import annotations

import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from training.corpus_prep import CorpusPrepResult, RawDocument, prepare_corpus
from training.errors import TrainingError
from training.experiments.real_corpus_experiment import (
    REAL_CORPUS_LICENSE_POLICY,
    SEED,
    VAL_FRACTION,
    _count_words,
    train_tokenizer_on_train_partition,
)

EXPERIMENT_NAME = "tokenizer_vocabulary_scaling_investigation"
DEFAULT_VOCAB_SIZES: tuple[int, ...] = (384, 512, 1024, 2048, 4096, 8192)
BASELINE_VOCAB_SIZE = 384
SPECIAL_TOKENS: tuple[str, ...] = ("bos", "eos")
RARE_THRESHOLDS: tuple[int, ...] = (5, 10, 100)
FIT_MODES = ("prefix", "independent")
DEFAULT_VERIFY_PREFIX_UP_TO = 1024
_NUM_BASE_BYTES = 256
_MAX_LISTED_UNOBSERVED_IDS = 25

# Reference values from the previous controlled real-corpus run (given
# in the task statement). Used only to REPORT agreement/disagreement.
PREVIOUS_RUN_REFERENCE: dict[str, Any] = {
    "accepted_content_hash": "ad304dc739ece9fce7656aa645d26a523a3fbc11a50d2884aa18b04ee3a9b5fe",
    "vocab_size": 384,
    "tokenizer_identity": "sha256:78d6028adc8d05b84434c4a1f15a934217b1304be989236e626ad950c55bfbb3",
    "train_document_count": 800,
    "validation_document_count": 200,
    "train_total_tokens": 2_266_296,
    "validation_total_tokens": 595_617,
}

LIMITATIONS: tuple[str, ...] = (
    "Pure-Python BPE training/encoding cost grows with vocabulary size; wall-clock at 4096/8192 on the full corpus is unmeasured until a real run.",
    "Utilization is measured on a ~4M-character corpus; it says nothing about how utilization would look on a larger corpus.",
    "Tokenization efficiency is NOT model quality: no Transformer is trained here, so no conclusion about loss or generation follows from these numbers.",
    "Word count is a whitespace split, not a linguistic word count.",
    "Wikibooks-only shard sample (per the prior experiment); results may not transfer to other domains.",
    "In prefix fit mode, prefix==independent-training equivalence is verified only for sizes <= verify_prefix_up_to; larger sizes rely on the algorithm's determinism.",
)


class VocabScalingConfigError(TrainingError):
    """Invalid experiment configuration (bad sizes, missing baseline, ...)."""


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _pct_reduction(baseline: float | int | None, value: float | int | None) -> float | None:
    """``(baseline - value) / baseline * 100``; None if undefined."""
    if baseline is None or value is None or baseline == 0:
        return None
    return (baseline - value) / baseline * 100.0


def _nearest_rank(sorted_values: Sequence[int], q: float) -> int:
    """Nearest-rank percentile (deterministic, no interpolation)."""
    n = len(sorted_values)
    rank = max(1, math.ceil(q * n))
    return sorted_values[rank - 1]


def validate_vocab_sizes(vocab_sizes: Sequence[int], baseline: int, num_specials: int = len(SPECIAL_TOKENS)) -> list[int]:
    """Validate and return the sizes sorted ascending.

    Raises :class:`VocabScalingConfigError` for empty/duplicate/non-int
    sizes, sizes below 256 + specials, or a baseline not in the list.
    """
    sizes = list(vocab_sizes)
    if not sizes:
        raise VocabScalingConfigError("vocab_sizes must not be empty")
    for s in sizes:
        if isinstance(s, bool) or not isinstance(s, int):
            raise VocabScalingConfigError(f"vocab size must be int, got {s!r}")
    if len(set(sizes)) != len(sizes):
        raise VocabScalingConfigError(f"vocab_sizes contains duplicates: {sizes}")
    minimum = _NUM_BASE_BYTES + num_specials
    too_small = [s for s in sizes if s < minimum]
    if too_small:
        raise VocabScalingConfigError(f"vocab sizes {too_small} are below the minimum {minimum}")
    if baseline not in sizes:
        raise VocabScalingConfigError(f"baseline vocab size {baseline} must be one of vocab_sizes {sorted(sizes)}")
    return sorted(sizes)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def tokenizer_from_merge_prefix(full_tokenizer, target_vocab_size: int):
    """Assemble the tokenizer that independent training at
    ``target_vocab_size`` would produce, from the first K merges of a
    larger already-trained tokenizer (public constructor + public
    ``to_dict``; the tokenizer class is not modified).

    K = min(available merges, target - 256 - num_specials). Special
    tokens get ids after the merges, as ``ByteLevelBpeTokenizer.train``
    assigns them.
    """
    from yug.tokenizer.bpe import ByteLevelBpeTokenizer

    data = full_tokenizer.to_dict()
    names = sorted(data["special_tokens"], key=lambda n: data["special_tokens"][n])
    k = max(0, min(len(data["merges"]), target_vocab_size - _NUM_BASE_BYTES - len(names)))
    merges = [tuple(m) for m in data["merges"][:k]]
    specials = {name: _NUM_BASE_BYTES + k + i for i, name in enumerate(names)}
    return ByteLevelBpeTokenizer(merges=merges, special_tokens=specials)


def _train_independent(train_texts: list[str], vocab_size: int):
    from yug.tokenizer.bpe import ByteLevelBpeTokenizer

    return ByteLevelBpeTokenizer.train(corpus=train_texts, vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure_partition(tokenizer, texts: list[str]) -> tuple[dict[str, Any], Counter]:
    """Single encode pass over ``texts``: efficiency/correctness stats
    plus the token-id frequency counter.

    Stats keys and definitions match
    ``real_corpus_experiment.compute_tokenizer_stats`` (that function is
    verified equal on fixtures by a test), plus ``round_trip_mismatches``
    (``decode(encode(t)) != t``) and ``ids_out_of_range`` (emitted ids
    not in ``[0, vocab_size)`` or equal to a special-token id).
    """
    from yug.tokenizer.errors import DecodeError

    special_ids = set(tokenizer.special_tokens.values())
    vocab_size = tokenizer.vocab_size
    counts: Counter = Counter()
    chars = words = tokens = enc_fail = dec_fail = mismatches = out_of_range = 0

    for text in texts:
        chars += len(text)
        words += _count_words(text)
        try:
            ids = tokenizer.encode(text)
        except Exception:  # noqa: BLE001 - recorded, not fatal
            enc_fail += 1
            continue
        tokens += len(ids)
        counts.update(ids)
        try:
            decoded = tokenizer.decode(ids)
            if decoded != text:
                mismatches += 1
        except DecodeError:
            dec_fail += 1
    for tid in counts:
        if not (0 <= tid < vocab_size) or tid in special_ids:
            out_of_range += 1

    stats = {
        "document_count": len(texts),
        "total_characters": chars,
        "total_words": words,
        "total_tokens": tokens,
        "tokens_per_word": (tokens / words) if words else None,
        "tokens_per_character": (tokens / chars) if chars else None,
        "encoding_failures": enc_fail,
        "decoding_failures": dec_fail,
        "round_trip_mismatches": mismatches,
        "ids_out_of_range": out_of_range,
    }
    return stats, counts


def compute_utilization(
    counts: Counter,
    *,
    num_merges: int,
    num_specials: int,
    rare_thresholds: Sequence[int] = RARE_THRESHOLDS,
) -> dict[str, Any]:
    """Vocabulary-utilization summary for one partition's id counts.

    Universe = non-special ids ``0 .. 256 + num_merges - 1``. See module
    docstring for every definition.
    """
    universe_size = _NUM_BASE_BYTES + num_merges
    special_lo = universe_size
    special_ids = set(range(special_lo, special_lo + num_specials))

    in_universe = {tid: c for tid, c in counts.items() if 0 <= tid < universe_size and c > 0}
    observed = len(in_universe)
    never = [tid for tid in range(universe_size) if tid not in in_universe]
    total = sum(in_universe.values())

    freqs = sorted(in_universe.values())
    summary: dict[str, Any]
    if freqs:
        probs = [c / total for c in freqs]
        entropy = -sum(p * math.log2(p) for p in probs)
        desc = sorted(freqs, reverse=True)
        summary = {
            "min": freqs[0],
            "max": freqs[-1],
            "mean": total / observed,
            "median": _nearest_rank(freqs, 0.5),
            "p10": _nearest_rank(freqs, 0.10),
            "p90": _nearest_rank(freqs, 0.90),
            "p99": _nearest_rank(freqs, 0.99),
            "top10_token_share": sum(desc[:10]) / total,
            "top100_token_share": sum(desc[:100]) / total,
            "unigram_entropy_bits": entropy,
        }
    else:
        summary = {k: None for k in ("min", "max", "mean", "median", "p10", "p90", "p99", "top10_token_share", "top100_token_share", "unigram_entropy_bits")}

    never_base = sum(1 for t in never if t < _NUM_BASE_BYTES)
    return {
        "universe_size_non_special": universe_size,
        "tokens_observed": observed,
        "percent_observed": (observed / universe_size * 100.0) if universe_size else None,
        "tokens_never_observed": len(never),
        "never_observed_base_bytes": never_base,
        "never_observed_merges": len(never) - never_base,
        "never_observed_first_ids": never[:_MAX_LISTED_UNOBSERVED_IDS],
        "frequency_summary_observed_tokens": summary,
        "rare_token_counts": {
            f"count_lt_{t}": sum(1 for tid in range(universe_size) if in_universe.get(tid, 0) < t)
            for t in rare_thresholds
        },
        "special_tokens_emitted": sum(c for tid, c in counts.items() if tid in special_ids),
    }


def _fit_corpus_fingerprint(prep: CorpusPrepResult) -> str:
    import hashlib

    h = hashlib.sha256()
    for doc in prep.train:
        h.update(doc.content_hash.encode("utf-8"))
    return "sha256:" + h.hexdigest()


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_tokenizer_vocab_scaling(
    documents: list[RawDocument],
    *,
    vocab_sizes: Sequence[int] = DEFAULT_VOCAB_SIZES,
    baseline_vocab_size: int = BASELINE_VOCAB_SIZE,
    fit_mode: str = "prefix",
    verify_prefix_up_to: int = DEFAULT_VERIFY_PREFIX_UP_TO,
    progress: Any = None,
) -> dict[str, Any]:
    """Run the tokenizer-only vocabulary scaling experiment.

    ``documents`` is a plain ``list[RawDocument]`` (from acquisition or a
    fixture). Nothing here trains a Transformer. Per-size failures are
    recorded in ``errors`` and do not abort other sizes; configuration
    errors raise :class:`VocabScalingConfigError` before any work.

    ``progress``: optional ``callable(str)`` for status lines (kept out
    of the deterministic result).
    """
    import time

    if fit_mode not in FIT_MODES:
        raise VocabScalingConfigError(f"fit_mode must be one of {FIT_MODES}, got {fit_mode!r}")
    sizes = validate_vocab_sizes(vocab_sizes, baseline_vocab_size)
    say = progress if callable(progress) else (lambda _m: None)

    prep = prepare_corpus(
        documents, license_policy=REAL_CORPUS_LICENSE_POLICY, val_fraction=VAL_FRACTION, seed=SEED
    )
    manifest = prep.manifest.to_dict()
    train_texts = [d.text for d in prep.train]
    val_texts = [d.text for d in prep.validation]

    train_ids = {d.document_id for d in prep.train}
    val_ids = {d.document_id for d in prep.validation}
    train_hashes = {d.content_hash for d in prep.train}
    val_hashes = {d.content_hash for d in prep.validation}
    leakage = {
        "tokenizer_fit_partition": "train_only",
        "train_document_count": len(prep.train),
        "validation_document_count": len(prep.validation),
        "document_id_overlap_train_validation": len(train_ids & val_ids),
        "content_hash_overlap_train_validation": len(train_hashes & val_hashes),
        "train_fit_corpus_fingerprint": _fit_corpus_fingerprint(prep),
    }

    result: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "not_a_milestone": True,
        "nb13_assigned": False,
        "trains_transformer": False,
        "configuration": {
            "seed": SEED,
            "val_fraction": VAL_FRACTION,
            "vocab_sizes": sizes,
            "baseline_vocab_size": baseline_vocab_size,
            "special_tokens": list(SPECIAL_TOKENS),
            "fit_mode": fit_mode,
            "verify_prefix_up_to": verify_prefix_up_to,
            "rare_thresholds_count_lt": list(RARE_THRESHOLDS),
            "tokenizer": "yug.tokenizer.ByteLevelBpeTokenizer (unmodified, byte-level, no <unk>)",
            "word_count_definition": "len(text.split())",
            "unknown_token_note": "not applicable: byte-level BPE has no <unk> concept",
        },
        "corpus": {
            "requested_document_count": len(documents),
            "accepted_content_hash": manifest.get("accepted_content_hash"),
            "manifest": manifest,
            "previous_run_comparison": {
                "reference": PREVIOUS_RUN_REFERENCE,
                "accepted_content_hash_matches": manifest.get("accepted_content_hash")
                == PREVIOUS_RUN_REFERENCE["accepted_content_hash"],
                "train_document_count_matches": len(prep.train) == PREVIOUS_RUN_REFERENCE["train_document_count"],
                "validation_document_count_matches": len(prep.validation)
                == PREVIOUS_RUN_REFERENCE["validation_document_count"],
            },
        },
        "leakage_checks": leakage,
        "tokenizers": [],
        "comparison_to_baseline": {},
        "errors": [],
        "limitations": list(LIMITATIONS),
        "timings_seconds_non_deterministic": {},
    }
    timings = result["timings_seconds_non_deterministic"]

    # --- fit -------------------------------------------------------------
    fitted: dict[int, Any] = {}
    fit_info: dict[int, dict[str, Any]] = {}
    try:
        if fit_mode == "prefix":
            t0 = time.perf_counter()
            say(f"fitting tokenizer once at max size {sizes[-1]} (train partition only)")
            full = train_tokenizer_on_train_partition(prep, vocab_size=sizes[-1])
            timings["fit_max_size"] = time.perf_counter() - t0
            for s in sizes:
                fitted[s] = tokenizer_from_merge_prefix(full, s)
                fit_info[s] = {"mode": "prefix_of_max_size_fit", "prefix_equals_independent_training": None}
    except Exception as exc:  # noqa: BLE001
        result["errors"].append({"stage": "fit_prefix", "error": f"{type(exc).__name__}: {exc}"})

    for s in sizes:
        try:
            need_independent = fit_mode == "independent" or (
                fit_mode == "prefix" and s in fitted and s <= verify_prefix_up_to
            )
            independent = None
            if need_independent:
                t0 = time.perf_counter()
                say(f"independent training at vocab {s}")
                independent = _train_independent(train_texts, s)
                timings[f"independent_fit_{s}"] = time.perf_counter() - t0
            if fit_mode == "independent":
                fitted[s] = independent
                fit_info[s] = {"mode": "independent", "prefix_equals_independent_training": None}
            elif independent is not None and s in fitted:
                fit_info[s]["prefix_equals_independent_training"] = independent.identity == fitted[s].identity
        except Exception as exc:  # noqa: BLE001
            result["errors"].append({"stage": "fit", "requested_vocab_size": s, "error": f"{type(exc).__name__}: {exc}"})

    # --- determinism: retrain smallest size, compare identity -------------
    smallest = sizes[0]
    if smallest in fitted:
        try:
            again = _train_independent(train_texts, smallest)
            result["determinism_check"] = {
                "requested_vocab_size": smallest,
                "retrain_identity_matches": again.identity == fitted[smallest].identity,
            }
        except Exception as exc:  # noqa: BLE001
            result["errors"].append({"stage": "determinism", "error": f"{type(exc).__name__}: {exc}"})

    # --- measure ---------------------------------------------------------
    for s in sizes:
        tok = fitted.get(s)
        if tok is None:
            continue
        try:
            t0 = time.perf_counter()
            say(f"measuring vocab {s} (actual {tok.vocab_size})")
            train_stats, train_counts = measure_partition(tok, train_texts)
            val_stats, val_counts = measure_partition(tok, val_texts)
            timings[f"measure_{s}"] = time.perf_counter() - t0

            num_specials = len(tok.special_tokens)
            num_merges = tok.vocab_size - _NUM_BASE_BYTES - num_specials
            util_train = compute_utilization(train_counts, num_merges=num_merges, num_specials=num_specials)
            util_val = compute_utilization(val_counts, num_merges=num_merges, num_specials=num_specials)
            util_val["observed_in_validation_not_in_train"] = sum(
                1 for tid, c in val_counts.items() if c > 0 and train_counts.get(tid, 0) == 0
            )

            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "tok.json"
                tok.save(path)
                from yug.tokenizer.bpe import ByteLevelBpeTokenizer

                reloaded_identity = ByteLevelBpeTokenizer.load(path).identity

            result["tokenizers"].append(
                {
                    "requested_vocab_size": s,
                    "actual_vocab_size": tok.vocab_size,
                    "stopped_short_of_request": tok.vocab_size < s,
                    "num_merges": num_merges,
                    "tokenizer_identity": tok.identity,
                    "fit": fit_info.get(s, {}),
                    "train": train_stats,
                    "validation": val_stats,
                    "utilization": {"train": util_train, "validation": util_val},
                    "checks": {
                        "encoding_ok": train_stats["encoding_failures"] + val_stats["encoding_failures"] == 0,
                        "decoding_ok": train_stats["decoding_failures"] + val_stats["decoding_failures"] == 0,
                        "round_trip_ok": train_stats["round_trip_mismatches"] + val_stats["round_trip_mismatches"] == 0,
                        "no_unk_token": "unk" not in tok.special_tokens and "<unk>" not in tok.special_tokens,
                        "no_special_or_out_of_range_ids_emitted": train_stats["ids_out_of_range"]
                        + val_stats["ids_out_of_range"]
                        == 0,
                        "save_load_identity_preserved": reloaded_identity == tok.identity,
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001
            result["errors"].append({"stage": "measure", "requested_vocab_size": s, "error": f"{type(exc).__name__}: {exc}"})

    # --- comparison to baseline -----------------------------------------
    by_req = {t["requested_vocab_size"]: t for t in result["tokenizers"]}
    base = by_req.get(baseline_vocab_size)
    if base is None:
        result["errors"].append({"stage": "comparison", "error": f"baseline vocab {baseline_vocab_size} has no result"})
    else:
        for s, t in sorted(by_req.items()):
            result["comparison_to_baseline"][str(s)] = {
                "baseline_vocab_size": baseline_vocab_size,
                "train_token_count_reduction_pct": _pct_reduction(base["train"]["total_tokens"], t["train"]["total_tokens"]),
                "validation_token_count_reduction_pct": _pct_reduction(
                    base["validation"]["total_tokens"], t["validation"]["total_tokens"]
                ),
                "train_tokens_per_word_reduction_pct": _pct_reduction(
                    base["train"]["tokens_per_word"], t["train"]["tokens_per_word"]
                ),
                "validation_tokens_per_word_reduction_pct": _pct_reduction(
                    base["validation"]["tokens_per_word"], t["validation"]["tokens_per_word"]
                ),
                "train_tokens_per_character_reduction_pct": _pct_reduction(
                    base["train"]["tokens_per_character"], t["train"]["tokens_per_character"]
                ),
                "validation_tokens_per_character_reduction_pct": _pct_reduction(
                    base["validation"]["tokens_per_character"], t["validation"]["tokens_per_character"]
                ),
            }
        if baseline_vocab_size == PREVIOUS_RUN_REFERENCE["vocab_size"]:
            result["corpus"]["previous_run_comparison"].update(
                {
                    "baseline_tokenizer_identity_matches": base["tokenizer_identity"]
                    == PREVIOUS_RUN_REFERENCE["tokenizer_identity"],
                    "baseline_train_tokens_match": base["train"]["total_tokens"]
                    == PREVIOUS_RUN_REFERENCE["train_total_tokens"],
                    "baseline_validation_tokens_match": base["validation"]["total_tokens"]
                    == PREVIOUS_RUN_REFERENCE["validation_total_tokens"],
                }
            )
    return result


def strip_nondeterministic(result: dict[str, Any]) -> dict[str, Any]:
    """Copy of ``result`` without wall-clock timings (for equality checks)."""
    return {k: v for k, v in result.items() if k != "timings_seconds_non_deterministic"}


if __name__ == "__main__":
    import argparse
    import json
    import sys

    from training.errors import RealCorpusAcquisitionError
    from training.experiments.real_corpus_experiment import acquire_common_pile_wikimedia_sample

    parser = argparse.ArgumentParser(description="Tokenizer vocabulary scaling investigation (not a milestone).")
    parser.add_argument("--max-documents", type=int, default=1000)
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=list(DEFAULT_VOCAB_SIZES))
    parser.add_argument("--baseline", type=int, default=BASELINE_VOCAB_SIZE)
    parser.add_argument("--fit-mode", choices=FIT_MODES, default="prefix")
    parser.add_argument("--verify-prefix-up-to", type=int, default=DEFAULT_VERIFY_PREFIX_UP_TO)
    parser.add_argument("--output", default=None, help="write UTF-8 JSON here (avoids PowerShell UTF-16 redirection)")
    args = parser.parse_args()

    def _log(message: str) -> None:
        print(f"[vocab-scaling] {message}", file=sys.stderr, flush=True)

    try:
        docs, acquisition_report = acquire_common_pile_wikimedia_sample(args.max_documents)
    except RealCorpusAcquisitionError as exc:
        print(f"ACQUISITION FAILED: {exc}", file=sys.stderr)
        sys.exit(2)

    try:
        outcome = run_tokenizer_vocab_scaling(
            docs,
            vocab_sizes=args.vocab_sizes,
            baseline_vocab_size=args.baseline,
            fit_mode=args.fit_mode,
            verify_prefix_up_to=args.verify_prefix_up_to,
            progress=_log,
        )
    except VocabScalingConfigError as exc:
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        sys.exit(3)

    acq = acquisition_report.to_dict()
    acq.pop("provenance", None)  # keep the artifact small; per-document provenance is not needed here
    outcome["acquisition"] = acq
    payload = json.dumps(outcome, indent=2, sort_keys=True, default=str)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        print(f"result written to {args.output}")
    else:
        print(payload)
    sys.exit(0 if not outcome["errors"] else 1)