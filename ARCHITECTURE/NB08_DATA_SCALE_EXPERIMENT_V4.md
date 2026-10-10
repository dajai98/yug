# NB-08 V4 — Controlled Data-Scale Experiment

**Status: BLOCKED (execution). Corpus generation, tokenizer-lock,
leakage checks, word coverage, and baselines for all 4 scales are
[VERIFIED BY TEST] (genuinely executed, no PyTorch needed). Model
training/validation/checkpoint/generation stages are [UNKNOWN] —
written, not executed (no PyTorch in this sandbox).**

## 1. Research Question

[PROPOSAL] Does increasing deterministic training-data scale, holding
model architecture, tokenizer, optimizer, and evaluation methodology
fixed, materially improve generalization beyond what V3 achieved?

## 2. Previous Evidence

```
V1: 221,760 params, final_val_loss=15.9820, unigram_baseline=4.7020 -> FAIL — GENERALIZATION
V2: 55,536 params,  final_val_loss=13.4730, unigram_baseline=4.7020, best_val=8.9852@step50 -> FAIL — GENERALIZATION
V3: 55,536 params, larger/coverage-fixed corpus, final_val_loss=8.0932,
    best_val_loss=5.2548@step100, unigram_baseline=4.5188 -> PARTIAL — DATA HELPS, GENERALIZATION INSUFFICIENT
```

## 3. Repository Diagnosis

**[VERIFIED IN REPOSITORY]** — fresh inspection this turn confirmed
`training/experiments/` contains: `nb08_corpus.py` (V1/V2, 40
sentences), `nb08_v3_corpus.py` (V3, 72 sentences, single-index
template cycling), `nb08_baseline.py` (torch-free baselines, reused
unchanged), `nb08_experiment.py`/`_v2.py`/`_v3.py` (prior experiment
runners). `ByteLevelBpeTokenizer` (`yug/tokenizer/bpe.py`) has zero
PyTorch dependency, confirmed by inspecting its imports — this is what
makes the tokenizer-lock-by-reuse strategy (Section 11) possible.

**Critical finding**: V3's corpus generator (`nb08_v3_corpus.py`)
cycles a single shared index through 12-word vocabulary lists. This
scheme is mathematically periodic with period 12 — it is **incapable**
of producing more than 12 unique sentences per template, regardless of
range. V3's generator could not itself be reused to build a larger
scale; see Section 7.

## 4. External Research

**[WEB-RESEARCH]**

- The Chinchilla scaling law work (Hoffmann et al., cited via multiple
  secondary sources found this turn) establishes ~20 tokens per
  parameter as the compute-optimal ratio for large language models.
  V3's 682 train tokens against 55,536 parameters is **≈0.0123
  tokens/parameter — roughly 1,600× below** that reference point. Even
  V4's largest scale (Section 9) reaches only ≈6.5 tokens/parameter,
  still ~3× below Chinchilla-optimal. This is offered as an orienting
  reference, not a claim that this tiny synthetic experiment should be
  expected to reach it — it exists to calibrate how large a "large"
  change actually is here.
- Multiple independently retrieved sources (a diffusion-language-model
  scaling paper; "MiniGPT: Rebuilding GPT from First Principles";
  general ML-education sources) confirm the training-loss-down/
  validation-loss-up pattern V2 exhibited is the standard, well-known
  overfitting signature, and that selecting the **best validation
  checkpoint**, not the final one, is standard practice — directly
  motivating this experiment's `best_val_loss`/`best_val_step` metrics
  (already present in V3, carried forward here).
- A retrieved unigram-perplexity methodology description
  (metricgate.com) matches this project's own `unigram_baseline_loss`
  implementation essentially exactly: build frequency counts from
  training data only, apply add-one (Laplace) smoothing, never peek at
  held-out data — confirming the existing baseline implementation
  follows standard, established methodology.

## 5. Hypothesis

[PROPOSAL] If data scale is a major limiting factor, validation loss
(particularly `best_val_loss`) should improve monotonically or
near-monotonically as scale increases, potentially approaching or
beating the (per-scale-recalculated) unigram baseline at the largest
scale.

## 6. Experimental Controls

**[VERIFIED IN REPOSITORY]**, all locked and reused unmodified across
every scale:
- Model: `hidden_size=48, num_layers=2, num_heads=4, ffn_size=96` → 55,536 parameters (`test_v4_uses_identical_architecture_to_v2_v3`).
- Tokenizer: one `ByteLevelBpeTokenizer` instance, trained once on V3's training split, **reused unchanged** (never retrained) for every scale — verified this turn to reproduce V3's real-run identity hash exactly (`test_locked_tokenizer_matches_v3_real_run_identity`).
- Training config: seed 1234, AdamW lr=3e-3, batch_size=4, sequence_length=16, **max_steps=500 for every scale**.
- Evaluation methodology: `Trainer.evaluate()` for every reported loss.
- Curve checkpoints: `[0, 50, 100, 200, 300, 500]`, identical across scales.

## 7. Independent Variable

**Training data scale only.** Necessary implementation change, declared
per this milestone's own Section 8: V3's single-index corpus generator
cannot produce more than 12 sentences/template (Section 3). A new
module, `nb08_v4_corpus.py`, generalizes the exact same vocabulary
lists and sentence templates (imported from `nb08_v3_corpus.py`, not
duplicated) to a 2-index combinatorial scheme, enumerating up to
12×12=144 unique combinations per template. This change affects only
the corpus generator's internal enumeration scheme — it does not touch
the independent variable's meaning, nor any locked control.

## 8. Dataset Generation Method

**[VERIFIED BY TEST]** — `generate_corpus(pairs_per_template)` takes
the first N `(i, j)` index pairs (deterministic nested-loop order, no
randomness) for each of the same 6 grammatical templates as V3. Two
semantically primary slots are driven directly by `i`/`j`; remaining
slots use a deterministic linear mix of both indices. Verified this
turn: generation is deterministic (`test_generation_is_deterministic`),
larger scales are strict supersets of smaller ones at the sentence
level (`test_larger_scale_is_superset_of_smaller_scale_sentences`), and
zero duplicate-collision losses occur at any configured scale.

## 9. Scale Ladder

**[VERIFIED BY TEST]**, all four scales, real execution this turn:

| Scale | Sentences (train/val) | Train tokens | Val tokens | Word coverage | Sentence leakage | params/token |
|---|---:|---:|---:|---:|---:|---:|
| scale_0_v3_baseline | 58 / 14 | 682 | 207 | 94.2% | 0 | 81.43 |
| scale_1 | 116 / 28 | 1395 | 338 | 100.0% | 0 | 39.81 |
| scale_2 | 288 / 72 | 3419 | 898 | 100.0% | 0 | 16.24 |
| scale_3 | 692 / 172 | 8613 | 2149 | 100.0% | 0 | 6.45 |

Zero exact train/validation sentence collisions at every scale
(`test_no_exact_sentence_leakage_at_any_scale`, parametrized over all
three new scales). Word coverage reaches 100% at scales 1–3 (up from
V3's already-improved 94.2%).

## 10. Leakage Analysis

**[VERIFIED BY TEST]** — exact-sentence intersection is 0 at every
scale (Section 9 table, last column). Structural relationship: every
validation sentence at every scale uses a template also present many
times in training (by construction — no template is scale- or
split-exclusive), and (per Section 9) every validation word appears in
training at scales 1–3, and 94.2% do at scale 0.

## 11. Tokenizer Identity

**[VERIFIED BY TEST]**: `sha256:076c103ee402a41947e7911ee9f2908b00ac0af57b09e7c5c44180a4abf741e7`
— identical across all 4 scales (built once, reused, never retrained),
and confirmed to exactly match V3's own real-run identity.
`vocab_size=384`, unchanged.

## 12. Model Architecture

`hidden_size=48, num_layers=2, num_heads=4, ffn_size=96,
context_length=32` → **55,536 parameters**, identical to V2/V3,
enforced by a regression test (`test_v4_uses_identical_architecture_to_v2_v3`).

## 13. Training Budget

**[DECISION]**: Option A — fixed step count (500) for every scale, not
a fixed token-budget. Per Section 4's research, this is the standard
way to isolate a pure data-scaling effect: it holds compute constant
while data varies, meaning a larger dataset is seen for fewer
repetitions per unique example under the same optimizer-step budget —
exactly the comparison this experiment needs. A token-matched design
(Option B) would have required deciding, in advance, how many
epochs-equivalent each scale gets, which risks smuggling in an
unprincipled choice; fixed steps avoids that.

## 14. Baselines

**[VERIFIED BY TEST]**, recalculated per scale (never reused from V3
blindly, per this milestone's own explicit requirement):

```
scale_0_v3_baseline: uniform=5.950643  unigram=4.518758
scale_1:              uniform=5.950643  unigram=4.436531
scale_2:              uniform=5.950643  unigram=4.468572
scale_3:              uniform=5.950643  unigram=4.494875
```

Uniform baseline is identical across scales (same `vocab_size=384`,
locked tokenizer). Unigram baseline fluctuates narrowly (4.44–4.52)
across scales — expected, since it depends on each scale's specific
token-frequency distribution.

## 15. Metrics

Per-scale metric collection implemented in `run_single_scale()`:
`parameter_count`, sentence/token/word-vocabulary counts, `word_coverage`,
`train_val_sentence_intersection`, `initial_train_loss`,
`final_train_loss_full_eval`, `best_val_loss`, `best_val_step`,
`final_val_loss`, both baselines, `best_model_minus_unigram`,
`final_model_minus_unigram`, full `training_curve`, checkpoint path,
and `generated_text` (via the existing `safe_decode_generation` guard).

## 16. Actual Results

**[UNKNOWN]** — requires PyTorch execution. Not fabricated.

## 17. Training Curves

**[UNKNOWN]** — requires PyTorch execution, for all 4 scales.

## 18. Generalization Analysis

**[UNKNOWN]** — pending execution. To be filled in using the
Result A/B/C/D interpretation framework once real numbers exist.

## 19. Limitations

- Even scale_3 (6.45 params/token) remains ~3× below Chinchilla-optimal
  (~20 tokens/param, i.e. ~0.05 params/token) — this experiment cannot
  by itself rule out "data scale matters, just not yet enough scale
  tested," only "within this range, scale did/didn't help."
- The corpus remains fully synthetic and template-based; findings here
  say nothing about natural-language corpora at any scale.
- `max_steps=500` fixed across scales means scale_3's ~692 training
  sentences are seen for far fewer effective passes than scale_0's 58
  — an intentional, declared consequence of the Option A choice
  (Section 13), not an oversight.

## 20. Decision

**BLOCKED**

Every torch-free stage — corpus generation and its determinism, the
tokenizer lock (verified to exactly reproduce V3's real hash), leakage
checks, word coverage, and all four scales' baselines — is genuinely
executed and verified. The actual training runs, for all 4 scales,
have never executed.

**Exact commands for the real environment:**

```bash
python -m pytest tests/test_nb08_v4_corpus.py -q          # already known-good here
python -m pytest tests/test_nb08_experiment_v4.py -q
python -m training.experiments.nb08_experiment_v4          # runs ALL 4 scales
python -m pytest tests -q
python -m compileall -q yug training tests
```

## 21. One Next Recommendation

Only after real results exist: if `best_val_loss` improves
monotonically with scale and approaches/beats the per-scale unigram
baseline by scale_3, the next experiment should hold data scale fixed
at whichever scale first crosses that bar and instead vary training
steps/epochs (was 500 steps enough, or would more help further at that
scale?) — not jump to yet another data-scale increase, since the
current 4-point ladder already spans a meaningful, deliberately-chosen
range (81.4 → 6.5 params/token). If instead validation loss does not
improve meaningfully with scale, the next experiment should return to
regularization (dropout/weight_decay), the one lever explicitly held
constant across V1–V4 so far. This recommendation is not implemented
here, per this milestone's hard stop.
