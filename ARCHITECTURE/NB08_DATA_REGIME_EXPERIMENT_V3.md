# NB-08 V3 — Controlled Data-Regime Experiment

**Status: BLOCKED (execution). Corpus generation, diagnosis, and split
are [VERIFIED BY TEST] (genuinely executed, no PyTorch needed). Model
training/validation/checkpoint/generation stages are [UNKNOWN] —
written, not executed (no PyTorch in this sandbox).**

## 1. Objective

Test whether V1/V2's failure to generalize was primarily caused by an
inadequate data regime — specifically, insufficient word/pattern
repetition for the model to learn reusable structure from — rather
than architecture or training configuration (both already
investigated and adjusted in V2 without success).

## 2. Previous Evidence

```
V1: parameters=221760, final_val_loss=15.9820, unigram_baseline=4.7020
    -> FAIL — GENERALIZATION

V2: parameters=55536, final_val_loss=13.4730, unigram_baseline=4.7020
    training curve: val loss improves to 8.9852 at step 50, then
    degrades monotonically through step 500 (13.4730) — classic
    overfitting, and even the BEST checkpoint (8.9852) never beat the
    uniform baseline (5.9506), let alone the unigram baseline (4.7020).
    -> FAIL — GENERALIZATION
```

## 3. Diagnosis

**[VERIFIED BY TEST]** (genuinely executed against the real V1/V2
corpus this turn): a structural analysis using function-word skeleton
extraction found the V1/V2 corpus is **more** syntactically varied
than initially assumed — 37 distinct skeletons out of 40 sentences.
The real problem is sharper and different: **word-level coverage**.
Running the actual `split_sentences(seed=1234)` against the real
corpus: validation contains 44 unique words, only **19 (43%)** ever
appear anywhere in the 32 training sentences. The other 25 words
(`fall`, `noon`, `soft`, `flowed`, `barked`, `sandcastle`, etc.) are
completely unseen by the model at training time. No model can predict
a word it has never encountered — this alone is sufficient to explain
much of the observed validation loss inflation, independent of
architecture or overfitting dynamics.

**Conclusion**: the corpus's problem is not repetitiveness (it isn't
repetitive) — it's that each of its 40 sentences contributes mostly
novel vocabulary, giving the model too few repeated exposures to any
given word or pattern to learn generalizable structure.

## 4. Controlled Variables (unchanged from V2)

- Model architecture: `hidden_size=48, num_layers=2, num_heads=4, ffn_size=96` → 55,536 parameters (verified identical via `test_v3_uses_identical_architecture_to_v2`).
- Training configuration: `build_training_config()` reused unmodified — seed 1234, learning rate 3e-3, batch size 4, sequence length 16, max_steps 500, eval_interval 50, checkpoint_interval 500, grad_clip_norm 1.0, device cpu.
- Evaluation methodology: `Trainer.evaluate()` for all three headline losses (initial/final-train/val), matching the V1 fix — never the raw last-step training-mode loss.
- Baselines: same `uniform_baseline_loss`/`unigram_baseline_loss` functions, unmodified.
- `training/loop.py`, `training/checkpoint.py`: untouched.

## 5. Changed Variable

**The corpus only.** `training/experiments/nb08_v3_corpus.py` replaces
`nb08_corpus.py`'s 40 hand-written sentences with 72 sentences
generated deterministically (no randomness anywhere in generation —
pure index-cycling) from 6 grammatical templates × a small, fixed,
heavily-reused 12-word-per-slot vocabulary. See that module's
docstring for the full design rationale, directly responding to
Section 3's diagnosis.

**Necessary side effect, explicitly declared** (per this milestone's
own Section 6 requirement): the tokenizer must be retrained on the new
corpus, producing a **different identity hash** than V1/V2's. The
tokenizer **algorithm and hyperparameter** (`ByteLevelBpeTokenizer`,
`vocab_size=384`) are unchanged — only the resulting hash, an
unavoidable consequence of training on different text, differs. This
is not an independent, hidden variable; it is a direct, documented
consequence of the one variable actually being changed (the corpus).

## 6. Corpus Specification

**[VERIFIED BY TEST]**, all numbers from actually running the real
code this turn:

```
total sentences: 72 (0 duplicates)
train sentences: 58   val sentences: 14   (split_sentences, seed=1234, val_fraction=0.2)

word-level coverage: 49/52 (94.2%) of validation words appear in training
  (missing: squirrels, teachers, farmers -- plural forms of words that
   DO appear, singular, in training -- a 3x improvement over V1's 43%)

train tokens: 682   val tokens: 207        (vocab_size=384, same hyperparameter as V1/V2)
tokenizer_identity: sha256:076c103ee402a41947e7911ee9f2908b00ac0af57b09e7c5c44180a4abf741e7
  (necessarily different from V1/V2's sha256:ac40c48e... -- see Section 5)

train sequences (sequence_length=16, chunk_len=17): 40   (up from V1's 27)
val sequences: 12                                          (up from V1's 8)

token-level coverage (for continuity with V1's original metric): 80/82 types (97.6%), 99.0% of occurrences
```

## 7. Model

Identical to V2, verified by `test_v3_uses_identical_architecture_to_v2`
(constructs the exact config and asserts `count_parameters() == 55536`):

```
hidden_size: 48   num_layers: 2   num_heads: 4   ffn_size: 96
context_length: 32   vocab_size: 384 (from V3's own retrained tokenizer)
parameters: 55,536 (same as V2 -- architecture unchanged)
```

## 8. Training

```
seed: 1234 (unchanged)
optimizer: AdamW, lr=3e-3, weight_decay=0.0 (unchanged)
max_steps: 500 (unchanged -- no principled reason yet to increase it;
  the corpus is larger but not so much larger as to obviously need more
  steps, and changing step count would confound the single-variable
  design)
eval semantics: Trainer.evaluate(), full pass, eval mode, no grad (unchanged)
curve checkpoints: [0, 50, 100, 200, 300, 500] (unchanged)
```

## 9. Baselines

**[VERIFIED BY TEST]**, real execution against the actual V3
corpus/split/tokenizer:

```
uniform_baseline_loss:  5.950642552587727   (unchanged -- same vocab_size)
unigram_baseline_loss:  4.518757994941216   (V1/V2 was 4.701993336814625;
  slightly lower here, consistent with V3's more templated/predictable
  structure giving the frequency baseline a marginally easier target too)
```

## 10. Results

**[UNKNOWN]** — requires PyTorch execution. Not fabricated.

## 11. Training Curve

**[UNKNOWN]** — requires PyTorch execution. Captured via
`run_experiment_v3`'s own loop (same mechanism as V2 — direct
`Trainer.train_step()`/`evaluate()` calls, `training/loop.py` itself
untouched) at steps `[0, 50, 100, 200, 300, 500]`.

## 12. Generalization Analysis

**[UNKNOWN]** — pending execution. To be filled in with the real
curve, using the same Case A/B/C/D framework from the V3 prompt.

## 13. Failure Analysis

**[UNKNOWN]** — pending execution.

## 14. Decision

**BLOCKED**

Every stage that could be verified without PyTorch — corpus
generation, its determinism, the word-coverage improvement that is
this experiment's entire premise, the real tokenizer run, real token
counts, real baselines — is done and genuinely verified. The actual
training run has never executed.

**Exact commands for the real environment:**

```bash
python -m pytest tests/test_nb08_v3_corpus.py -q          # already known-good here
python -m pytest tests/test_nb08_experiment_v3.py -q
python -m training.experiments.nb08_experiment_v3
python -m pytest tests -q
python -m compileall -q yug training tests
```

## 15. Next-Step Recommendation

Once real V3 numbers exist: if `final_val_loss` (or `best_val_loss`)
beats the unigram baseline (4.5188) by a meaningful margin, that would
be the first genuine positive generalization evidence in this
milestone's history, and the next experiment should investigate
*where* on the train/validation curve that happens (does it still
overfit past a point, per V2's pattern?) before considering any
further changes. If it does not beat the baseline, the next
single-variable experiment should be regularization (dropout and/or
weight_decay) at V2/V3's architecture — the one lever explicitly held
constant across V1, V2, and V3 so far — rather than a further corpus
change, since two consecutive data-regime interventions (V2's capacity
reduction, V3's coverage fix) would need to have been tried before
concluding data/capacity alone can't fix this.
