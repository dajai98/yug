# NB-09 — Minimal Evaluation Harness

Status: **IMPLEMENTATION VERIFIED / CHECKPOINT-BASED ACCEPTANCE PENDING**
(see Section 13; do not read this as a full "PASS" against the
roadmap's own NB-09 exit criteria — the checkpoint-evaluation half of
those criteria is not yet satisfiable, for reasons this document does
not treat as a defect).

Authorized by: `YUG — NB-09 IMPLEMENTATION AUTHORIZATION` (this
session). Scope proposed in this session's prior architecture audit,
itself grounded in `ARCHITECTURE/POST_NB07_ROADMAP.md` §6, NB-09.

---

## 1. Objective

Provide a standalone way to measure a checkpoint's language-modeling
quality (loss, perplexity) independent of any specific training run in
progress. `training.loop.Trainer.evaluate()` only computes eval loss
*during* an active training run, using whatever model/optimizer object
the caller already has in memory; nothing in the repository before
this milestone could answer "is this checkpoint, sitting on disk,
any good?" after the fact, without reconstructing that context by
hand.

## 2. Scope

Implemented in `training/evaluate.py`:

- `evaluate_checkpoint(checkpoint_dir, tokenizer, eval_sentences, *, sequence_length, batch_size, seed=0) -> EvaluationReport`
- `perplexity_from_loss(loss: float) -> float`
- `EvaluationReport` (frozen dataclass) with `.to_dict()`, `.to_json()`,
  `.to_human_readable()`
- `EvaluationError` (a `training.errors.TrainingError` subclass), for
  the one evaluation-specific failure mode this module itself
  introduces (tokenizer-identity mismatch)

Nothing else was created or modified. `yug/`, NB-08's experiment
files, `training/checkpoint.py`, `training/loop.py`, and every other
existing module are untouched.

## 3. Non-goals

Per the roadmap and this milestone's authorization, explicitly NOT
part of NB-09: next-token accuracy, top-k accuracy, BLEU/ROUGE,
exact-match toy tasks, generation sanity checks, a benchmark suite, a
leaderboard, a model-comparison framework, any change to
`BrainInterface`/tokenizer semantics/`YugTransformer`/NB-08/checkpoint
format, and `YugNativeAdapter` (that is NB-10).

## 4. Existing infrastructure reused

No logic was duplicated. This module calls, unmodified:

| Reused | From | For |
|---|---|---|
| `load_checkpoint` | `training.checkpoint` | Reading `model_config.json`/`metadata.json`/`state.pt` |
| `encode_corpus`, `make_sequences`, `iter_batches` | `training.dataset` | Turning `eval_sentences` into batches |
| `causal_lm_loss` | `training.loss` | The loss itself |
| `ModelConfig.load`, `.identity` | `yug.native.config` | Model/tokenizer identity, config reconstruction |
| `YugTransformer`, `.count_parameters()` | `yug.native.model` | Model construction, parameter count |

`training.split.split_sentences` and `training.experiments.nb08_baseline`
were inspected but are not called directly by `evaluate_checkpoint`
itself — held-out splitting is the caller's responsibility (this
function takes `eval_sentences` already split), and the
uniform/unigram baselines are a separate, already-existing comparison
point that a caller can compute alongside this harness's output rather
than this module recomputing them.

## 5. Evaluation contract

```
checkpoint (checkpoint_dir)
    +
eval_sentences (already held-out, sentence-level list[str])
    +
tokenizer (already loaded/trained instance)
    +
sequence_length, batch_size, seed
    ↓
EvaluationReport
```

The evaluator does not train (no `optimizer.step()` is ever called —
the optimizer constructed internally exists solely to satisfy
`load_checkpoint`'s existing signature; see §8). It does not modify
the checkpoint (no write happens anywhere in `evaluate_checkpoint`).
It runs entirely under `model.eval()` + `torch.no_grad()`.

## 6. Result schema

`EvaluationReport` fields:

```
checkpoint_dir        : str
checkpoint_step        : int     (from the checkpoint's own metadata)
model_identity          : str     (ModelConfig.identity, sha256:...)
model_parameter_count   : int     (YugTransformer.count_parameters())
tokenizer_identity      : str     (tokenizer.identity, sha256:...)
eval_token_count        : int
eval_sequence_count     : int
sequence_length         : int
batch_size              : int
seed                    : int
loss                    : float
perplexity              : float
```

`.to_json()` serializes exactly these fields (sorted keys, 2-space
indent). `.to_human_readable()` renders them as labeled lines. No
additional identity scheme was invented; every identity/config field
above is read from an existing repository mechanism (§4).

## 7. Determinism behavior

Evaluation runs under `model.eval()`, which sets
`CausalSelfAttention.dropout_p` to `0.0` regardless of
`ModelConfig.dropout` (see `yug/native/model.py`, `dropout_p=(self.dropout_p if self.training else 0.0)`).
Dropout is the model's only source of randomness. Consequently the
forward pass in `evaluate_checkpoint` is a deterministic function of
the loaded weights and the input batches — `seed` is recorded on the
report for bookkeeping/config-tracking consistency with this project's
`TrainingConfig.seed` convention, but does not itself change the
computed loss.

**Known, deliberately-unaddressed side effect**: constructing
`YugTransformer(model_config)` draws initial random weights from the
global PyTorch RNG before `load_checkpoint`'s `load_state_dict` call
immediately overwrites them. This is an existing pattern already
present in `tests/test_training_checkpoint.py` (this module did not
introduce it), and does not affect the final loaded weights or this
function's own output determinism — but a caller who cares about the
global RNG state should know `evaluate_checkpoint` does not leave it
untouched. `restore_rng_state=False` is passed to `load_checkpoint`
deliberately: restoring a checkpoint's saved *training-time* RNG
snapshot has no meaning for a read-only evaluation call.

## 8. Checkpoint loading behavior

`training.checkpoint.load_checkpoint()` requires an already-constructed
`model` **and** `optimizer` (by design — see its own docstring: "this
module is an integration boundary, not another model implementation").
This module does not build a second, parallel checkpoint-reading path
to avoid that requirement. Instead it constructs a throwaway
`torch.optim.AdamW` purely to satisfy `load_checkpoint`'s existing
signature; that optimizer's state is loaded and then discarded —
`optimizer.step()` is never called anywhere in this module. This was
the smallest change that reuses, rather than duplicates, existing
checkpoint infrastructure (per this milestone's explicit instruction
to inspect the existing implementation before adding a second one).
No checkpoint format change was made or considered necessary.

## 9. Tests

`tests/test_training_evaluate.py`, six tests (the five required, plus
one directly exercising the tokenizer-identity-mismatch failure mode
this module itself introduces):

1. `test_perplexity_equals_exp_loss_on_known_case` — three
   hand-computable cases (`loss=0 → ppl=1`, `loss=ln4 → ppl=4`,
   `loss=2 → ppl=e²`), not merely "returns a finite number."
2. `test_evaluate_checkpoint_round_trip` — builds a tiny
   `YugTransformer`, saves it via the real
   `training.checkpoint.save_checkpoint`, loads it back through
   `evaluate_checkpoint`, asserts a finite loss/perplexity related by
   `perplexity == exp(loss)`.
3. `test_evaluate_is_deterministic` — same checkpoint + same eval data
   run twice, asserts identical `EvaluationReport.to_dict()`.
4. `test_evaluate_reports_identity_fields` — asserts every identity/
   config field is present and correct, including round-tripping
   through `.to_json()`.
5. `test_evaluate_rejects_missing_checkpoint` — asserts
   `ModelConfigSerializationError` on a nonexistent checkpoint
   directory (no fake metric returned).
6. `test_evaluate_rejects_tokenizer_identity_mismatch` — asserts
   `EvaluationError` when the supplied tokenizer does not match the
   checkpoint's recorded `tokenizer_identity`.

## 10. Actual test results

**[VERIFIED BY ACTUAL RUN — PROJECT-OWNER-SUPPLIED, Windows/PyTorch
environment, `PyTorch: 2.12.0+cpu`, `CUDA: False`.]** Not executed by
the assistant; supplied by the project owner from `C:\YUG\yug1`, real
`pytest` (not the shim), run twice for repeatability:

```
PS C:\YUG\yug1> python -m pytest tests/test_training_evaluate.py -q
......                                                             [100%]
6 passed in 11.52s

PS C:\YUG\yug1> python -m pytest tests -q
596 passed, 1 skipped in 20.46s
```

Repeated:

```
PS C:\YUG\yug1> python -m pytest tests/test_training_evaluate.py -q
......                                                             [100%]
6 passed in 6.58s

PS C:\YUG\yug1> python -m pytest tests -q
596 passed, 1 skipped in 17.74s
```

All 6 tests in `tests/test_training_evaluate.py` pass, both runs. The
existing skip is unchanged and unrelated to NB-09:

```
SKIPPED [1] tests\test_brain_llama_cpp_adapter.py:136:
REQUIRES REAL MODEL: set the YUG_TEST_MODEL_PATH environment variable
to a local .gguf file to run this test.
```

**Consistency check**: the full-suite count prior to NB-09 (accepted
earlier this session as project-owner-supplied evidence) was 590
passed, 1 skipped. `590 + 6 (new NB-09 tests) = 596` — matches the
figure above exactly. This is a coherent, not merely asserted, delta;
it does not independently re-verify the pre-NB-09 590 figure, but it
is consistent with it.

The two repeated runs producing identical pass counts (6/6, then
596/1) supports — but on its own does not prove beyond doubt — the
determinism claim in Section 7; it is real evidence of repeatability
under the same environment, not a formal proof of bit-for-bit
determinism across arbitrary environments/hardware, which this
document does not claim.

This satisfies the "real-execution tests pass" half of
`POST_NB07_ROADMAP.md`'s NB-09 exit criteria. It does **not** by
itself satisfy the other half (Section 11, unchanged: no real NB-08
checkpoint exists in this repository to evaluate against).

## 11. Actual evaluation result

**[UNKNOWN — no real NB-08 checkpoint exists in this workspace; none
fabricated.]**

Per this milestone's authorization (Directive Section 14): the
workspace was inspected for a real NB-08 checkpoint before writing
this document, and again at closure-audit time.

```
find . -iname "*.pt" -o -iname "*.safetensors" -o -iname "*checkpoint*" -type d
→ (no results)
```

Independently, on the real Windows/PyTorch environment where the
tests above actually ran:

```
PS C:\YUG\yug1> Get-ChildItem -Recurse -Include *.pt,*.safetensors
→ (no results)
```

Two independent checks, in two different environments, agree: no
checkpoint artifact — from NB-08 V4 or otherwise — exists anywhere in
this repository. The NB-08 V4 numbers accepted
earlier in this session
(`train_tokens`/`best_val_loss`/`final_val_loss`/`unigram_baseline_loss`
per scale) are **project-owner-supplied evidence from a separate
Windows/PyTorch environment**, not something this harness has been run
against, and the checkpoint file that would let it be run against them
is not present here.

Consequently: **no evaluate_checkpoint() report exists yet, real or
synthetic-for-testing, because it has never been executed in this
sandbox.** This document does not fabricate one. The correctness
evidence in §10's Test 1 fragment is the only actual numeric
verification performed this session.

## 12. Known limitations

- ~~No torch/network in this sandbox~~ — **resolved**: real execution
  on a Windows/PyTorch environment (`PyTorch 2.12.0+cpu`) confirms all
  6 `tests/test_training_evaluate.py` tests pass and the full suite is
  unaffected (596 passed, 1 skipped, twice). This limitation no longer
  applies; retained here struck through for the historical record.
- **No real NB-08 checkpoint in this package** (§11) — the sole
  remaining gap. Even with PyTorch available, there is currently nothing to run
  `evaluate_checkpoint` against beyond the synthetic checkpoints the
  test suite constructs for its own purposes. If the project owner's
  Windows/PyTorch environment that produced the NB-08 V4 numbers also
  wrote a checkpoint to disk, that checkpoint needs to be copied into
  this workspace (or evaluated there directly) before §11 can be
  completed for real.
- **Throwaway-optimizer overhead** (§8) — accepted as the smallest
  correct integration with the existing checkpoint API; not free, but
  not a new checkpoint format either.
- **RNG side effect of model construction** (§7) — pre-existing project
  pattern, not new here, but worth knowing about if a caller chains
  multiple `evaluate_checkpoint` calls with other RNG-sensitive code
  in between.

## 13. Exit decision

Per this milestone's own governing instruction ("optimize for
producing a small evaluation tool whose numbers can be trusted," not
for declaring the milestone complete):

```
NB-09 VERIFIED IMPLEMENTATION / CHECKPOINT ACCEPTANCE PENDING
```

- Code and tests exist, reuse existing infrastructure without
  duplication, and have now been run for real: 6/6 pass on a
  Windows/PyTorch 2.12.0+cpu environment, twice, with the full 596/1
  suite unaffected both times (Section 10).
- `POST_NB07_ROADMAP.md`'s NB-09 exit criteria has two clauses:
  "real-execution tests pass" (now satisfied) **and** "a report exists
  showing this run against NB-08's checkpoint" (not satisfied — no
  checkpoint exists anywhere in this repository, confirmed
  independently in two environments, Section 11).
- This is not a full `PASS` against the roadmap's own stated exit
  criteria, and it is not a `FAIL` — the implementation has not
  failed anything it has been tested against; the checkpoint half of
  the criteria is simply not yet satisfiable with what currently
  exists in this repository.
- **Next authorized action**: obtain or regenerate the real NB-08
  checkpoint and run `evaluate_checkpoint` against it. NB-10 is not
  authorized until that report exists.
