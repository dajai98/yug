"""NB-12: Real-Checkpoint Evaluation & Adapter Exercise.

Thin orchestration driver ONLY — per
``ARCHITECTURE/POST_NB07_ROADMAP.md``'s NB-12 definition, this reuses
NB-08 V4's existing, unmodified ``run_single_scale()`` /
``build_locked_tokenizer()`` / ``SCALE_CORPORA["scale_3"]``, NB-09's
``evaluate_checkpoint()``, NB-10's ``YugNativeAdapter``, and NB-11's
``YugCore`` exactly as authorized. No training logic, no evaluation
metric, no adapter/runtime logic is reimplemented here — this file
contains orchestration and acceptance-criteria checking only.

VERIFICATION STATUS: written and syntax-checked without a PyTorch
installation available in this sandbox (no network access to install
one — the same disclosed constraint as every prior milestone this
session). NOT executed by this assistant. Running this for real
requires the project's real Windows/PyTorch environment; see
``ARCHITECTURE/NB12_REAL_CHECKPOINT_EVALUATION.md`` for exactly what
has and hasn't been verified, and by whom.

Usage:
    python -m training.experiments.nb12_checkpoint_closure

Produces one checkpoint in a fresh temporary directory (never inside
this repository — per the roadmap, the checkpoint binary itself is
not a committed deliverable), runs the full evaluation/adapter/runtime
chain against it exactly once, and prints a JSON result to stdout.
Exit code is 0 for PASS or RESULT/OBSERVATION, 1 for FAIL.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path


def _check_finite(values: dict[str, float]) -> dict[str, float]:
    """Returns the subset of ``values`` that is NOT finite (empty dict
    means all finite). A successful Python computation does not by
    itself guarantee a finite result (NaN/Inf are valid floats)."""
    return {k: v for k, v in values.items() if not math.isfinite(v)}


def _check_state_dict_finite(state_dict) -> tuple[bool, str]:
    """Checks every tensor in a raw model state_dict for NaN/Inf. A
    successful ``torch.load()`` does not by itself guarantee this —
    NaN/Inf are perfectly loadable float32 values."""
    import torch

    bad = [name for name, tensor in state_dict.items() if not torch.isfinite(tensor).all()]
    if bad:
        return False, f"NaN/Inf found in checkpoint parameters: {bad}"
    return True, "all checkpoint parameters finite"


def run() -> dict:
    """Run the complete NB-12 pipeline once and return a structured
    result dict (also written to
    ARCHITECTURE/NB12_REAL_CHECKPOINT_EVALUATION.md by the caller).

    Every *expected* classification outcome (PASS / RESULT-OBSERVATION
    / FAIL for a defined acceptance-criterion reason) is recorded in
    the returned dict's "verdict"/"verdict_reason" fields, not raised
    as an exception — only a genuinely unexpected crash in this
    orchestration code itself would propagate as a real Python
    exception out of this function.
    """
    import asyncio

    import torch

    from training.evaluate import evaluate_checkpoint
    from training.experiments.nb08_experiment_v4 import (
        SCALE_CORPORA,
        SEED,
        SEQUENCE_LENGTH,
        VAL_FRACTION,
        build_locked_tokenizer,
        build_training_config,
        run_single_scale,
    )
    from training.split import split_sentences
    from yug.brain.adapters.yug_native import YugNativeAdapter
    from yug.brain.errors import BrainError
    from yug.memory.working_memory import WorkingMemory
    from yug.runtime.core import YugCore

    result: dict = {"milestone": "NB-12", "verdict": None, "verdict_reason": None}

    # -- Steps 1-3: locked tokenizer, existing scale_3 corpus, real
    #    training via the existing, UNMODIFIED run_single_scale(). ---
    tokenizer = build_locked_tokenizer()
    corpus = SCALE_CORPORA["scale_3"]

    checkpoint_root = Path(tempfile.mkdtemp(prefix="nb12_"))  # never inside the repo
    checkpoint_dir = checkpoint_root / "scale_3"
    try:
        experiment = run_single_scale("scale_3", corpus, tokenizer, str(checkpoint_dir))
    except Exception as exc:  # noqa: BLE001 - orchestration-level safety net
        result["verdict"] = "FAIL"
        result["verdict_reason"] = (
            f"run_single_scale() itself raised: {type(exc).__name__}: {exc}"
        )
        return result

    # Step 4/7: record the complete return value verbatim -- this is
    # every field run_single_scale() computes, not a summary.
    result["experiment"] = dict(experiment)

    # -- Step 5: checkpoint actually exists. ---------------------------
    state_path = Path(experiment["checkpoint_dir"]) / "state.pt"
    if not state_path.exists():
        result["verdict"] = "FAIL"
        result["verdict_reason"] = f"checkpoint state file not found: {state_path}"
        return result

    # -- Steps 6/8: finite losses, no NaN/Inf in the saved state. -----
    loss_fields = (
        "initial_train_loss",
        "final_train_loss_full_eval",
        "final_val_loss",
        "best_val_loss",
        "uniform_baseline_loss",
        "unigram_baseline_loss",
    )
    non_finite = _check_finite({k: experiment[k] for k in loss_fields})
    if non_finite:
        result["verdict"] = "FAIL"
        result["verdict_reason"] = f"non-finite loss value(s) from run_single_scale: {non_finite}"
        return result

    raw_state = torch.load(state_path, map_location="cpu")
    state_ok, state_detail = _check_state_dict_finite(raw_state["model"])
    result["checkpoint_integrity"] = state_detail
    if not state_ok:
        result["verdict"] = "FAIL"
        result["verdict_reason"] = state_detail
        return result

    # -- Step 9: best_val_loss < initial_train_loss (NB-08 V4's own
    #    evidence condition for "actual learning happened"). ----------
    learning_demonstrated = experiment["best_val_loss"] < experiment["initial_train_loss"]
    result["best_val_loss_lt_initial_train_loss"] = learning_demonstrated
    if not learning_demonstrated:
        result["verdict"] = "FAIL"
        result["verdict_reason"] = (
            f"best_val_loss ({experiment['best_val_loss']}) is not less than "
            f"initial_train_loss ({experiment['initial_train_loss']}) -- training "
            f"evidence does not satisfy the NB-08 V4 condition"
        )
        return result

    # -- Step 10: baseline comparison, recorded (not itself a hard
    #    gate beyond the roadmap's stated comparison requirement). ---
    result["best_val_loss_lt_uniform_baseline"] = (
        experiment["best_val_loss"] < experiment["uniform_baseline_loss"]
    )
    result["best_val_loss_lt_unigram_baseline"] = (
        experiment["best_val_loss"] < experiment["unigram_baseline_loss"]
    )

    # -- Step 11: NB-09 evaluation of the SAME real checkpoint, on the
    #    identical held-out split run_single_scale() itself used
    #    (same corpus, same VAL_FRACTION, same SEED -> same split). --
    val_sentences = split_sentences(corpus, val_fraction=VAL_FRACTION, seed=SEED)[1]
    training_config = build_training_config()
    try:
        eval_report = evaluate_checkpoint(
            experiment["checkpoint_dir"],
            tokenizer,
            val_sentences,
            sequence_length=SEQUENCE_LENGTH,
            batch_size=training_config.batch_size,
            seed=SEED,
            data_provenance={
                "corpus_identity": None, "split_name": "validation",
                "split_method": "training.split.split_sentences",
                "split_seed": SEED, "split_fraction": VAL_FRACTION,
            },
        )
        evaluation_report = eval_report.to_dict()
    except Exception as exc:  # noqa: BLE001
        result["verdict"] = "FAIL"
        result["verdict_reason"] = f"evaluate_checkpoint() raised: {type(exc).__name__}: {exc}"
        return result

    result["evaluation"] = {
        "report": evaluation_report,
        "loss": eval_report.loss,
        "perplexity": eval_report.perplexity,
        "loss_finite": math.isfinite(eval_report.loss),
        "perplexity_finite": math.isfinite(eval_report.perplexity),
    }
    if not (math.isfinite(eval_report.loss) and math.isfinite(eval_report.perplexity)):
        result["verdict"] = "FAIL"
        result["verdict_reason"] = "evaluate_checkpoint() produced a non-finite loss/perplexity"
        return result

    # -- Steps 12-13: adapter load (checked in isolation, BEFORE
    #    respond() -- the roadmap's own evidence list treats "adapter
    #    loads" (9) and "respond() is called" (10) as sequential,
    #    separately-verified steps, not one combined call) + YugCore
    #    construction. ---------------------------------------------
    adapter = YugNativeAdapter(checkpoint_dir=experiment["checkpoint_dir"], tokenizer=tokenizer)
    try:
        asyncio.run(adapter._ensure_loaded())  # noqa: SLF001 - deliberate isolated load check,
        # mirroring tests/test_brain_yug_native_adapter.py's own established
        # pattern of calling this directly to check load success in isolation.
    except BrainError as exc:
        result["adapter_load"] = f"FAILED: {type(exc).__name__}: {exc}"
        result["verdict"] = "FAIL"
        result["verdict_reason"] = f"YugNativeAdapter failed to load the real checkpoint: {exc}"
        return result
    result["adapter_load"] = "SUCCESS"

    # max_tokens=8 (not YugCore's default of 1024): NB-08 V4's own
    # deliberately tiny context_length=32 model cannot support an
    # unbounded generation budget. The formatted prompt for the fixed
    # request below ("user: Hello.\nassistant:") is 21 tokens with the
    # real locked tokenizer -- 21 + 8 = 29, safely inside 32 with
    # margin. This is the exact, narrow remediation for this
    # milestone's original Classification-B failure (a driver
    # invocation error, not a production defect) -- see
    # ARCHITECTURE/NB12_REAL_CHECKPOINT_EVALUATION.md's remediation
    # history for the full trace.
    core = YugCore(adapter, WorkingMemory(), max_tokens=8)

    # -- Step 14: respond() exactly once. -----------------------------
    response = core.respond("Hello.")
    result["respond_success"] = response.success
    result["respond_content"] = response.content
    result["respond_error"] = response.error

    # -- Step 15: classify. --------------------------------------------
    # NOTE on the string-matching below: yug.runtime.core.Response
    # deliberately does not expose the original exception object or
    # its __cause__ chain (a structured-failure design choice, see
    # that module's docstring) -- error is always
    # f"{type(exc).__name__}: {exc}". Distinguishing "the permitted
    # DecodeError->GenerationError boundary" from any other BrainError
    # therefore has to match on that string's stable-at-time-of-writing
    # wording (yug/brain/adapters/yug_native.py's own
    # `raise GenerationError(f"Generation failed: {exc}") from exc`,
    # where exc's message always contains "not valid UTF-8" per
    # yug/tokenizer/bpe.py's DecodeError raise site). This is
    # necessarily string-based, not exception-based -- flagged here
    # rather than hidden, and worth re-checking if either message
    # ever changes.
    if response.success:
        result["verdict"] = "PASS"
        result["verdict_reason"] = "real checkpoint loaded and generated a real response"
    elif (
        response.error is not None
        and response.error.startswith("GenerationError:")
        and "not valid UTF-8" in response.error
    ):
        result["verdict"] = "RESULT / OBSERVATION"
        result["verdict_reason"] = (
            "real pipeline executed successfully end-to-end; greedy decoding "
            "produced a token sequence that is not valid UTF-8 -- the known, "
            "already-verified DecodeError -> GenerationError boundary "
            "(yug/brain/adapters/yug_native.py). This is a legitimate "
            "model-level outcome, not an infrastructure failure."
        )
    else:
        result["verdict"] = "FAIL"
        result["verdict_reason"] = (
            f"YugCore.respond() failed with an error other than the permitted "
            f"DecodeError->GenerationError boundary: {response.error}"
        )

    return result


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(outcome, indent=2, default=str))
    sys.exit(0 if outcome["verdict"] in ("PASS", "RESULT / OBSERVATION") else 1)
