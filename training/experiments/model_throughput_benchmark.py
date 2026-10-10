"""Local CPU throughput benchmark for the proposed 5M-class 4096-vs-8192 pair.

NOT a milestone: no NB-13 (or any numbered milestone) is assigned. This is
NOT the co-design experiment, NOT a model-quality comparison, NOT a
tokenizer decision, and NOT YUG-100M. Its only question: can this machine
practically execute real training steps of the proposed pair, and at what
measured speed?

It reuses, unmodified: ``yug.native.config.ModelConfig``,
``yug.native.model.YugTransformer``, ``training.loop.Trainer.train_step``
(real forward -> loss -> backward -> optimizer step -> zero_grad),
``training.checkpoint.save_checkpoint/load_checkpoint``,
``training.dataset`` and ``training.determinism``. Nothing in ``yug/`` or
the rest of ``training/`` is modified, and ``yug/`` does not import this file.

DATA MODES
  synthetic (default): deterministic uniform-random token ids in [0, V).
      Dense-Transformer step cost does not depend on token values, so this
      is valid for THROUGHPUT only. ModelConfig demands a ``sha256:`` tokenizer
      identity, so a clearly-labelled PLACEHOLDER identity is used; it is NOT a
      real tokenizer identity and ``tokenizer_identity_verified`` is False.
  supplied tokenizer + corpus (opt-in): ``--tokenizer-4096/--tokenizer-8192``
      (files written by ``ByteLevelBpeTokenizer.save``) plus ``--corpus-dir``
      (UTF-8 ``*.txt``). Vocab size must equal the tokenizer's actual vocab
      size, and identities are compared to expected ones when requested.
      Any mismatch raises ``BenchmarkBlockedError`` (no silent continuation).

Byte accounting: ``bytes_per_second`` is always ``"UNKNOWN"`` here; the
existing pipeline concatenates documents into one token stream and keeps no
per-window byte mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from training.checkpoint import load_checkpoint, save_checkpoint
from training.config import TrainingConfig
from training.dataset import encode_corpus, iter_batches, make_sequences
from training.determinism import seed_everything
from training.errors import TrainingError
from training.loop import Trainer
from yug.native.config import ModelConfig, estimate_parameter_count
from yug.native.model import YugTransformer
from yug.tokenizer.bpe import ByteLevelBpeTokenizer

EXPERIMENT_NAME = "model_throughput_benchmark"

PRIMARY_ARCH = {"hidden_size": 256, "num_layers": 6, "num_heads": 8, "ffn_size": 768}
PRIMARY_VOCAB_SIZES = (4096, 8192)
PRIMARY_SEQUENCE_LENGTH = 128
PRIMARY_CONTEXT_LENGTH = 128
PRIMARY_EXPECTED_PARAMETERS = {4096: 4_984_064, 8192: 6_032_640}
PRIMARY_EXPECTED_BODY_PARAMETERS = 3_935_488

# Historical values from a prior real execution, supplied by the project owner.
# They are EXPECTATIONS to compare against, never "current truth".
PRIOR_TOKENIZER_IDENTITIES = {
    4096: "sha256:a1adc71e96ac48816d25012590eebba10d419d0f9c36717c3569779ea85f64a3",
    8192: "sha256:545072a1b410cbc3001c778ce105ca0f3282532f99b749de419a5a1a17679e5c",
}

DEFAULT_SEED = 1234
DEFAULT_BATCH_SIZE = 8
DEFAULT_WARMUP_STEPS = 30
DEFAULT_MEASURED_STEPS = 100
DEFAULT_EVAL_BATCHES = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
UNKNOWN = "UNKNOWN"


class BenchmarkBlockedError(TrainingError):
    """A stop condition was hit; the benchmark must not continue."""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def expected_parameter_count(vocab_size: int, hidden_size: int, num_layers: int, ffn_size: int) -> int:
    """P = V*h + L*(4h^2 + 2h*f + 2h) + h (tied, RMSNorm, RoPE, 2-proj FFN, no bias)."""
    h = hidden_size
    return vocab_size * h + num_layers * (4 * h * h + 2 * h * ffn_size + 2 * h) + h


def placeholder_tokenizer_identity(vocab_size: int) -> str:
    digest = hashlib.sha256(f"yug-throughput-benchmark-placeholder:{vocab_size}".encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def build_model_config(
    vocab_size: int,
    tokenizer_identity: str,
    *,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    ffn_size: int,
    context_length: int,
) -> ModelConfig:
    """Repository-default norm/positional/activation/bias/tying; nothing overridden."""
    return ModelConfig(
        vocab_size=vocab_size,
        tokenizer_identity=tokenizer_identity,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_size=ffn_size,
        context_length=context_length,
    )


def summarize_steps(step_seconds: Sequence[float]) -> dict[str, float]:
    return {
        "mean_step_seconds": statistics.fmean(step_seconds),
        "median_step_seconds": statistics.median(step_seconds),
        "min_step_seconds": min(step_seconds),
        "max_step_seconds": max(step_seconds),
        "stdev_step_seconds": statistics.stdev(step_seconds) if len(step_seconds) > 1 else 0.0,
    }


def peak_process_memory_bytes() -> int | None:
    """Peak process working set / RSS via stdlib only, else None. Process-wide and
    cumulative (a later config's value includes earlier configs); not tensor memory."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _PMC()
            counters.cb = ctypes.sizeof(_PMC)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return None
            return int(counters.PeakWorkingSetSize)
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024
    except Exception:  # noqa: BLE001 - memory is best-effort
        return None


def collect_environment() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or UNKNOWN,
        "processor_identifier_env": os.environ.get("PROCESSOR_IDENTIFIER", UNKNOWN),
        "logical_cpu_count": os.cpu_count(),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_used": "cpu",
        "torch_num_threads_default": torch.get_num_threads(),
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS", UNKNOWN),
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _load_corpus_texts(corpus_dir: Path) -> tuple[list[str], dict[str, Any]]:
    files = sorted(Path(corpus_dir).rglob("*.txt"))
    if not files:
        raise BenchmarkBlockedError(f"no *.txt files under corpus dir {corpus_dir}")
    digest = hashlib.sha256()
    texts: list[str] = []
    for f in files:
        raw = f.read_bytes()
        digest.update(raw)
        texts.append(raw.decode("utf-8"))
    return texts, {"corpus_file_count": len(files), "corpus_files_sha256": "sha256:" + digest.hexdigest(),
                   "corpus_utf8_bytes": sum(len(t.encode("utf-8")) for t in texts)}


def synthetic_token_ids(vocab_size: int, count: int, seed: int) -> list[int]:
    rng = random.Random(seed * 1_000_003 + vocab_size)
    return [rng.randrange(vocab_size) for _ in range(count)]


def build_batches(token_ids: list[int], sequence_length: int, batch_size: int, *, shuffle_seed: int | None):
    """Full-size batches only (short last batch dropped) so tokens/step is constant."""
    sequences = make_sequences(token_ids, sequence_length)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(sequences)
    batches = [b for b in iter_batches(sequences, batch_size) if b[0].shape[0] == batch_size]
    if not batches:
        raise BenchmarkBlockedError(
            f"only {len(sequences)} sequences available; need at least batch_size={batch_size}"
        )
    return batches


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def time_training_steps(
    trainer,
    batches,
    *,
    warmup_steps: int,
    measured_steps: int,
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run real training steps via ``trainer.train_step``. Warm-up steps are executed
    but never timed (the timer is not called for them). Exactly two timer calls per
    measured step."""
    idx = 0
    warmup_losses: list[float] = []
    for _ in range(warmup_steps):
        x, y = batches[idx % len(batches)]
        idx += 1
        warmup_losses.append(trainer.train_step(x, y))
    step_seconds: list[float] = []
    measured_losses: list[float] = []
    for _ in range(measured_steps):
        x, y = batches[idx % len(batches)]
        idx += 1
        t0 = timer()
        loss = trainer.train_step(x, y)
        step_seconds.append(timer() - t0)
        measured_losses.append(loss)
    initial = warmup_losses[0] if warmup_losses else measured_losses[0]
    return {
        "step_seconds": step_seconds,
        "initial_loss": initial,
        "final_loss": measured_losses[-1],
    }


def checkpoint_roundtrip(model, optimizer, model_config, training_config, identity, step, timer) -> dict[str, Any]:
    out: dict[str, Any] = {"save_success": False, "reload_success": False, "save_seconds": UNKNOWN,
                           "reload_seconds": UNKNOWN, "checkpoint_bytes": UNKNOWN, "error": None}
    with tempfile.TemporaryDirectory(prefix="yug_bench_ckpt_") as tmp:
        try:
            t0 = timer()
            save_checkpoint(tmp, model=model, optimizer=optimizer, model_config=model_config,
                            training_config=training_config, step=step, tokenizer_identity=identity)
            out["save_seconds"] = timer() - t0
            out["save_success"] = True
            out["checkpoint_bytes"] = sum(p.stat().st_size for p in Path(tmp).iterdir())
            fresh = YugTransformer(model_config)
            fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            t0 = timer()
            meta = load_checkpoint(tmp, model=fresh, optimizer=fresh_opt)
            out["reload_seconds"] = timer() - t0
            a, b = model.state_dict(), fresh.state_dict()
            same = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
            out["reload_success"] = bool(same and meta["step"] == step)
        except Exception as exc:  # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# ---------------------------------------------------------------------------
# One configuration
# ---------------------------------------------------------------------------


def _run_one_config(vocab_size, *, arch, sequence_length, context_length, batch_size, warmup_steps,
                    measured_steps, eval_batches, seed, tokenizer_path, corpus, expected_identity, timer):
    seed_everything(seed)
    needed_sequences = (warmup_steps + measured_steps) * batch_size

    if tokenizer_path is None:
        token_ids = synthetic_token_ids(vocab_size, needed_sequences * (sequence_length + 1), seed)
        identity = placeholder_tokenizer_identity(vocab_size)
        data = {"data_source": "synthetic_uniform_random_ids", "throughput_only": True,
                "tokenizer_identity_kind": "benchmark_placeholder_not_a_real_tokenizer",
                "tokenizer_identity_verified": False, "expected_identity": None}
        shuffle_seed = None
    else:
        tok = ByteLevelBpeTokenizer.load(tokenizer_path)
        if tok.vocab_size != vocab_size:
            raise BenchmarkBlockedError(
                f"tokenizer at {tokenizer_path} has actual vocab_size {tok.vocab_size}, expected {vocab_size}")
        if expected_identity is not None and tok.identity != expected_identity:
            raise BenchmarkBlockedError(
                f"tokenizer identity mismatch for V={vocab_size}: expected {expected_identity}, got {tok.identity}")
        texts, corpus_info = corpus
        token_ids = encode_corpus(tok, texts)
        identity = tok.identity
        data = {"data_source": "supplied_tokenizer_and_corpus", "throughput_only": True,
                "tokenizer_identity_kind": "real", "tokenizer_identity_verified": expected_identity is not None,
                "expected_identity": expected_identity, "corpus_token_count": len(token_ids), **corpus_info}
        shuffle_seed = seed
    batches = build_batches(token_ids, sequence_length, batch_size, shuffle_seed=shuffle_seed)

    config = build_model_config(vocab_size, identity, context_length=context_length, **arch)
    model = YugTransformer(config)
    actual = model.count_parameters()
    estimate = estimate_parameter_count(config).total
    formula = expected_parameter_count(vocab_size, arch["hidden_size"], arch["num_layers"], arch["ffn_size"])
    hardcoded = PRIMARY_EXPECTED_PARAMETERS.get(vocab_size) if arch == PRIMARY_ARCH else None
    if not (actual == estimate == formula) or (hardcoded is not None and actual != hardcoded):
        raise BenchmarkBlockedError(
            f"parameter mismatch V={vocab_size}: actual={actual}, config_estimate={estimate}, "
            f"formula={formula}, hardcoded_expected={hardcoded}")
    vocab_params = vocab_size * arch["hidden_size"]
    body_params = actual - vocab_params
    if arch == PRIMARY_ARCH and body_params != PRIMARY_EXPECTED_BODY_PARAMETERS:
        raise BenchmarkBlockedError(f"body parameter mismatch: {body_params} != {PRIMARY_EXPECTED_BODY_PARAMETERS}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    training_config = TrainingConfig(
        seed=seed, learning_rate=LEARNING_RATE, batch_size=batch_size, sequence_length=sequence_length,
        max_steps=warmup_steps + measured_steps, eval_interval=10**9, checkpoint_interval=10**9,
        grad_clip_norm=GRAD_CLIP_NORM, weight_decay=WEIGHT_DECAY, device="cpu")
    trainer = Trainer(model, optimizer, training_config, config, identity)

    timing = time_training_steps(trainer, batches, warmup_steps=warmup_steps,
                                 measured_steps=measured_steps, timer=timer)
    steps = timing["step_seconds"]
    total = sum(steps)
    tokens_per_step = batch_size * sequence_length
    training = {
        "warmup_steps": warmup_steps, "measured_steps": measured_steps, "total_measured_seconds": total,
        **summarize_steps(steps), "training_tokens_per_second": tokens_per_step * measured_steps / total if total > 0 else UNKNOWN,
        "bytes_per_second": UNKNOWN,
        "bytes_accounting_note": "Existing pipeline keeps no per-window UTF-8 byte mapping; not estimated.",
        "initial_loss_sanity_only": timing["initial_loss"], "final_loss_sanity_only": timing["final_loss"],
        "step_seconds": steps,
    }

    evaluation: dict[str, Any] = {"measured": False}
    if eval_batches > 0:
        eb = [batches[i % len(batches)] for i in range(eval_batches)]
        t0 = timer()
        trainer.evaluate(eb)
        dt = timer() - t0
        evaluation = {"measured": True, "batches": eval_batches, "seconds": dt,
                      "eval_tokens_per_second": eval_batches * tokens_per_step / dt if dt > 0 else UNKNOWN,
                      "note": "forward-only over training-stream batches; EVALUATION throughput, not training"}

    peak = peak_process_memory_bytes()
    checkpoint = checkpoint_roundtrip(model, optimizer, config, training_config, identity, trainer.step, timer)

    return {
        "vocab_size": vocab_size, "hidden_size": arch["hidden_size"], "num_layers": arch["num_layers"],
        "num_heads": arch["num_heads"], "head_dim": arch["hidden_size"] // arch["num_heads"],
        "ffn_size": arch["ffn_size"], "context_length": context_length, "sequence_length": sequence_length,
        "architecture_flags": {"normalization": config.normalization, "positional_encoding": config.positional_encoding,
                               "activation": config.activation, "bias": config.bias,
                               "tie_embeddings": config.tie_embeddings, "dropout": config.dropout},
        "tokenizer_identity": identity, "data": data,
        "parameters": {"actual": actual, "estimate_from_config": estimate, "expected_formula": formula,
                       "expected_primary_hardcoded": hardcoded, "vocab_dependent": vocab_params,
                       "body": body_params, "vocab_share": vocab_params / actual, "body_share": body_params / actual},
        "batch": {"batch_size": batch_size, "sequence_length": sequence_length, "tokens_per_step": tokens_per_step,
                  "gradient_accumulation_steps": 1},
        "optimizer": {"name": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                      "grad_clip_norm": GRAD_CLIP_NORM},
        "training": training, "evaluation": evaluation,
        "memory": {"peak_process_rss_bytes": peak if peak is not None else UNKNOWN,
                   "note": "process-wide cumulative peak (includes earlier configs); not exact tensor memory"},
        "checkpoint": checkpoint,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_benchmark(
    *,
    vocab_sizes: Sequence[int] = PRIMARY_VOCAB_SIZES,
    hidden_size: int = PRIMARY_ARCH["hidden_size"],
    num_layers: int = PRIMARY_ARCH["num_layers"],
    num_heads: int = PRIMARY_ARCH["num_heads"],
    ffn_size: int = PRIMARY_ARCH["ffn_size"],
    sequence_length: int = PRIMARY_SEQUENCE_LENGTH,
    context_length: int = PRIMARY_CONTEXT_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    measured_steps: int = DEFAULT_MEASURED_STEPS,
    eval_batches: int = DEFAULT_EVAL_BATCHES,
    threads: int | None = None,
    seed: int = DEFAULT_SEED,
    tokenizer_paths: dict[int, Path | str] | None = None,
    corpus_dir: Path | str | None = None,
    expected_identities: dict[int, str] | None = None,
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    if measured_steps < 1 or warmup_steps < 0 or batch_size < 1 or eval_batches < 0:
        raise BenchmarkBlockedError("measured_steps/batch_size must be >= 1; warmup_steps/eval_batches >= 0")
    if threads is not None and threads < 1:
        raise BenchmarkBlockedError("threads must be >= 1")
    arch = {"hidden_size": hidden_size, "num_layers": num_layers, "num_heads": num_heads, "ffn_size": ffn_size}
    if sequence_length > context_length:
        raise BenchmarkBlockedError("sequence_length exceeds context_length")

    real = bool(tokenizer_paths) or corpus_dir is not None
    corpus = None
    if real:
        if not tokenizer_paths or corpus_dir is None or any(v not in tokenizer_paths for v in vocab_sizes):
            raise BenchmarkBlockedError("real-data mode needs a tokenizer path for every vocab size AND --corpus-dir")
        corpus = _load_corpus_texts(Path(corpus_dir))

    previous_threads = torch.get_num_threads()
    if threads is not None:
        torch.set_num_threads(threads)
    try:
        environment = collect_environment()
        threading = {"requested": threads, "actual": torch.get_num_threads()}
        configs: dict[str, Any] = {}
        for v in vocab_sizes:
            configs[str(v)] = _run_one_config(
                v, arch=arch, sequence_length=sequence_length, context_length=context_length,
                batch_size=batch_size, warmup_steps=warmup_steps, measured_steps=measured_steps,
                eval_batches=eval_batches, seed=seed,
                tokenizer_path=(tokenizer_paths or {}).get(v) if real else None, corpus=corpus,
                expected_identity=(expected_identities or {}).get(v), timer=timer)
    finally:
        torch.set_num_threads(previous_threads)

    comparison: dict[str, Any] = {}
    if len(vocab_sizes) == 2:
        a, b = (configs[str(v)]["training"] for v in vocab_sizes)
        comparison = {
            "ratio_definition": f"{vocab_sizes[1]} / {vocab_sizes[0]}",
            "mean_step_time_ratio": b["mean_step_seconds"] / a["mean_step_seconds"],
            "median_step_time_ratio": b["median_step_seconds"] / a["median_step_seconds"],
            "tokens_per_second_ratio": (b["training_tokens_per_second"] / a["training_tokens_per_second"]
                                        if isinstance(a["training_tokens_per_second"], float)
                                        and isinstance(b["training_tokens_per_second"], float) else UNKNOWN),
            "note": "Throughput/cost ratios only. Not a model-quality result.",
        }

    return {
        "experiment_name": EXPERIMENT_NAME, "not_a_milestone": True, "nb13_assigned": False,
        "trains_for_quality": False, "model_quality_verdict": "NOT_GENERATED",
        "environment": environment, "threading": threading,
        "corpus": {"mode": "supplied_tokenizer_and_corpus" if real else "synthetic_uniform_random_ids",
                   "train_documents": UNKNOWN, "validation_documents": UNKNOWN,
                   "prior_frozen_corpus_reference": {"train_documents": 143, "validation_documents": 35,
                                                     "source": "owner-supplied prior real execution; not re-derived here"},
                   "prior_tokenizer_identities_expected_unverified": {str(k): v for k, v in PRIOR_TOKENIZER_IDENTITIES.items()},
                   **(corpus[1] if corpus else {})},
        "configs": configs, "comparison": comparison,
    }


def _print_summary(result: dict[str, Any]) -> None:
    print(f"{result['experiment_name']}  (not a milestone; throughput only)")
    print(f"  threads requested/actual: {result['threading']['requested']}/{result['threading']['actual']}")
    for v, c in result["configs"].items():
        t = c["training"]
        print(f"  V={v}: params={c['parameters']['actual']:,}  mean_step={t['mean_step_seconds']:.3f}s  "
              f"median={t['median_step_seconds']:.3f}s  tokens/s={t['training_tokens_per_second']}  "
              f"ckpt_ok={c['checkpoint']['save_success'] and c['checkpoint']['reload_success']}")
    if result["comparison"]:
        print(f"  step-time ratio ({result['comparison']['ratio_definition']}): "
              f"{result['comparison']['mean_step_time_ratio']:.3f}")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Local CPU throughput benchmark (not a milestone).")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    p.add_argument("--measured-steps", type=int, default=DEFAULT_MEASURED_STEPS)
    p.add_argument("--eval-batches", type=int, default=DEFAULT_EVAL_BATCHES)
    p.add_argument("--threads", type=int, default=None, help="torch.set_num_threads for this process only")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--tokenizer-4096", default=None)
    p.add_argument("--tokenizer-8192", default=None)
    p.add_argument("--corpus-dir", default=None)
    p.add_argument("--expect-prior-identities", action="store_true",
                   help="require supplied tokenizers to match the prior 4096/8192 identities")
    p.add_argument("--output", default="model_throughput_benchmark_result.json")
    args = p.parse_args(argv)

    paths = {v: t for v, t in ((4096, args.tokenizer_4096), (8192, args.tokenizer_8192)) if t}
    try:
        result = run_benchmark(
            batch_size=args.batch_size, warmup_steps=args.warmup_steps, measured_steps=args.measured_steps,
            eval_batches=args.eval_batches, threads=args.threads, seed=args.seed,
            tokenizer_paths=paths or None, corpus_dir=args.corpus_dir,
            expected_identities=PRIOR_TOKENIZER_IDENTITIES if args.expect_prior_identities else None)
    except BenchmarkBlockedError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _print_summary(result)
    print(f"result written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())