"""DIAGNOSTIC ONLY: bounded preprocessing measurements, never model training.

Run as a module. Local source/documentation text is a surrogate, NOT the planned
corpus or an approved dataset mixture. Fresh workers isolate process memory;
timed stages exclude correctness checks and separately instrumented cProfile runs.
Production tokenizer, corpus, split, window and batch functions are unchanged.
"""
from __future__ import annotations

import argparse
import cProfile
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "ARCHITECTURE/NATIVE_BRAIN_ARCHITECTURE.md",
    "ARCHITECTURE/NB08_DATA_SCALE_EXPERIMENT_V4.md",
    "REPORT/YUG_FINAL_ARCHITECTURE_DECISION_REPORT.md",
    "yug/tokenizer/bpe.py", "yug/native/model.py", "training/corpus_prep.py",
    "training/experiments/real_corpus_experiment.py",
)
CHAR_SIZES = (32768, 65536, 131072)
TOKEN_SIZES = (250000, 1000000, 2000000)
VOCABS = (384, 1024)
SEED = 1234
SEQUENCE_LENGTH = 128
BATCH_SIZE = 8
MAX_CHARS = 131072
MAX_TOKENS = 2000000


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def local_documents(characters):
    """Nested round-robin prefixes of <=1024-character surrogate documents."""
    if not 0 < characters <= MAX_CHARS:
        raise ValueError("Diagnostic character cap exceeded")
    from training.corpus_prep import RawDocument
    texts = [(ROOT / name).read_text(encoding="utf-8") for name in SOURCES]
    documents = []
    remaining = characters
    for offset in range(0, max(map(len, texts)), 1024):
        for name, text in zip(SOURCES, texts):
            part = text[offset:offset + min(1024, remaining)]
            if part:
                documents.append(RawDocument(
                    document_id=f"diagnostic:{name}:{offset}", text=part,
                    source_id=name, source_name="local repository surrogate",
                    license="DIAGNOSTIC_FIXTURE_ONLY", language="mixed",
                    primary_domain="diagnostic_not_dataset_mixture",
                ))
                remaining -= len(part)
            if remaining == 0:
                return documents
    raise ValueError("Local surrogate has insufficient characters")


def prepare(documents):
    from training.corpus_prep import LicenseDecision, LicensePolicy, prepare_corpus
    return prepare_corpus(
        documents, license_policy=LicensePolicy({"DIAGNOSTIC_FIXTURE_ONLY": LicenseDecision.ALLOW}),
        val_fraction=0.2, seed=SEED,
    )


def memory():
    """OS process readings; peaks are lifetime cumulative, NOT stage allocations."""
    if sys.platform != "win32":
        return {"current_rss_bytes": None, "peak_rss_bytes": None,
                "error": "This diagnostic implements Windows process-memory readings only"}
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (key, ctypes.c_size_t) for key in (
                "peak", "current", "paged_peak", "paged", "nonpaged_peak",
                "nonpaged", "pagefile", "pagefile_peak",
            )]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return {"current_rss_bytes": int(counters.current), "peak_rss_bytes": int(counters.peak),
            "error": None}


def measured(fn):
    before = memory()
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    result = fn()
    wall, cpu = time.perf_counter() - wall_start, time.process_time() - cpu_start
    after = memory()
    return result, {"wall_seconds": wall, "cpu_seconds": cpu,
                    "memory_before": before, "memory_after": after}


def add_rates(metrics, **counts):
    metrics["work_units"] = counts
    metrics["per_second"] = {key: value / metrics["wall_seconds"] for key, value in counts.items()}


def profile(fn):
    profiler = cProfile.Profile()
    result = profiler.runcall(fn)
    stats = pstats.Stats(profiler)
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
        rows.append({"function": f"{filename}:{line}:{name}", "calls": calls,
                     "primitive_calls": primitive, "self_seconds": own,
                     "cumulative_seconds": cumulative})
    return result, {"instrumented_not_benchmark_timing": True,
                    "total_seconds": stats.total_tt, "total_calls": stats.total_calls,
                    "top_cumulative": sorted(rows, key=lambda r: r["cumulative_seconds"], reverse=True)[:18],
                    "top_self": sorted(rows, key=lambda r: r["self_seconds"], reverse=True)[:12]}


def batch_checks(ids, sequences, batches):
    """Independent exact slicing oracle, bounded to one batch's list temporary."""
    assert len(sequences) == len(ids) // (SEQUENCE_LENGTH + 1)
    index = 0
    for x, y in batches:
        length = x.shape[0]
        expected = [ids[(index + i) * (SEQUENCE_LENGTH + 1):
                        (index + i + 1) * (SEQUENCE_LENGTH + 1)] for i in range(length)]
        assert x.tolist() == [s[:-1] for s in expected]
        assert y.tolist() == [s[1:] for s in expected]
        index += length
    assert index == len(sequences)
    return {"all_batches_match_flat_token_oracle": True,
            "scored_targets": index * SEQUENCE_LENGTH,
            "discarded_tail_tokens": len(ids) % (SEQUENCE_LENGTH + 1)}


def dataset_stages(ids, stages):
    from training.dataset import iter_batches, make_sequences
    sequences, stages["windows"] = measured(lambda: make_sequences(ids, SEQUENCE_LENGTH))
    add_rates(stages["windows"], tokens=len(ids), sequences=len(sequences))
    batches, stages["batches"] = measured(lambda: iter_batches(sequences, BATCH_SIZE))
    add_rates(stages["batches"], sequences=len(sequences), scored_targets=len(sequences) * SEQUENCE_LENGTH)
    tensor_bytes = sum(t.numel() * t.element_size() for batch in batches for t in batch)
    return {"tokens": len(ids), "sequences": len(sequences), "batches": len(batches),
            "tensor_storage_bytes": tensor_bytes,
            "flat_list_container_bytes": sys.getsizeof(ids),
            "window_list_container_bytes": sys.getsizeof(sequences) + sum(map(sys.getsizeof, sequences)),
            "memory_before_checks": memory(), "correctness": batch_checks(ids, sequences, batches)}


def worker(kind, size, vocab, tokenizer_path):
    # Imports are deliberately outside stage timing, and before the RSS baseline.
    import torch
    from training.dataset import encode_corpus
    from yug.tokenizer.bpe import ByteLevelBpeTokenizer
    stages = {}
    out = {"kind": kind, "size": size, "requested_vocab": vocab, "stages": stages,
           "environment": {"python": sys.version, "torch": str(torch.__version__),
                           "platform": platform.platform(), "cpu_count": os.cpu_count(),
                           "torch_threads": torch.get_num_threads()}, "baseline_memory": memory()}
    if kind == "dataset":
        if size not in TOKEN_SIZES:
            raise ValueError("Diagnostic token cap exceeded")
        # Controlled allocation stress, not corpus encoding. IDs are legal at V=8192.
        ids, stages["synthetic_ids"] = measured(lambda: [258 + (i % 7934) for i in range(size)])
        out["synthetic_non_cached_integer_objects"] = True
        out["counts"] = dataset_stages(ids, stages)
        return out
    docs, stages["local_input"] = measured(lambda: local_documents(size))
    out["input"] = {"documents": len(docs), "characters": sum(len(d.text) for d in docs),
                    "utf8_bytes": sum(len(d.text.encode("utf-8")) for d in docs),
                    "identity": digest([d.text for d in docs])}
    prep, stages["prepare_including_split"] = measured(lambda: prepare(docs))
    add_rates(stages["prepare_including_split"], documents=len(docs),
              characters=out["input"]["characters"], utf8_bytes=out["input"]["utf8_bytes"])
    train = [d.text for d in prep.train]
    validation = [d.text for d in prep.validation]
    out["prepared"] = {"accepted": len(prep.accepted), "rejected": len(prep.rejected),
                       "review": len(prep.review), "train_documents": len(train),
                       "validation_documents": len(validation),
                       "train_characters": sum(map(len, train)),
                       "train_utf8_bytes": sum(len(s.encode("utf-8")) for s in train),
                       "train_identity": digest(train), "validation_identity": digest(validation)}
    if kind in ("fit", "profile_fit"):
        if vocab not in (*VOCABS, 8192):
            raise ValueError("Diagnostic vocabulary cap exceeded")
        fit = lambda: ByteLevelBpeTokenizer.train(train, vocab_size=vocab)
        if kind == "fit":
            tok, stages["fit"] = measured(fit)
            add_rates(stages["fit"], characters=out["prepared"]["train_characters"],
                      utf8_bytes=out["prepared"]["train_utf8_bytes"], documents=len(train))
        else:
            tok, out["profile"] = profile(fit)
        out["tokenizer"] = tok.to_dict()
        return out
    tok = ByteLevelBpeTokenizer.load(tokenizer_path)
    out["tokenizer_identity"] = tok.identity
    out["actual_vocab"] = tok.vocab_size
    if kind == "profile_encode":
        ids, out["profile"] = profile(lambda: encode_corpus(tok, train))
        out["encoded_tokens"] = len(ids)
        return out
    for name, texts in (("train", train), ("validation", validation)):
        ids, stages[f"encode_{name}"] = measured(lambda: encode_corpus(tok, texts))
        add_rates(stages[f"encode_{name}"], characters=sum(map(len, texts)),
                  utf8_bytes=sum(len(s.encode("utf-8")) for s in texts),
                  documents=len(texts), tokens=len(ids))
        out[f"{name}_counts"] = dataset_stages(ids, stages.setdefault(name, {}))
        # Untimed document round trips also check no tokenizer/state drift.
        assert all(tok.decode(tok.encode(text)) == text for text in texts)
    out["round_trips_ok"] = True
    out["note"] = "Partitions measured separately; this does not measure retaining BOTH batch sets together."
    return out


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in (ROOT / "yug", ROOT / "training") for p in sorted(directory.rglob("*.py"))}


def run_suite(output, repeats):
    if repeats not in (2, 3):
        raise ValueError("Use two or three repetitions")
    output = output.resolve()
    if output.exists():
        raise ValueError("Refusing to overwrite an existing diagnostic run")
    output.mkdir(parents=True)
    original_hashes = source_hashes()
    records = []
    tokenizer_path = output / "diagnostic_tokenizer_1024.json"

    def run(kind, size, vocab=1024, repeat=0):
        command = [sys.executable, "-B", "-m", "training.experiments.preprocessing_scalability_probe",
                   "--worker", kind, "--size", str(size), "--vocab", str(vocab),
                   "--tokenizer", str(tokenizer_path)]
        print(f"{kind} size={size} vocab={vocab} repeat={repeat}", flush=True)
        try:
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                       encoding="utf-8", timeout=90, check=True)
            result = json.loads(completed.stdout)
            result["repeat"] = repeat
            result["command"] = command
            if "tokenizer" in result:
                tok = result.pop("tokenizer")
                result["tokenizer_identity"] = tok["identity"]
                result["actual_vocab"] = tok["vocab_size"]
                if kind == "fit" and size == MAX_CHARS and vocab == 1024 and repeat == 0:
                    tokenizer_path.write_text(json.dumps(tok, indent=2), encoding="utf-8")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            result = {"kind": kind, "size": size, "requested_vocab": vocab, "repeat": repeat,
                      "error": str(exc), "stderr": getattr(exc, "stderr", None), "command": command}
        records.append(result)
        (output / "raw_results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    for size in CHAR_SIZES:
        for vocab in VOCABS:
            for repeat in range(repeats):
                run("fit", size, vocab, repeat)
    if tokenizer_path.exists():
        for size in CHAR_SIZES:
            for repeat in range(repeats):
                run("encode", size, repeat=repeat)
        run("profile_encode", 65536)
    run("profile_fit", 65536)
    # Probe the historical vocabulary on a SMALL input only; not its historical workload.
    for repeat in range(repeats):
        run("fit", 32768, 8192, repeat)
    for size in TOKEN_SIZES:
        for repeat in range(repeats):
            run("dataset", size, repeat=repeat)
    manifest = {
        "schema": "yug-diagnostic-preprocessing-v1", "diagnostic_only": True,
        "trains_transformer": False, "downloads_data": False,
        "input_description": "Local repository prose/code surrogate; not representative corpus or dataset mixture",
        "sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES},
        "document_max_characters": 1024, "sequence_length": SEQUENCE_LENGTH, "batch_size": BATCH_SIZE,
        "repeats": repeats, "seed": SEED, "worker_timeout_seconds": 90,
        "source_hashes_before": original_hashes, "source_hashes_after": source_hashes(),
        "source_files_unchanged": original_hashes == source_hashes(),
        "memory_note": "Windows working set; fresh worker per case; peaks cumulative including imports and prior stages. Not Python heap or GPU memory.",
        "timing_note": "Stage wall/process CPU seconds; imports excluded; checks excluded; cProfile separate.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if any("error" in result for result in records) or not manifest["source_files_unchanged"]:
        raise RuntimeError("Incomplete diagnostic run: inspect raw_results.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--worker", choices=("fit", "encode", "dataset", "profile_fit", "profile_encode"))
    parser.add_argument("--size", type=int, default=32768)
    parser.add_argument("--vocab", type=int, default=1024)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.size, args.vocab, args.tokenizer), allow_nan=False))
    elif args.output:
        run_suite(args.output, args.repeats)
    else:
        parser.error("Provide --output for a suite or --worker for one bounded case")


if __name__ == "__main__":
    main()
