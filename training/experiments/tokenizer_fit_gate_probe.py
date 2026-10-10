"""Capped native tokenizer FIT/ENCODE diagnostics; local surrogate, not corpus policy."""
from __future__ import annotations

import argparse
from collections import Counter
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import subprocess
import sys
import time

from training.experiments.preprocessing_scalability_probe import local_documents, memory

CASES = ((32768, 1024), (65536, 1024), (131072, 1024),
         (32768, 4096), (32768, 8192), (65536, 4096), (65536, 8192))


def worker(chars, vocab, progress, profiled=False, legacy_count=False):
    from yug.tokenizer import bpe
    texts = [d.text for d in local_documents(chars)]
    raw = sum(len(text.encode("utf-8")) for text in texts)
    corpus_hash = hashlib.sha256(json.dumps(texts, ensure_ascii=True).encode()).hexdigest()
    native = bpe._get_adjacent_pairs
    def legacy(sequences):
        counts = Counter()
        for seq in sequences:
            for a, b in zip(seq, seq[1:]):
                counts[(a, b)] += 1
        return counts
    original = legacy if legacy_count else native
    rounds = 0
    start, cpu = time.perf_counter(), time.process_time()
    before = memory()
    def observed(sequences):
        nonlocal rounds
        if rounds % 128 == 0:
            progress.write_text(json.dumps(dict(rounds_started=rounds, wall_seconds=time.perf_counter()-start,
                cpu_seconds=time.process_time()-cpu, memory=memory(), live_tokens=sum(map(len, sequences)))), encoding="utf-8")
        rounds += 1
        return original(sequences)
    bpe._get_adjacent_pairs = observed
    profiler = cProfile.Profile() if profiled else None
    try:
        if profiler:
            profiler.enable()
        tok = bpe.ByteLevelBpeTokenizer.train(texts, vocab)
    finally:
        if profiler:
            profiler.disable()
        bpe._get_adjacent_pairs = native
    fit_wall, fit_cpu = time.perf_counter()-start, time.process_time()-cpu
    fit_memory = memory()
    start, cpu = time.perf_counter(), time.process_time()
    encoded = 0
    digest = hashlib.sha256()
    for text in texts:
        ids = tok.encode(text)
        encoded += len(ids)
        digest.update(json.dumps(ids, separators=(",", ":")).encode())
    encode_wall, encode_cpu = time.perf_counter()-start, time.process_time()-cpu
    # Decoding/serialization verification is outside both stage timers.
    assert all(tok.decode(tok.encode(text)) == text for text in texts)
    result = dict(status="complete", profiled=profiled, legacy_count=legacy_count, documents=len(texts), characters=chars,
        utf8_bytes=raw, initial_byte_tokens=raw, final_encoded_tokens=encoded,
        corpus_sha256=corpus_hash, target_vocab=vocab, actual_vocab=tok.vocab_size,
        merges_completed=len(tok.to_dict()["merges"]), tokenizer_identity=tok.identity,
        encoded_sha256=digest.hexdigest(), fit_wall_seconds=fit_wall, fit_cpu_seconds=fit_cpu,
        fit_cpu_percent_one_core=100*fit_cpu/fit_wall, fit_bytes_per_second=raw/fit_wall,
        encode_wall_seconds=encode_wall, encode_cpu_seconds=encode_cpu,
        memory_before=before, fit_memory=fit_memory, encode_memory=memory())
    if profiler:
        rows = []
        for (filename, line, name), (_, calls, own, cumulative, _) in pstats.Stats(profiler).stats.items():
            rows.append(dict(function=f"{filename}:{line}:{name}", calls=calls,
                             self_seconds=own, cumulative_seconds=cumulative))
        result["profile_top_self"] = sorted(rows, key=lambda x:x["self_seconds"], reverse=True)[:12]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--chars", type=int)
    parser.add_argument("--vocab", type=int)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--legacy-count", action="store_true")
    parser.add_argument("--matched", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        if (args.chars, args.vocab) not in CASES:
            raise ValueError("Unapproved diagnostic size")
        print(json.dumps(worker(args.chars, args.vocab, args.progress, args.profile, args.legacy_count)))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    cases = ([(c,v,False,old) for c,v in ((32768,1024), (65536,1024), (131072,1024),
              (65536,4096), (65536,8192)) for old in (True,False)] if args.matched else
             [(c,v,False,False) for c,v in CASES] + [(32768,1024,False,False), (32768,1024,True,False)])
    for i, (chars, vocab, profile, legacy) in enumerate(cases):
        progress = args.output / f"progress-{i}.json"
        command = [sys.executable, "-B", "-m", __spec__.name, "--worker", "--chars", str(chars),
                   "--vocab", str(vocab), "--progress", str(progress)]
        if profile:
            command.append("--profile")
        if legacy:
            command.append("--legacy-count")
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
            row = json.loads(run.stdout)
        except subprocess.TimeoutExpired:
            row = dict(status="timeout", characters=chars, target_vocab=vocab,
                       timeout_seconds=60, profiled=profile, legacy_count=legacy,
                       last_progress=json.loads(progress.read_text()) if progress.exists() else None)
        row["command"] = command
        rows.append(row)
        (args.output / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"{chars} chars / {vocab} vocab / profile={profile}: {row['status']}", flush=True)
    complete = [r for r in rows if r["status"] == "complete" and r["characters"] == 32768 and r["target_vocab"] == 1024]
    if complete:
        assert len({r["tokenizer_identity"] for r in complete}) == 1
        assert len({r["encoded_sha256"] for r in complete}) == 1
    if args.matched:
        for chars, vocab in {(r["characters"], r["target_vocab"]) for r in rows}:
            pair = [r for r in rows if r["characters"] == chars and r["target_vocab"] == vocab]
            if all(r["status"] == "complete" for r in pair):
                for key in ("corpus_sha256", "tokenizer_identity", "encoded_sha256", "merges_completed"):
                    assert pair[0][key] == pair[1][key], key


if __name__ == "__main__":
    main()
