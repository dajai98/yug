"""Local diagnostic only: fresh-process eager/bounded comparison, no training.

Text cases reuse <=1024-character LOCAL SURROGATE documents, not corpus data.
Token cases isolate materialization with deterministic generated token IDs.
No benchmark output is a source manifest or a production token shard.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from training.experiments.preprocessing_scalability_probe import local_documents, memory


def worker(kind, size, mode):
    import torch
    from training.dataset import (encode_corpus, iter_batches, iter_encoded_tokens,
                                  iter_lazy_batches, iter_sequences, make_sequences)
    from yug.tokenizer.bpe import ByteLevelBpeTokenizer
    torch.set_num_threads(1)
    if kind == "text":
        if size not in (1, 8, 32):
            raise ValueError("Unsafe text scale")
        texts = [d.text for d in local_documents(32768)]
        tok = ByteLevelBpeTokenizer.train(texts, 384)
        source = lambda: (text for _ in range(size) for text in texts)
        source_bytes = sum(len(t.encode("utf-8")) for t in texts) * size
        chars = sum(map(len, texts)) * size
        docs = len(texts) * size
        identity = tok.identity
    else:
        if size not in (250000, 1000000, 2000000):
            raise ValueError("Unsafe token scale")
        source = lambda: (i % 4096 for i in range(size))
        source_bytes = chars = docs = identity = None
    start_memory = memory()
    timings = {}
    started = time.perf_counter()
    encoded = 0
    if mode == "eager":
        ids = encode_corpus(tok, list(source())) if kind == "text" else list(source())
        encoded = len(ids)
        timings["encode_or_generate_seconds"] = time.perf_counter() - started
        mark = time.perf_counter()
        sequences = make_sequences(ids, 128)
        timings["windows_seconds"] = time.perf_counter() - mark
        mark = time.perf_counter()
        batches = iter_batches(sequences, 8)
        timings["batches_seconds"] = time.perf_counter() - mark
    else:
        tokens = iter_encoded_tokens(tok, source()) if kind == "text" else source()
        def counted():
            nonlocal encoded
            for token in tokens:
                encoded += 1
                yield token
        batches = iter_lazy_batches(iter_sequences(counted(), 128), 8)
    batch_count = sequence_count = 0
    digest = hashlib.sha256()
    for x, y in batches:
        # Same bounded checksum work in both modes, included in total timing.
        for tensor in (x, y):
            values = array("q", tensor.reshape(-1).tolist())
            if sys.byteorder != "little":
                values.byteswap()
            digest.update(values.tobytes())
        batch_count += 1
        sequence_count += x.shape[0]
    elapsed = time.perf_counter() - started
    return dict(kind=kind, size=size, mode=mode, source_bytes=source_bytes,
                characters=chars, documents=docs, tokenizer_identity=identity,
                encoded_tokens=encoded, sequences=sequence_count, batches=batch_count,
                scored_targets=sequence_count * 128, tail_tokens=encoded % 129,
                value_sha256=digest.hexdigest(), wall_seconds=elapsed,
                tokens_per_second=encoded / elapsed, stages=timings,
                memory_before=start_memory, memory_after=memory())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("text", "tokens"))
    parser.add_argument("--size", type=int)
    parser.add_argument("--mode", choices=("eager", "bounded"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.size, args.mode)))
        return
    if args.output is None:
        parser.error("--output is required")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for repeat in range(2):
        for kind, sizes in (("text", (1, 8, 32)), ("tokens", (250000, 1000000, 2000000))):
            for size in sizes:
                pair = []
                for mode in ("eager", "bounded"):
                    command = [sys.executable, "-B", "-m", __spec__.name, "--worker", kind,
                               "--size", str(size), "--mode", mode]
                    run = subprocess.run(command, capture_output=True, text=True,
                                         check=True, timeout=90)
                    row = json.loads(run.stdout)
                    row.update(repeat=repeat, command=command)
                    rows.append(row)
                    pair.append(row)
                    print(f"{repeat} {kind} {size} {mode}: {row['wall_seconds']:.3f}s", flush=True)
                    (args.output / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
                for key in ("encoded_tokens", "sequences", "batches", "scored_targets",
                            "tail_tokens", "value_sha256", "tokenizer_identity"):
                    if pair[0][key] != pair[1][key]:
                        raise AssertionError(f"Eager/bounded mismatch: {key}")


if __name__ == "__main__":
    main()
