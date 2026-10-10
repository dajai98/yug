"""Capped local pipeline/replay measurements; no downloads or model training."""
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

SIZES = (64, 512, 2048)


def fixture(size):
    from training.corpus_prep import RawDocument
    base = local_documents(32768)
    for i in range(size):
        text = base[i % len(base)].text + f"\nDiagnostic unique document {i}."
        yield RawDocument(str(i), text, "local-surrogate", "repository text fixture",
                          "DIAGNOSTIC", "mixed", "diagnostic")


def build(path, size, resume=False):
    from training.corpus_prep import LicenseDecision, LicensePolicy
    from training.production_corpus import PreparedCorpus, identity, prepare_stream, read_json
    from training.token_shards import build_token_shards
    from yug.tokenizer.bpe import ByteLevelBpeTokenizer
    tok = ByteLevelBpeTokenizer.train([d.text for d in local_documents(32768)], 270)
    before = memory()
    start = time.perf_counter()
    if resume:
        pm = read_json(path / "prepared/manifest.json")
        prep = PreparedCorpus(path / "prepared", expected_identity=pm["prepared_corpus_identity"])
    else:
        prep = prepare_stream(fixture(size), path / "prepared",
            license_policy=LicensePolicy({"DIAGNOSTIC": LicenseDecision.ALLOW}),
            source_identity=identity(dict(fixture="local-surrogate-v1", documents=size)),
            sources={"local-surrogate": {"revision": "local snapshot; see gate source hashes"}},
            tokenizer_identity=tok.identity, val_fraction=.2, seed=1234)
    preparation_seconds = time.perf_counter() - start
    prepared_memory = memory()
    start = time.perf_counter()
    calls = []
    encode = tok.encode
    def counted(text):
        calls.append(len(text))  # diagnostic metadata only, no payload retention
        return encode(text)
    tok.encode = counted
    manifest = build_token_shards(prep, tok, path / "tokens")
    build_seconds = time.perf_counter() - start
    tok.save(path / "tokenizer.json")
    return dict(mode="resume" if resume else "build", documents=size, preparation_seconds=preparation_seconds,
                token_build_seconds=build_seconds, encode_calls=len(calls),
                memory_before=before, memory_after_preparation=prepared_memory,
                memory_after=memory(), encoded_tokens=manifest["total_token_count"],
                shard_bytes=sum(p.stat().st_size for p in (path / "tokens").glob("*.bin")),
                shard_count=manifest["shard_count"],
                manifest_bytes=(path / "tokens/manifest.json").stat().st_size,
                spool_bytes=sum(p.stat().st_size for p in (path / "tokens").glob("*.work")),
                prepared_payload_bytes=sum((path / "prepared" / name).stat().st_size
                                          for name in ("accepted.jsonl", "rejected.jsonl")),
                prepared_index_bytes=(path / "prepared/index.json").stat().st_size)


def replay(path, mode, size):
    import torch
    from training.dataset import DocumentSequences, iter_lazy_batches
    from training.production_corpus import PreparedCorpus, read_json
    from training.token_shards import TokenShardStream
    from yug.tokenizer.bpe import ByteLevelBpeTokenizer
    torch.set_num_threads(1)
    pm = read_json(path / "prepared/manifest.json")
    before = memory()
    started = time.perf_counter()
    calls = 0
    if mode == "documents":
        prep = PreparedCorpus(path / "prepared", expected_identity=pm["prepared_corpus_identity"])
        tok = ByteLevelBpeTokenizer.load(path / "tokenizer.json")
        encode = tok.encode
        def counted(text):
            nonlocal calls
            calls += 1
            return encode(text)
        tok.encode = counted
        sequences = DocumentSequences(tok, lambda: prep.documents("train"), 128)
    else:
        replay = TokenShardStream(path / "tokens", split="train",
                    tokenizer_identity=pm["tokenizer_identity"],
                    prepared_corpus_identity=pm["prepared_corpus_identity"])
        sequences = replay.sequences(128)
    open_seconds = time.perf_counter() - started
    digest = hashlib.sha256()
    start = time.perf_counter()
    count = batches = 0
    for x, y in iter_lazy_batches(sequences, 8):
        for tensor in (x, y):
            values = array("q", tensor.reshape(-1).tolist())
            if sys.byteorder != "little":
                values.byteswap()
            digest.update(values.tobytes())
        count += x.shape[0]
        batches += 1
    elapsed = time.perf_counter() - start
    m = read_json(path / "tokens/manifest.json")
    tokens = m["splits"]["train"]["total_token_count"]
    return dict(mode=mode, documents=size, open_validation_seconds=open_seconds,
                replay_seconds=elapsed, encoded_tokens=tokens, sequences=count, batches=batches,
                tokens_per_second=tokens / elapsed, encode_calls=calls,
                value_sha256=digest.hexdigest(), memory_before=before, memory_after=memory())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("build", "resume", "documents", "shards"))
    parser.add_argument("--size", type=int)
    parser.add_argument("--path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.size not in SIZES:
            raise ValueError("Diagnostic cap exceeded")
        result = (build(args.path, args.size, resume=args.worker == "resume")
                  if args.worker in ("build", "resume") else replay(args.path, args.worker, args.size))
        print(json.dumps(result))
        return
    if args.output is None:
        parser.error("--output is required")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for size in SIZES:
        case = args.output / f"case-{size}"
        for repeat, modes in ((-1, ("build",)), (0, ("documents", "shards")), (1, ("documents", "shards"))):
            pair = []
            for mode in modes:
                command = [sys.executable, "-B", "-m", __spec__.name,
                           "--worker", mode, "--size", str(size), "--path", str(case)]
                run = subprocess.run(command, text=True, capture_output=True, timeout=180)
                if run.returncode:
                    (args.output / "failure.json").write_text(json.dumps(dict(
                        command=command, returncode=run.returncode, stderr=run.stderr), indent=2), encoding="utf-8")
                    raise RuntimeError(run.stderr)
                row = json.loads(run.stdout)
                row.update(repeat=repeat, command=command)
                rows.append(row)
                pair.append(row)
                (args.output / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
                print(f"completed {size} {repeat} {mode}", flush=True)
            if repeat >= 0:
                assert pair[0]["value_sha256"] == pair[1]["value_sha256"]
                assert pair[1]["encode_calls"] == 0


if __name__ == "__main__":
    main()
