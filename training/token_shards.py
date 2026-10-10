"""Deterministic uint32-LE tokens, single-writer resumable build, verified replay.

No model/tokenizer implementation changes. No sequence boundaries on disk.
Temporary split spools enable document-boundary restart without re-encoding
committed documents. They are retained for recovery (disk overhead is explicit).
"""
from __future__ import annotations

from array import array
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import struct
import sys

from training.production_corpus import (PipelineError, atomic_json, atomic_replace, check_identity,
                                       file_digest, identity, read_json, require)

SCHEMA = "yug-token-shards-v1"
DEFAULT_SHARD_TOKENS = 1048576  # 4 MiB: bounded verification, ~1908 shards for 2B.
MAX_SHARD_TOKENS = 16777216  # Explicit 64 MiB maximum verified-shard buffer.


@contextmanager
def _writer_lock(directory):
    """OS advisory lock released on process exit, including interrupted builds."""
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    path = directory.parent / ("." + directory.name + ".writer.lock")
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise PipelineError("Token build directory already has a writer") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _publish_bytes(path, data):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    atomic_replace(temporary, path)


def _prefix_digest(handle, size):
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        block = handle.read(min(65536, remaining))
        require(bool(block), "Truncated committed token spool")
        digest.update(block)
        remaining -= len(block)
    return digest


def _pack(ids, vocab_size):
    for start in range(0, len(ids), 16384):
        block = ids[start:start + 16384]
        require(all(_integer(i) and i < vocab_size for i in block), "Invalid token ID")
        yield struct.pack("<" + "I" * len(block), *block)


def _spool(prepared, tokenizer, directory, split, request_identity):
    progress_path = directory / f"{split}.progress.json"
    spool = directory / f"{split}.tokens.work"
    total_docs = len(prepared.index["splits"][split])
    if progress_path.exists():
        progress = read_json(progress_path)
        check = dict(progress)
        stored = check.pop("identity", None)
        require(stored == identity(check), "Corrupt progress identity")
        require(progress.get("request_identity") == request_identity, "Incompatible progress")
        done, token_count = progress.get("documents"), progress.get("tokens")
        require(_integer(done) and done <= total_docs and _integer(token_count), "Invalid progress counts")
        require(spool.exists(), "Missing committed token spool")
    else:
        done, token_count = 0, 0
        progress = None
    # Uncommitted tail (including a crashed first document) is discarded only
    # within this task-owned spool, never a completed shard or another path.
    with spool.open("r+b" if spool.exists() else "w+b") as handle:
        digest = _prefix_digest(handle, token_count * 4)
        if progress:
            require("sha256:" + digest.hexdigest() == progress["checksum"], "Committed spool checksum mismatch")
        handle.truncate(token_count * 4)
        handle.seek(token_count * 4)
        for text in prepared.documents(split, start=done):
            require(tokenizer.identity == prepared.manifest["tokenizer_identity"], "Tokenizer changed during build")
            ids = tokenizer.encode(text)  # exactly once per newly committed document
            for data in _pack(ids, tokenizer.vocab_size):
                handle.write(data)
                digest.update(data)
            token_count += len(ids)
            del ids
            handle.flush()
            os.fsync(handle.fileno())
            done += 1
            progress = dict(request_identity=request_identity, documents=done, tokens=token_count,
                            checksum="sha256:" + digest.hexdigest())
            progress["identity"] = identity(progress)
            atomic_json(progress_path, progress)
    return spool, token_count, "sha256:" + digest.hexdigest()


def build_token_shards(prepared, tokenizer, directory, *, shard_tokens=DEFAULT_SHARD_TOKENS):
    """Serialize writers with an OS lock; resume only a matching verified job."""
    with _writer_lock(directory):
        return _build_token_shards(prepared, tokenizer, directory, shard_tokens=shard_tokens)


def _build_token_shards(prepared, tokenizer, directory, *, shard_tokens):
    """Build or resume only a matching job; final manifest publishes last.

    On a process interruption,
    at most the uncommitted document is re-encoded. Existing completed work is
    verified, never trusted solely because a filename exists. No production fit.
    """
    require(_integer(shard_tokens, 1) and shard_tokens <= MAX_SHARD_TOKENS, "Invalid shard size")
    require(_integer(tokenizer.vocab_size, 1) and tokenizer.vocab_size <= 2**32,
            "Vocabulary does not fit uint32")
    require(tokenizer.identity == prepared.manifest["tokenizer_identity"], "Tokenizer identity mismatch")
    request = dict(schema_version=SCHEMA, prepared_corpus_identity=prepared.identity,
                   tokenizer_identity=tokenizer.identity, vocab_size=tokenizer.vocab_size,
                   split_identity=prepared.manifest["split_identity"], shard_tokens=shard_tokens,
                   token_dtype="uint32", endianness="little",
                   tokenization_semantics="per-document-concat-no-separators-v1")
    directory = Path(directory)
    if directory.exists():
        require(read_json(directory / "request.json") == request, "Incompatible resume request")
    else:
        directory.mkdir(parents=True)
        atomic_json(directory / "request.json", request)
    if (directory / "manifest.json").exists():
        for split in ("train", "validation"):
            TokenShardStream(directory, split=split, tokenizer_identity=tokenizer.identity,
                             prepared_corpus_identity=prepared.identity)
        return read_json(directory / "manifest.json")
    splits = {}
    request_id = identity(request)
    for split in ("train", "validation"):
        spool, token_count, stream_id = _spool(prepared, tokenizer, directory, split, request_id)
        shards = []
        with spool.open("rb") as handle:
            index = 0
            while data := handle.read(shard_tokens * 4):
                name = f"{split}-{index:06d}.bin"
                checksum = "sha256:" + hashlib.sha256(data).hexdigest()
                path = directory / name
                if path.exists():
                    require(path.stat().st_size == len(data) and file_digest(path) == checksum,
                            "Existing completed shard is corrupt")
                else:
                    _publish_bytes(path, data)
                shards.append(dict(path=name, index=index, split=split, token_count=len(data) // 4,
                                   byte_size=len(data), checksum=checksum))
                index += 1
        splits[split] = dict(total_token_count=token_count, shard_count=len(shards),
                             ordered_token_stream_identity=stream_id, shards=shards)
    manifest = dict(request, request_identity=request_id, prepared_manifest=prepared.manifest,
                    splits=splits, total_token_count=sum(s["total_token_count"] for s in splits.values()),
                    shard_count=sum(s["shard_count"] for s in splits.values()))
    manifest["manifest_identity"] = identity(manifest)
    atomic_json(directory / "manifest.json", manifest)
    return manifest


class TokenShardStream:
    """Reusable tokens for ONE explicit split; no tokenizer object required.

    Validates all selected shards before returning from construction. Each pass
    verifies a bounded shard before yielding any of that shard's tokens; later
    modification therefore fails rather than silently changing data. Files must
    remain immutable. SHA256 is integrity checking, not publisher authentication.
    """

    def __init__(self, directory, *, split, tokenizer_identity, prepared_corpus_identity):
        self.directory = Path(directory).resolve()
        check_identity(tokenizer_identity)
        check_identity(prepared_corpus_identity)
        require(split in ("train", "validation"), "Unknown replay split")
        self.split = split
        self.manifest = read_json(self.directory / "manifest.json")
        m = self.manifest
        body = dict(m)
        stored = body.pop("manifest_identity", None)
        require(stored == identity(body), "Manifest checksum mismatch")
        require(m.get("schema_version") == SCHEMA, "Unsupported shard schema")
        require(m.get("tokenizer_identity") == tokenizer_identity, "Tokenizer identity mismatch")
        require(m.get("prepared_corpus_identity") == prepared_corpus_identity, "Prepared corpus identity mismatch")
        require(m.get("token_dtype") == "uint32" and m.get("endianness") == "little", "Unsupported token representation")
        require(m.get("tokenization_semantics") == "per-document-concat-no-separators-v1", "Wrong tokenization semantics")
        request_keys = ("schema_version", "prepared_corpus_identity", "tokenizer_identity",
                        "vocab_size", "split_identity", "shard_tokens", "token_dtype",
                        "endianness", "tokenization_semantics")
        require(m.get("request_identity") == identity({key: m.get(key) for key in request_keys}),
                "Request identity mismatch")
        require(_integer(m.get("vocab_size"), 1) and m["vocab_size"] <= 2**32, "Invalid vocabulary range")
        require(_integer(m.get("shard_tokens"), 1) and m["shard_tokens"] <= MAX_SHARD_TOKENS, "Invalid shard size")
        prepared = dict(m.get("prepared_manifest", {}))
        found = prepared.pop("prepared_corpus_identity", None)
        require(found == prepared_corpus_identity == identity(prepared), "Prepared provenance mismatch")
        require(prepared.get("tokenizer_identity") == tokenizer_identity and
                prepared.get("split_identity") == m.get("split_identity"), "Split/provenance identity mismatch")
        splits = m.get("splits", {})
        require(set(splits) == {"train", "validation"}, "Missing or mixed splits")
        for name, part in splits.items():
            shards = part.get("shards", [])
            require(_integer(part.get("shard_count")) and part["shard_count"] == len(shards), "Shard count mismatch")
            check_identity(part.get("ordered_token_stream_identity"))
            for i, entry in enumerate(shards):
                require(_integer(entry.get("index")) and entry["index"] == i and entry.get("split") == name and
                        entry.get("path") == f"{name}-{i:06d}.bin", "Wrong shard order/path/split")
                count = entry.get("token_count")
                require(_integer(count, 1) and count <= m["shard_tokens"], "Invalid shard token count")
                require(i == len(shards) - 1 or count == m["shard_tokens"], "Short interior shard")
                require(entry.get("byte_size") == count * 4, "Shard byte/token count mismatch")
                check_identity(entry.get("checksum"))
            require(part.get("total_token_count") == sum(s["token_count"] for s in shards), "Split token count mismatch")
        require(m.get("total_token_count") == sum(s["total_token_count"] for s in splits.values()), "Total token count mismatch")
        require(m.get("shard_count") == sum(s["shard_count"] for s in splits.values()), "Total shard count mismatch")
        self._part = splits[split]
        # Eager integrity preflight, bounded payload; no list(self).
        for _ in self:
            pass

    def __iter__(self):
        digest = hashlib.sha256()
        for shard in self._part["shards"]:
            path = self.directory / shard["path"]
            require(path.resolve().parent == self.directory, "Shard path escapes directory")
            try:
                with path.open("rb") as handle:
                    data = handle.read(shard["byte_size"] + 1)
            except OSError as exc:
                raise PipelineError(f"Missing/unreadable shard {path}") from exc
            require(len(data) == shard["byte_size"], "Truncated or oversized shard")
            require("sha256:" + hashlib.sha256(data).hexdigest() == shard["checksum"], "Shard checksum mismatch")
            digest.update(data)
            values = array("I")
            require(values.itemsize == 4, "Platform uint32 array is not 4 bytes")
            values.frombytes(data)
            if sys.byteorder != "little":
                values.byteswap()
            require(all(value < self.manifest["vocab_size"] for value in values), "Token outside vocabulary")
            yield from values
        require("sha256:" + digest.hexdigest() == self._part["ordered_token_stream_identity"], "Ordered token stream checksum mismatch")

    def sequences(self, sequence_length):
        return _ShardSequences(self, sequence_length)


class _ShardSequences:
    def __init__(self, tokens, length):
        self.tokens, self.length = tokens, length

    def __iter__(self):
        from training.dataset import iter_sequences
        return iter_sequences(self.tokens, self.length)
