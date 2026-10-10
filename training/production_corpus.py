"""Disk-backed preparation; legacy payload semantics, O(document-count) metadata.

Single writer, caller-provided ordered RawDocument stream. Source acquisition and
source caps remain upstream. No tokenizer fitting, near dedup or policy invention.
Preparation publishes its manifest last; incomplete preparation requires a fresh
directory. A completed prepared corpus is reusable by resumable tokenization.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import time

from training.corpus_prep import (LicenseDecision, QualityConfig, RejectionReason,
                                  check_quality, content_hash, normalize_text)
from training.errors import CorpusPrepError
from training.split import split_sentences

SCHEMA = "yug-prepared-corpus-v1"
SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")


class PipelineError(CorpusPrepError):
    """Invalid, incompatible, incomplete or corrupt preparation/replay data."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def identity(value):
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise PipelineError(message)


def check_identity(value):
    require(isinstance(value, str) and SHA.fullmatch(value), "Invalid SHA256 identity")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(canonical(value))
        handle.flush()
        os.fsync(handle.fileno())
    atomic_replace(temporary, path)


def atomic_replace(temporary, path):
    """Bounded retry for Windows sharing/access contention; never delete target.

    A failed attempt leaves the last committed file in place. Persistent access
    errors still propagate; no non-atomic copy fallback is permitted.
    """
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 7:
                raise
            time.sleep(min(.01 * 2**attempt, .1))


def read_json(path):
    try:
        return json.loads(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise PipelineError(f"Cannot read JSON {path}: {exc}") from exc


def file_digest(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(65536), b""):
                digest.update(block)
    except OSError as exc:
        raise PipelineError(f"Cannot read {path}: {exc}") from exc
    return "sha256:" + digest.hexdigest()


def _split(indices, fraction, seed):
    # split_sentences shuffles references without inspecting payload values.
    if len(indices) < 2:
        return list(indices), []
    return split_sentences(indices, fraction, seed)


def prepare_stream(documents, directory, *, license_policy, source_identity,
                   sources, tokenizer_identity, val_fraction, seed,
                   quality_config=None, text_normalizer=None,
                   normalization_identity=None, category_order=None):
    """Spool accepted and rejected payloads once; preserve default or mini split.

    sources maps every source_id to explicit version/revision/provenance metadata
    (use null when unavailable). Custom normalizers require a versioned identity.
    category_order selects the existing per-category split, not a new split rule.
    Callers must use a new directory; reopen finished work with PreparedCorpus.
    """
    check_identity(source_identity)
    check_identity(tokenizer_identity)
    require(isinstance(sources, dict) and bool(sources), "Missing source provenance")
    if text_normalizer is not None:
        require(isinstance(normalization_identity, str) and bool(normalization_identity),
                "Custom normalization requires an explicit semantics identity")
    else:
        require(normalization_identity is None, "Default normalization identity is fixed")
        normalization_identity = "training.corpus_prep.normalize_text-v1"
    quality = quality_config or QualityConfig()
    categories = None if category_order is None else list(category_order)
    if categories is not None:
        require(len(set(categories)) == len(categories), "Duplicate categories")
    policy = dict(license_decisions={k: v.value for k, v in license_policy.decisions.items()},
                  license_default=license_policy.default.value, quality=asdict(quality),
                  normalization=normalization_identity,
                  preparation="license-normalize-quality-exact-first-seen-v1",
                  split_method="category-stratified-v1" if categories is not None else
                  "training.split.split_sentences-v1", category_order=categories,
                  split_seed=seed, split_fraction=val_fraction)
    directory = Path(directory)
    require(not directory.exists(), "Preparation directory already exists; use PreparedCorpus to reopen completed work")
    directory.mkdir(parents=True)
    entries, seen = [], set()
    counts, reasons = Counter(), Counter()
    raw_digest = hashlib.sha256()
    with (directory / "accepted.jsonl").open("wb") as accepted, \
            (directory / "rejected.jsonl").open("wb") as rejected:
        for doc in documents:
            require(doc.source_id in sources, f"Missing source provenance: {doc.source_id}")
            raw = asdict(doc)
            raw_digest.update(canonical(raw) + b"\n")
            counts["input"] += 1
            decision = license_policy.decide(doc.license)
            normalized = None
            if decision is LicenseDecision.REJECT:
                reason = RejectionReason.LICENSE_REJECTED
            elif decision is LicenseDecision.REVIEW:
                reason = RejectionReason.LICENSE_REVIEW_REQUIRED
            else:
                normalized = normalize_text(doc.text) if text_normalizer is None else text_normalizer(doc)
                reason = check_quality(normalized, quality)
                if reason is None:
                    digest = content_hash(normalized)
                    if digest in seen:
                        reason = RejectionReason.EXACT_DUPLICATE
            if reason is not None:
                counts["review" if decision is LicenseDecision.REVIEW else "rejected"] += 1
                reasons[reason.value] += 1
                rejected.write(canonical(dict(document=raw, reason=reason.value)) + b"\n")
                continue
            seen.add(digest)
            record = dict(raw, text=normalized, content_hash=digest)
            data = canonical(record) + b"\n"
            entries.append(dict(offset=accepted.tell(), bytes=len(data),
                                identity=identity(record), domain=doc.primary_domain))
            accepted.write(data)
            counts["accepted"] += 1
        for handle in (accepted, rejected):
            handle.flush()
            os.fsync(handle.fileno())
    require(counts["input"] > 0, "documents must contain at least one entry")
    if categories is None:
        train, validation = _split(list(range(len(entries))), val_fraction, seed)
    else:
        require(all(e["domain"] in categories for e in entries), "Unmapped category")
        train, validation = [], []
        for cat in categories:
            a, b = _split([i for i, e in enumerate(entries) if e["domain"] == cat], val_fraction, seed)
            train.extend(a)
            validation.extend(b)
    index = dict(entries=entries, splits={"train": train, "validation": validation})
    atomic_json(directory / "index.json", index)
    split_identity = identity(dict(policy=policy, ordered_documents={
        split: [entries[i]["identity"] for i in indices]
        for split, indices in index["splits"].items()}))
    body = dict(schema_version=SCHEMA, source_identity=source_identity, sources=sources,
                ordered_input_identity="sha256:" + raw_digest.hexdigest(), policy=policy,
                filtering_policy_identity=identity(policy), split_identity=split_identity,
                tokenizer_identity=tokenizer_identity,
                tokenization_semantics="per-document-concat-no-separators-v1",
                shard_schema="yug-token-shards-v1", counts=dict(counts),
                rejection_reason_counts=dict(reasons),
                files={name: file_digest(directory / name) for name in
                       ("accepted.jsonl", "rejected.jsonl", "index.json")})
    body["prepared_corpus_identity"] = identity(body)
    atomic_json(directory / "manifest.json", body)
    return PreparedCorpus(directory, expected_identity=body["prepared_corpus_identity"])


class PreparedCorpus:
    """Verified on-disk documents with in-memory offsets and split metadata."""

    def __init__(self, directory, *, expected_identity):
        self.directory = Path(directory)
        check_identity(expected_identity)
        self.manifest = read_json(self.directory / "manifest.json")
        body = dict(self.manifest)
        found = body.pop("prepared_corpus_identity", None)
        require(found == expected_identity == identity(body), "Prepared corpus identity mismatch")
        require(body.get("schema_version") == SCHEMA, "Unsupported prepared schema")
        require(set(body.get("files", {})) == {"accepted.jsonl", "rejected.jsonl", "index.json"},
                "Invalid prepared files")
        for name, checksum in body["files"].items():
            require(file_digest(self.directory / name) == checksum, "Prepared file checksum mismatch")
        self.index = read_json(self.directory / "index.json")
        entries = self.index["entries"]
        splits = self.index["splits"]
        require(set(splits) == {"train", "validation"}, "Invalid prepared splits")
        joined = splits["train"] + splits["validation"]
        require(sorted(joined) == list(range(len(entries))), "Invalid split partition")
        self.identity = expected_identity

    def documents(self, split, start=0):
        require(split in ("train", "validation"), "Unknown split")
        require(type(start) is int and start >= 0, "Invalid document offset")
        with (self.directory / "accepted.jsonl").open("rb") as handle:
            for i in self.index["splits"][split][start:]:
                entry = self.index["entries"][i]
                handle.seek(entry["offset"])
                data = handle.read(entry["bytes"])
                try:
                    record = json.loads(data)
                except ValueError as exc:
                    raise PipelineError("Corrupt prepared document") from exc
                require(identity(record) == entry["identity"], "Prepared document identity mismatch")
                yield record["text"]
