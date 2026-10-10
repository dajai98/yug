"""Tests for the bounded Common Pile acquisition boundary in
training.experiments.real_corpus_experiment.

VERIFICATION STATUS: every test here is torch-free and network-free.
The stream is an in-memory gzip blob injected through ``open_stream``
(or a patched ``urllib.request.urlopen``), so the REAL decompression,
JSON-lines parsing, mapping, stopping, and error paths execute; only
the HTTP transport itself is replaced. No live request to Hugging Face
is made by any test. The live read has not been verified by the
author of this change.

The records below are invented placeholder sentences shaped like the
schema read from the official dataset card (top-level id/text/source/
created, metadata.license/namespace/title/url/provenance/wiki/
authors). They are NOT real Wikimedia text.
"""

from __future__ import annotations

import gzip
import io
import json
import random
import urllib.error
import urllib.request

import pytest

from training.corpus_prep import AcceptedDocument, LicenseDecision, prepare_corpus
from training.errors import RealCorpusAcquisitionError
from training.experiments.real_corpus_experiment import (
    COMMON_PILE_DATASET_ID,
    COMMON_PILE_DATASET_REVISION,
    REAL_CORPUS_LICENSE_POLICY,
    TOKENIZER_VOCAB_SIZE,
    VAL_FRACTION,
    WIKIMEDIA_LICENSE,
    WIKIMEDIA_LICENSE_SOURCE_STRING,
    acquire_common_pile_wikimedia_sample,
    load_common_pile_wikimedia_sample,
    train_tokenizer_on_train_partition,
)

_SOURCE = "wiki/dump/wikibooks.com"


def _record(i: int, *, namespace: str = "0", license_: str | None = WIKIMEDIA_LICENSE_SOURCE_STRING, text: str | None = None) -> dict:
    metadata: dict = {
        "authors": [["placeholder-editor", str(1000 + i)]],
        "namespace": namespace,
        "provenance": f"wikimedia-0000.json.gz:{i}",
        "title": f"Placeholder Title {i}",
        "url": f"https://example.invalid/wiki/Placeholder_{i}",
        "wiki": "wikibooks.com",
    }
    if license_ is not None:
        metadata["license"] = license_
    return {
        "added": "2024-07-27T16:40:57.324173",
        "created": "2021-04-25T19:00:03",
        "id": f"{namespace}-{i}",
        "metadata": metadata,
        "source": _SOURCE,
        "text": text
        if text is not None
        else f"Invented placeholder paragraph {i} describing an imaginary subject at some length for testing.",
    }


def _jsonl(records: list) -> bytes:
    return ("\n".join(json.dumps(r) for r in records) + "\n").encode("utf-8")


def _gz(records: list) -> bytes:
    return gzip.compress(_jsonl(records))


class _FakeServer:
    """Serves in-memory blobs by URL and records every URL requested."""

    def __init__(self, blobs: dict[str, bytes]):
        self.blobs = blobs
        self.requested: list[str] = []

    def __call__(self, url: str):
        self.requested.append(url)
        return io.BytesIO(self.blobs[url])


def _shard_url(index: int) -> str:
    return (
        f"https://huggingface.co/datasets/{COMMON_PILE_DATASET_ID}/resolve/"
        f"{COMMON_PILE_DATASET_REVISION}/wikimedia-{index:04d}.json.gz"
    )


def _single_shard(records: list) -> _FakeServer:
    return _FakeServer({_shard_url(0): _gz(records)})


# ---------------------------------------------------------------------------
# Schema mapping, provenance, license
# ---------------------------------------------------------------------------


def test_record_maps_to_raw_document_fields():
    docs, _ = acquire_common_pile_wikimedia_sample(1, open_stream=_single_shard([_record(7)]))
    doc = docs[0]
    assert doc.document_id == f"{_SOURCE}:0-7"
    assert doc.text == _record(7)["text"]  # verbatim: normalization is corpus_prep's job
    assert doc.source_id == _SOURCE
    assert doc.source_name == f"{COMMON_PILE_DATASET_ID}@{COMMON_PILE_DATASET_REVISION[:12]}"
    assert doc.language == "en"
    assert doc.primary_domain == "wikibooks.com"
    assert doc.license == WIKIMEDIA_LICENSE


def test_real_source_license_string_maps_exactly_to_yug_license_id():
    assert WIKIMEDIA_LICENSE_SOURCE_STRING == (
        "Creative Commons - Attribution Share-Alike - https://creativecommons.org/licenses/by-sa/4.0/"
    )
    assert REAL_CORPUS_LICENSE_POLICY.decide(WIKIMEDIA_LICENSE) is LicenseDecision.ALLOW


def test_provenance_is_preserved_per_document_and_only_from_real_fields():
    docs, report = acquire_common_pile_wikimedia_sample(1, open_stream=_single_shard([_record(7)]))
    prov = report.provenance[docs[0].document_id]
    assert prov["dataset"] == COMMON_PILE_DATASET_ID
    assert prov["revision"] == COMMON_PILE_DATASET_REVISION
    assert prov["shard_file"] == "wikimedia-0000.json.gz"
    assert prov["record_id"] == "0-7"
    assert prov["title"] == "Placeholder Title 7"
    assert prov["url"] == "https://example.invalid/wiki/Placeholder_7"
    assert prov["provenance_locator"] == "wikimedia-0000.json.gz:7"
    assert prov["wiki"] == "wikibooks.com"
    assert prov["namespace"] == "0"
    assert prov["authors"] == [["placeholder-editor", "1007"]]
    assert prov["created"] == "2021-04-25T19:00:03"
    assert prov["license_source_string"] == WIKIMEDIA_LICENSE_SOURCE_STRING
    assert prov["license_yug_id"] == WIKIMEDIA_LICENSE


def test_absent_optional_metadata_is_omitted_not_fabricated():
    rec = _record(3)
    for key in ("title", "url", "provenance", "wiki", "authors"):
        del rec["metadata"][key]
    del rec["created"]
    docs, report = acquire_common_pile_wikimedia_sample(1, open_stream=_single_shard([rec]))
    prov = report.provenance[docs[0].document_id]
    for key in ("title", "url", "provenance_locator", "wiki", "authors", "created"):
        assert key not in prov


def test_unrecognized_license_passes_verbatim_and_gate_quarantines_it():
    records = [_record(1), _record(2, license_="Some Other License v9")]
    docs, _ = acquire_common_pile_wikimedia_sample(2, open_stream=_single_shard(records))
    assert docs[1].license == "Some Other License v9"  # not relabelled

    result = prepare_corpus(docs, license_policy=REAL_CORPUS_LICENSE_POLICY, val_fraction=0.5, seed=1)
    assert [d.document_id for d in result.review] == [docs[1].document_id]
    assert docs[1].document_id not in {d.document_id for d in result.accepted}


def test_record_without_license_is_skipped_and_counted_never_invented():
    records = [_record(1), _record(2, license_=None), _record(3)]
    docs, report = acquire_common_pile_wikimedia_sample(2, open_stream=_single_shard(records))
    assert [d.document_id for d in docs] == [f"{_SOURCE}:0-1", f"{_SOURCE}:0-3"]
    assert report.skipped == {"missing_license": 1}


def test_too_many_missing_licenses_refuses_rather_than_bypassing_gate():
    records = [_record(i, license_=None) for i in range(1, 8)] + [_record(99)]
    with pytest.raises(RealCorpusAcquisitionError) as excinfo:
        acquire_common_pile_wikimedia_sample(1, max_unusable_records=3, open_stream=_single_shard(records))
    assert "license" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Selection: namespace filter, duplicates, malformed records
# ---------------------------------------------------------------------------


def test_main_namespace_only_by_default():
    records = [_record(1), _record(2, namespace="2"), _record(3, namespace="1"), _record(4)]
    docs, report = acquire_common_pile_wikimedia_sample(2, open_stream=_single_shard(records))
    assert [d.document_id for d in docs] == [f"{_SOURCE}:0-1", f"{_SOURCE}:0-4"]
    assert report.skipped == {"wrong_namespace": 2}
    assert report.main_namespace_only is True


def test_namespace_filter_can_be_disabled():
    records = [_record(1), _record(2, namespace="2")]
    docs, report = acquire_common_pile_wikimedia_sample(
        2, main_namespace_only=False, open_stream=_single_shard(records)
    )
    assert len(docs) == 2
    assert report.skipped == {}


def test_duplicate_document_ids_are_skipped_first_wins():
    first, dup = _record(1), _record(1)
    dup["text"] = "A different body that reuses the same identifier and must be skipped as a duplicate id."
    docs, report = acquire_common_pile_wikimedia_sample(2, open_stream=_single_shard([first, dup, _record(2)]))
    assert [d.document_id for d in docs] == [f"{_SOURCE}:0-1", f"{_SOURCE}:0-2"]
    assert docs[0].text == first["text"]
    assert report.skipped == {"duplicate_id": 1}


def test_malformed_records_are_skipped_and_counted():
    bad = [
        ["not", "a", "dict"],
        {"id": "0-1", "source": _SOURCE, "metadata": {"namespace": "0", "license": "x"}},  # no text
        {"id": "", "text": "t", "source": _SOURCE, "metadata": {"namespace": "0", "license": "x"}},
        {"id": "0-2", "text": "t", "source": _SOURCE, "metadata": "not-a-dict"},
        {"id": "0-3", "text": "t", "source": _SOURCE, "metadata": {"license": "x"}},  # no namespace
    ]
    docs, report = acquire_common_pile_wikimedia_sample(1, open_stream=_single_shard(bad + [_record(9)]))
    assert len(docs) == 1
    assert report.skipped == {"malformed": 5}


def test_schema_mismatch_beyond_tolerance_raises():
    bad = [{"unexpected": i} for i in range(20)] + [_record(1)]
    with pytest.raises(RealCorpusAcquisitionError):
        acquire_common_pile_wikimedia_sample(1, max_unusable_records=5, open_stream=_single_shard(bad))


def test_invalid_json_line_raises_clearly():
    blob = gzip.compress(_jsonl([_record(1)]) + b"{this is not json\n")
    server = _FakeServer({_shard_url(0): blob})
    with pytest.raises(RealCorpusAcquisitionError) as excinfo:
        acquire_common_pile_wikimedia_sample(2, open_stream=server)
    assert "not valid JSON" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Bounded behaviour
# ---------------------------------------------------------------------------


def test_stops_at_max_documents_without_reading_further():
    # A poisoned (invalid-JSON) line follows the 5th valid record. If
    # the reader looked one line past the bound, this would raise.
    blob = gzip.compress(_jsonl([_record(i) for i in range(1, 6)]) + b"{POISON: reading past the bound\n")
    docs, report = acquire_common_pile_wikimedia_sample(5, open_stream=_FakeServer({_shard_url(0): blob}))
    assert len(docs) == 5
    assert report.records_scanned == 5
    assert report.records_selected == 5


def test_compressed_bytes_read_is_a_small_fraction_of_a_large_shard():
    rng = random.Random(0)
    words = [f"w{rng.randrange(10**6)}" for _ in range(5000)]  # poorly compressible
    records = [
        _record(i, text=" ".join(rng.choice(words) for _ in range(40))) for i in range(1, 20001)
    ]
    blob = _gz(records)
    assert len(blob) > 1_000_000  # a genuinely large stream for this test

    docs, report = acquire_common_pile_wikimedia_sample(10, open_stream=_FakeServer({_shard_url(0): blob}))
    assert len(docs) == 10
    assert report.compressed_bytes_read < len(blob) * 0.05


def test_scan_cap_raises_instead_of_reading_unboundedly():
    records = [_record(i, namespace="2") for i in range(1, 50)]  # nothing qualifies
    with pytest.raises(RealCorpusAcquisitionError) as excinfo:
        acquire_common_pile_wikimedia_sample(1, max_records_scanned=10, open_stream=_single_shard(records))
    assert "scan cap" in str(excinfo.value)


def test_exhausted_shard_with_too_few_documents_raises_with_counts():
    with pytest.raises(RealCorpusAcquisitionError) as excinfo:
        acquire_common_pile_wikimedia_sample(10, open_stream=_single_shard([_record(1), _record(2)]))
    message = str(excinfo.value)
    assert "2 of 10" in message


# ---------------------------------------------------------------------------
# Pinned revision, shard selection, determinism
# ---------------------------------------------------------------------------


def test_requests_pinned_revision_url_for_shard_zero_by_default():
    server = _single_shard([_record(1)])
    acquire_common_pile_wikimedia_sample(1, open_stream=server)
    assert server.requested == [_shard_url(0)]
    assert COMMON_PILE_DATASET_REVISION in server.requested[0]
    assert len(COMMON_PILE_DATASET_REVISION) == 40


def test_second_shard_is_opened_only_when_the_first_is_insufficient():
    server = _FakeServer(
        {
            _shard_url(0): _gz([_record(1), _record(2)]),
            _shard_url(1): _gz([_record(3), _record(4)]),
        }
    )
    docs, report = acquire_common_pile_wikimedia_sample(3, shard_indices=(0, 1), open_stream=server)
    assert server.requested == [_shard_url(0), _shard_url(1)]
    assert [d.document_id for d in docs] == [f"{_SOURCE}:0-{n}" for n in (1, 2, 3)]
    assert report.shard_files == ("wikimedia-0000.json.gz", "wikimedia-0001.json.gz")

    server2 = _FakeServer({_shard_url(0): _gz([_record(1), _record(2)])})
    acquire_common_pile_wikimedia_sample(2, shard_indices=(0, 1), open_stream=server2)
    assert server2.requested == [_shard_url(0)]  # shard 1 never touched


def test_acquisition_is_deterministic():
    records = [_record(i) for i in range(1, 30)]
    docs1, rep1 = acquire_common_pile_wikimedia_sample(12, open_stream=_single_shard(records))
    docs2, rep2 = acquire_common_pile_wikimedia_sample(12, open_stream=_single_shard(records))
    assert docs1 == docs2
    assert rep1.to_dict() == rep2.to_dict()
    assert json.dumps(rep1.to_dict(), sort_keys=True) == json.dumps(rep2.to_dict(), sort_keys=True)


# ---------------------------------------------------------------------------
# Failure handling: every failure surfaces as RealCorpusAcquisitionError
# ---------------------------------------------------------------------------


def test_invalid_arguments_are_rejected():
    server = _single_shard([_record(1)])
    with pytest.raises(RealCorpusAcquisitionError):
        acquire_common_pile_wikimedia_sample(0, open_stream=server)
    with pytest.raises(RealCorpusAcquisitionError):
        acquire_common_pile_wikimedia_sample(1, shard_indices=(), open_stream=server)
    with pytest.raises(RealCorpusAcquisitionError):
        acquire_common_pile_wikimedia_sample(1, shard_indices=(35,), open_stream=server)
    with pytest.raises(RealCorpusAcquisitionError):
        acquire_common_pile_wikimedia_sample(1, shard_indices=(-1,), open_stream=server)
    assert server.requested == []  # validation happens before any I/O


def test_network_failure_is_wrapped():
    def failing_opener(url):
        raise urllib.error.URLError("simulated network outage")

    with pytest.raises(RealCorpusAcquisitionError) as excinfo:
        acquire_common_pile_wikimedia_sample(1, open_stream=failing_opener)
    assert "simulated network outage" in str(excinfo.value)


def test_http_error_status_is_reported():
    def forbidden(url):
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)

    with pytest.raises(RealCorpusAcquisitionError) as excinfo:
        acquire_common_pile_wikimedia_sample(1, open_stream=forbidden)
    assert "HTTP 403" in str(excinfo.value)


def test_non_gzip_response_is_rejected():
    server = _FakeServer({_shard_url(0): b"<html>this is an error page, not gzip</html>"})
    with pytest.raises(RealCorpusAcquisitionError):
        acquire_common_pile_wikimedia_sample(1, open_stream=server)


def test_truncated_gzip_stream_is_rejected():
    blob = _gz([_record(i) for i in range(1, 6)])
    server = _FakeServer({_shard_url(0): blob[:-6]})  # trailer cut off
    with pytest.raises(RealCorpusAcquisitionError):
        acquire_common_pile_wikimedia_sample(50, open_stream=server)  # needs to read to EOF


def test_invalid_utf8_is_rejected():
    blob = gzip.compress(b'{"id": "0-1", "text": "\xff\xfe broken"}\n')
    with pytest.raises(RealCorpusAcquisitionError):
        acquire_common_pile_wikimedia_sample(1, open_stream=_FakeServer({_shard_url(0): blob}))


def test_default_opener_issues_a_pinned_urllib_request(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["user_agent"] = request.get_header("User-agent")
        captured["timeout"] = timeout
        return io.BytesIO(_gz([_record(1)]))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    docs, _ = acquire_common_pile_wikimedia_sample(1)  # no open_stream: default opener path
    assert len(docs) == 1
    assert captured["url"] == _shard_url(0)
    assert captured["user_agent"]
    assert captured["timeout"] and captured["timeout"] > 0


def test_default_opener_network_failure_is_wrapped(monkeypatch):
    def offline(request, timeout=None):
        raise urllib.error.URLError("no route to host (simulated)")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    with pytest.raises(RealCorpusAcquisitionError):
        acquire_common_pile_wikimedia_sample(1)


def test_load_wrapper_returns_documents_only(monkeypatch):
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout=None: io.BytesIO(_gz([_record(1), _record(2)]))
    )
    docs = load_common_pile_wikimedia_sample(2)
    assert isinstance(docs, list)
    assert len(docs) == 2


# ---------------------------------------------------------------------------
# Acquisition -> corpus_prep integration (existing gate, unchanged)
# ---------------------------------------------------------------------------


def test_acquired_documents_flow_through_corpus_prep_with_provenance_intact():
    records = [_record(i) for i in range(1, 31)]
    records.append(_record(31, text=records[0]["text"]))  # exact duplicate text of record 1
    records.append(_record(32, license_="Unlisted Licence"))
    records.append(_record(33, namespace="2"))
    docs, report = acquire_common_pile_wikimedia_sample(32, open_stream=_single_shard(records))

    result = prepare_corpus(docs, license_policy=REAL_CORPUS_LICENSE_POLICY, val_fraction=VAL_FRACTION, seed=1234)

    assert result.manifest.input_count == 32
    assert result.manifest.duplicate_count == 1
    assert result.manifest.review_count == 1
    assert result.manifest.accepted_count == 30
    assert result.manifest.license_distribution == {WIKIMEDIA_LICENSE: 31, "Unlisted Licence": 1}
    assert result.manifest.source_distribution == {_SOURCE: 32}
    assert result.manifest.domain_distribution == {"wikibooks.com": 32}
    assert result.manifest.language_distribution == {"en": 32}

    train_ids = {d.document_id for d in result.train}
    val_ids = {d.document_id for d in result.validation}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {d.document_id for d in result.accepted}
    # every accepted document traces back to real acquisition provenance
    assert {d.document_id for d in result.accepted} <= set(report.provenance)
    for doc in result.accepted:
        assert doc.license == WIKIMEDIA_LICENSE
        assert doc.source_id == _SOURCE


# ---------------------------------------------------------------------------
# Tokenizer trains on the TRAIN partition only
# ---------------------------------------------------------------------------


def _accepted(doc_id: str, text: str) -> AcceptedDocument:
    return AcceptedDocument(
        document_id=doc_id, text=text, source_id=_SOURCE, source_name="test",
        license=WIKIMEDIA_LICENSE, language="en", primary_domain="wikibooks.com", content_hash=doc_id,
    )


class _Partition:
    def __init__(self, train, validation):
        self.train = train
        self.validation = validation


def test_validation_documents_do_not_participate_in_tokenizer_training(monkeypatch):
    from yug.tokenizer.bpe import ByteLevelBpeTokenizer

    train_docs = [
        _accepted("t1", "the quick brown fox jumps over the lazy dog near the quiet river bank"),
        _accepted("t2", "a gentle rain fell across the old town square just before midnight"),
    ]
    val_docs = [_accepted("v1", "zqxj " * 200)]  # a pattern that would create new merges if it leaked in
    partition = _Partition(train_docs, val_docs)

    train_texts = [d.text for d in train_docs]
    expected_train_only = ByteLevelBpeTokenizer.train(
        corpus=train_texts, vocab_size=TOKENIZER_VOCAB_SIZE, special_tokens=("bos", "eos")
    )
    leaked = ByteLevelBpeTokenizer.train(
        corpus=train_texts + [d.text for d in val_docs],
        vocab_size=TOKENIZER_VOCAB_SIZE,
        special_tokens=("bos", "eos"),
    )
    assert expected_train_only.identity != leaked.identity  # the sentinel really would change the tokenizer

    seen = {}
    original_train = ByteLevelBpeTokenizer.train

    def spy(corpus, vocab_size, special_tokens=("bos", "eos")):
        seen["corpus"] = list(corpus)
        return original_train(corpus=corpus, vocab_size=vocab_size, special_tokens=special_tokens)

    monkeypatch.setattr(ByteLevelBpeTokenizer, "train", spy)
    tokenizer = train_tokenizer_on_train_partition(partition)

    assert seen["corpus"] == train_texts
    assert not any("zqxj" in text for text in seen["corpus"])
    assert tokenizer.identity == expected_train_only.identity
    assert tokenizer.identity != leaked.identity


def test_tokenizer_helper_reports_actual_vocab_not_the_requested_target():
    docs = [_accepted("t1", "short training text with few distinct byte pairs to merge here")]
    tokenizer = train_tokenizer_on_train_partition(_Partition(docs, []))
    # A tiny corpus cannot reach the 384 target; the ACTUAL value must be
    # what is observable, so callers report it rather than assuming 384.
    assert tokenizer.vocab_size < TOKENIZER_VOCAB_SIZE
