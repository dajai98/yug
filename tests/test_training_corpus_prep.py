"""Tests for training.corpus_prep.

VERIFICATION STATUS: training.corpus_prep and training.split have no
PyTorch dependency, so every test in this file executes directly in
this environment -- except
``test_accepted_text_feeds_training_dataset_encode_corpus``, which
imports ``training.dataset`` (module-level ``import torch``) and is
gated with ``pytest.importorskip("torch")``, matching every other
torch-dependent test already in this suite. Exact pass/fail/skip
counts for a real run are reported alongside this task's other results,
not asserted here.
"""

from __future__ import annotations

import json

import pytest

from training.corpus_prep import (
    AcceptedDocument,
    CorpusPrepError,
    LicenseDecision,
    LicensePolicy,
    QualityConfig,
    RawDocument,
    RejectionReason,
    check_quality,
    normalize_text,
    prepare_corpus,
)

# ---------------------------------------------------------------------------
# Fixture corpus (Section 17): tiny, original, covers every listed case.
# ---------------------------------------------------------------------------

_LICENSE_POLICY = LicensePolicy(
    decisions={
        "cc-by-4.0": LicenseDecision.ALLOW,
        "proprietary-no-license": LicenseDecision.REJECT,
        # "unknown-license-xyz" deliberately absent -- exercises the
        # policy's default (REVIEW), i.e. the "unrecognized license"
        # case, not an explicitly-configured REVIEW entry.
    }
)


def _fixture_documents() -> list[RawDocument]:
    return [
        RawDocument(
            document_id="doc-001",
            text="The quick brown fox jumps over the lazy dog near the river bank today.",
            source_id="source-a",
            source_name="Source A Corpus",
            license="cc-by-4.0",
            language="en",
            primary_domain="news",
        ),
        RawDocument(
            # Whitespace-normalized duplicate of doc-001.
            document_id="doc-002",
            text="  The quick brown fox   jumps over the lazy dog near the river bank today.  \n",
            source_id="source-a",
            source_name="Source A Corpus",
            license="cc-by-4.0",
            language="en",
            primary_domain="news",
        ),
        RawDocument(
            document_id="doc-003",
            text="A gentle rain fell across the old town square just before midnight fell.",
            source_id="source-b",
            source_name="Source B Archive",
            license="cc-by-4.0",
            language="en",
            primary_domain="fiction",
        ),
        RawDocument(
            document_id="doc-004",
            text="This document contains proprietary text that cannot be used for training at all.",
            source_id="source-a",
            source_name="Source A Corpus",
            license="proprietary-no-license",
            language="en",
            primary_domain="news",
        ),
        RawDocument(
            document_id="doc-005",
            text="This document has an unrecognized license identifier that requires manual review.",
            source_id="source-b",
            source_name="Source B Archive",
            license="unknown-license-xyz",
            language="en",
            primary_domain="fiction",
        ),
        RawDocument(
            document_id="doc-006",
            text="   \n\t  ",
            source_id="source-a",
            source_name="Source A Corpus",
            license="cc-by-4.0",
            language="en",
            primary_domain="news",
        ),
        RawDocument(
            document_id="doc-007",
            text="Too short.",
            source_id="source-b",
            source_name="Source B Archive",
            license="cc-by-4.0",
            language="en",
            primary_domain="fiction",
        ),
        RawDocument(
            document_id="doc-008",
            text="spam spam spam spam spam spam spam spam",
            source_id="source-a",
            source_name="Source A Corpus",
            license="cc-by-4.0",
            language="en",
            primary_domain="news",
        ),
        RawDocument(
            document_id="doc-009",
            text="Historians have long debated the causes of the sudden economic shift that year.",
            source_id="source-b",
            source_name="Source B Archive",
            license="cc-by-4.0",
            language="en",
            primary_domain="news",
        ),
        RawDocument(
            document_id="doc-010",
            text="Deep within the forest, the travelers discovered a hidden path leading to the lake.",
            source_id="source-a",
            source_name="Source A Corpus",
            license="cc-by-4.0",
            language="en",
            primary_domain="fiction",
        ),
    ]


def _prepare(seed: int = 42, val_fraction: float = 0.25):
    return prepare_corpus(
        _fixture_documents(),
        license_policy=_LICENSE_POLICY,
        val_fraction=val_fraction,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_accepted_documents_retain_full_provenance():
    result = _prepare()
    by_id = {doc.document_id: doc for doc in result.accepted}

    doc1 = by_id["doc-001"]
    assert doc1.source_id == "source-a"
    assert doc1.source_name == "Source A Corpus"
    assert doc1.license == "cc-by-4.0"
    assert doc1.language == "en"
    assert doc1.primary_domain == "news"
    assert isinstance(doc1, AcceptedDocument)


# ---------------------------------------------------------------------------
# License policy
# ---------------------------------------------------------------------------


def test_allowed_license_document_is_accepted():
    result = _prepare()
    assert "doc-001" in {d.document_id for d in result.accepted}


def test_rejected_license_document_excluded_from_accepted_and_review():
    result = _prepare()
    accepted_ids = {d.document_id for d in result.accepted}
    review_ids = {d.document_id for d in result.review}
    assert "doc-004" not in accepted_ids
    assert "doc-004" not in review_ids

    rejected = {d.document_id: d for d in result.rejected}
    assert rejected["doc-004"].reason is RejectionReason.LICENSE_REJECTED


def test_unrecognized_license_defaults_to_review_not_accepted():
    result = _prepare()
    accepted_ids = {d.document_id for d in result.accepted}
    assert "doc-005" not in accepted_ids

    review = {d.document_id: d for d in result.review}
    assert "doc-005" in review
    assert review["doc-005"].reason is RejectionReason.LICENSE_REVIEW_REQUIRED


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalization_is_deterministic():
    text = "  hello\r\n   world  \t\n\n  again  "
    assert normalize_text(text) == normalize_text(text)


def test_normalization_collapses_whitespace_variants_to_same_string():
    a = "The quick brown fox jumps over the lazy dog near the river bank today."
    b = "  The quick brown fox   jumps over the lazy dog near the river bank today.  \n"
    assert normalize_text(a) == normalize_text(b)


# ---------------------------------------------------------------------------
# Exact deduplication
# ---------------------------------------------------------------------------


def test_whitespace_normalized_duplicate_is_rejected_not_accepted():
    result = _prepare()
    accepted_ids = {d.document_id for d in result.accepted}
    assert "doc-001" in accepted_ids  # first occurrence wins
    assert "doc-002" not in accepted_ids  # later duplicate rejected

    rejected = {d.document_id: d for d in result.rejected}
    assert rejected["doc-002"].reason is RejectionReason.EXACT_DUPLICATE


def test_manifest_duplicate_count_matches():
    result = _prepare()
    assert result.manifest.duplicate_count == 1


# ---------------------------------------------------------------------------
# Rejection-reason tracking
# ---------------------------------------------------------------------------


def test_every_rejected_and_review_document_has_an_explicit_reason():
    result = _prepare()
    for doc in result.rejected + result.review:
        assert isinstance(doc.reason, RejectionReason)


def test_quality_rejection_reasons_are_specific():
    result = _prepare()
    rejected = {d.document_id: d for d in result.rejected}
    assert rejected["doc-006"].reason is RejectionReason.EMPTY_AFTER_NORMALIZATION
    assert rejected["doc-007"].reason is RejectionReason.BELOW_MIN_LENGTH
    assert rejected["doc-008"].reason is RejectionReason.PATHOLOGICAL_REPETITION


@pytest.mark.parametrize(
    ("text", "expected_reason"),
    [
        ("", RejectionReason.EMPTY_AFTER_NORMALIZATION),
        ("   \n\t  ", RejectionReason.EMPTY_AFTER_NORMALIZATION),
        ("short", RejectionReason.BELOW_MIN_LENGTH),
        ("word word word word word word word", RejectionReason.PATHOLOGICAL_REPETITION),
    ],
)
def test_check_quality_unit_level(text, expected_reason):
    assert check_quality(normalize_text(text), QualityConfig()) is expected_reason


def test_check_quality_passes_good_text():
    good = "This is a perfectly ordinary sentence of reasonable length."
    assert check_quality(normalize_text(good), QualityConfig()) is None


# ---------------------------------------------------------------------------
# Deterministic split
# ---------------------------------------------------------------------------


def test_split_is_deterministic_for_same_input_and_config():
    result1 = _prepare(seed=42)
    result2 = _prepare(seed=42)
    ids1 = ([d.document_id for d in result1.train], [d.document_id for d in result1.validation])
    ids2 = ([d.document_id for d in result2.train], [d.document_id for d in result2.validation])
    assert ids1 == ids2


def test_split_differs_for_different_seed():
    result1 = _prepare(seed=1)
    result2 = _prepare(seed=2)
    ids1 = ([d.document_id for d in result1.train], [d.document_id for d in result1.validation])
    ids2 = ([d.document_id for d in result2.train], [d.document_id for d in result2.validation])
    assert ids1 != ids2


def test_train_and_validation_have_no_document_overlap():
    result = _prepare()
    train_ids = {d.document_id for d in result.train}
    val_ids = {d.document_id for d in result.validation}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {d.document_id for d in result.accepted}


def test_split_raises_before_dedup_is_impossible_by_construction():
    # Directive Section 9: "duplicates must already have been removed
    # before splitting." Verified structurally: split_sentences is only
    # ever called (in _split_accepted) on the already-deduplicated
    # `accepted` list, never on raw/candidate documents. This test
    # confirms the observable consequence: the known duplicate
    # (doc-002) never appears in either split.
    result = _prepare()
    all_split_ids = {d.document_id for d in result.train + result.validation}
    assert "doc-002" not in all_split_ids


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_counts_match_actual_results():
    result = _prepare()
    m = result.manifest
    assert m.input_count == 10
    assert m.accepted_count == len(result.accepted) == 4
    assert m.rejected_count == len(result.rejected) == 5
    assert m.review_count == len(result.review) == 1
    assert m.train_count == len(result.train)
    assert m.validation_count == len(result.validation)
    assert m.train_count + m.validation_count == m.accepted_count


def test_manifest_distributions_match_input():
    result = _prepare()
    m = result.manifest
    assert m.source_distribution == {"source-a": 6, "source-b": 4}
    assert m.license_distribution == {
        "cc-by-4.0": 8,
        "proprietary-no-license": 1,
        "unknown-license-xyz": 1,
    }
    assert m.language_distribution == {"en": 10}
    assert m.domain_distribution == {"fiction": 4, "news": 6}


def test_manifest_rejection_reason_counts():
    result = _prepare()
    counts = result.manifest.rejection_reason_counts
    assert counts[RejectionReason.EXACT_DUPLICATE.value] == 1
    assert counts[RejectionReason.LICENSE_REJECTED.value] == 1
    assert counts[RejectionReason.LICENSE_REVIEW_REQUIRED.value] == 1
    assert counts[RejectionReason.EMPTY_AFTER_NORMALIZATION.value] == 1
    assert counts[RejectionReason.BELOW_MIN_LENGTH.value] == 1
    assert counts[RejectionReason.PATHOLOGICAL_REPETITION.value] == 1


def test_manifest_json_is_deterministic_and_valid_json():
    result1 = _prepare()
    result2 = _prepare()
    json1 = result1.manifest.to_json()
    json2 = result2.manifest.to_json()
    assert json1 == json2
    parsed = json.loads(json1)  # must actually be valid JSON
    assert parsed["schema_version"]
    assert "timestamp" not in json1 and "time" not in parsed


def test_manifest_accepted_content_hash_independent_of_which_duplicate_survives():
    # doc-001 and doc-002 have identical normalized text, so whichever
    # one dedup happens to keep, the set of accepted content_hashes --
    # and therefore accepted_content_hash -- is unaffected by input
    # order in this respect.
    reordered = _fixture_documents()
    reordered[0], reordered[1] = reordered[1], reordered[0]  # doc-002 now wins dedup
    result_original = _prepare()
    result_reordered = prepare_corpus(
        reordered, license_policy=_LICENSE_POLICY, val_fraction=0.25, seed=42
    )
    assert (
        result_original.manifest.accepted_content_hash
        == result_reordered.manifest.accepted_content_hash
    )
    # But the document_id that actually survived did change:
    assert {d.document_id for d in result_original.accepted} != {
        d.document_id for d in result_reordered.accepted
    }


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_empty_document_list_rejected():
    with pytest.raises(CorpusPrepError):
        prepare_corpus([], license_policy=_LICENSE_POLICY, val_fraction=0.25, seed=1)


def test_corpus_prep_module_does_not_import_yug():
    """Architectural constraint (Section 15): production yug/ must not
    depend on training/. This module doesn't need to depend on yug/
    either (it has no tokenizer/model dependency), so the simplest way
    to keep the boundary unambiguous is for it to import nothing from
    yug/ at all -- checked directly."""
    import inspect

    import training.corpus_prep as corpus_prep_module

    source = inspect.getsource(corpus_prep_module)
    assert "import yug" not in source
    assert "from yug" not in source


# ---------------------------------------------------------------------------
# Existing training/dataset.py integration (Section 11 / 13) -- real API,
# not mocked. Gated: training.dataset imports torch at module scope.
# ---------------------------------------------------------------------------


def test_accepted_text_feeds_training_dataset_encode_corpus():
    pytest.importorskip("torch")

    from yug.tokenizer.bpe import ByteLevelBpeTokenizer
    from training.dataset import encode_corpus, make_sequences

    result = _prepare()
    accepted_texts = [doc.text for doc in result.accepted]
    assert len(accepted_texts) == 4  # sanity: fixture didn't silently change shape

    tokenizer = ByteLevelBpeTokenizer.train(accepted_texts, vocab_size=280)

    # The real, unmodified training.dataset API, given exactly the
    # AcceptedDocument.text this prototype produces -- no adapter, no
    # mock, no shim.
    token_ids = encode_corpus(tokenizer, accepted_texts)
    assert len(token_ids) > 0

    sequences = make_sequences(token_ids, sequence_length=4)
    assert len(sequences) > 0
    assert all(len(seq) == 5 for seq in sequences)  # sequence_length + 1
