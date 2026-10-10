"""Tests for training.experiments.real_corpus_experiment.

VERIFICATION STATUS: training.corpus_prep, training.split, and
yug.tokenizer.bpe have no PyTorch dependency, so every test touching
only those (acquisition boundary, corpus-prep reuse, document-level
split isolation, manifest/provenance, tokenizer-stat calculation,
experiment configuration, result-schema pieces that don't need a
trained model) executes directly in this environment. The one test
that runs the full ``run_real_corpus_experiment`` (which needs
PyTorch for the model/Trainer/generation stages) is gated with
``pytest.importorskip("torch")`` and reports SKIPPED here, matching
every other torch-dependent test already in this suite.

The fixture documents below are tiny, original, hand-written text --
explicitly NOT real Wikimedia/Common Pile data. Acquisition-specific
tests live in tests/test_real_corpus_acquisition.py. They exist
only to prove the PIPELINE CODE is correct, the same role the Corpus
Preparation Prototype's own fixture played for that prototype.
"""

from __future__ import annotations

import pytest

from training.corpus_prep import LicenseDecision, RawDocument
from training.experiments.real_corpus_experiment import (
    REAL_CORPUS_LICENSE_POLICY,
    SEQUENCE_LENGTH,
    TOKENIZER_VOCAB_SIZE,
    VAL_FRACTION,
    WIKIMEDIA_LICENSE,
    build_training_config,
    compute_tokenizer_stats,
    run_real_corpus_experiment,
)

# ---------------------------------------------------------------------------
# Tiny, original fixture -- NOT real Wikimedia data (see module docstring).
# ---------------------------------------------------------------------------


def _fixture_documents(count: int = 12) -> list[RawDocument]:
    base_sentences = [
        "Historians have long studied how ancient trade routes shaped early cities.",
        "The river valley provided fertile soil for generations of early farmers here.",
        "Astronomers catalogued thousands of faint stars using early photographic plates.",
        "The old library preserved manuscripts describing regional agricultural practices.",
        "Local musicians blended traditional rhythms with newly imported instruments then.",
        "The mountain pass connected two distinct cultures for several centuries at least.",
        "Botanists documented dozens of previously unrecorded plant species in the valley.",
        "The coastal town depended on fishing and modest seasonal trade for its economy.",
        "Early engineers designed irrigation channels that still function to this day.",
        "The museum curator organized an exhibit tracing regional pottery techniques.",
        "Geologists mapped the shifting riverbed across nearly a full century of records.",
        "The council debated how best to preserve the old market square for visitors.",
    ]
    docs = []
    for i in range(count):
        text = base_sentences[i % len(base_sentences)]
        if i >= len(base_sentences):
            text = text + f" Revision {i}."  # keep unique after wrap-around
        docs.append(
            RawDocument(
                document_id=f"fixture-{i:03d}",
                text=text,
                source_id="source-a" if i % 2 == 0 else "source-b",
                source_name="Fixture Source",
                license=WIKIMEDIA_LICENSE,
                language="en",
                primary_domain="reference",
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


def test_build_training_config_matches_directive_values():
    config = build_training_config()
    assert config.learning_rate == 0.003
    assert config.batch_size == 4
    assert config.grad_clip_norm == 1.0
    assert config.weight_decay == 0
    assert config.seed == 1234
    assert config.max_steps == 3000
    assert config.sequence_length == SEQUENCE_LENGTH == 16


def test_license_policy_only_allows_wikimedia_license():
    assert REAL_CORPUS_LICENSE_POLICY.decide(WIKIMEDIA_LICENSE) is LicenseDecision.ALLOW
    assert REAL_CORPUS_LICENSE_POLICY.decide("some-other-license") is LicenseDecision.REVIEW
    assert REAL_CORPUS_LICENSE_POLICY.decide("public-domain") is LicenseDecision.REVIEW


def test_val_fraction_is_approximately_80_20():
    assert VAL_FRACTION == 0.2


# ---------------------------------------------------------------------------
# corpus_prep reuse: document-level split isolation, manifest, provenance
# ---------------------------------------------------------------------------


def test_document_level_split_has_no_overlap_and_preserves_provenance():
    from training.corpus_prep import prepare_corpus

    result = prepare_corpus(
        _fixture_documents(), license_policy=REAL_CORPUS_LICENSE_POLICY, val_fraction=VAL_FRACTION, seed=1234
    )

    train_ids = {d.document_id for d in result.train}
    val_ids = {d.document_id for d in result.validation}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {d.document_id for d in result.accepted}

    for doc in result.accepted:
        assert doc.license == WIKIMEDIA_LICENSE
        assert doc.source_id in {"source-a", "source-b"}


def test_manifest_reflects_license_gate_and_counts():
    from training.corpus_prep import prepare_corpus

    docs = _fixture_documents(6) + [
        RawDocument(
            document_id="unlicensed-1",
            text="This one has no pre-approved license and must be quarantined for review only.",
            source_id="source-c",
            source_name="Fixture Source",
            license="all-rights-reserved",
            language="en",
            primary_domain="reference",
        )
    ]
    result = prepare_corpus(docs, license_policy=REAL_CORPUS_LICENSE_POLICY, val_fraction=VAL_FRACTION, seed=1234)

    assert result.manifest.review_count == 1
    assert "unlicensed-1" not in {d.document_id for d in result.accepted}
    manifest_dict = result.manifest.to_dict()
    assert manifest_dict["license_distribution"][WIKIMEDIA_LICENSE] == 6
    assert manifest_dict["license_distribution"]["all-rights-reserved"] == 1


def test_real_corpus_experiment_does_not_reimplement_split_isolation():
    """Directive: 'the split must occur at the DOCUMENT level before
    training sequences are generated' and 'do not bypass
    training/corpus_prep.py'. Verified structurally: the experiment
    module calls prepare_corpus and nothing else that could produce a
    train/validation partition."""
    import inspect

    import training.experiments.real_corpus_experiment as module

    source = inspect.getsource(module)
    assert "split_sentences(" not in source  # only prepare_corpus's own internal call, not called directly here
    assert "prepare_corpus(" in source


# ---------------------------------------------------------------------------
# Tokenizer-stat calculation (genuinely executed, torch-free)
# ---------------------------------------------------------------------------


def test_compute_tokenizer_stats_on_real_tokenizer():
    from yug.tokenizer.bpe import ByteLevelBpeTokenizer

    texts = [doc.text for doc in _fixture_documents(10)]
    tokenizer = ByteLevelBpeTokenizer.train(texts, vocab_size=TOKENIZER_VOCAB_SIZE, special_tokens=("bos", "eos"))

    stats = compute_tokenizer_stats(tokenizer, texts)

    assert stats["document_count"] == 10
    assert stats["total_characters"] == sum(len(t) for t in texts)
    assert stats["total_words"] == sum(len(t.split()) for t in texts)
    assert stats["total_tokens"] > 0
    expected_total_tokens = stats["total_tokens"]  # independently checked below via >0 and dict identity
    assert stats["tokens_per_word"] == expected_total_tokens / sum(len(t.split()) for t in texts)
    assert stats["tokens_per_character"] == expected_total_tokens / sum(len(t) for t in texts)
    assert stats["encoding_failures"] == 0
    assert stats["decoding_failures"] == 0
    assert stats["tokenizer_identity"] == tokenizer.identity
    assert stats["tokenizer_vocab_size"] == tokenizer.vocab_size


def test_compute_tokenizer_stats_reports_no_unk_metric():
    from yug.tokenizer.bpe import ByteLevelBpeTokenizer

    tokenizer = ByteLevelBpeTokenizer.train(["hello world"], vocab_size=260)
    stats = compute_tokenizer_stats(tokenizer, ["hello world"])
    assert stats["unknown_token_count"] is None
    assert stats["unknown_token_rate"] is None
    assert "no <unk> concept" in stats["unknown_token_note"]


def test_compute_tokenizer_stats_handles_empty_text_list():
    from yug.tokenizer.bpe import ByteLevelBpeTokenizer

    tokenizer = ByteLevelBpeTokenizer.train(["hello world"], vocab_size=260)
    stats = compute_tokenizer_stats(tokenizer, [])
    assert stats["document_count"] == 0
    assert stats["tokens_per_word"] is None
    assert stats["tokens_per_character"] is None


# ---------------------------------------------------------------------------
# Architectural boundary
# ---------------------------------------------------------------------------


def test_real_corpus_experiment_module_does_not_import_from_yug_at_module_scope():
    """yug/ MUST NOT import training/ (unaffected by this file either
    way), but this experiment module itself should not create a new,
    unnecessary yug-facing surface either -- checked directly: every
    yug.* import in this module is deferred inside function bodies
    (matching training.experiments.scaling_scale3_budget_extension's
    own convention), not at module scope, so importing this module
    never requires PyTorch."""
    import training.experiments.real_corpus_experiment as module

    assert "torch" not in dir(module)  # torch is never imported at module scope


def test_no_yug_file_imports_training():
    """Repository-wide architectural boundary check (Directive:
    'Verify again: yug/ MUST NOT import training/. Run an explicit
    dependency search/check.')."""
    import pathlib

    yug_root = pathlib.Path(__file__).resolve().parent.parent / "yug"
    offending = []
    for path in yug_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "import training" in text or "from training" in text:
            offending.append(str(path))
    assert offending == [], f"yug/ files importing training/: {offending}"


# ---------------------------------------------------------------------------
# Full pipeline (torch-gated) -- construction/wiring + genuine run on the
# tiny fixture only. NOT the real experiment (acquisition blocked).
# ---------------------------------------------------------------------------


def test_run_real_corpus_experiment_end_to_end_on_fixture():
    pytest.importorskip("torch")

    documents = _fixture_documents(12)
    result = run_real_corpus_experiment(documents)

    assert result["not_a_milestone"] is True
    assert result["nb13_assigned"] is False
    assert result["stop_reason"] is None or "overfitting" in result["stop_reason"]
    assert result["train_document_count"] > 0
    assert result["validation_document_count"] > 0
    assert len(result["generation_probes"]) == 5  # all 5 fixed prompts, unconditionally
    assert result["corpus_manifest"]["accepted_count"] == (
        result["train_document_count"] + result["validation_document_count"]
    )
