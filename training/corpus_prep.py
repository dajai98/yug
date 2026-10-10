"""Corpus Preparation Prototype.

VERIFICATION STATUS: this module itself has no PyTorch dependency (only
``dataclasses``, ``enum``, ``hashlib``, ``json``, ``re`` -- stdlib) and
was executed directly in this environment (see the accompanying test
file's own VERIFICATION STATUS note for exact results). Only the one
test proving compatibility with ``training.dataset``'s real API needs
PyTorch (that module imports ``torch`` at module scope) and is gated
with ``pytest.importorskip("torch")``, unexecuted here, matching every
other torch-dependent test already in this repository.

Scope (deliberately small -- this is a correctness prototype, not the
future large-scale pipeline; see this module's own docstring sections
below for exactly what is and is not implemented):

    Raw documents (RawDocument)
          |
          v
    License gate (ALLOW / REJECT / REVIEW, explicit policy only)
          |
          v
    Normalization (conservative, deterministic)
          |
          v
    Quality filter (empty / below-min-length / pathological repetition)
          |
          v
    Exact deduplication (SHA-256 of normalized text, first-seen wins)
          |
          v
    Deterministic train/validation split (reuses training.split.split_sentences)
          |
          v
    AcceptedDocument list + CorpusManifest
          |
          v
    (caller feeds AcceptedDocument.text into training.dataset.encode_corpus,
     unmodified -- see this module's docstring section on integration)

Explicitly NOT implemented (out of scope for this prototype, per its
own directive's hard boundaries): ML quality/toxicity/perplexity
classifiers, near-duplicate detection (MinHash/LSH/embeddings),
web-scale ingestion, a dataset UI, external services. All rejection
decisions are simple, deterministic, and explainable from the input
text and an explicitly configured license policy alone.

Architectural boundary: this module lives under ``training/`` and may
import ``yug`` (it doesn't need to -- it has no dependency on
``yug.native``/``yug.tokenizer`` at all). Nothing under ``yug/``
imports this module or anything else under ``training/``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

from training.errors import CorpusPrepError
from training.split import split_sentences

_SCHEMA_VERSION = "corpus-prep-prototype-v1"

# ---------------------------------------------------------------------------
# Document model (Section 4) -- minimal, only the fields the directive
# asked for. No "giant metadata framework": one flat frozen dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawDocument:
    """One raw input document, before any processing.

    All seven fields are the ones the directive named explicitly; none
    added beyond that (no "genuinely necessary" additional field was
    identified -- everything the pipeline needs to decide ALLOW/REJECT/
    REVIEW, normalize, filter, dedup, and attribute provenance is
    already covered by these seven).
    """

    document_id: str
    text: str
    source_id: str
    source_name: str
    license: str
    language: str
    primary_domain: str


class RejectionReason(Enum):
    """Machine-readable reason a document did not reach ``accepted``.

    One reason per document (the first check that fails wins -- see
    ``prepare_corpus``'s docstring for the exact, fixed check order).
    """

    LICENSE_REJECTED = "license_rejected"
    LICENSE_REVIEW_REQUIRED = "license_review_required"
    EMPTY_AFTER_NORMALIZATION = "empty_after_normalization"
    BELOW_MIN_LENGTH = "below_min_length"
    PATHOLOGICAL_REPETITION = "pathological_repetition"
    EXACT_DUPLICATE = "exact_duplicate"


@dataclass(frozen=True)
class AcceptedDocument:
    """A document that survived every gate, holding enough provenance
    to trace it back to its source (Section 4's "processed
    representation must retain enough provenance" requirement) and
    ``content_hash`` for integrity/reproducibility checks.

    ``text`` here is the NORMALIZED text -- this is what
    ``training.dataset.encode_corpus`` should actually be given (see
    module docstring), not ``RawDocument.text``.
    """

    document_id: str
    text: str
    source_id: str
    source_name: str
    license: str
    language: str
    primary_domain: str
    content_hash: str


@dataclass(frozen=True)
class RejectedDocument:
    """A document that did not reach ``accepted`` -- covers both hard
    rejections and review-required documents (the ``reason`` says
    which; see ``CorpusPrepResult`` for which bucket it lands in).
    Raw (not normalized) text is retained for manual audit/review --
    the entire point of a REVIEW bucket is that a human can look at
    exactly what was excluded and why."""

    document_id: str
    source_id: str
    license: str
    reason: RejectionReason
    text: str


# ---------------------------------------------------------------------------
# License gate (Section 5)
# ---------------------------------------------------------------------------


class LicenseDecision(Enum):
    ALLOW = "allow"
    REJECT = "reject"
    REVIEW = "review"


@dataclass(frozen=True)
class LicensePolicy:
    """An explicit, human-authored mapping from license identifier
    string to :class:`LicenseDecision`. This is a lookup table, not a
    copyright classifier -- exactly what the directive asked for.

    ``default`` governs any license string not present in
    ``decisions``. It defaults to ``REVIEW``, never ``ALLOW`` --
    an unrecognized license is exactly the case the directive says
    "must NOT silently enter the accepted training corpus", so the
    one safe default that satisfies that for every caller (even one
    that forgot to configure a default) is REVIEW, not ALLOW.
    """

    decisions: Mapping[str, LicenseDecision]
    default: LicenseDecision = LicenseDecision.REVIEW

    def decide(self, license_id: str) -> LicenseDecision:
        return self.decisions.get(license_id, self.default)


# ---------------------------------------------------------------------------
# Normalization (Section 6) -- conservative and deterministic.
# ---------------------------------------------------------------------------

_HORIZONTAL_WHITESPACE_RUN = re.compile(r"[ \t]+")


def normalize_text(text: str) -> str:
    """Conservative, deterministic normalization:

    1. CRLF/CR -> LF (newline normalization).
    2. Runs of spaces/tabs collapsed to one space (controlled
       whitespace normalization) -- newlines themselves are NOT
       collapsed, so paragraph structure survives.
    3. Each line stripped of leading/trailing whitespace.
    4. The whole result stripped of leading/trailing whitespace.

    Deliberately NOT done (per the directive's "do not aggressively
    rewrite natural language" / "do not modify code semantics"): no
    case-folding, no Unicode normalization (NFC/NFKC), no punctuation
    changes, no line joining/unwrapping. See this module's "Known
    limitations" note (also in the final report) on why this makes the
    prototype natural-language-oriented, not safe for source code
    corpora as-is (per-line stripping would alter significant
    indentation).
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HORIZONTAL_WHITESPACE_RUN.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Quality filtering (Section 7) -- simple, deterministic, no ML.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityConfig:
    min_length: int = 20
    """Minimum length, in characters, of the NORMALIZED text."""

    min_repeated_tokens_for_pathological: int = 6
    """A document whose normalized text is nothing but the same single
    whitespace-separated token repeated at least this many times is
    rejected as pathological repetition. Deliberately the simplest
    possible trivial check the directive allows ("if trivial to
    detect") -- not a general repetition/compression-ratio detector."""

    def __post_init__(self) -> None:
        if self.min_length < 0:
            raise CorpusPrepError(f"min_length must be >= 0, got {self.min_length}")
        if self.min_repeated_tokens_for_pathological < 1:
            raise CorpusPrepError(
                "min_repeated_tokens_for_pathological must be >= 1, "
                f"got {self.min_repeated_tokens_for_pathological}"
            )


def _is_pathological_repetition(normalized_text: str, config: QualityConfig) -> bool:
    tokens = normalized_text.split()
    return (
        len(tokens) >= config.min_repeated_tokens_for_pathological
        and len(set(tokens)) == 1
    )


def check_quality(
    normalized_text: str, config: QualityConfig
) -> RejectionReason | None:
    """Return the :class:`RejectionReason` this text fails, or
    ``None`` if it passes every quality check.

    An originally-empty and an originally-whitespace-only document are
    indistinguishable once normalized (both normalize to ``""``), so
    both directive examples ("empty document", "whitespace-only
    document") share the single ``EMPTY_AFTER_NORMALIZATION`` reason
    rather than two reasons with no observable difference to assign
    between them.
    """
    if normalized_text == "":
        return RejectionReason.EMPTY_AFTER_NORMALIZATION
    if len(normalized_text) < config.min_length:
        return RejectionReason.BELOW_MIN_LENGTH
    if _is_pathological_repetition(normalized_text, config):
        return RejectionReason.PATHOLOGICAL_REPETITION
    return None


# ---------------------------------------------------------------------------
# Exact deduplication (Section 8)
# ---------------------------------------------------------------------------


def content_hash(normalized_text: str) -> str:
    """SHA-256 hex digest of the normalized text's UTF-8 bytes --
    the directive's suggested stable cryptographic hash; nothing in
    this repository already provides an equivalent (checked: no
    existing content-hashing utility under ``yug/`` or ``training/``
    before adding this)."""
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Manifest (Section 10)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusManifest:
    schema_version: str
    input_count: int
    accepted_count: int
    rejected_count: int
    review_count: int
    duplicate_count: int
    train_count: int
    validation_count: int
    rejection_reason_counts: dict[str, int]
    source_distribution: dict[str, int]
    license_distribution: dict[str, int]
    language_distribution: dict[str, int]
    domain_distribution: dict[str, int]
    config: dict[str, Any]
    accepted_content_hash: str
    """SHA-256 over the sorted, newline-joined list of every accepted
    document's own ``content_hash`` -- a single reproducibility
    checksum for what actually ended up in the accepted corpus.
    Computed from content hashes (not document_ids), so it is the same
    across two runs even if dedup happened to keep a different
    duplicate's document_id (their normalized text, and therefore
    their content_hash, is identical by construction)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "input_count": self.input_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "review_count": self.review_count,
            "duplicate_count": self.duplicate_count,
            "train_count": self.train_count,
            "validation_count": self.validation_count,
            "rejection_reason_counts": self.rejection_reason_counts,
            "source_distribution": self.source_distribution,
            "license_distribution": self.license_distribution,
            "language_distribution": self.language_distribution,
            "domain_distribution": self.domain_distribution,
            "config": self.config,
            "accepted_content_hash": self.accepted_content_hash,
        }

    def to_json(self) -> str:
        """Deterministic JSON: sorted keys, no timestamps, no other
        nondeterministic values -- two runs on the same input with the
        same configuration produce byte-identical output."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)


def _sorted_counts(items: list[str]) -> dict[str, int]:
    counts = Counter(items)
    return {key: counts[key] for key in sorted(counts)}


# ---------------------------------------------------------------------------
# The pipeline itself
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusPrepResult:
    accepted: list[AcceptedDocument]
    rejected: list[RejectedDocument]
    review: list[RejectedDocument]
    train: list[AcceptedDocument]
    validation: list[AcceptedDocument]
    manifest: CorpusManifest


def prepare_corpus(
    documents: list[RawDocument],
    *,
    license_policy: LicensePolicy,
    quality_config: QualityConfig | None = None,
    val_fraction: float,
    seed: int,
    text_normalizer: Callable[[RawDocument], str] | None = None,
) -> CorpusPrepResult:
    """Run the full Corpus Preparation Prototype pipeline.

    Fixed check order per document (first failing check wins -- this
    is deliberate and documented, not an implementation accident):

        1. License gate (on ``RawDocument.license``, raw text untouched)
        2. Normalization (on ``RawDocument.text``)
        3. Quality filter (on the normalized text)
        4. Exact deduplication (across all documents that passed 1-3,
           in ``documents`` input order -- first occurrence of a given
           normalized text wins as accepted, every later occurrence is
           rejected as EXACT_DUPLICATE)

    Then: the surviving accepted documents' (normalized) text is
    passed to ``training.split.split_sentences`` (reused, not
    reimplemented) for the deterministic train/validation split.

    ``text_normalizer`` (optional, backward-compatible): a function
    ``RawDocument -> str`` used INSTEAD of :func:`normalize_text` at
    step 2. ``None`` (the default) keeps the original prose-oriented
    behaviour byte-for-byte. It exists because ``normalize_text``
    strips per-line indentation and collapses space runs, which is
    destructive for source code and markdown; content-type-aware
    callers (see ``training.experiments.representative_mini_corpus``)
    supply a whitespace-preserving normalizer for those documents.

    Raises:
        CorpusPrepError: ``documents`` is empty.
    """
    if not documents:
        raise CorpusPrepError("documents must contain at least one entry")
    quality_config = quality_config or QualityConfig()

    accepted: list[AcceptedDocument] = []
    rejected: list[RejectedDocument] = []
    review: list[RejectedDocument] = []
    seen_hashes: set[str] = set()
    duplicate_count = 0

    for doc in documents:
        decision = license_policy.decide(doc.license)

        if decision is LicenseDecision.REJECT:
            rejected.append(
                RejectedDocument(
                    document_id=doc.document_id,
                    source_id=doc.source_id,
                    license=doc.license,
                    reason=RejectionReason.LICENSE_REJECTED,
                    text=doc.text,
                )
            )
            continue

        if decision is LicenseDecision.REVIEW:
            review.append(
                RejectedDocument(
                    document_id=doc.document_id,
                    source_id=doc.source_id,
                    license=doc.license,
                    reason=RejectionReason.LICENSE_REVIEW_REQUIRED,
                    text=doc.text,
                )
            )
            continue

        normalized = normalize_text(doc.text) if text_normalizer is None else text_normalizer(doc)
        quality_failure = check_quality(normalized, quality_config)
        if quality_failure is not None:
            rejected.append(
                RejectedDocument(
                    document_id=doc.document_id,
                    source_id=doc.source_id,
                    license=doc.license,
                    reason=quality_failure,
                    text=doc.text,
                )
            )
            continue

        doc_hash = content_hash(normalized)
        if doc_hash in seen_hashes:
            duplicate_count += 1
            rejected.append(
                RejectedDocument(
                    document_id=doc.document_id,
                    source_id=doc.source_id,
                    license=doc.license,
                    reason=RejectionReason.EXACT_DUPLICATE,
                    text=doc.text,
                )
            )
            continue
        seen_hashes.add(doc_hash)

        accepted.append(
            AcceptedDocument(
                document_id=doc.document_id,
                text=normalized,
                source_id=doc.source_id,
                source_name=doc.source_name,
                license=doc.license,
                language=doc.language,
                primary_domain=doc.primary_domain,
                content_hash=doc_hash,
            )
        )

    train, validation = _split_accepted(accepted, val_fraction=val_fraction, seed=seed)

    manifest = _build_manifest(
        documents=documents,
        accepted=accepted,
        rejected=rejected,
        review=review,
        duplicate_count=duplicate_count,
        train=train,
        validation=validation,
        quality_config=quality_config,
        val_fraction=val_fraction,
        seed=seed,
    )

    return CorpusPrepResult(
        accepted=accepted,
        rejected=rejected,
        review=review,
        train=train,
        validation=validation,
        manifest=manifest,
    )


def _split_accepted(
    accepted: list[AcceptedDocument], *, val_fraction: float, seed: int
) -> tuple[list[AcceptedDocument], list[AcceptedDocument]]:
    """Deterministic split, reusing ``training.split.split_sentences``
    rather than reimplementing splitting logic. That function operates
    on ``list[str]``; texts are guaranteed unique here (deduplication
    already ran), so mapping each returned string back to its
    originating :class:`AcceptedDocument` by exact text match is safe
    and unambiguous."""
    by_text = {doc.text: doc for doc in accepted}
    assert len(by_text) == len(accepted), (
        "accepted documents must have unique text at this point -- "
        "deduplication should have already guaranteed this"
    )

    if len(accepted) < 2:
        # split_sentences requires n_val >= 1 and at least 1 remaining
        # training sentence, i.e. at least 2 inputs. A one-(or-zero)-
        # document accepted corpus cannot be meaningfully split; every
        # accepted document goes to train, validation is empty. This
        # is a fixture/small-corpus edge case, not a silent departure
        # from determinism -- the same input always produces this same
        # outcome.
        return list(accepted), []

    train_texts, val_texts = split_sentences(
        [doc.text for doc in accepted], val_fraction=val_fraction, seed=seed
    )
    return [by_text[t] for t in train_texts], [by_text[t] for t in val_texts]


def _build_manifest(
    *,
    documents: list[RawDocument],
    accepted: list[AcceptedDocument],
    rejected: list[RejectedDocument],
    review: list[RejectedDocument],
    duplicate_count: int,
    train: list[AcceptedDocument],
    validation: list[AcceptedDocument],
    quality_config: QualityConfig,
    val_fraction: float,
    seed: int,
) -> CorpusManifest:
    rejection_reason_counts = _sorted_counts(
        [r.reason.value for r in rejected] + [r.reason.value for r in review]
    )
    accepted_hashes_sorted = sorted(doc.content_hash for doc in accepted)
    accepted_content_hash = hashlib.sha256(
        "\n".join(accepted_hashes_sorted).encode("utf-8")
    ).hexdigest()

    return CorpusManifest(
        schema_version=_SCHEMA_VERSION,
        input_count=len(documents),
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        review_count=len(review),
        duplicate_count=duplicate_count,
        train_count=len(train),
        validation_count=len(validation),
        rejection_reason_counts=rejection_reason_counts,
        source_distribution=_sorted_counts([d.source_id for d in documents]),
        license_distribution=_sorted_counts([d.license for d in documents]),
        language_distribution=_sorted_counts([d.language for d in documents]),
        domain_distribution=_sorted_counts([d.primary_domain for d in documents]),
        config={
            "val_fraction": val_fraction,
            "seed": seed,
            "min_length": quality_config.min_length,
            "min_repeated_tokens_for_pathological": (
                quality_config.min_repeated_tokens_for_pathological
            ),
        },
        accepted_content_hash=accepted_content_hash,
    )