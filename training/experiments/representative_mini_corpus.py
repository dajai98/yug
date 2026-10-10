"""Representative Mini-Corpus Investigation.

NOT a milestone: no NB-13 (or any numbered milestone) is assigned. This
is a tokenizer/data diagnostic. It trains NO Transformer and selects NO
final tokenizer vocabulary size or winner.

QUESTION: how do candidate tokenizer vocabulary sizes behave, PER
CATEGORY and globally, across English prose, source code, C#/Unity-
ecosystem code, technical documentation, Nepali, and English-Nepali
code-switching?

Category is carried in the existing ``RawDocument.primary_domain``
field (no schema change). See ``CATEGORIES``.

SOURCES (every source has an explicit :class:`SourceSpec`; see
``SOURCE_REGISTRY``; BLOCKED/REVIEW/not-yet-verified entries are
reported, never silently dropped or accepted):
  - general_english: Common Pile ``wikimedia_filtered`` via the EXISTING
    ``acquire_common_pile_wikimedia_sample`` (pinned revision).
  - code / csharp_gamedev / software_docs: an explicit list of files at
    PINNED COMMITS of permissively licensed GitHub repositories
    (``mini_corpus_github_files.json``), fetched from
    ``raw.githubusercontent.com`` and verified against each file's git
    blob SHA-1 (tamper/drift evidence). No crawling, no API key.
  - nepali: Nepali Wikipedia (``wikimedia/wikipedia``, config
    ``20231101.ne``) through the public Hugging Face dataset-viewer
    ``/rows`` API (bounded, deterministic offsets, stdlib only).
    Provenance = the ``ne`` Wikipedia project, NOT script detection.
  - nepali_english_codeswitch: NATURAL SOURCE NOT YET VERIFIED. A small
    hand-written SYNTHETIC diagnostic set exists; it is measured
    separately, labelled synthetic, and is NEVER used to fit a
    tokenizer or counted as natural data.
  - Unity documentation / Asset Store / Academy: BLOCKED (not scraped).

CONTENT-TYPE-AWARE NORMALIZATION (the one change to shared code):
``training.corpus_prep.normalize_text`` strips per-line indentation and
collapses space runs, which is destructive for code and markdown.
``prepare_corpus`` gained an OPTIONAL ``text_normalizer`` hook (default
``None`` = unchanged behaviour). This module supplies
:func:`normalize_document`: prose categories keep ``normalize_text``;
code and technical-documentation categories use
:func:`normalize_structured_text` (CRLF->LF, trailing space/tab removal,
leading/trailing blank lines removed; indentation, tabs and internal
space runs preserved).

SPLIT: ``prepare_corpus``'s own split is global (a small category can
land with zero validation documents). This module therefore reuses its
gate/normalize/quality/exact-dedup stages and re-splits the accepted
documents PER CATEGORY with the existing ``training.split.split_sentences``
(same seed and fraction). ``corpus_prep_manifest.train_count`` /
``validation_count`` describe prepare_corpus's global split and are NOT
what tokenizer fitting uses; ``split`` describes the split actually used.

TOKENIZER: the existing ``ByteLevelBpeTokenizer`` (unmodified), fitted
ONLY on training documents, via the same prefix-of-largest-fit approach
as ``tokenizer_vocab_scaling`` (its building blocks are imported, not
duplicated). Fit mixture = natural concatenation of every category's
train partition, so byte-heavy categories dominate merge learning; this
is reported (``train_composition``), not corrected.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from training.corpus_prep import (
    LicenseDecision,
    LicensePolicy,
    RawDocument,
    normalize_text,
    prepare_corpus,
)
from training.errors import TrainingError
from training.experiments import tokenizer_vocab_scaling as tvs
from training.experiments.real_corpus_experiment import (
    COMMON_PILE_DATASET_ID,
    COMMON_PILE_DATASET_REVISION,
    SEED,
    VAL_FRACTION,
    WIKIMEDIA_LICENSE,
    acquire_common_pile_wikimedia_sample,
)
from training.split import split_sentences

EXPERIMENT_NAME = "representative_mini_corpus_investigation"

CAT_ENGLISH = "general_english"
CAT_CODE = "code"
CAT_CSHARP = "csharp_gamedev"
CAT_DOCS = "software_docs"
CAT_NEPALI = "nepali"
CAT_CODESWITCH = "nepali_english_codeswitch"
CATEGORIES: tuple[str, ...] = (CAT_ENGLISH, CAT_CODE, CAT_CSHARP, CAT_DOCS, CAT_NEPALI, CAT_CODESWITCH)

CONTENT_TYPE: dict[str, str] = {
    CAT_ENGLISH: "prose",
    CAT_CODE: "code",
    CAT_CSHARP: "code",
    CAT_DOCS: "technical_documentation",
    CAT_NEPALI: "nepali_prose",
    CAT_CODESWITCH: "code_switched_text",
}
_STRUCTURED_CONTENT_TYPES = frozenset({"code", "technical_documentation"})

DEFAULT_VOCAB_SIZES: tuple[int, ...] = (1024, 2048, 4096, 8192)
DEFAULT_ENGLISH_MAX_DOCUMENTS = 60
DEFAULT_ENGLISH_MAX_DOC_CHARS = 12_000
DEFAULT_ENGLISH_MAX_TOTAL_CHARS = 180_000
DEFAULT_NEPALI_MAX_TOTAL_CHARS = 60_000
DEFAULT_NEPALI_MAX_DOC_CHARS = 8_000
DEFAULT_NEPALI_MIN_DOC_CHARS = 300
DEFAULT_NEPALI_MAX_ROWS_SCANNED = 2_000

LICENSE_SYNTHETIC = "synthetic-diagnostic"
_GITHUB_MANIFEST_PATH = Path(__file__).with_name("mini_corpus_github_files.json")
_HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
_HF_NEPALI_DATASET = "wikimedia/wikipedia"
_HF_NEPALI_CONFIG = "20231101.ne"
_USER_AGENT = "yug-mini-corpus-investigation/0.1"
_HTTP_TIMEOUT_SECONDS = 60.0

NATURAL_CODESWITCH_STATUS = "NATURAL SOURCE NOT YET VERIFIED"

MINI_CORPUS_LICENSE_POLICY = LicensePolicy(
    decisions={
        "CC-BY-SA-4.0": LicenseDecision.ALLOW,
        "CC-BY-SA-3.0": LicenseDecision.ALLOW,
        "CC-BY-4.0": LicenseDecision.ALLOW,
        "CC-BY-3.0": LicenseDecision.ALLOW,
        "MIT": LicenseDecision.ALLOW,
        "Apache-2.0": LicenseDecision.ALLOW,
        "MS-PL": LicenseDecision.ALLOW,
    },
    default=LicenseDecision.REVIEW,  # unknown / synthetic / custom => REVIEW, never ALLOW
)


class MiniCorpusError(TrainingError):
    """Invalid mini-corpus input (unknown category, bad configuration, ...)."""


class MiniCorpusAcquisitionError(MiniCorpusError):
    """A live source could not be acquired or verified."""


# ---------------------------------------------------------------------------
# Source registry (provenance / license status)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    category: str
    name: str
    upstream: str
    canonical_id: str
    license_id: str
    license_evidence: str
    status: str  # ACCEPT | REVIEW | BLOCKED | NATURAL SOURCE NOT YET VERIFIED | SYNTHETIC_DIAGNOSTIC_ONLY
    language: str
    acquisition_method: str
    revision: str
    redistribution_notes: str
    attribution: str
    automated: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _gh(source_id, category, repo, license_id, evidence, commit, notes, reason, language="zxx"):
    return SourceSpec(
        source_id=source_id,
        category=category,
        name=f"{repo} (selected files)",
        upstream=repo,
        canonical_id=f"https://github.com/{repo}",
        license_id=license_id,
        license_evidence=evidence,
        status="ACCEPT",
        language=language,
        acquisition_method="explicit file list at pinned commit via raw.githubusercontent.com; git blob SHA-1 verified",
        revision=commit,
        redistribution_notes=notes,
        attribution="Retain upstream copyright/license notices; nothing is redistributed by this experiment (fetched locally, not committed).",
        automated=True,
        reason=reason,
    )


def _load_github_manifest(path: Path = _GITHUB_MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _commit_for(source_id: str) -> str:
    for s in _load_github_manifest()["sources"]:
        if s["source_id"] == source_id:
            return s["commit"]
    raise MiniCorpusError(f"source {source_id!r} missing from GitHub file manifest")


def build_source_registry() -> tuple[SourceSpec, ...]:
    c = _commit_for
    repo_note = "License read from the repository LICENSE file at the pinned commit; repo-level, individual files not audited for bundled third-party code."
    specs = [
        SourceSpec(
            source_id="cp-wikimedia-filtered",
            category=CAT_ENGLISH,
            name="Common Pile - Wikimedia Filtered (Wikibooks in shard 0)",
            upstream="common-pile/wikimedia_filtered",
            canonical_id=f"hf:{COMMON_PILE_DATASET_ID}@{COMMON_PILE_DATASET_REVISION}",
            license_id=WIKIMEDIA_LICENSE,
            license_evidence="per-record license string mapped by exact match in real_corpus_experiment",
            status="ACCEPT",
            language="en",
            acquisition_method="existing acquire_common_pile_wikimedia_sample (bounded streaming read, pinned revision)",
            revision=COMMON_PILE_DATASET_REVISION,
            redistribution_notes="CC BY-SA 4.0: attribution + share-alike. Not committed to the repository.",
            attribution="Wikimedia contributors; see per-document provenance in the acquisition report.",
            automated=True,
            reason="Already executed by Rai in the first real-corpus experiment; same shard, so the first N documents are a prefix of that sample.",
        ),
        _gh("gh-psf-requests", CAT_CODE, "psf/requests", "Apache-2.0", "LICENSE = Apache License 2.0", c("gh-psf-requests"), repo_note, "Python source"),
        _gh("gh-expressjs-express", CAT_CODE, "expressjs/express", "MIT", "LICENSE = MIT", c("gh-expressjs-express"), repo_note, "JavaScript source"),
        _gh("gh-gin-gonic-gin", CAT_CODE, "gin-gonic/gin", "MIT", "LICENSE = MIT", c("gh-gin-gonic-gin"), repo_note, "Go source"),
        _gh(
            "gh-burntsushi-ripgrep", CAT_CODE, "BurntSushi/ripgrep", "MIT",
            "COPYING: dual-licensed Unlicense OR MIT; MIT chosen", c("gh-burntsushi-ripgrep"), repo_note, "Rust source",
        ),
        _gh("gh-cysharp-unitask", CAT_CSHARP, "Cysharp/UniTask", "MIT", "LICENSE = MIT", c("gh-cysharp-unitask"), repo_note,
            "Third-party Unity-ecosystem C# library (UnityEngine APIs); source code only, NOT Unity documentation"),
        _gh("gh-mirrornetworking-mirror", CAT_CSHARP, "MirrorNetworking/Mirror", "MIT",
            "LICENSE = MIT (copyright Unity Technologies 2015 and Mirror contributors 2019)", c("gh-mirrornetworking-mirror"), repo_note,
            "Third-party Unity game-networking C#; source code only, NOT Unity documentation"),
        _gh("gh-monogame-monogame", CAT_CSHARP, "MonoGame/MonoGame", "MS-PL", "LICENSE.txt = Microsoft Public License (Ms-PL)", c("gh-monogame-monogame"),
            repo_note + " Restricted to MonoGame.Framework/ (ThirdParty/ excluded).", "Game-development C# (not Unity)"),
        _gh("gh-google-eng-practices", CAT_DOCS, "google/eng-practices", "CC-BY-3.0", "LICENSE = Creative Commons Attribution 3.0 Unported", c("gh-google-eng-practices"),
            repo_note, "Engineering-practice documentation (code review)", language="en"),
        _gh("gh-microsoft-api-guidelines", CAT_DOCS, "microsoft/api-guidelines", "CC-BY-4.0", "license.txt = Creative Commons Attribution 4.0 International", c("gh-microsoft-api-guidelines"),
            repo_note, "API design documentation", language="en"),
        _gh("gh-rust-lang-book", CAT_DOCS, "rust-lang/book", "MIT", "LICENSE-MIT + LICENSE-APACHE (dual MIT OR Apache-2.0); MIT chosen", c("gh-rust-lang-book"),
            repo_note, "Programming-language documentation", language="en"),
        SourceSpec(
            source_id="hf-wikipedia-ne-20231101",
            category=CAT_NEPALI,
            name="Nepali Wikipedia, 20231101 dump (cleaned articles)",
            upstream="wikimedia/wikipedia, config 20231101.ne (Wikipedia project ne)",
            canonical_id=f"hf:{_HF_NEPALI_DATASET}#{_HF_NEPALI_CONFIG}",
            license_id="CC-BY-SA-3.0",
            license_evidence="[SOURCE-DERIVED] dataset card: text licensed under GFDL and CC BY-SA 3.0; some text may be CC-only or carry additional licenses",
            status="ACCEPT",
            language="ne",
            acquisition_method="Hugging Face dataset-viewer /rows API, sequential offsets from 0, <=100 rows/request, stdlib urllib",
            revision="snapshot label 20231101 (the /rows API exposes no git revision; num_rows_total is recorded at run time)",
            redistribution_notes="CC BY-SA 3.0 (+GFDL): attribution + share-alike. Not committed to the repository.",
            attribution="Nepali Wikipedia contributors; per-article title/url recorded in the acquisition report.",
            automated=True,
            reason="[PROPOSAL] ACCEPT on the dataset card's stated license; final legal call rests with the project owner. Live path NOT executed by the author (sandbox has no access).",
        ),
        SourceSpec(
            source_id="synthetic-codeswitch-diagnostic",
            category=CAT_CODESWITCH,
            name="Hand-written synthetic English-Nepali code-switching diagnostic lines",
            upstream="this repository (representative_mini_corpus.py)",
            canonical_id="repo:training/experiments/representative_mini_corpus.py#SYNTHETIC_CODESWITCH_LINES",
            license_id=LICENSE_SYNTHETIC,
            license_evidence="original text written by the implementation engineer for diagnostics",
            status="SYNTHETIC_DIAGNOSTIC_ONLY",
            language="ne-en",
            acquisition_method="in-repo constant",
            revision="n/a",
            redistribution_notes="Synthetic; not natural data; never used to fit a tokenizer or counted as training data.",
            attribution="n/a",
            automated=True,
            reason="Not reviewed by a native Nepali speaker [UNKNOWN]. Measured separately as `synthetic_diagnostic`.",
        ),
        SourceSpec(
            source_id="natural-codeswitch-none",
            category=CAT_CODESWITCH,
            name="Natural English-Nepali code-switching corpus",
            upstream="none identified",
            canonical_id="none",
            license_id="unknown",
            license_evidence="no source with verified license and provenance identified",
            status=NATURAL_CODESWITCH_STATUS,
            language="ne-en",
            acquisition_method="none",
            revision="n/a",
            redistribution_notes="n/a",
            attribution="n/a",
            automated=False,
            reason="Not acquired. Candidates must be license-verified before entering the gate.",
        ),
        SourceSpec(
            source_id="unity-docs",
            category=CAT_CSHARP,
            name="Unity Manual / Scripting API documentation",
            upstream="Unity Technologies",
            canonical_id="https://docs.unity3d.com",
            license_id="unknown",
            license_evidence="not verified as training-safe",
            status="BLOCKED",
            language="en",
            acquisition_method="none (not scraped)",
            revision="n/a",
            redistribution_notes="n/a",
            attribution="n/a",
            automated=False,
            reason="Directive: do not assume Unity documentation is training-safe; do not scrape.",
        ),
        SourceSpec(
            source_id="unity-asset-store-academy",
            category=CAT_CSHARP,
            name="Unity Asset Store / Unity Academy content",
            upstream="Unity Technologies",
            canonical_id="n/a",
            license_id="unknown",
            license_evidence="n/a",
            status="BLOCKED",
            language="en",
            acquisition_method="none",
            revision="n/a",
            redistribution_notes="n/a",
            attribution="n/a",
            automated=False,
            reason="Directive: must not be used.",
        ),
        SourceSpec(
            source_id="gh-unity-csreference",
            category=CAT_CSHARP,
            name="Unity-Technologies/UnityCsReference",
            upstream="Unity Technologies",
            canonical_id="https://github.com/Unity-Technologies/UnityCsReference",
            license_id="Unity Reference-Only License",
            license_evidence="[VERIFIED BY REAL FETCH] LICENSE.md points to unity3d.com/legal/licenses/Unity_Reference_Only_License",
            status="BLOCKED",
            language="zxx",
            acquisition_method="none",
            revision="n/a",
            redistribution_notes="reference-only terms; not an open-source license",
            attribution="n/a",
            automated=False,
            reason="Reference-only license is not a training-use grant.",
        ),
        SourceSpec(
            source_id="gh-unity-ml-agents",
            category=CAT_CSHARP,
            name="Unity-Technologies/ml-agents",
            upstream="Unity Technologies",
            canonical_id="https://github.com/Unity-Technologies/ml-agents",
            license_id="Apache-2.0",
            license_evidence="[VERIFIED BY REAL FETCH] LICENSE.md = Apache License 2.0",
            status="REVIEW",
            language="zxx",
            acquisition_method="not automated",
            revision="n/a",
            redistribution_notes="Apache-2.0 at repo level",
            attribution="Retain notices",
            automated=False,
            reason="License is explicit, but the project's existing Unity policy is not visible in the uploaded repository [UNKNOWN]; held for the owner's decision rather than accepted.",
        ),
        SourceSpec(
            source_id="web-crawl-nepali",
            category=CAT_NEPALI,
            name="Web-crawl-derived Nepali corpora (e.g. CC-100/OSCAR/mC4 ne)",
            upstream="various",
            canonical_id="n/a",
            license_id="unknown",
            license_evidence="[ASSUMPTION] crawl-derived; per-document licenses not established",
            status="REVIEW",
            language="ne",
            acquisition_method="none",
            revision="n/a",
            redistribution_notes="n/a",
            attribution="n/a",
            automated=False,
            reason="Unclear per-document licensing => REVIEW, not ACCEPT.",
        ),
    ]
    return tuple(specs)


# ---------------------------------------------------------------------------
# Categories + normalization
# ---------------------------------------------------------------------------


def category_of(doc: RawDocument) -> str:
    """Category of ``doc`` (its ``primary_domain``). Raises
    :class:`MiniCorpusError` for anything not in :data:`CATEGORIES`."""
    if doc.primary_domain not in CONTENT_TYPE:
        raise MiniCorpusError(
            f"document {doc.document_id!r} has unknown category {doc.primary_domain!r}; expected one of {list(CATEGORIES)}"
        )
    return doc.primary_domain


def normalize_structured_text(text: str) -> str:
    """Whitespace-preserving normalization for code / markdown.

    CRLF/CR -> LF; trailing spaces/tabs removed per line; leading and
    trailing blank lines removed. Leading indentation, tabs, and
    internal space runs are preserved. NOT done: Unicode normalization,
    case changes, line joining. Known effect: trailing whitespace that
    is semantically meaningful (e.g. markdown hard line breaks written
    as two trailing spaces) is dropped.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip(" \t") for line in text.split("\n")]
    return "\n".join(lines).strip("\n")


def normalize_document(doc: RawDocument) -> str:
    """Content-type-aware normalizer passed to ``prepare_corpus``."""
    if CONTENT_TYPE[category_of(doc)] in _STRUCTURED_CONTENT_TYPES:
        return normalize_structured_text(doc.text)
    return normalize_text(doc.text)


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------


def _default_open_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:  # noqa: S310
        return response.read()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324 - git object id, not security


def acquire_github_files(
    manifest: dict[str, Any] | None = None,
    *,
    registry: Sequence[SourceSpec] | None = None,
    open_bytes: Callable[[str], bytes] | None = None,
) -> tuple[list[RawDocument], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Fetch every file in the pinned manifest and verify its git blob SHA-1.

    Returns ``(documents, provenance, errors)``. A fetch failure, blob
    SHA mismatch, or non-UTF-8 file is recorded in ``errors`` and that
    file is skipped (never accepted unverified).
    """
    manifest = manifest if manifest is not None else _load_github_manifest()
    specs = {s.source_id: s for s in (registry if registry is not None else build_source_registry())}
    fetch = open_bytes or _default_open_bytes
    docs: list[RawDocument] = []
    provenance: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    for source in manifest["sources"]:
        spec = specs.get(source["source_id"])
        if spec is None:
            errors.append({"stage": "github", "source_id": source["source_id"], "error": "source not in registry"})
            continue
        repo = spec.upstream
        commit = source["commit"]
        for entry in source["files"]:
            path = entry["path"]
            url = f"https://raw.githubusercontent.com/{repo}/{commit}/{urllib.parse.quote(path)}"
            try:
                raw = fetch(url)
            except Exception as exc:  # noqa: BLE001
                errors.append({"stage": "github", "source_id": spec.source_id, "path": path, "error": f"fetch failed: {type(exc).__name__}: {exc}"})
                continue
            actual = git_blob_sha1(raw)
            if actual != entry["git_blob_sha1"]:
                errors.append({"stage": "github", "source_id": spec.source_id, "path": path,
                               "error": f"git blob SHA-1 mismatch: expected {entry['git_blob_sha1']}, got {actual}"})
                continue
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                errors.append({"stage": "github", "source_id": spec.source_id, "path": path, "error": f"not valid UTF-8: {exc}"})
                continue
            doc_id = f"{spec.source_id}:{path}@{commit[:12]}"
            docs.append(
                RawDocument(
                    document_id=doc_id,
                    text=text,
                    source_id=spec.source_id,
                    source_name=spec.name,
                    license=spec.license_id,
                    language=spec.language,
                    primary_domain=spec.category,
                )
            )
            provenance[doc_id] = {
                "repo": repo, "commit": commit, "path": path, "url": url,
                "git_blob_sha1": entry["git_blob_sha1"], "bom_stripped": raw.startswith(b"\xef\xbb\xbf"),
                "license_id": spec.license_id,
            }
    return docs, provenance, errors


def acquire_english(
    max_documents: int = DEFAULT_ENGLISH_MAX_DOCUMENTS,
    *,
    max_doc_chars: int = DEFAULT_ENGLISH_MAX_DOC_CHARS,
    max_total_chars: int = DEFAULT_ENGLISH_MAX_TOTAL_CHARS,
    acquire: Callable[..., Any] | None = None,
) -> tuple[list[RawDocument], dict[str, Any]]:
    """General English via the EXISTING Common Pile acquisition, then a
    deterministic size bound (documents longer than ``max_doc_chars``
    are skipped, not truncated; stop once ``max_total_chars`` reached)."""
    fn = acquire or acquire_common_pile_wikimedia_sample
    raw_docs, report = fn(max_documents)
    kept: list[RawDocument] = []
    total = 0
    skipped_long = 0
    for doc in raw_docs:
        if len(doc.text) > max_doc_chars:
            skipped_long += 1
            continue
        if total + len(doc.text) > max_total_chars:
            break
        kept.append(dataclasses.replace(doc, primary_domain=CAT_ENGLISH))
        total += len(doc.text)
    info = {
        "acquired_before_bounds": len(raw_docs),
        "kept": len(kept),
        "skipped_over_max_doc_chars": skipped_long,
        "total_chars_kept": total,
        "acquisition_report": {k: v for k, v in report.to_dict().items() if k != "provenance"},
        "original_source_ids": sorted({d.source_id for d in kept}),
    }
    return kept, info


def _default_open_json(url: str) -> dict[str, Any]:
    return json.loads(_default_open_bytes(url).decode("utf-8"))


def devanagari_fraction(text: str) -> float | None:
    """Fraction of non-whitespace characters in U+0900..U+097F. A
    DIAGNOSTIC only (e.g. to spot markup contamination); it is never
    used to decide language."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return None
    return sum(1 for c in chars if "\u0900" <= c <= "\u097f") / len(chars)


def acquire_nepali_wikipedia(
    *,
    max_total_chars: int = DEFAULT_NEPALI_MAX_TOTAL_CHARS,
    max_doc_chars: int = DEFAULT_NEPALI_MAX_DOC_CHARS,
    min_doc_chars: int = DEFAULT_NEPALI_MIN_DOC_CHARS,
    max_rows_scanned: int = DEFAULT_NEPALI_MAX_ROWS_SCANNED,
    open_json: Callable[[str], dict[str, Any]] | None = None,
) -> tuple[list[RawDocument], dict[str, dict[str, Any]], dict[str, Any]]:
    """Bounded, deterministic read of Nepali Wikipedia rows (offset 0
    upward, 100 per request). Rows whose ``text`` cell is truncated by
    the API, too short, or too long are skipped and counted.

    Raises :class:`MiniCorpusAcquisitionError` on transport/schema
    failure or if no usable document was found.
    """
    fetch = open_json or _default_open_json
    docs: list[RawDocument] = []
    provenance: dict[str, dict[str, Any]] = {}
    skipped: Counter = Counter()
    total = 0
    scanned = 0
    offset = 0
    num_rows_total: int | None = None
    requests_made = 0

    while total < max_total_chars and scanned < max_rows_scanned:
        query = urllib.parse.urlencode(
            {"dataset": _HF_NEPALI_DATASET, "config": _HF_NEPALI_CONFIG, "split": "train", "offset": offset, "length": 100}
        )
        try:
            payload = fetch(f"{_HF_ROWS_URL}?{query}")
            requests_made += 1
            rows = payload["rows"]
            num_rows_total = payload.get("num_rows_total", num_rows_total)
        except Exception as exc:  # noqa: BLE001
            raise MiniCorpusAcquisitionError(f"Nepali Wikipedia rows request failed at offset {offset}: {type(exc).__name__}: {exc}") from exc
        if not rows:
            break
        for item in rows:
            scanned += 1
            try:
                row = item["row"]
                text, rid = row["text"], str(row["id"])
            except (KeyError, TypeError) as exc:
                raise MiniCorpusAcquisitionError(f"unexpected row schema at offset {offset}: {exc!r}") from exc
            if "text" in (item.get("truncated_cells") or []):
                skipped["text_truncated_by_api"] += 1
                continue
            if not isinstance(text, str) or len(text) < min_doc_chars:
                skipped["below_min_doc_chars"] += 1
                continue
            if len(text) > max_doc_chars:
                skipped["over_max_doc_chars"] += 1
                continue
            if total + len(text) > max_total_chars:
                skipped["would_exceed_budget"] += 1
                continue
            doc_id = f"wikipedia-ne:{rid}"
            docs.append(
                RawDocument(
                    document_id=doc_id, text=text, source_id="hf-wikipedia-ne-20231101",
                    source_name=f"{_HF_NEPALI_DATASET}#{_HF_NEPALI_CONFIG}", license="CC-BY-SA-3.0",
                    language="ne", primary_domain=CAT_NEPALI,
                )
            )
            provenance[doc_id] = {
                "row_idx": item.get("row_idx"), "title": row.get("title"), "url": row.get("url"),
                "devanagari_fraction_diagnostic": devanagari_fraction(text),
            }
            total += len(text)
            if total >= max_total_chars:
                break
        offset += len(rows)

    if not docs:
        raise MiniCorpusAcquisitionError(f"no usable Nepali Wikipedia documents found (scanned={scanned}, skipped={dict(skipped)})")
    fractions = sorted(p["devanagari_fraction_diagnostic"] for p in provenance.values() if p["devanagari_fraction_diagnostic"] is not None)
    info = {
        "rows_scanned": scanned, "requests_made": requests_made, "documents_kept": len(docs), "total_chars_kept": total,
        "skipped": dict(sorted(skipped.items())), "num_rows_total_reported_by_api": num_rows_total,
        "devanagari_fraction_min": fractions[0] if fractions else None,
        "devanagari_fraction_median": fractions[len(fractions) // 2] if fractions else None,
    }
    return docs, provenance, info


# Hand-written SYNTHETIC diagnostic lines. NOT natural data. Not reviewed
# by a native speaker [UNKNOWN].
SYNTHETIC_CODESWITCH_LINES: tuple[str, ...] = (
    "आज मैले नयाँ laptop किनें, तर battery life निकै कमजोर छ।",
    "यो function ले array को length फर्काउँछ।",
    "कृपया file लाई GitHub मा push गर्नुहोस्।",
    "मेरो code मा bug छ, debug गर्न मद्दत चाहियो।",
    "Unity मा player को movement कसरी गर्ने भनेर म सिक्दै छु।",
    "भोलि meeting मा हामी project को deadline बारे छलफल गर्नेछौं।",
    "यो database query धेरै slow छ, index थप्नुपर्छ।",
    "Nepal को capital Kathmandu हो र यो शहर धेरै busy छ।",
    "मलाई Python सिक्न मन लाग्छ किनभने यो सजिलो छ।",
    "server down भएकाले website खुल्दैन।",
    "कृपया password नबिर्सनुहोस्, नत्र account lock हुन सक्छ।",
    "उसले exam मा राम्रो score ल्यायो र सबै खुसी भए।",
)


def synthetic_codeswitch_documents() -> list[RawDocument]:
    """The synthetic diagnostic set, explicitly labelled (source_id,
    license and name all say synthetic). Its license is not in the ALLOW
    policy, so it can never pass the license gate into the corpus."""
    return [
        RawDocument(
            document_id=f"synthetic-codeswitch:{i:03d}", text=line, source_id="synthetic-codeswitch-diagnostic",
            source_name="SYNTHETIC diagnostic (not natural data)", license=LICENSE_SYNTHETIC,
            language="ne-en", primary_domain=CAT_CODESWITCH,
        )
        for i, line in enumerate(SYNTHETIC_CODESWITCH_LINES)
    ]


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------


def _sorted_counts(items: Sequence[str]) -> dict[str, int]:
    c = Counter(items)
    return {k: c[k] for k in sorted(c)}


def stratified_split(accepted, *, val_fraction: float, seed: int):
    """Per-category deterministic split reusing ``split_sentences``.
    Returns ``(train, validation)`` lists in category order. A category
    with fewer than 2 accepted documents goes entirely to train."""
    train, validation = [], []
    for cat in CATEGORIES:
        docs = [d for d in accepted if d.primary_domain == cat]
        if len(docs) < 2:
            train.extend(docs)
            continue
        by_text = {d.text: d for d in docs}
        tr, va = split_sentences([d.text for d in docs], val_fraction=val_fraction, seed=seed)
        train.extend(by_text[t] for t in tr)
        validation.extend(by_text[t] for t in va)
    return train, validation


def build_corpus(documents: list[RawDocument], *, val_fraction: float = VAL_FRACTION, seed: int = SEED) -> dict[str, Any]:
    """Run gate/normalize/quality/exact-dedup (``prepare_corpus`` with the
    content-type-aware normalizer), then the per-category split.

    Raises :class:`MiniCorpusError` for an unknown category (fail fast,
    before any processing)."""
    for doc in documents:
        category_of(doc)
    prep = prepare_corpus(
        documents, license_policy=MINI_CORPUS_LICENSE_POLICY, val_fraction=val_fraction, seed=seed,
        text_normalizer=normalize_document,
    )
    train, validation = stratified_split(prep.accepted, val_fraction=val_fraction, seed=seed)
    cat_by_id = {d.document_id: d.primary_domain for d in documents}

    per_category: dict[str, dict[str, Any]] = {}
    for cat in CATEGORIES:
        rej = [r for r in prep.rejected if cat_by_id[r.document_id] == cat]
        rev = [r for r in prep.review if cat_by_id[r.document_id] == cat]
        per_category[cat] = {
            "content_type": CONTENT_TYPE[cat],
            "input": sum(1 for d in documents if d.primary_domain == cat),
            "accepted": sum(1 for d in prep.accepted if d.primary_domain == cat),
            "rejected": len(rej),
            "review": len(rev),
            "duplicates": sum(1 for r in rej if r.reason.value == "exact_duplicate"),
            "rejection_reasons": _sorted_counts([r.reason.value for r in rej + rev]),
            "train": sum(1 for d in train if d.primary_domain == cat),
            "validation": sum(1 for d in validation if d.primary_domain == cat),
            "accepted_content_hash": hashlib.sha256(
                "\n".join(sorted(d.content_hash for d in prep.accepted if d.primary_domain == cat)).encode("utf-8")
            ).hexdigest(),
        }
    return {"prep": prep, "train": train, "validation": validation, "per_category": per_category}


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

_SUM_KEYS = (
    "document_count", "total_characters", "total_words", "total_tokens", "total_utf8_bytes",
    "encoding_failures", "decoding_failures", "round_trip_mismatches", "ids_out_of_range",
)


def _finish(stats: dict[str, Any]) -> dict[str, Any]:
    out = dict(stats)
    tok = out["total_tokens"]
    out["tokens_per_word"] = tok / out["total_words"] if out["total_words"] else None
    out["tokens_per_character"] = tok / out["total_characters"] if out["total_characters"] else None
    out["tokens_per_utf8_byte"] = tok / out["total_utf8_bytes"] if out["total_utf8_bytes"] else None
    return out


def measure_group(tokenizer, texts: list[str]) -> tuple[dict[str, Any], Counter]:
    """``measure_partition`` (reused) plus UTF-8 byte totals."""
    stats, counts = tvs.measure_partition(tokenizer, texts)
    stats = dict(stats)
    stats["total_utf8_bytes"] = sum(len(t.encode("utf-8")) for t in texts)
    return _finish(stats), counts


def combine_stats(parts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Sum additive fields; recompute ratios from the sums."""
    total = {k: sum(p[k] for p in parts) for k in _SUM_KEYS}
    return _finish(total)


def category_dispersion(per_category: dict[str, dict[str, Any]], metric: str = "tokens_per_character") -> dict[str, Any]:
    """Descriptive spread of ``combined.<metric>`` across natural
    categories that have data. Not a ranking or selection."""
    vals = {c: v["combined"][metric] for c, v in per_category.items() if v.get("combined") and v["combined"][metric] is not None}
    if not vals:
        return {"metric": metric, "categories_measured": 0}
    lo = min(vals, key=lambda c: (vals[c], c))
    hi = max(vals, key=lambda c: (vals[c], c))
    return {
        "metric": metric, "categories_measured": len(vals), "values": dict(sorted(vals.items())),
        "min": {"category": lo, "value": vals[lo]}, "max": {"category": hi, "value": vals[hi]},
        "max_over_min_ratio": (vals[hi] / vals[lo]) if vals[lo] else None,
    }


def _reduction_block(base: dict[str, Any] | None, cur: dict[str, Any] | None) -> dict[str, Any] | None:
    if base is None or cur is None:
        return None
    return {
        "token_count_reduction_pct": tvs._pct_reduction(base["total_tokens"], cur["total_tokens"]),
        "tokens_per_word_reduction_pct": tvs._pct_reduction(base["tokens_per_word"], cur["tokens_per_word"]),
        "tokens_per_character_reduction_pct": tvs._pct_reduction(base["tokens_per_character"], cur["tokens_per_character"]),
    }


LIMITATIONS: tuple[str, ...] = (
    "Tokenization statistics are not model-quality evidence; no Transformer is trained and no multilingual capability is claimed.",
    "Source diversity is small: a few repositories per code/docs category, one Wikibooks shard for English, Nepali Wikipedia only for Nepali.",
    "Natural English-Nepali code-switching data is NOT available (NATURAL SOURCE NOT YET VERIFIED); the synthetic set is measured separately and is not natural.",
    "Unity-specific coverage is limited to third-party MIT C# source (UniTask, Mirror); Unity documentation is BLOCKED and no Unity-authored material is included.",
    "Repo-level licenses were read from LICENSE files; individual files were not audited for bundled third-party code.",
    "Per-category validation sets are small; per-category validation statistics are noisy. `combined` (train+validation) is provided for comparison.",
    "The tokenizer is fitted on the natural concatenation of category train partitions, so byte-heavy categories dominate merge learning.",
    "Structured normalization drops trailing whitespace (e.g. markdown hard breaks) and does not touch Unicode normalization.",
    "Nepali Wikipedia is encyclopedic formal prose; it does not represent colloquial Nepali. Language is by Wikipedia-project provenance, not script detection.",
    "The Nepali live acquisition path was written from the dataset-viewer documentation and has not been executed by the author.",
    "Pure-Python BPE cost grows with vocabulary size; wall-clock at 8192 is unmeasured until a real run.",
)


def run_representative_mini_corpus(
    documents: list[RawDocument],
    *,
    synthetic_documents: list[RawDocument] | None = None,
    vocab_sizes: Sequence[int] = DEFAULT_VOCAB_SIZES,
    baseline_vocab_size: int | None = None,
    verify_prefix_up_to: int = tvs.DEFAULT_VERIFY_PREFIX_UP_TO,
    acquisition_info: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Tokenizer-only, per-category vocabulary diagnostic. No Transformer.

    ``documents`` are natural documents (category in ``primary_domain``).
    ``synthetic_documents`` (default: the built-in synthetic code-switch
    set) are measured but never fitted on. Per-size failures are
    recorded in ``errors`` without aborting other sizes.
    """
    import tempfile
    import time

    from yug.tokenizer.bpe import ByteLevelBpeTokenizer

    baseline = baseline_vocab_size if baseline_vocab_size is not None else min(vocab_sizes)
    sizes = tvs.validate_vocab_sizes(vocab_sizes, baseline)
    say = progress if callable(progress) else (lambda _m: None)
    synthetic = synthetic_codeswitch_documents() if synthetic_documents is None else synthetic_documents
    for s in synthetic:
        category_of(s)
        if s.license != LICENSE_SYNTHETIC:
            raise MiniCorpusError(f"synthetic document {s.document_id!r} must carry license {LICENSE_SYNTHETIC!r}")

    built = build_corpus(documents)
    prep, train, validation = built["prep"], built["train"], built["validation"]
    manifest = prep.manifest.to_dict()
    registry = build_source_registry()

    train_ids = {d.document_id for d in train}
    val_ids = {d.document_id for d in validation}
    leakage = {
        "tokenizer_fit_partition": "train_only",
        "document_id_overlap_train_validation": len(train_ids & val_ids),
        "content_hash_overlap_train_validation": len({d.content_hash for d in train} & {d.content_hash for d in validation}),
        "synthetic_documents_in_fit_corpus": sum(1 for d in train if d.license == LICENSE_SYNTHETIC or d.source_id.startswith("synthetic")),
        "train_fit_corpus_fingerprint": "sha256:" + hashlib.sha256("".join(d.content_hash for d in train).encode("utf-8")).hexdigest(),
        "categories_without_validation_documents": [c for c, v in built["per_category"].items() if v["accepted"] and not v["validation"]],
    }
    comp = {c: sum(len(d.text) for d in train if d.primary_domain == c) for c in CATEGORIES}
    comp_bytes = {c: sum(len(d.text.encode("utf-8")) for d in train if d.primary_domain == c) for c in CATEGORIES}
    tot_b = sum(comp_bytes.values()) or 1

    result: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "not_a_milestone": True,
        "nb13_assigned": False,
        "trains_transformer": False,
        "configuration": {
            "seed": SEED, "val_fraction": VAL_FRACTION, "vocab_sizes": sizes, "baseline_vocab_size": baseline,
            "verify_prefix_up_to": verify_prefix_up_to, "fit_mode": "prefix_of_largest_fit",
            "special_tokens": list(tvs.SPECIAL_TOKENS), "rare_thresholds_count_lt": list(tvs.RARE_THRESHOLDS),
            "categories": list(CATEGORIES), "content_types": CONTENT_TYPE,
            "normalization": {"prose": "corpus_prep.normalize_text", "structured (code, technical_documentation)": "normalize_structured_text"},
            "split": "per-category, training.split.split_sentences, same seed/fraction",
            "word_count_definition": "len(text.split())",
            "no_winner_selected": True,
        },
        "source_manifest": [s.to_dict() for s in registry],
        "source_license_status": _sorted_counts([s.status for s in registry]),
        "acquisition": acquisition_info or {},
        "category_distribution": built["per_category"],
        "language_distribution": _sorted_counts([d.language for d in prep.accepted]),
        "split": {"train_documents": len(train), "validation_documents": len(validation), "seed": SEED, "val_fraction": VAL_FRACTION},
        "corpus_counts": {
            "input": manifest["input_count"], "accepted": manifest["accepted_count"], "rejected": manifest["rejected_count"],
            "review": manifest["review_count"], "duplicates": manifest["duplicate_count"],
            "rejection_reason_counts": manifest["rejection_reason_counts"],
        },
        "corpus_prep_manifest": manifest,
        "corpus_hashes": {"accepted_content_hash": manifest["accepted_content_hash"],
                          "per_category_accepted_content_hash": {c: v["accepted_content_hash"] for c, v in built["per_category"].items()}},
        "train_composition": {c: {"characters": comp[c], "utf8_bytes": comp_bytes[c], "share_of_train_bytes": comp_bytes[c] / tot_b} for c in CATEGORIES},
        "leakage_checks": leakage,
        "tokenizers": [],
        "global_metrics_by_vocab": {},
        "per_category_metrics_by_vocab": {},
        "comparison_to_baseline": {},
        "errors": [],
        "limitations": list(LIMITATIONS),
        "timings_seconds_non_deterministic": {},
    }
    timings = result["timings_seconds_non_deterministic"]

    train_texts = [d.text for d in train]
    if not train_texts:
        result["errors"].append({"stage": "fit", "error": "no training documents"})
        return result

    fitted: dict[int, Any] = {}
    fit_info: dict[int, dict[str, Any]] = {}
    try:
        t0 = time.perf_counter()
        say(f"fitting tokenizer once at {sizes[-1]} on train partition only")
        full = ByteLevelBpeTokenizer.train(corpus=train_texts, vocab_size=sizes[-1], special_tokens=tvs.SPECIAL_TOKENS)
        timings["fit_max_size"] = time.perf_counter() - t0
        for s in sizes:
            fitted[s] = tvs.tokenizer_from_merge_prefix(full, s)
            fit_info[s] = {"mode": "prefix_of_max_size_fit", "prefix_equals_independent_training": None}
    except Exception as exc:  # noqa: BLE001
        result["errors"].append({"stage": "fit", "error": f"{type(exc).__name__}: {exc}"})

    for s in sizes:
        if s in fitted and s <= verify_prefix_up_to:
            try:
                say(f"verifying prefix equivalence at {s}")
                fit_info[s]["prefix_equals_independent_training"] = tvs._train_independent(train_texts, s).identity == fitted[s].identity
            except Exception as exc:  # noqa: BLE001
                result["errors"].append({"stage": "verify_prefix", "requested_vocab_size": s, "error": f"{type(exc).__name__}: {exc}"})

    by_cat_train = {c: [d.text for d in train if d.primary_domain == c] for c in CATEGORIES}
    by_cat_val = {c: [d.text for d in validation if d.primary_domain == c] for c in CATEGORIES}
    per_size_blocks: dict[int, dict[str, Any]] = {}

    for s in sizes:
        tok = fitted.get(s)
        if tok is None:
            continue
        try:
            t0 = time.perf_counter()
            say(f"measuring vocab {s} (actual {tok.vocab_size})")
            num_specials = len(tok.special_tokens)
            num_merges = tok.vocab_size - 256 - num_specials
            per_cat: dict[str, Any] = {}
            train_parts, val_parts, train_counter, val_counter = [], [], Counter(), Counter()
            for cat in CATEGORIES:
                if not by_cat_train[cat] and not by_cat_val[cat]:
                    per_cat[cat] = {
                        "status": NATURAL_CODESWITCH_STATUS if cat == CAT_CODESWITCH else "NO DOCUMENTS ACCEPTED",
                        "train": None, "validation": None, "combined": None, "utilization": None,
                    }
                    continue
                tr, trc = measure_group(tok, by_cat_train[cat])
                va, vac = measure_group(tok, by_cat_val[cat])
                train_parts.append(tr), val_parts.append(va)
                train_counter.update(trc), val_counter.update(vac)
                per_cat[cat] = {
                    "status": "measured",
                    "train": tr, "validation": va, "combined": combine_stats([tr, va]),
                    "utilization": {
                        "train": tvs.compute_utilization(trc, num_merges=num_merges, num_specials=num_specials),
                        "validation": tvs.compute_utilization(vac, num_merges=num_merges, num_specials=num_specials),
                    },
                }
            g_train, g_val = combine_stats(train_parts), combine_stats(val_parts)
            global_block = {
                "train": g_train, "validation": g_val, "combined": combine_stats([g_train, g_val]),
                "utilization": {
                    "train": tvs.compute_utilization(train_counter, num_merges=num_merges, num_specials=num_specials),
                    "validation": tvs.compute_utilization(val_counter, num_merges=num_merges, num_specials=num_specials),
                },
            }

            syn_stats, _ = measure_group(tok, [d.text for d in synthetic]) if synthetic else (None, Counter())
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "tok.json"
                tok.save(path)
                reloaded = ByteLevelBpeTokenizer.load(path).identity

            def _ok(field: str) -> bool:
                blocks = [b for c in per_cat.values() if c["status"] == "measured" for b in (c["train"], c["validation"])]
                blocks += [syn_stats] if syn_stats else []
                return all(b[field] == 0 for b in blocks)

            per_size_blocks[s] = {"global": global_block, "per_category": per_cat}
            result["tokenizers"].append({
                "requested_vocab_size": s, "actual_vocab_size": tok.vocab_size, "stopped_short_of_request": tok.vocab_size < s,
                "num_merges": num_merges, "tokenizer_identity": tok.identity, "fit": fit_info.get(s, {}),
                "global": global_block, "per_category": per_cat,
                "category_dispersion": {
                    "tokens_per_character": category_dispersion(per_cat, "tokens_per_character"),
                    "tokens_per_utf8_byte": category_dispersion(per_cat, "tokens_per_utf8_byte"),
                },
                "synthetic_diagnostic": {
                    "is_synthetic": True, "natural_data": False, "used_for_tokenizer_fit": False,
                    "category": CAT_CODESWITCH, "metrics": syn_stats,
                },
                "checks": {
                    "encoding_ok": _ok("encoding_failures"), "decoding_ok": _ok("decoding_failures"),
                    "round_trip_ok": _ok("round_trip_mismatches"), "no_out_of_range_or_special_ids": _ok("ids_out_of_range"),
                    "no_unk_token": "unk" not in tok.special_tokens and "<unk>" not in tok.special_tokens,
                    "save_load_identity_preserved": reloaded == tok.identity,
                },
            })
            timings[f"measure_{s}"] = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            result["errors"].append({"stage": "measure", "requested_vocab_size": s, "error": f"{type(exc).__name__}: {exc}"})

    for t in result["tokenizers"]:
        key = str(t["requested_vocab_size"])
        result["global_metrics_by_vocab"][key] = {k: t["global"][k] for k in ("train", "validation", "combined")}
        result["per_category_metrics_by_vocab"][key] = {
            c: ({k: v[k] for k in ("train", "validation", "combined")} if v["status"] == "measured" else {"status": v["status"]})
            for c, v in t["per_category"].items()
        }

    base_block = per_size_blocks.get(baseline)
    if base_block is None:
        result["errors"].append({"stage": "comparison", "error": f"baseline vocab {baseline} has no result"})
    else:
        for s, blk in sorted(per_size_blocks.items()):
            entry = {"baseline_vocab_size": baseline, "global": _reduction_block(base_block["global"]["combined"], blk["global"]["combined"]), "per_category": {}}
            for cat in CATEGORIES:
                b, c = base_block["per_category"][cat], blk["per_category"][cat]
                entry["per_category"][cat] = _reduction_block(b["combined"], c["combined"]) if b["status"] == "measured" and c["status"] == "measured" else None
            result["comparison_to_baseline"][str(s)] = entry
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def assemble_natural_documents(
    *,
    english_max_documents: int = DEFAULT_ENGLISH_MAX_DOCUMENTS,
    nepali_max_total_chars: int = DEFAULT_NEPALI_MAX_TOTAL_CHARS,
    allow_missing_categories: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[RawDocument], dict[str, Any]]:
    """Live acquisition of all automated natural sources. Fails (raises
    :class:`MiniCorpusAcquisitionError`) if any required natural
    category ends up empty, unless ``allow_missing_categories``."""
    say = progress if callable(progress) else (lambda _m: None)
    docs: list[RawDocument] = []
    info: dict[str, Any] = {"errors": []}

    say("acquiring English (Common Pile Wikimedia)")
    try:
        en_docs, info["english"] = acquire_english(english_max_documents)
        docs += en_docs
    except Exception as exc:  # noqa: BLE001
        info["errors"].append({"stage": "english", "error": f"{type(exc).__name__}: {exc}"})

    say("acquiring pinned GitHub files")
    gh_docs, gh_prov, gh_errors = acquire_github_files()
    docs += gh_docs
    info["github"] = {"documents": len(gh_docs), "files_failed": len(gh_errors), "provenance": gh_prov}
    info["errors"] += gh_errors

    say("acquiring Nepali Wikipedia")
    try:
        ne_docs, ne_prov, info["nepali"] = acquire_nepali_wikipedia(max_total_chars=nepali_max_total_chars)
        info["nepali"]["provenance"] = ne_prov
        docs += ne_docs
    except MiniCorpusAcquisitionError as exc:
        info["errors"].append({"stage": "nepali", "error": str(exc)})

    present = {d.primary_domain for d in docs}
    required = [CAT_ENGLISH, CAT_CODE, CAT_CSHARP, CAT_DOCS, CAT_NEPALI]
    missing = [c for c in required if c not in present]
    info["missing_required_categories"] = missing
    if missing and not allow_missing_categories:
        raise MiniCorpusAcquisitionError(f"required categories have no documents: {missing}; errors: {info['errors'][:5]}")
    return docs, info


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Representative mini-corpus investigation (not a milestone).")
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=list(DEFAULT_VOCAB_SIZES))
    parser.add_argument("--baseline", type=int, default=None, help="default: smallest requested size")
    parser.add_argument("--english-max-documents", type=int, default=DEFAULT_ENGLISH_MAX_DOCUMENTS)
    parser.add_argument("--nepali-max-chars", type=int, default=DEFAULT_NEPALI_MAX_TOTAL_CHARS)
    parser.add_argument("--verify-prefix-up-to", type=int, default=tvs.DEFAULT_VERIFY_PREFIX_UP_TO)
    parser.add_argument("--allow-missing-categories", action="store_true")
    parser.add_argument("--acquire-only", action="store_true", help="acquire + prepare corpus, report counts, no tokenizer work")
    parser.add_argument("--output", default=None, help="write UTF-8 JSON here (avoids PowerShell UTF-16 redirection)")
    args = parser.parse_args()

    def _log(message: str) -> None:
        print(f"[mini-corpus] {message}", file=sys.stderr, flush=True)

    try:
        natural_docs, acq = assemble_natural_documents(
            english_max_documents=args.english_max_documents, nepali_max_total_chars=args.nepali_max_chars,
            allow_missing_categories=args.allow_missing_categories, progress=_log,
        )
    except MiniCorpusAcquisitionError as exc:
        print(f"ACQUISITION FAILED: {exc}", file=sys.stderr)
        sys.exit(2)

    for section in ("github", "nepali"):  # keep the artifact compact
        if section in acq and "provenance" in acq[section]:
            acq[section]["provenance_document_count"] = len(acq[section].pop("provenance"))

    if args.acquire_only:
        built = build_corpus(natural_docs)
        payload = json.dumps({"acquisition": acq, "category_distribution": built["per_category"],
                              "corpus_prep_manifest": built["prep"].manifest.to_dict()}, indent=2, sort_keys=True, default=str)
    else:
        try:
            outcome = run_representative_mini_corpus(
                natural_docs, vocab_sizes=args.vocab_sizes, baseline_vocab_size=args.baseline,
                verify_prefix_up_to=args.verify_prefix_up_to, acquisition_info=acq, progress=_log,
            )
        except (tvs.VocabScalingConfigError, MiniCorpusError) as exc:
            print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
            sys.exit(3)
        payload = json.dumps(outcome, indent=2, sort_keys=True, default=str)
        code = 0 if not outcome["errors"] else 1
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        print(f"result written to {args.output}")
    else:
        print(payload)
    sys.exit(0 if args.acquire_only else code)