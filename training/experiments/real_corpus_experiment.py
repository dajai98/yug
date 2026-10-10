"""First Real-Corpus Experiment.

NOT a milestone: no NB-13 is assigned, no roadmap file is touched by
this script or its execution. Chief-architect-authorized controlled
experiment, same status as
``training.experiments.scaling_scale3_budget_extension``.

OBJECTIVE: test whether real English prose can flow through the
already-verified pipeline (corpus_prep -> tokenizer -> dataset ->
Trainer -> evaluate -> checkpoint -> generation) unchanged, and
measure how the existing Scale-3 architecture behaves on it.

============================================================
DATASET ACQUISITION: bounded, pinned-revision streaming read
============================================================

Source: ``common-pile/wikimedia_filtered`` (Hugging Face), the
"filtered" Wikimedia subset of Common Pile v0.1. Official Wikimedia
wikis are CC BY-SA per the dataset card; the per-record license string
in the data itself is what this module actually maps (see below), not
the card's summary.

Mechanism (:func:`acquire_common_pile_wikimedia_sample`):
  - Python standard library only (``urllib.request``, ``gzip``,
    ``json``) -- NO new dependency, no ``datasets``/``pyarrow``.
  - Streams ``wikimedia-0000.json.gz`` (one of 35 shards, each
    roughly 500-800 MB compressed) from the dataset's files at a
    PINNED git revision, decompressing on the fly, one JSON record
    per line. It stops reading, and closes the connection, the moment
    ``max_documents`` qualifying records have been collected -- the
    remainder of the shard is never requested by this code.
  - Bounded a second way: ``max_records_scanned`` caps how many lines
    may be examined, so a shard that yields few qualifying records
    cannot turn into an unbounded read.
  - Deterministic: the result is a pure function of (pinned revision,
    ``shard_indices``, ``main_namespace_only``, ``max_documents``) --
    records are taken in shard/line order, with no randomness.
  - Selection rule: only main-namespace pages (``metadata.namespace``
    == "0", i.e. encyclopedic/article pages) by default, so user pages,
    talk pages and project-administration pages are not counted as
    "real prose/reference text". This is a schema-based rule, not a
    quality judgement; ``corpus_prep`` still applies its own quality
    filter and exact deduplication afterwards.

Record schema actually used (read from the official dataset card /
viewer, not assumed): top-level ``id``, ``text``, ``source``;
``metadata.license``, ``metadata.namespace``, and (optional, kept as
provenance when present) ``metadata.title``, ``metadata.url``,
``metadata.provenance``, ``metadata.wiki``, ``metadata.authors``,
plus top-level ``created``.

License handling (the existing gate is NOT bypassed):
  - The real per-record value is
    ``"Creative Commons - Attribution Share-Alike -
    https://creativecommons.org/licenses/by-sa/4.0/"``. It is mapped
    by EXACT string match to YUG's ``CC-BY-SA-4.0`` identifier.
  - Any other license string is passed through to ``corpus_prep``
    VERBATIM, unmapped, so the existing policy quarantines it to
    REVIEW instead of it being relabelled as allowed.
  - A record with no license string is skipped and counted; a license
    is never invented for it.
  - The dataset card itself warns some licenses may be assigned
    incorrectly upstream ("license laundering"); this module cannot
    detect that, and does not claim to.

Provenance: ``RawDocument`` has no slot for title/URL/locator, and
``corpus_prep`` is deliberately not being redesigned here. The full
per-document provenance therefore travels in
:class:`AcquisitionReport.provenance`, keyed by ``document_id``
(``"<source>:<record id>"``), and is written into the result artifact
under ``"acquisition"``. Nothing downloaded is stored in the repository.

VERIFICATION STATUS OF ACQUISITION: written against the schema read
from the official dataset card, and exercised in tests with an
injected in-memory gzip stream (so the real decompression, JSON-lines
parsing, mapping, stopping and error paths run) -- but the LIVE
network read has NOT been executed by the author of this change,
because the authoring sandbox has no outbound network. First real
execution is Rai's.

Everything downstream of acquisition --
:func:`run_real_corpus_experiment` and its helpers -- takes a plain
``list[RawDocument]`` and does not know where the documents came from.

VERIFICATION STATUS OF THE REST: this module has no PyTorch dependency
at import time (only inside ``run_real_corpus_experiment``, matching
``scaling_scale3_budget_extension.py``'s convention). The torch-free
parts (acquisition, ``compute_tokenizer_stats``,
``train_tokenizer_on_train_partition``, ``build_training_config``, and
the corpus_prep-facing stage) are covered by tests that run without
PyTorch. The training/evaluation/generation stages require PyTorch and
were verified independently on Rai's Windows environment against a tiny
fixture; a real-corpus training run is not claimed by this module.

FROZEN CONTROLS reused via import, never duplicated or modified (same
components ``scaling_scale3_budget_extension.py`` reuses):
  - architecture: hidden_size=48, num_layers=2, num_heads=4, ffn_size=96
  - sequence_length=16, context_length=32
  - optimizer: AdamW, lr=3e-3, weight_decay=0.0, grad_clip_norm=1.0,
    batch_size=4 (``training.experiments.nb08_experiment.build_training_config``'s
    values, reused as literals here -- see this module's own
    ``build_training_config``)
  - seed=1234
  - tokenizer vocab_size target=384
    (``training.experiments.nb08_experiment.TOKENIZER_VOCAB_SIZE``,
    reused as a literal, matching that module's own convention of not
    importing across independent experiment scripts)
  - baselines: ``training.experiments.nb08_baseline.{uniform_baseline_loss,
    unigram_baseline_loss}``
  - decode-safety: ``training.experiments.nb08_experiment.safe_decode_generation``
  - sustained-overfitting stop condition: copied verbatim from
    ``training.experiments.scaling_scale3_budget_extension.run`` (see
    inline comment at its use below) -- not a competing definition.

NEW here (this experiment's own, genuinely new work): document-level
license-gated corpus preparation via ``training.corpus_prep`` (Scale-3
and NB-08 both used pre-approved raw sentence lists with no license
gate at all), and :func:`compute_tokenizer_stats`, which nothing
existing already provides.
"""

from __future__ import annotations

import contextlib
import gzip
import http.client
import io
import json
import math
import urllib.request
import zlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable

from training.corpus_prep import (
    LicenseDecision,
    LicensePolicy,
    RawDocument,
    prepare_corpus,
)
from training.errors import RealCorpusAcquisitionError

SEED = 1234
VAL_FRACTION = 0.2  # ~80/20, document-level (Directive "SPLIT REQUIREMENT")
TOKENIZER_VOCAB_SIZE = 384  # see module docstring; verify against the
# real tokenizer's own .vocab_size once acquisition is unblocked --
# BPE training can stop short of a target it can't reach (see
# yug.tokenizer.bpe.ByteLevelBpeTokenizer.train's own docstring), so
# this is a target, not an assumed achieved value.
SEQUENCE_LENGTH = 16
CONTEXT_LENGTH = 32
HIDDEN_SIZE = 48
NUM_LAYERS = 2
NUM_HEADS = 4
FFN_SIZE = 96
MAX_STEPS = 3000
EVAL_CHECKPOINTS = [0, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
RETAINED_CHECKPOINTS = [500, 1000, 1500, 2000, 3000]  # full save+reload-eval
GENERATION_PROMPTS = ["The", "People", "Science", "A small", "In the"]

WIKIMEDIA_LICENSE = "CC-BY-SA-4.0"
# Only this one license is pre-approved; anything else (including a
# document this experiment did not itself expect) is quarantined for
# REVIEW, never silently accepted -- same default corpus_prep.LicensePolicy
# already uses on its own, made explicit here rather than relied on
# implicitly.
REAL_CORPUS_LICENSE_POLICY = LicensePolicy(
    decisions={WIKIMEDIA_LICENSE: LicenseDecision.ALLOW},
    default=LicenseDecision.REVIEW,
)

# --- Acquisition constants (see module docstring, "DATASET ACQUISITION") ---
COMMON_PILE_DATASET_ID = "common-pile/wikimedia_filtered"
# Git commit of the dataset repository's ``main`` at the time this was
# written (read from the repository's file listing). Pinning it makes
# the sample reproducible even if ``main`` later changes.
COMMON_PILE_DATASET_REVISION = "0641bb84bd9b7162bcddf8be7836822161a9a342"
_SHARD_URL_TEMPLATE = (
    "https://huggingface.co/datasets/{dataset}/resolve/{revision}/wikimedia-{index:04d}.json.gz"
)
_SHARD_COUNT = 35  # wikimedia-0000 .. wikimedia-0034, per the repository listing
_HTTP_TIMEOUT_SECONDS = 60.0
_USER_AGENT = "yug-real-corpus-experiment/0.1"

# The exact license string the dataset's own records carry (observed in
# the official viewer). Mapped by EXACT match only; anything else is
# passed through verbatim so the license gate quarantines it.
WIKIMEDIA_LICENSE_SOURCE_STRING = (
    "Creative Commons - Attribution Share-Alike - "
    "https://creativecommons.org/licenses/by-sa/4.0/"
)
_SOURCE_LICENSE_TO_YUG_LICENSE = {WIKIMEDIA_LICENSE_SOURCE_STRING: WIKIMEDIA_LICENSE}
DATASET_LANGUAGE = "en"  # dataset-level: the card declares English; records carry no language field


@dataclass(frozen=True)
class AcquisitionReport:
    """What acquisition actually did -- recorded, not assumed. Written
    into the result artifact under ``"acquisition"``.

    ``provenance`` maps ``document_id`` to the real source metadata
    (title, URL, locator, wiki, namespace, authors, the source's own
    license string, ...) that ``RawDocument`` has no fields for. Only
    metadata actually present in the source record is included."""

    dataset_id: str
    dataset_revision: str
    shard_files: tuple[str, ...]
    max_documents: int
    main_namespace_only: bool
    records_scanned: int
    records_selected: int
    skipped: dict[str, int]
    compressed_bytes_read: int
    provenance: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "shard_files": list(self.shard_files),
            "max_documents": self.max_documents,
            "main_namespace_only": self.main_namespace_only,
            "records_scanned": self.records_scanned,
            "records_selected": self.records_selected,
            "skipped": dict(sorted(self.skipped.items())),
            "compressed_bytes_read": self.compressed_bytes_read,
            "provenance": self.provenance,
        }


class _CountingReader:
    """Minimal read-only wrapper that counts compressed bytes actually
    pulled off the wire -- evidence that the read really was bounded."""

    def __init__(self, raw: BinaryIO) -> None:
        self._raw = raw
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        data = self._raw.read(size)
        self.bytes_read += len(data)
        return data


def _open_shard_stream(url: str) -> BinaryIO:
    """Default network opener: a plain HTTPS GET, streamed. Only this
    function touches the network, so tests replace it wholesale."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    return urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS)  # noqa: S310


_SKIP_MALFORMED = "malformed"
_SKIP_MISSING_LICENSE = "missing_license"
_SKIP_WRONG_NAMESPACE = "wrong_namespace"
_SKIP_DUPLICATE_ID = "duplicate_id"


def _present_str(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _map_record(
    record: object, *, shard_file: str, main_namespace_only: bool
) -> tuple[str, tuple[RawDocument, dict[str, Any]] | None]:
    """Map one source record to ``(status, payload)``. ``status`` is
    ``"ok"`` (payload = (RawDocument, provenance)) or a skip reason
    (payload None). Never invents a value for a missing field."""
    if not isinstance(record, dict):
        return _SKIP_MALFORMED, None

    record_id = _present_str(record.get("id"))
    source = _present_str(record.get("source"))
    text = record.get("text")
    metadata = record.get("metadata")
    if record_id is None or source is None or not isinstance(text, str) or not isinstance(metadata, dict):
        return _SKIP_MALFORMED, None

    namespace = metadata.get("namespace")
    if main_namespace_only:
        if namespace is None:
            return _SKIP_MALFORMED, None
        if str(namespace) != "0":
            return _SKIP_WRONG_NAMESPACE, None

    license_source_string = _present_str(metadata.get("license"))
    if license_source_string is None:
        return _SKIP_MISSING_LICENSE, None
    # Exact-match mapping only; an unrecognized string is kept verbatim
    # so the existing license gate quarantines it (never relabelled).
    license_id = _SOURCE_LICENSE_TO_YUG_LICENSE.get(license_source_string, license_source_string)

    document_id = f"{source}:{record_id}"
    document = RawDocument(
        document_id=document_id,
        text=text,
        source_id=source,
        source_name=f"{COMMON_PILE_DATASET_ID}@{COMMON_PILE_DATASET_REVISION[:12]}",
        license=license_id,
        language=DATASET_LANGUAGE,
        primary_domain=source.rsplit("/", 1)[-1],
    )

    provenance: dict[str, Any] = {
        "dataset": COMMON_PILE_DATASET_ID,
        "revision": COMMON_PILE_DATASET_REVISION,
        "shard_file": shard_file,
        "record_id": record_id,
        "source": source,
        "license_source_string": license_source_string,
        "license_yug_id": license_id,
    }
    for key in ("title", "url", "provenance", "wiki"):
        value = _present_str(metadata.get(key))
        if value is not None:
            provenance["provenance_locator" if key == "provenance" else key] = value
    if namespace is not None:
        provenance["namespace"] = str(namespace)
    if isinstance(metadata.get("authors"), list):
        provenance["authors"] = metadata["authors"]
    created = _present_str(record.get("created"))
    if created is not None:
        provenance["created"] = created
    return "ok", (document, provenance)


def acquire_common_pile_wikimedia_sample(
    max_documents: int = 1000,
    *,
    shard_indices: tuple[int, ...] = (0,),
    main_namespace_only: bool = True,
    max_records_scanned: int = 100_000,
    max_unusable_records: int = 100,
    open_stream: Callable[[str], BinaryIO] | None = None,
) -> tuple[list[RawDocument], AcquisitionReport]:
    """Acquire a small, deterministic, bounded sample of
    ``common-pile/wikimedia_filtered`` as :class:`RawDocument` objects.

    See the module docstring for the mechanism, selection rule, schema,
    and license handling. ``open_stream`` exists so tests can supply an
    in-memory stream; production callers leave it ``None``.

    Raises:
        RealCorpusAcquisitionError: bad arguments; any network/HTTP/
            decompression/decoding failure; a line that is not valid
            JSON; too many unusable records (schema mismatch or missing
            licenses -- refusing to bypass the license gate); the scan
            cap reached, or every requested shard exhausted, before
            ``max_documents`` qualifying records were found.
    """
    if max_documents < 1:
        raise RealCorpusAcquisitionError(f"max_documents must be >= 1, got {max_documents}")
    if not shard_indices:
        raise RealCorpusAcquisitionError("shard_indices must contain at least one shard index")
    for index in shard_indices:
        if not 0 <= index < _SHARD_COUNT:
            raise RealCorpusAcquisitionError(
                f"shard index {index} out of range (valid: 0..{_SHARD_COUNT - 1})"
            )
    if max_records_scanned < 1 or max_unusable_records < 0:
        raise RealCorpusAcquisitionError("max_records_scanned must be >= 1 and max_unusable_records >= 0")

    opener = open_stream if open_stream is not None else _open_shard_stream
    documents: list[RawDocument] = []
    provenance: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    skipped: Counter[str] = Counter()
    shards_opened: list[str] = []
    records_scanned = 0
    compressed_bytes_read = 0

    try:
        for index in shard_indices:
            if len(documents) >= max_documents:
                break
            shard_file = f"wikimedia-{index:04d}.json.gz"
            url = _SHARD_URL_TEMPLATE.format(
                dataset=COMMON_PILE_DATASET_ID, revision=COMMON_PILE_DATASET_REVISION, index=index
            )
            shards_opened.append(shard_file)

            with contextlib.closing(opener(url)) as raw:
                counting = _CountingReader(raw)
                try:
                    with io.TextIOWrapper(
                        gzip.GzipFile(fileobj=counting, mode="rb"), encoding="utf-8"
                    ) as lines:
                        for line in lines:
                            if not line.strip():
                                continue
                            if records_scanned >= max_records_scanned:
                                raise RealCorpusAcquisitionError(
                                    f"scan cap reached: examined {records_scanned} records without "
                                    f"collecting {max_documents} qualifying documents "
                                    f"(collected {len(documents)}); raise max_records_scanned, "
                                    f"add shard_indices, or lower max_documents"
                                )
                            records_scanned += 1
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError as exc:
                                raise RealCorpusAcquisitionError(
                                    f"{shard_file}: record #{records_scanned} is not valid JSON "
                                    f"({exc}); the source format differs from the expected "
                                    f"one-JSON-object-per-line layout"
                                ) from exc

                            status, payload = _map_record(
                                record, shard_file=shard_file, main_namespace_only=main_namespace_only
                            )
                            if status != "ok":
                                skipped[status] += 1
                                if skipped[_SKIP_MALFORMED] + skipped[_SKIP_MISSING_LICENSE] > max_unusable_records:
                                    raise RealCorpusAcquisitionError(
                                        f"{shard_file}: more than {max_unusable_records} records were "
                                        f"unusable (malformed={skipped[_SKIP_MALFORMED]}, "
                                        f"missing_license={skipped[_SKIP_MISSING_LICENSE]}) -- either "
                                        f"the dataset schema differs from what this code expects or "
                                        f"license metadata is absent; refusing to bypass the license gate"
                                    )
                                continue

                            document, record_provenance = payload
                            if document.document_id in seen_ids:
                                skipped[_SKIP_DUPLICATE_ID] += 1
                                continue
                            seen_ids.add(document.document_id)
                            documents.append(document)
                            provenance[document.document_id] = record_provenance
                            if len(documents) >= max_documents:
                                break
                finally:
                    compressed_bytes_read += counting.bytes_read
    except RealCorpusAcquisitionError:
        raise
    except (OSError, EOFError, http.client.HTTPException, zlib.error, UnicodeDecodeError) as exc:
        status_code = getattr(exc, "code", None)
        detail = f"HTTP {status_code}: " if status_code is not None else ""
        raise RealCorpusAcquisitionError(
            f"could not read {COMMON_PILE_DATASET_ID}@{COMMON_PILE_DATASET_REVISION[:12]} "
            f"({shards_opened[-1] if shards_opened else 'no shard opened'}): "
            f"{detail}{type(exc).__name__}: {exc}"
        ) from exc

    if len(documents) < max_documents:
        raise RealCorpusAcquisitionError(
            f"requested shards exhausted after {records_scanned} records with only "
            f"{len(documents)} of {max_documents} qualifying documents "
            f"(skipped: {dict(sorted(skipped.items()))}); add shard_indices or lower max_documents"
        )

    report = AcquisitionReport(
        dataset_id=COMMON_PILE_DATASET_ID,
        dataset_revision=COMMON_PILE_DATASET_REVISION,
        shard_files=tuple(shards_opened),
        max_documents=max_documents,
        main_namespace_only=main_namespace_only,
        records_scanned=records_scanned,
        records_selected=len(documents),
        skipped=dict(skipped),
        compressed_bytes_read=compressed_bytes_read,
        provenance=provenance,
    )
    return documents, report


def load_common_pile_wikimedia_sample(max_documents: int = 1000) -> list[RawDocument]:
    """Convenience wrapper returning only the documents. Use
    :func:`acquire_common_pile_wikimedia_sample` when the
    :class:`AcquisitionReport` (provenance, counts, bytes read) is
    needed -- ``__main__`` does, so it lands in the result artifact."""
    documents, _report = acquire_common_pile_wikimedia_sample(max_documents)
    return documents


def _count_words(text: str) -> int:
    """The one documented word-count definition this experiment uses:
    whitespace-split token count on already-normalized text. Simple
    and deterministic by construction (``str.split()`` with no
    arguments splits on any run of whitespace and discards empty
    strings) -- not a linguistic tokenizer, deliberately."""
    return len(text.split())


def compute_tokenizer_stats(tokenizer, texts: list[str]) -> dict:
    """Tokenizer efficiency/health statistics over ``texts`` (already-
    normalized, accepted document text). Genuinely new: nothing
    existing in this repository already computes this.

    Does NOT report an unknown-token count/rate: verified directly
    (``yug/tokenizer/bpe.py``'s own docstring) that
    ``ByteLevelBpeTokenizer`` has no ``<unk>`` concept at all -- every
    byte is already a valid base symbol, so an unknown-token fallback
    would be dead code, per that module's own documented design
    decision. Inventing a metric here for a concept the tokenizer
    doesn't have would be exactly what this directive's TOKENIZER
    section forbids, so the key is omitted rather than reported as an
    always-zero placeholder.
    """
    from yug.tokenizer.errors import DecodeError

    total_characters = 0
    total_words = 0
    total_tokens = 0
    encoding_failures = 0
    decoding_failures = 0

    for text in texts:
        total_characters += len(text)
        total_words += _count_words(text)
        try:
            token_ids = tokenizer.encode(text)
        except Exception:  # noqa: BLE001 - record, do not crash the stat pass
            encoding_failures += 1
            continue
        total_tokens += len(token_ids)
        try:
            tokenizer.decode(token_ids)
        except DecodeError:
            # Not expected for a direct encode->decode round trip (see
            # tokenizer's own DecodeError docstring: this is for
            # arbitrary/generated token sequences that don't land on a
            # valid UTF-8 byte boundary, not for decoding a sequence
            # the tokenizer itself just produced from valid text) --
            # measured honestly rather than assumed to always be 0.
            decoding_failures += 1

    return {
        "tokenizer_identity": tokenizer.identity,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "document_count": len(texts),
        "total_characters": total_characters,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "tokens_per_word": (total_tokens / total_words) if total_words else None,
        "tokens_per_character": (total_tokens / total_characters) if total_characters else None,
        "encoding_failures": encoding_failures,
        "decoding_failures": decoding_failures,
        "unknown_token_count": None,
        "unknown_token_rate": None,
        "unknown_token_note": (
            "not applicable: ByteLevelBpeTokenizer has no <unk> concept "
            "(every byte is a valid base symbol) -- see yug/tokenizer/bpe.py"
        ),
    }


def _sequence_utilization(num_sequences: int, sequence_length: int, total_tokens: int) -> float | None:
    """Fraction of encoded tokens that actually ended up inside a
    training sequence. ``make_sequences`` truncates a final partial
    chunk (see ``training/dataset.py``), so this is always <= 1.0 and
    directly derivable from its own documented chunking behavior --
    not a new metric definition, just this experiment's own bookkeeping
    of an existing function's already-documented truncation."""
    if total_tokens == 0:
        return None
    return (num_sequences * (sequence_length + 1)) / total_tokens


def train_tokenizer_on_train_partition(prep, *, vocab_size: int = TOKENIZER_VOCAB_SIZE):
    """Train the EXISTING ``ByteLevelBpeTokenizer`` on the TRAINING
    partition of a ``CorpusPrepResult`` only. Validation documents are
    never passed to tokenizer training (Directive: "Validation
    documents MUST NOT participate in tokenizer training").

    Behaviour-preserving extraction of what
    :func:`run_real_corpus_experiment` already did inline; pulled out
    only so this property is testable without PyTorch. ``vocab_size``
    is a target, not a guarantee -- BPE can stop short if the corpus
    has no more mergeable pairs -- so callers must report the actual
    ``tokenizer.vocab_size`` (``run_real_corpus_experiment`` does).
    """
    from yug.tokenizer.bpe import ByteLevelBpeTokenizer

    train_texts = [doc.text for doc in prep.train]
    return ByteLevelBpeTokenizer.train(
        corpus=train_texts, vocab_size=vocab_size, special_tokens=("bos", "eos")
    )


def build_training_config():
    """Historical Scale-3 training configuration, reused verbatim (same
    values as ``training.experiments.nb08_experiment.build_training_config``
    and ``scaling_scale3_budget_extension.run``), with this experiment's
    own authorized ``max_steps=3000`` (directive's "Maximum training
    budget") in place of NB-08's 500 / the extension's 5000."""
    from training.config import TrainingConfig

    return TrainingConfig(
        seed=SEED,
        learning_rate=3e-3,
        batch_size=4,
        sequence_length=SEQUENCE_LENGTH,
        max_steps=MAX_STEPS,
        eval_interval=50,
        checkpoint_interval=MAX_STEPS,
        grad_clip_norm=1.0,
        device="cpu",
    )


def _check_state_dict_finite(state_dict) -> tuple[bool, str]:
    """Copied verbatim from
    ``training.experiments.scaling_scale3_budget_extension._check_state_dict_finite``
    (itself copied from ``training.experiments.nb12_checkpoint_closure``'s
    equivalent helper) -- not re-derived, not imported across
    independent experiment scripts, matching both files' own stated
    reason for keeping this local."""
    import torch

    bad = [name for name, t in state_dict.items() if not torch.isfinite(t).all()]
    if bad:
        return False, f"NaN/Inf found in checkpoint parameters: {bad}"
    return True, "all checkpoint parameters finite"


def run_real_corpus_experiment(documents: list[RawDocument]) -> dict:
    """Run the complete first-real-corpus pipeline on ``documents``.

    Reuses ``training.corpus_prep.prepare_corpus`` for the ENTIRE
    provenance/license/normalize/filter/dedup/split stage (Directive:
    "Do not bypass training/corpus_prep.py") -- the document-level
    train/validation split happens there, before any tokenization, so
    "content from the same source document must never appear in both
    train and validation" is corpus_prep's own already-tested
    guarantee, not reimplemented here.

    Raises:
        CorpusPrepError: (from ``prepare_corpus``) ``documents`` is
            empty, or too few documents survive to produce both a
            non-empty train and validation split.
    """
    import torch

    from training.checkpoint import save_checkpoint
    from training.dataset import encode_corpus, iter_batches, make_sequences
    from training.determinism import seed_everything
    from training.evaluate import evaluate_checkpoint
    from training.experiments.nb08_baseline import unigram_baseline_loss, uniform_baseline_loss
    from training.experiments.nb08_experiment import safe_decode_generation
    from training.loop import Trainer
    from yug.native.config import ModelConfig
    from yug.native.generation import generate
    from yug.native.model import YugTransformer

    result: dict = {
        "experiment": "first_real_corpus_experiment",
        "not_a_milestone": True,
        "nb13_assigned": False,
        "dataset_source": "Common Pile - Wikimedia Filtered",
        "dataset_license": WIKIMEDIA_LICENSE,
        "requested_document_count": len(documents),
        "stop_reason": None,
    }

    prep = prepare_corpus(
        documents,
        license_policy=REAL_CORPUS_LICENSE_POLICY,
        val_fraction=VAL_FRACTION,
        seed=SEED,
    )
    result["corpus_manifest"] = prep.manifest.to_dict()

    train_texts = [doc.text for doc in prep.train]
    val_texts = [doc.text for doc in prep.validation]

    tokenizer = train_tokenizer_on_train_partition(prep)
    result["tokenizer_training"] = {
        "trained_on": "train_partition_only",
        "train_document_count": len(prep.train),
        "requested_vocab_size": TOKENIZER_VOCAB_SIZE,
        "actual_vocab_size": tokenizer.vocab_size,
        "tokenizer_identity": tokenizer.identity,
    }
    result["tokenizer_stats"] = {
        "train": compute_tokenizer_stats(tokenizer, train_texts),
        "validation": compute_tokenizer_stats(tokenizer, val_texts),
    }

    train_token_ids = encode_corpus(tokenizer, train_texts)
    val_token_ids = encode_corpus(tokenizer, val_texts)
    train_sequences = make_sequences(train_token_ids, sequence_length=SEQUENCE_LENGTH)
    val_sequences = make_sequences(val_token_ids, sequence_length=SEQUENCE_LENGTH)

    result["sequence_utilization"] = {
        "train": _sequence_utilization(len(train_sequences), SEQUENCE_LENGTH, len(train_token_ids)),
        "validation": _sequence_utilization(len(val_sequences), SEQUENCE_LENGTH, len(val_token_ids)),
    }

    if not train_sequences or not val_sequences:
        result["stop_reason"] = (
            f"insufficient tokens for at least one sequence: "
            f"train_sequences={len(train_sequences)}, val_sequences={len(val_sequences)}"
        )
        return result

    model_config = ModelConfig.from_tokenizer(
        tokenizer,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ffn_size=FFN_SIZE,
        context_length=CONTEXT_LENGTH,
    )
    training_config = build_training_config()

    seed_everything(SEED)
    model = YugTransformer(model_config)
    parameter_count = model.count_parameters()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay
    )
    trainer = Trainer(model, optimizer, training_config, model_config, tokenizer.identity)

    train_batches = iter_batches(train_sequences, training_config.batch_size)
    val_batches = iter_batches(val_sequences, training_config.batch_size)

    def _current_losses() -> tuple[float, float]:
        return trainer.evaluate(train_batches), trainer.evaluate(val_batches)

    import tempfile
    from pathlib import Path

    checkpoint_root = Path(tempfile.mkdtemp(prefix="real_corpus_experiment_"))

    training_curve: list[dict] = []
    checkpoints_saved: dict[int, dict] = {}
    generation_samples: dict[int, dict[str, str]] = {}
    stop_reason: str | None = None

    initial_train_loss, initial_val_loss = _current_losses()
    training_curve.append({"step": 0, "train_loss": initial_train_loss, "val_loss": initial_val_loss})
    if not (math.isfinite(initial_train_loss) and math.isfinite(initial_val_loss)):
        stop_reason = "non-finite loss at step 0 (before any training)"

    batch_idx = 0
    while stop_reason is None and trainer.step < training_config.max_steps:
        input_ids, target_ids = train_batches[batch_idx % len(train_batches)]
        step_train_loss = trainer.train_step(input_ids, target_ids)
        batch_idx += 1

        if not math.isfinite(step_train_loss):
            stop_reason = f"non-finite training-step loss at step {trainer.step}: {step_train_loss}"
            break

        if trainer.step not in EVAL_CHECKPOINTS:
            continue

        train_loss, val_loss = _current_losses()
        training_curve.append({"step": trainer.step, "train_loss": train_loss, "val_loss": val_loss})

        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            stop_reason = f"non-finite loss at step {trainer.step}: train={train_loss} val={val_loss}"
            break

        if trainer.step in RETAINED_CHECKPOINTS:
            ckpt_dir = checkpoint_root / f"step_{trainer.step}"
            save_checkpoint(
                ckpt_dir,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                training_config=training_config,
                step=trainer.step,
                tokenizer_identity=tokenizer.identity,
            )
            raw_state = torch.load(ckpt_dir / "state.pt", map_location="cpu")
            state_ok, state_detail = _check_state_dict_finite(raw_state["model"])

            eval_ok, eval_detail, eval_loss, eval_ppl = True, None, None, None
            evaluation_report = None
            try:
                eval_report = evaluate_checkpoint(
                    str(ckpt_dir),
                    tokenizer,
                    val_texts,
                    sequence_length=SEQUENCE_LENGTH,
                    batch_size=training_config.batch_size,
                    seed=SEED,
                    data_provenance={
                        "corpus_identity": None, "split_name": "validation",
                        "split_method": "training.split.split_sentences",
                        "split_seed": SEED, "split_fraction": VAL_FRACTION,
                    },
                )
                evaluation_report = eval_report.to_dict()
                eval_loss, eval_ppl = eval_report.loss, eval_report.perplexity
                if not (math.isfinite(eval_loss) and math.isfinite(eval_ppl)):
                    eval_ok = False
                    eval_detail = f"non-finite: loss={eval_loss} perplexity={eval_ppl}"
            except Exception as exc:  # noqa: BLE001 - record, don't crash the experiment
                eval_ok = False
                eval_detail = f"{type(exc).__name__}: {exc}"

            checkpoints_saved[trainer.step] = {
                "checkpoint_dir": str(ckpt_dir),
                "state_finite": state_ok,
                "state_detail": state_detail,
                "evaluation_report": evaluation_report if eval_ok else None,
                "reload_eval_success": eval_ok,
                "reload_eval_loss": eval_loss,
                "reload_eval_perplexity": eval_ppl,
                "reload_eval_detail": eval_detail,
            }
            if not state_ok or not eval_ok:
                stop_reason = (
                    f"checkpoint integrity/reload-evaluation failure at step "
                    f"{trainer.step}: state_ok={state_ok}, eval_ok={eval_ok}"
                )

        # Sustained-overfitting stop condition, copied verbatim from
        # training.experiments.scaling_scale3_budget_extension.run
        # (Directive: "reuse it rather than implementing a competing
        # definition"). Requires a consistent trend across 3
        # consecutive transitions in the last 4 recorded curve points,
        # not a single blip.
        if stop_reason is None and len(training_curve) >= 4:
            last4 = training_curve[-4:]
            val_rising = all(last4[i]["val_loss"] < last4[i + 1]["val_loss"] for i in range(3))
            train_falling = all(last4[i]["train_loss"] > last4[i + 1]["train_loss"] for i in range(3))
            if val_rising and train_falling:
                stop_reason = (
                    f"sustained overfitting detected: val loss rose and train "
                    f"loss fell across the last 4 recorded checkpoints "
                    f"(steps {[c['step'] for c in last4]})"
                )

    # Generation probes: ALL fixed prompts, unconditionally recorded
    # (Directive: "Do not select only successful-looking generations"),
    # run once at the end of training (or at the point training
    # stopped).
    for prompt in GENERATION_PROMPTS:
        prompt_ids = torch.tensor([encode_corpus(tokenizer, [prompt])], dtype=torch.long)
        generated_ids = generate(model, prompt_ids, max_new_tokens=10)
        generation_samples[prompt] = safe_decode_generation(tokenizer, generated_ids[0].tolist())

    final_train_loss_full_eval, final_val_loss = _current_losses()
    best_point = min(training_curve, key=lambda c: c["val_loss"])

    result.update(
        {
            "seed": SEED,
            "model_config": {
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "num_heads": NUM_HEADS,
                "ffn_size": FFN_SIZE,
                "context_length": CONTEXT_LENGTH,
                "sequence_length": SEQUENCE_LENGTH,
                "vocab_size": model_config.vocab_size,
            },
            "parameter_count": parameter_count,
            "training_config": {
                "learning_rate": training_config.learning_rate,
                "batch_size": training_config.batch_size,
                "max_steps": training_config.max_steps,
                "grad_clip_norm": training_config.grad_clip_norm,
                "weight_decay": training_config.weight_decay,
            },
            "train_document_count": len(prep.train),
            "validation_document_count": len(prep.validation),
            "train_tokens": len(train_token_ids),
            "val_tokens": len(val_token_ids),
            "initial_train_loss": initial_train_loss,
            "final_train_loss_full_eval": final_train_loss_full_eval,
            "final_val_loss": final_val_loss,
            "best_val_loss": best_point["val_loss"],
            "best_val_step": best_point["step"],
            "uniform_baseline_loss": uniform_baseline_loss(model_config.vocab_size),
            "unigram_baseline_loss": unigram_baseline_loss(train_token_ids, val_token_ids),
            "training_curve": training_curve,
            "checkpoints_saved": checkpoints_saved,
            "generation_probes": generation_samples,
            "num_steps_completed": trainer.step,
            "reached_max_steps": trainer.step >= training_config.max_steps and stop_reason is None,
            "stop_reason": stop_reason,
        }
    )
    return result


if __name__ == "__main__":
    import argparse
    import pathlib
    import sys

    parser = argparse.ArgumentParser(description="First real-corpus experiment (not a milestone).")
    parser.add_argument("--max-documents", type=int, default=1000)
    parser.add_argument(
        "--output",
        default=None,
        help="write the UTF-8 JSON result to this path instead of stdout "
        "(avoids Windows PowerShell's UTF-16 redirection)",
    )
    parser.add_argument(
        "--acquire-only",
        action="store_true",
        help="run ONLY the bounded acquisition and print a summary (no PyTorch needed); "
        "use this to verify network access and the dataset schema before a long training run",
    )
    args = parser.parse_args()

    try:
        real_documents, acquisition_report = acquire_common_pile_wikimedia_sample(args.max_documents)
    except RealCorpusAcquisitionError as exc:
        print(f"ACQUISITION FAILED: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.acquire_only:
        summary = acquisition_report.to_dict()
        summary["provenance_sample"] = dict(list(summary.pop("provenance").items())[:3])
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        sys.exit(0)

    outcome = run_real_corpus_experiment(real_documents)
    outcome["acquisition"] = acquisition_report.to_dict()
    payload = json.dumps(outcome, indent=2, sort_keys=True, default=str)
    if args.output:
        pathlib.Path(args.output).write_text(payload, encoding="utf-8")
        print(f"result written to {args.output}")
    else:
        print(payload)
    sys.exit(0 if outcome["stop_reason"] is None or "overfitting" in (outcome["stop_reason"] or "") else 1)
