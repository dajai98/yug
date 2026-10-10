"""YUG's own byte-level BPE tokenizer.

Design decisions made in this module (see accompanying NB-02 report and
``docs/architecture/NATIVE_BRAIN_ARCHITECTURE.md`` for the formal
record):

- **Algorithm:** byte-level BPE. The base vocabulary is the 256
  possible byte values (id == byte value). Every valid Unicode string
  (any script -- Latin, Devanagari/Nepali, CJK, emoji, or otherwise) is
  therefore representable at minimum as its raw UTF-8 bytes, even
  before any merge applies. This is a structural guarantee, not a
  best-effort one.
- **No <unk> token.** Because every byte is already a valid base
  token, there is no input that ``encode()`` can fail to represent.
  An unknown-token fallback would be dead code. (Decision, not
  deferred -- this is a property of the algorithm, independent of any
  future model's vocabulary-size decision.)
- **No word-level pre-tokenization** (unlike GPT-2-style tokenizers,
  which first split text with a regex before applying BPE). Merges
  here operate directly on the raw byte stream. This is simpler,
  fully deterministic, and avoids reproducing a pretokenization scheme
  designed for a different, external tokenizer. Trade-off: likely
  somewhat less compression-efficient on e.g. whitespace-heavy text
  than a word-aware scheme -- acceptable for a correctness-first
  prototype (NB-02 does not require performance tuning).
- **Vocabulary size is a training-time parameter, not a constant.**
  Nothing in this module hardcodes a production vocabulary size; that
  decision is explicitly deferred to when the model architecture's
  parameter budget is fixed (see architecture doc, Open Decisions).
- **Special tokens are a configurable, named set** passed to
  :meth:`train`, not a fixed hardcoded list. ``bos``/``eos`` are used
  as the default prototype set in this codebase's tests; a ``pad``
  token is deliberately NOT included by default -- whether batched
  training needs one is a training-pipeline decision (NB-08), not a
  tokenizer one.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from yug.tokenizer.errors import DecodeError, SerializationError, TokenizerError, VocabularyError

_FORMAT_VERSION = "1"
_ALGORITHM = "byte_level_bpe"
_NUM_BASE_BYTES = 256


def _get_adjacent_pairs(sequences: list[list[int]]) -> Counter:
    counts: Counter = Counter()
    for seq in sequences:
        # Bulk counting preserves first encounter order (including ties) while
        # avoiding a Python-level increment for every adjacent pair.
        counts.update(zip(seq, seq[1:]))
    return counts


def _apply_merge(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` in ``seq``
    with ``new_id``, scanning left to right."""
    out: list[int] = []
    i = 0
    n = len(seq)
    while i < n:
        if i < n - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def _learn_merges(sequences: list[list[int]], target_merges: int) -> list[tuple[int, int]]:
    """Cache document counts; recount/rebuild only documents containing a merge.

    Counter key order is the first occurrence in the CURRENT document. Walking
    these counters in corpus order therefore reproduces the original global
    Counter's first-occurrence tie rule, independently of totals/index order.
    """
    merge_indices = range(target_merges)
    if not merge_indices:
        return []
    document_counts = [Counter(zip(seq, seq[1:])) for seq in sequences]
    totals: Counter = Counter()
    pair_documents: dict[tuple[int, int], set[int]] = {}
    for document, counts in enumerate(document_counts):
        totals.update(counts)
        for pair in counts:
            pair_documents.setdefault(pair, set()).add(document)

    merges: list[tuple[int, int]] = []
    for i in merge_indices:
        highest = max(totals.values(), default=0)
        if highest < 2:
            break
        # Never use totals' insertion order to break a tie: updates can change it.
        best_pair = next(pair for counts in document_counts for pair in counts
                         if totals[pair] == highest)
        merges.append(best_pair)
        # Snapshot membership before updating the index; document order stays fixed.
        for document in sorted(pair_documents[best_pair]):
            old_counts = document_counts[document]
            seq = _apply_merge(sequences[document], best_pair, _NUM_BASE_BYTES + i)
            new_counts = Counter(zip(seq, seq[1:]))
            for pair, count in old_counts.items():
                totals[pair] -= count
                pair_documents[pair].remove(document)
                if totals[pair] == 0:
                    del totals[pair]
                    del pair_documents[pair]
            for pair, count in new_counts.items():
                totals[pair] += count
                pair_documents.setdefault(pair, set()).add(document)
            sequences[document] = seq
            document_counts[document] = new_counts
    return merges


class ByteLevelBpeTokenizer:
    """A YUG-owned byte-level BPE tokenizer.

    Construct via :meth:`train` (learns merges from a corpus) or
    :meth:`load` (restores a previously saved tokenizer). The
    constructor itself is a low-level assembly point and is not the
    primary public API.
    """

    def __init__(
        self,
        merges: Sequence[tuple[int, int]],
        special_tokens: dict[str, int],
    ) -> None:
        self._merges: tuple[tuple[int, int], ...] = tuple(tuple(m) for m in merges)
        self._special_tokens: dict[str, int] = dict(special_tokens)

        # merge (pair) -> rank (earlier learned = lower rank = applied first)
        self._pair_rank: dict[tuple[int, int], int] = {
            pair: rank for rank, pair in enumerate(self._merges)
        }
        # merge (pair) -> resulting new token id
        self._pair_to_id: dict[tuple[int, int], int] = {
            pair: _NUM_BASE_BYTES + rank for rank, pair in enumerate(self._merges)
        }
        self._special_id_set: set[int] = set(self._special_tokens.values())

        # Lazily populated: token id -> raw bytes it expands to.
        self._expand_cache: dict[int, bytes] = {}

    # -- construction -----------------------------------------------

    @classmethod
    def train(
        cls,
        corpus: Sequence[str],
        vocab_size: int,
        special_tokens: Sequence[str] = ("bos", "eos"),
    ) -> "ByteLevelBpeTokenizer":
        """Learn BPE merges from ``corpus`` up to (at most) ``vocab_size``
        total tokens (256 base bytes + merges + ``len(special_tokens)``).

        Training may stop before reaching ``vocab_size`` if the corpus
        is exhausted (no adjacent pair repeats) -- the actual resulting
        vocabulary size is never silently padded to match the request;
        check :attr:`vocab_size` on the returned instance.

        Determinism: for a fixed ``corpus`` (as an ordered sequence)
        and ``vocab_size``, training always produces identical merges.
        Ties between equally-frequent pairs are broken by first
        occurrence in the current corpus. Ordered per-document counters
        preserve that rule even as global counts are updated incrementally;
        arbitrary hash or index insertion order never decides ties.

        Raises:
            VocabularyError: if ``vocab_size`` is too small to hold the
                256 base bytes plus the requested special tokens.
        """
        reserved = len(special_tokens)
        minimum = _NUM_BASE_BYTES + reserved
        if vocab_size < minimum:
            raise VocabularyError(
                f"vocab_size={vocab_size} is too small: need at least "
                f"{minimum} ({_NUM_BASE_BYTES} base bytes + {reserved} "
                f"special token(s))."
            )

        target_merges = vocab_size - minimum
        sequences: list[list[int]] = [list(text.encode("utf-8")) for text in corpus]

        merges = _learn_merges(sequences, target_merges)

        special_token_ids = {
            name: _NUM_BASE_BYTES + len(merges) + idx
            for idx, name in enumerate(special_tokens)
        }
        return cls(merges=merges, special_tokens=special_token_ids)

    # -- properties ---------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Actual total vocabulary size: base bytes + merges + specials."""
        return _NUM_BASE_BYTES + len(self._merges) + len(self._special_tokens)

    @property
    def special_tokens(self) -> dict[str, int]:
        """Mapping of special-token name -> token id (a copy)."""
        return dict(self._special_tokens)

    @property
    def identity(self) -> str:
        """A stable content hash identifying this exact vocabulary +
        merge table + special-token set.

        Computed as ``"sha256:" + hexdigest`` of the canonical JSON
        (sorted keys, compact separators, no whitespace) of::

            {"algorithm": ..., "vocab_size": ..., "merges": [...],
             "special_tokens": {...}}

        Two tokenizer instances with identical merges and special
        tokens always produce the same identity, regardless of how
        they were constructed (trained fresh vs. loaded from disk).
        This is what a future checkpoint system would reference to
        detect an incompatible tokenizer (see architecture doc,
        "Tokenizer Identity").
        """
        return self._compute_identity(self._merges, self._special_tokens)

    @staticmethod
    def _compute_identity(
        merges: Sequence[tuple[int, int]], special_tokens: dict[str, int]
    ) -> str:
        payload = {
            "algorithm": _ALGORITHM,
            "vocab_size": _NUM_BASE_BYTES + len(merges) + len(special_tokens),
            "merges": [list(m) for m in merges],
            "special_tokens": special_tokens,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    # -- encode / decode -----------------------------------------------

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        """Encode ``text`` to a list of token IDs.

        Never raises for any valid Python ``str`` input, including
        empty strings, arbitrary Unicode, or emoji -- byte-level
        coverage is unconditional (see module docstring).

        Args:
            add_bos: prepend the ``"bos"`` special token id.
            add_eos: append the ``"eos"`` special token id.

        Raises:
            TypeError: if ``text`` is not a ``str``.
            TokenizerError: if ``add_bos``/``add_eos`` is requested but
                this tokenizer has no ``"bos"``/``"eos"`` special token.
        """
        if not isinstance(text, str):
            raise TypeError(f"encode() requires str, got {type(text).__name__}")

        ids = self._apply_merges(list(text.encode("utf-8")))

        if add_bos:
            ids = [self._require_special("bos")] + ids
        if add_eos:
            ids = ids + [self._require_special("eos")]
        return ids

    def _require_special(self, name: str) -> int:
        if name not in self._special_tokens:
            raise TokenizerError(
                f"This tokenizer has no special token named {name!r}. "
                f"Available: {sorted(self._special_tokens)}"
            )
        return self._special_tokens[name]

    def _apply_merges(self, ids: list[int]) -> list[int]:
        while True:
            best_pair = None
            best_rank = None
            for a, b in zip(ids, ids[1:]):
                rank = self._pair_rank.get((a, b))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = (a, b)
            if best_pair is None:
                return ids
            new_id = self._pair_to_id[best_pair]
            ids = _apply_merge(ids, best_pair, new_id)

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decode a sequence of token IDs back to text.

        Args:
            skip_special_tokens: if True (default), special-token IDs
                (bos/eos/etc.) are silently omitted from the output
                text. If False, encountering one raises
                :class:`DecodeError` (they have no byte representation).

        Raises:
            TypeError: if ``ids`` is not a sequence of integers.
            DecodeError: an id is outside this tokenizer's vocabulary,
                a special-token id was found with
                ``skip_special_tokens=False``, or the reassembled bytes
                are not valid UTF-8 (e.g. a slice that cuts a
                multi-byte character in half).
        """
        if isinstance(ids, (str, bytes)) or not hasattr(ids, "__iter__"):
            raise TypeError(f"decode() requires a sequence of ints, got {type(ids).__name__}")

        out = bytearray()
        for token_id in ids:
            if not isinstance(token_id, int):
                raise TypeError(f"decode() requires a sequence of ints, got element of type {type(token_id).__name__}")

            if token_id in self._special_id_set:
                if skip_special_tokens:
                    continue
                raise DecodeError(
                    f"Token id {token_id} is a special token and has no byte "
                    f"representation; pass skip_special_tokens=True to ignore it."
                )
            out.extend(self._expand(token_id))

        try:
            return bytes(out).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DecodeError(
                f"Decoded bytes are not valid UTF-8 (likely a token id slice "
                f"that split a multi-byte character): {exc}"
            ) from exc

    def _expand(self, token_id: int) -> bytes:
        """Recursively expand a token id to its raw bytes, memoized."""
        if token_id in self._expand_cache:
            return self._expand_cache[token_id]

        if 0 <= token_id < _NUM_BASE_BYTES:
            result = bytes([token_id])
        else:
            rank = token_id - _NUM_BASE_BYTES
            if 0 <= rank < len(self._merges):
                a, b = self._merges[rank]
                result = self._expand(a) + self._expand(b)
            else:
                raise DecodeError(
                    f"Token id {token_id} is outside this tokenizer's "
                    f"vocabulary (vocab_size={self.vocab_size})."
                )

        self._expand_cache[token_id] = result
        return result

    # -- serialization -----------------------------------------------

    def to_dict(self) -> dict:
        """Return the canonical JSON-serializable representation."""
        return {
            "yug_tokenizer_format_version": _FORMAT_VERSION,
            "algorithm": _ALGORITHM,
            "vocab_size": self.vocab_size,
            "num_base_bytes": _NUM_BASE_BYTES,
            "merges": [list(m) for m in self._merges],
            "special_tokens": dict(self._special_tokens),
            "identity": self.identity,
        }

    def save(self, path: str | Path) -> None:
        """Serialize to a JSON file (never pickle -- see architecture
        doc's rejection of pickle-based formats for the same reason
        applied there to model checkpoints: pickle can execute
        arbitrary code on load)."""
        data = self.to_dict()
        Path(path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteLevelBpeTokenizer":
        """Load a tokenizer previously written by :meth:`save`.

        Raises:
            SerializationError: file missing, invalid JSON, wrong/
                missing format version, structurally invalid, or its
                stored identity hash does not match a hash recomputed
                from its own contents (integrity check -- catches
                corruption or hand-edited files).
        """
        file_path = Path(path)
        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise SerializationError(f"Tokenizer file not found: {file_path}") from exc

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise SerializationError(f"Invalid JSON in tokenizer file {file_path}: {exc}") from exc

        if not isinstance(data, dict):
            raise SerializationError(f"Tokenizer file {file_path} must contain a JSON object")

        version = data.get("yug_tokenizer_format_version")
        if version != _FORMAT_VERSION:
            raise SerializationError(
                f"Unsupported tokenizer format version {version!r} in {file_path}; "
                f"expected {_FORMAT_VERSION!r}."
            )

        required = {"algorithm", "merges", "special_tokens", "identity"}
        missing = required - data.keys()
        if missing:
            raise SerializationError(
                f"Tokenizer file {file_path} is missing required field(s): {sorted(missing)}"
            )

        if data["algorithm"] != _ALGORITHM:
            raise SerializationError(
                f"Tokenizer file {file_path} has algorithm={data['algorithm']!r}, "
                f"this loader only supports {_ALGORITHM!r}."
            )

        try:
            merges = [tuple(pair) for pair in data["merges"]]
            special_tokens = dict(data["special_tokens"])
        except (TypeError, ValueError) as exc:
            raise SerializationError(f"Malformed merges/special_tokens in {file_path}: {exc}") from exc

        expected_identity = cls._compute_identity(merges, special_tokens)
        if data["identity"] != expected_identity:
            raise SerializationError(
                f"Identity hash mismatch in {file_path}: file claims "
                f"{data['identity']!r} but recomputes to {expected_identity!r}. "
                f"The file may be corrupted or hand-edited."
            )

        return cls(merges=merges, special_tokens=special_tokens)
