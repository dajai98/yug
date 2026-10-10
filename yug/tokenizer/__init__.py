"""YUG tokenizer subsystem.

Foundational, standalone infrastructure (NB-02 in
``docs/architecture/NATIVE_BRAIN_ARCHITECTURE.md``). Owned entirely by
YUG: no pretrained external tokenizer, vocabulary, or merge table is
used or depended upon.

This package must not import from ``yug.brain``, ``yug.native``, or
``training/`` -- the dependency direction is the reverse (those
consume the tokenizer, the tokenizer consumes nothing from them). See
the architecture document's "BOUNDARIES" section.
"""

from yug.tokenizer.bpe import ByteLevelBpeTokenizer
from yug.tokenizer.errors import (
    DecodeError,
    SerializationError,
    TokenizerError,
    VocabularyError,
)

__all__ = [
    "ByteLevelBpeTokenizer",
    "TokenizerError",
    "VocabularyError",
    "DecodeError",
    "SerializationError",
]
