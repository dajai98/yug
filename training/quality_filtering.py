"""Versioned production extension of the immutable Phase 1 quality checks.

The historical corpus_prep.py is hash-pinned evidence in production_corpus_v1.
This module calls its check_quality first and adds only the bounded Phase 2
structure/information checks. It does not create a separate admission route.
"""

from collections import Counter
from enum import Enum
import re
import unicodedata

from training.corpus_prep import QualityConfig, check_quality as baseline_check_quality


class QualityReason(Enum):
    EXCESSIVE_REPEATED_STRUCTURE = "excessive_repeated_structure"
    LOW_INFORMATION = "low_information"


# Expectations were declared before calibration in tests/test_quality_filtering.py.
# A semantic change requires a version bump; the identity also pins all limits.
QUALITY_POLICY_VERSION = "bounded-quality-v1"
QUALITY_LIMITS = {
    "line_min_characters": 12,
    "block_lines": 3,
    "block_min_characters": 48,
    "repeat_min_occurrences": 3,
    "structure_min_lines": 10,
    "structure_min_characters": 200,
    "repeated_line_fraction": 0.80,
    "repeated_block_fraction": 0.80,
    "dominant_line_fraction": 0.80,
    "lexical_min_words": 80,
    "lexical_max_unique_fraction": 0.05,
    "meaningful_min_prose_markdown": 0.20,
    "meaningful_min_code": 0.10,
    "meaningful_min_characters": 80,
    "invisible_min_characters": 80,
    "invisible_max_fraction": 0.20,
}


def quality_policy(config: QualityConfig, generated_markers: list[str]) -> dict:
    """Complete versioned contract for the effective production checks."""
    return {
        "version": QUALITY_POLICY_VERSION,
        "min_length": config.min_length,
        "min_repeated_tokens_for_pathological": config.min_repeated_tokens_for_pathological,
        "limits": dict(QUALITY_LIMITS),
        "generated_markers": list(generated_markers),
        "generated_scan": "whole_bounded_decoded_document",
        "normalization": "prose-v1/structured-code-markdown-v1",
        "content_types": ["code", "markdown", "prose"],
    }


def quality_metrics(normalized_text: str) -> dict[str, float | int]:
    """Fixed-size line-window metrics; O(n) time and memory per document."""
    lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]
    total_line_chars = sum(map(len, lines)) or 1
    eligible = Counter(line for line in lines if len(line) >= QUALITY_LIMITS["line_min_characters"])
    minimum = QUALITY_LIMITS["repeat_min_occurrences"]
    repeated_line_chars = sum(len(line) * count for line, count in eligible.items() if count >= minimum)
    dominant_line_chars = max((len(line) * count for line, count in eligible.items() if count >= minimum), default=0)
    width = QUALITY_LIMITS["block_lines"]
    blocks = Counter(tuple(lines[i:i + width]) for i in range(max(0, len(lines) - width + 1))
                     if sum(map(len, lines[i:i + width])) >= QUALITY_LIMITS["block_min_characters"])
    covered = bytearray(len(lines))
    for i in range(len(lines) - width + 1):
        block = tuple(lines[i:i + width])
        if blocks.get(block, 0) >= minimum:
            covered[i:i + width] = b"\x01" * width
    repeated_block_chars = sum(len(line) for line, mark in zip(lines, covered) if mark)
    words = re.findall(r"\w+", normalized_text.casefold())
    length = len(normalized_text) or 1
    invisible = sum(unicodedata.category(char) == "Cf" or
                    (unicodedata.category(char) == "Cc" and char not in "\n\r\t")
                    for char in normalized_text)
    return {
        "nonempty_lines": len(lines),
        "repeated_line_fraction": repeated_line_chars / total_line_chars,
        "repeated_block_fraction": repeated_block_chars / total_line_chars,
        "dominant_line_fraction": dominant_line_chars / total_line_chars,
        "word_count": len(words),
        "unique_word_fraction": len(set(words)) / len(words) if words else 0.0,
        "meaningful_character_fraction": sum(char.isalnum() for char in normalized_text) / length,
        "invisible_character_fraction": invisible / length,
    }


def check_quality(normalized_text: str, config: QualityConfig, *, content_type: str = "prose"):
    """Apply the existing checks, then conservative type-aware Phase 2 checks."""
    reason = baseline_check_quality(normalized_text, config)
    if reason is not None:
        return reason
    metrics = quality_metrics(normalized_text)
    limits = QUALITY_LIMITS
    if (metrics["nonempty_lines"] >= limits["structure_min_lines"]
            and len(normalized_text) >= limits["structure_min_characters"]
            and (metrics["repeated_line_fraction"] >= limits["repeated_line_fraction"]
                 or metrics["repeated_block_fraction"] >= limits["repeated_block_fraction"]
                 or metrics["dominant_line_fraction"] >= limits["dominant_line_fraction"])):
        return QualityReason.EXCESSIVE_REPEATED_STRUCTURE
    if (len(normalized_text) >= limits["invisible_min_characters"]
            and metrics["invisible_character_fraction"] >= limits["invisible_max_fraction"]):
        return QualityReason.LOW_INFORMATION
    meaningful_min = (limits["meaningful_min_code"] if content_type == "code"
                      else limits["meaningful_min_prose_markdown"])
    if (len(normalized_text) >= limits["meaningful_min_characters"]
            and metrics["meaningful_character_fraction"] < meaningful_min):
        return QualityReason.LOW_INFORMATION
    if (content_type in ("prose", "markdown")
            and metrics["word_count"] >= limits["lexical_min_words"]
            and metrics["unique_word_fraction"] <= limits["lexical_max_unique_fraction"]):
        return QualityReason.LOW_INFORMATION
    return None
