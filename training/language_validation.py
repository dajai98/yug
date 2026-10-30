"""Bounded script consistency for production language metadata.

This is not natural-language identification. In particular Devanagari does not
prove Nepali, and Latin text may be transliterated Nepali or another language.
"""
from __future__ import annotations

import unicodedata

from training.production_spec import need


NATURAL = ['en', 'ne', 'mixed', 'unknown']
# Only languages represented by retained production, the approved code scope,
# or the current frozen code subcategories. REVIEW-only Dart/Java are omitted.
PROGRAMMING = ['python', 'rust', 'go', 'csharp', 'javascript', 'typescript',
               'javascript_typescript', 'c', 'cpp', 'c_cpp']
POLICY = {
    'version': 'script-consistency-v1',
    'max_document_bytes': 1_048_576,
    'natural_labels': NATURAL,
    'programming_labels': PROGRAMMING,
    'code_labels': ['mixed', 'unknown'] + PROGRAMMING,
    'prose_markdown_labels': NATURAL,
    'latin_only_min': 80,
    'devanagari_only_min': 80,
    'opposite_max': 2,
    'dominant_share_min': 0.85,
    'mixed_min_each': 20,
    'mixed_minority_share': 0.20,
    'mixed_conflict_min_each': 30,
    'mixed_conflict_minority_share': 0.30,
    'relevant_share_min': 0.75,
    'normalization_input': 'production-normalized-document',
    'script_counts_are_language_identification': False,
}


def script_evidence(text: str) -> dict[str, int]:
    """Count bounded Unicode letters in one pass; never infer exact language."""
    need(isinstance(text,str) and len(text.encode('utf-8')) <= POLICY['max_document_bytes'],
         'Language document byte limit exceeded')
    latin = devanagari = other = 0
    for char in text:
        if not unicodedata.category(char).startswith('L'):
            continue
        if '\u0900' <= char <= '\u097f':
            devanagari += 1
        elif ('A' <= char <= 'Z' or 'a' <= char <= 'z'
              or unicodedata.name(char, '').startswith('LATIN ')):
            latin += 1
        else:
            other += 1
    return dict(latin_letters=latin, devanagari_letters=devanagari,
                other_letters=other, total_letters=latin+devanagari+other)


def validate_label(label, content_type: str) -> None:
    need(content_type in ('code', 'markdown', 'prose'), 'Unknown content type')
    need(type(label) is str and bool(label), 'Missing or unsupported language label')
    allowed = POLICY['code_labels'] if content_type == 'code' else NATURAL
    need(label in allowed, 'Unsupported language label for content type')


def validate_language(label, content_type: str, normalized_text: str) -> dict[str, int]:
    """Reject only clear script/label contradictions for natural-language text.

    Code is labeled by programming language (or mixed/unknown), not by its
    identifiers or comments. Unknown retains uncertainty, including other
    Latin-script languages. Thresholds are versioned in POLICY.
    """
    validate_label(label, content_type)
    metrics = script_evidence(normalized_text)
    if content_type == 'code' or label == 'unknown':
        return metrics
    latin, dev = metrics['latin_letters'], metrics['devanagari_letters']
    total = metrics['total_letters']
    relevant = latin + dev
    if not total or relevant / total < POLICY['relevant_share_min']:
        return metrics
    minority_share = min(latin, dev) / relevant if relevant else 0.0
    mixed = (latin >= POLICY['mixed_min_each'] and dev >= POLICY['mixed_min_each']
             and minority_share >= POLICY['mixed_minority_share'])
    conflict = (latin >= POLICY['mixed_conflict_min_each']
                and dev >= POLICY['mixed_conflict_min_each']
                and minority_share >= POLICY['mixed_conflict_minority_share'])
    if label == 'mixed':
        need(mixed, 'Language mixed label lacks sufficient script evidence')
    elif label == 'en':
        need(not (dev >= POLICY['devanagari_only_min']
                  and latin <= POLICY['opposite_max']
                  and dev / total >= POLICY['dominant_share_min']) and not conflict,
             'Language en contradicts clear script evidence')
    elif label == 'ne':
        need(not (latin >= POLICY['latin_only_min']
                  and dev <= POLICY['opposite_max']
                  and latin / total >= POLICY['dominant_share_min']) and not conflict,
             'Language ne contradicts clear script evidence')
    return metrics
