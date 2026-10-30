"""Versioned, text-bearing exclusions for the existing production dedup index.

This registry records what is actually protected; it cannot infer absent answers
or solutions from a question. Hash-only rows are not lexical references.
"""
import hashlib
import json
import re
from pathlib import Path

from training.corpus_prep import normalize_text
from training.experiments.representative_mini_corpus import normalize_structured_text
from training.production_spec import need

SCHEMA = 'yug-protected-reference-v1'
ROLES = frozenset(('question', 'answer', 'solution', 'reference'))
SPLITS = frozenset(('validation', 'test', 'evaluation'))
CONTENT_TYPES = frozenset(('code', 'markdown', 'prose'))
FIELDS = frozenset(('schema', 'work_id', 'work_version', 'item_id',
                    'reference_id', 'role', 'protected_split', 'aliases',
                    'provenance', 'text', 'content_type', 'language',
                    'raw_text_sha256', 'normalized_text_sha256'))
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}\Z')
_HASH = re.compile(r'sha256:[0-9a-f]{64}\Z')
_LANGUAGE = re.compile(r'[A-Za-z][A-Za-z0-9_-]{0,31}\Z')


def _digest(text):
    return 'sha256:' + hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalized_text(text, content_type):
    return normalize_structured_text(text) if content_type in ('code', 'markdown') else normalize_text(text)


def make_reference(*, work_id, work_version, item_id, reference_id, role,
                   protected_split, text, content_type, language, provenance,
                   aliases=()):
    """Construct a row, not an approval; registration still validates the inventory."""
    return dict(schema=SCHEMA, work_id=work_id, work_version=work_version,
                item_id=item_id, reference_id=reference_id, role=role,
                protected_split=protected_split, aliases=list(aliases),
                provenance=provenance, text=text, content_type=content_type,
                language=language, raw_text_sha256=_digest(text),
                normalized_text_sha256=_digest(normalized_text(text, content_type)))


def validate_records(records):
    """Stream validated rows; reject duplicates and inconsistent item relationships."""
    ids = set()
    bindings = set()
    item_splits = {}
    count = 0
    for row in records:
        need(type(row) is dict and row.get('schema') == SCHEMA,
             'Missing/unsupported protected-reference schema')
        need(row.keys() == FIELDS, 'Incomplete/unknown protected-reference fields')
        for field in ('work_id', 'work_version', 'item_id', 'reference_id'):
            need(type(row.get(field)) is str and _ID.fullmatch(row[field]),
                 'Invalid protected-reference ' + field)
        need(type(row.get('role')) is str and row['role'] in ROLES,
             'Missing/unsupported protected-reference role')
        need(type(row.get('protected_split')) is str and row['protected_split'] in SPLITS,
             'Invalid protected split')
        aliases = row.get('aliases')
        need(type(aliases) is list and all(type(alias) is str and _ID.fullmatch(alias) for alias in aliases)
             and len(aliases) == len(set(aliases)),
             'Malformed protected-reference aliases')
        provenance = row.get('provenance')
        need(type(provenance) is dict and provenance
             and all(type(key) is str and key.strip() for key in provenance)
             and any(isinstance(value, str) and value.strip() or isinstance(value, dict) and value
                     for value in provenance.values()),
             'Malformed protected-reference provenance')
        need(type(row.get('content_type')) is str and row['content_type'] in CONTENT_TYPES,
             'Unsupported reference content type')
        need(type(row.get('language')) is str and _LANGUAGE.fullmatch(row['language']),
             'Malformed reference language')
        text = row.get('text')
        need(type(text) is str and text.strip(), 'Protected reference requires text for lexical matching')
        for field in ('raw_text_sha256', 'normalized_text_sha256'):
            need(type(row.get(field)) is str and _HASH.fullmatch(row[field]),
                 'Missing/malformed protected-reference ' + field)
        need(row['raw_text_sha256'] == _digest(text), 'Protected reference raw hash mismatch')
        need(row['normalized_text_sha256'] == _digest(normalized_text(text, row['content_type'])),
             'Protected reference normalized hash mismatch')
        ref_id = row['reference_id']
        need(ref_id not in ids, 'Duplicate protected reference identity')
        ids.add(ref_id)
        item = (row['work_id'], row['work_version'], row['item_id'])
        binding = (*item, row['role'])
        need(binding not in bindings, 'Conflicting protected work/item/role binding')
        bindings.add(binding)
        previous = item_splits.setdefault(item, row['protected_split'])
        need(previous == row['protected_split'], 'Protected item split conflict')
        count += 1
        yield row
    need(count > 0, 'Production held-out inventory is empty')


def read_inventory(path):
    """Reject ambiguous JSON keys before validating the pinned text inventory."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, 'Duplicate protected-reference JSON field')
            result[key] = value
        return result

    def rows():
        with Path(path).open('rb') as stream:
            while line := stream.readline(1_048_577):
                need(len(line) <= 1_048_576, 'Oversized protected-reference row')
                if line.strip():
                    yield json.loads(line, object_pairs_hook=unique)
    return validate_records(rows())
