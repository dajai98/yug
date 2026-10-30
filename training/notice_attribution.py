"""Deterministic inventory of recorded corpus obligations, not a legal decision.

Source approval remains an external authority. This module checks that the
obligations carried into an export are exactly those in the retained accepted
items and their verified source clearances.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from training.conditional_license import validate_approval
from training.production_corpus import canonical, file_digest, identity
from training.production_spec import (local_path, need, seal, source_manifest,
                                      verify_seal)

SCHEMA = 'yug-notice-attribution-v1'
FILENAME = 'notice_attribution.json'
NOTICE_NAMES = {'notice', 'notice.txt', 'notice.md'}


def source_clearance(source, root):
    """Reopen the exact source grant; its self-seal alone is not an approval pin."""
    grant = json.loads(local_path(root, source['clearance_record']).read_bytes())
    verify_seal(grant)
    manifest = source_manifest(root, source)
    need(grant.get('source_id') == source['source_id'] and
         grant.get('revision') == source['revision'] and
         grant.get('file_manifest_identity') == manifest['identity'] and
         grant.get('decision') == 'ACCEPT', 'Source clearance scope changed')
    need(type(grant.get('attribution')) is str and bool(grant['attribution'].strip()),
         'Missing source attribution')
    references = grant.get('evidence')
    need(type(references) is list and bool(references), 'Missing source obligation evidence')
    for reference in references:
        need(type(reference) is dict and type(reference.get('path')) is str and
             type(reference.get('sha256')) is str and
             file_digest(local_path(root, reference['path'])) == reference['sha256'],
             'Source obligation evidence changed')
    return grant


def obligation(record, source, grant, root, *, allow_legacy=False):
    """The recorded obligations for one accepted item, including conditional use."""
    need(record['source_id'] == grant['source_id'] and
         record['revision'] == grant['revision'], 'Item/source obligation scope mismatch')
    need(record.get('source_clearance_identity') == grant['identity'] or
         (allow_legacy and record.get('source_clearance_identity') is None),
         'Item source-clearance identity missing or changed')
    need((allow_legacy or record.get('attribution') == grant['attribution']) and
         canonical(record.get('license_evidence')) == canonical(grant['evidence']),
         'Item attribution/evidence differs from source clearance')
    approval = validate_approval(record, source, root)
    refs = grant['evidence']
    notice_refs = [ref for ref in refs if Path(ref['path']).name.casefold() in NOTICE_NAMES]
    return dict(source_id=record['source_id'], revision=record['revision'],
        source_locator=source['locator'], license_id=record['license_id'],
        attribution=record['attribution'], license_evidence=refs,
        source_clearance_identity=grant['identity'],
        legacy_plan_bound_attribution=allow_legacy,
        notice_references=notice_refs,
        separate_notice_recorded=bool(notice_refs),
        separate_notice_applicability='RECORDED' if notice_refs else 'UNDETERMINED',
        conditional_approval_identity=approval['identity'] if approval else None,
        conditional_obligations=approval['obligations'] if approval else None)


def build_notice(rows, sources, root, *, plan_identity, admission_sha256,
                 legacy_record=None):
    """Aggregate accepted items only; callers provide a stable item order."""
    grouped = {}
    total = 0
    grants = {}
    for item, record in rows:
        source = sources[record['source_id']]
        if record['source_id'] not in grants:
            grants[record['source_id']] = source_clearance(source, root)
        grant = grants[record['source_id']]
        details = obligation(record, source, grant, root,
                             allow_legacy=legacy_record(record) if legacy_record else False)
        key = identity(details)
        if key not in grouped:
            grouped[key] = dict(obligation_identity=key, **details, accepted_items=0)
        grouped[key]['accepted_items'] += 1
        total += 1
    entries = sorted(grouped.values(), key=lambda row:(row['source_id'], row['revision'],
                     row['license_id'], row['obligation_identity']))
    return seal(dict(schema=SCHEMA, plan_identity=plan_identity,
                     admission_sha256=admission_sha256,
                     accepted_items=total, entries=entries))


def reference(notice):
    data = canonical(notice)
    return dict(path=FILENAME, schema=SCHEMA, identity=notice['identity'],
                sha256='sha256:'+hashlib.sha256(data).hexdigest())


def verify_file(directory, ref, *, expected=None):
    need(type(ref) is dict and set(ref)=={'path','schema','identity','sha256'} and
         ref['path']==FILENAME and ref['schema']==SCHEMA,
         'Missing or unsupported NOTICE binding')
    path = Path(directory)/FILENAME
    need(file_digest(path)==ref['sha256'], 'NOTICE artifact missing or changed')
    data = path.read_bytes()
    try:
        notice = json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise ValueError('Malformed NOTICE artifact') from exc
    need(canonical(notice)==data, 'Noncanonical NOTICE artifact')
    verify_seal(notice)
    need(notice.get('schema')==SCHEMA and notice['identity']==ref['identity'],
         'NOTICE identity/schema mismatch')
    if expected is not None:
        need(canonical(notice)==canonical(expected), 'NOTICE obligations differ from accepted items')
    return notice
