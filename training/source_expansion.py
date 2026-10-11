"""Deterministic offline source-catalog screening, never an admission grant.

Consumes the existing sealed registry schema, including future locally supplied
catalogs. Reuses source_prepare's license allowlist and eligibility checks. No
URLs, licenses, approvals, revisions or source caps are invented or promoted.
The emitted registry preserves source records exactly for source_prepare and
clearance_evidence. Candidate means eligible for review-cache preparation only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from training.production_corpus import atomic_json, file_digest
from training.production_spec import need, seal, verify_seal
from training.source_prepare import ROOT, LICENSES, classify
from training.token_shards import _writer_lock

VERSION = 'deterministic-source-expansion-v1'


def screen(source, spec):
    """Conservative discovery screen; source ACCEPT never clears any payload."""
    if source.get('license_id') not in LICENSES:
        return 'REJECTED', 'license absent, ambiguous, or outside preparation allowlist'
    eligibility = classify(source)
    if eligibility != 'CACHE_READY':
        return 'REJECTED', eligibility
    categories = {c['id']: c['subcategories'] for c in spec['categories']}
    if source.get('subcategory') not in categories.get(source.get('category'), []):
        return 'REJECTED', 'unknown category/subcategory'
    if not source.get('deny_paths') or not source.get('split_group'):
        return 'REJECTED', 'missing existing exclusion/split policy'
    return 'CANDIDATE', 'existing preparation checks passed; per-file clearance still required'


def build(registry, spec, output, *, resume=False):
    registry, spec, output = Path(registry).resolve(), Path(spec).resolve(), Path(output).resolve()
    for path in (registry, spec):
        need(not path.is_relative_to(output), 'Output overlaps immutable input')
    catalog = json.loads(registry.read_bytes()); policy = json.loads(spec.read_bytes())
    verify_seal(catalog); verify_seal(policy)
    need(catalog['schema'] == 'yug-production-source-registry-v1', 'Unknown registry schema')
    request = seal(dict(version=VERSION, registry_sha256=file_digest(registry),
                        spec_sha256=file_digest(spec), license_allowlist=sorted(LICENSES)))
    if (output/'request.json').exists():
        need(resume, 'Existing output requires --resume')
        need(json.loads((output/'request.json').read_bytes()) == request, 'Expansion inputs changed')
    else:
        need(not output.exists() or not any(output.iterdir()), 'Unrecognized output directory')
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output/'request.json', request)
    prior = None
    if (output/'complete.json').exists():
        prior = json.loads((output/'complete.json').read_bytes()); verify_seal(prior)
        need(prior['request_identity'] == request['identity'], 'Expansion receipt mismatch')
        for name, digest in prior['artifacts'].items():
            need(name in ('candidates.json', 'decisions.json'), 'Unexpected artifact')
            need(file_digest(output/name) == digest, 'Expansion output corrupt')
    candidates = []; decisions = []; seen = set(); origins = set()
    for source in sorted(catalog['sources'], key=lambda s: s['source_id']):
        sid = source['source_id']
        need(re.fullmatch(r'[A-Za-z0-9_-]+', sid) is not None, 'Unsafe source ID')
        need(sid not in seen, 'Duplicate source ID'); seen.add(sid)
        status, reason = screen(source, policy)
        origin = (str(source.get('locator')).casefold().rstrip('/'), source.get('revision'))
        if status == 'CANDIDATE':
            if origin in origins:
                status, reason = 'REJECTED', 'duplicate pinned repository'
            else:
                origins.add(origin); candidates.append(source)
        decisions.append(dict(source_id=sid, source_url=source.get('locator'),
                              locator=source.get('locator'), revision=source.get('revision'),
                              license_id=source.get('license_id'),
                              license_evidence=source.get('license_evidence'),
                              category=source.get('category'), subcategory=source.get('subcategory'),
                              status=status, reason=reason, training_allowed=False))
    candidate_registry = seal(dict(catalog, sources=candidates))
    decision_manifest = seal(dict(version=VERSION, request_identity=request['identity'],
                                  sources=decisions, training_allowed=False))
    if prior:
        need(json.loads((output/'candidates.json').read_bytes()) == candidate_registry
             and json.loads((output/'decisions.json').read_bytes()) == decision_manifest,
             'Expansion replay differs')
        return prior
    # Interrupted publication is regenerated from sealed inputs. Receipt is last.
    atomic_json(output/'candidates.json', candidate_registry)
    atomic_json(output/'decisions.json', decision_manifest)
    counts = Counter(r['status'] for r in decisions)
    result = seal(dict(version=VERSION, request_identity=request['identity'],
                       candidate_sources=counts['CANDIDATE'], rejected_sources=counts['REJECTED'],
                       training_allowed=False, artifacts={name: file_digest(output/name)
                       for name in ('candidates.json', 'decisions.json')}))
    atomic_json(output/'complete.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=ROOT/'training/specifications/source_registry_v1.json')
    parser.add_argument('--spec', type=Path, default=ROOT/'training/specifications/production_corpus_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        need(args.output.resolve().is_relative_to(ROOT/'artifacts'), 'Output must stay under artifacts')
        for path in (args.registry, args.spec):
            need(not path.resolve().is_relative_to(args.output.resolve()), 'Output overlaps immutable input')
        with _writer_lock(args.output.with_name(args.output.name+'.expansion')):
            result = build(**vars(args))
        print(json.dumps(result, sort_keys=True)); return 0
    except Exception as exc:
        print(f'Source expansion stopped: {type(exc).__name__}: {exc}'); return 2


if __name__ == '__main__':
    raise SystemExit(main())
