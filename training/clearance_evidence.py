"""Offline, content-bound evidence compiler; evidence is not admission authority.

Only existing validated per-file grants establish owned scope. Root licenses and
SPDX labels alone leave a file in REVIEW. No source preparation or network runs.
Resume rechecks immutable inputs and payloads, retaining committed decisions.
--limit caps total decisions (increase it on resume); --dry-run writes nothing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from training.production_acquisition import read_plan, item_key, validate_clearance
from training.production_corpus import atomic_json, atomic_replace, canonical, file_digest
from training.production_spec import (need, seal, verify_seal, validate, local_path,
                                      source_manifest, iter_manifest_files)
from training.source_admission import cached_sources, holdout_groups
from training.source_prepare import ROOT, screening
from training.token_shards import _writer_lock

POLICY = 'immutable-per-file-clearance-evidence-v1'
STATUSES = ('EVIDENCED', 'EXCLUDED', 'REVIEW')


def defaults(root=ROOT):
    root = Path(root)
    approved = root / 'artifacts/production_batch_1_v1/admission'
    return dict(root_registry=root/'training/specifications/source_registry_v1.json',
                root_spec=root/'training/specifications/production_corpus_v1.json',
                approved_registry=approved/'registry.json', approved_spec=approved/'spec.json',
                approved_plan=approved/'plan.jsonl',
                references=root/'artifacts/acquisition_readiness_gate_v1/heldout_references.jsonl')


def _check_output(root, cache, output, inputs):
    need(output.is_relative_to(root) and not output.is_relative_to(cache)
         and not cache.is_relative_to(output), 'Output must be separate from cache under repository')
    for p in inputs.values():
        need(not Path(p).resolve().is_relative_to(output), 'Output overlaps immutable inputs')


def _grants(db, registry, spec, plan, root, groups):
    sources = {s['source_id']: s for s in registry['sources']}
    for sid in sorted({r['source_id'] for r in read_plan(plan)}):
        need(sid in sources, 'Unknown grant source')
        for f in iter_manifest_files(root, source_manifest(root, sources[sid])):
            db.execute('INSERT INTO files VALUES (?,?,?)', (sid, f['path'], canonical(f).decode()))
    for record in read_plan(plan):
        source = sources[record['source_id']]
        row = db.execute('SELECT payload FROM files WHERE source=? AND path=?',
                         (record['source_id'], record['path'])).fetchone()
        need(row is not None, 'Grant missing from owned scope manifest')
        validate_clearance(record, source, spec, root, json.loads(row[0]))
        need(record['split_group'] not in groups, 'Existing grant conflicts with holdout group')
        db.execute('INSERT INTO grants VALUES (?,?)', (item_key(record), canonical(record).decode()))


def _decision(row, source, directory, done, spec, groups, db, request_identity):
    sid = source['source_id']
    need(row['source_id'] == sid and row['revision'] == source['revision']
         and row['locator'] == source['locator'], 'Cache row provenance mismatch')
    need(re.fullmatch('[0-9a-f]{40}', row['revision']) is not None, 'Unpinned revision')
    need(row['screening'] in ('UNRESOLVED', 'EXCLUDED'), 'Unknown cache screening state')
    data = b''
    if row['local_blob']:
        blob = local_path(directory, row['local_blob'])
        need(blob.is_relative_to(directory/'files') and blob.is_file(), 'Invalid cache payload location')
        need(blob.stat().st_size == row['bytes'] and file_digest(blob) == row['sha256'], 'Cache payload hash/size corrupt')
        if row.get('git_object'):
            digest = hashlib.sha1(f"blob {row['bytes']}\0".encode())
            with blob.open('rb') as stream:
                while block := stream.read(65536):
                    digest.update(block)
            need(digest.hexdigest() == row['git_object'], 'Git object/content mismatch')
        if row['bytes'] <= 1_048_576:
            data = blob.read_bytes()
    else:
        need(row['git_mode'] == '160000' and row['bytes'] is None
             and row['sha256'] is None, 'Missing cached file payload')
    status, reason = screening(row['path'], row['git_mode'], row['bytes'] or 0, data, source, spec)
    if row['screening'] == 'EXCLUDED':
        status, reason = 'EXCLUDED', row.get('reason') or 'retained cache exclusion'
    grant = None
    if status != 'EXCLUDED':
        if source['split_group'] in groups:
            status, reason = 'EXCLUDED', 'retained evaluation source group'
        else:
            status, reason = 'REVIEW', 'no exact owned-scope per-file grant; root license alone is insufficient'
            match = db.execute('SELECT payload FROM grants WHERE item=?', (item_key(row),)).fetchone()
            if match:
                grant = json.loads(match[0])
                need(grant['bytes'] == row['bytes'], 'Grant byte size mismatch')
                status, reason = 'EVIDENCED', 'validated immutable owned-scope per-file grant'
    return seal(dict(policy_version=POLICY, request_identity=request_identity,
                     source_id=sid, revision=row['revision'], path=row['path'],
                     sha256=row['sha256'], bytes=row['bytes'], status=status, reason=reason,
                     cache_inventory_sha256=done['inventory_sha256'],
                     license_id=grant['license_id'] if grant else None,
                     source_license_claim=source.get('license_id'),
                     license_evidence=grant['license_evidence'] if grant else [],
                     scope=dict(allow_rule=grant['allow_rule'], deny_rules_checked=grant['deny_rules_checked'],
                                clearance_identity=grant['identity']) if grant else None,
                     category=grant['owner'] if grant else row.get('category'),
                     subcategory=grant['subcategory'] if grant else row.get('subcategory'),
                     split_group=source['split_group'], training_allowed=False))


def build(cache, output, *, root=ROOT, resume=False, dry_run=False, limit=None, **paths):
    root, cache, output = Path(root).resolve(), Path(cache).resolve(), Path(output).resolve()
    need(limit is None or type(limit) is int and limit > 0, 'Limit must be positive')
    inputs = defaults(root); inputs.update(paths)
    _check_output(root, cache, output, inputs)
    base_reg = json.loads(Path(inputs['root_registry']).read_bytes())
    base_spec = json.loads(Path(inputs['root_spec']).read_bytes())
    verify_seal(base_reg); verify_seal(base_spec)
    registry = json.loads(Path(inputs['approved_registry']).read_bytes())
    spec = json.loads(Path(inputs['approved_spec']).read_bytes())
    validate(spec, registry, root)
    need(spec['categories'] == base_spec['categories'] and spec['three_d'] == base_spec['three_d'], 'Allocation differs')
    caches = list(cached_sources(cache, base_reg, base_spec, root))
    need(caches, 'No completed cache receipts')
    hashes = {Path(p).resolve().relative_to(root).as_posix(): file_digest(p) for p in inputs.values()}
    approved_sources = {s['source_id']: s for s in registry['sources']}
    for directory, source, inventory, done in caches:
        approved = approved_sources.get(source['source_id'])
        # Newly discovered sources have no production grants. They still need
        # screening, but cannot become EVIDENCED through _grants/_decision.
        if approved is not None:
            need(source['locator'] == approved['locator'], 'Grant repository origin mismatch')
        for p in (inventory, directory/'request.json', directory/'complete.json'):
            hashes[p.relative_to(root).as_posix()] = file_digest(p)
    request = seal(dict(policy_version=POLICY, inputs=hashes))
    if not dry_run:
        if (output/'request.json').exists():
            need(resume, 'Existing output requires --resume')
            need(json.loads((output/'request.json').read_bytes()) == request, 'Evidence inputs changed')
        else:
            need(not output.exists() or not any(output.iterdir()), 'Unrecognized output directory')
            output.mkdir(parents=True, exist_ok=True)
            atomic_json(output/'request.json', request)
        if (output/'complete.json').exists():
            verify_bundle(output, root)
    db = sqlite3.connect('' if dry_run else str(output/'state.sqlite'))
    db.execute('PRAGMA synchronous=FULL')
    db.executescript('CREATE TABLE IF NOT EXISTS decisions (position INTEGER PRIMARY KEY, payload TEXT NOT NULL);'
                     'CREATE TEMP TABLE files(source TEXT,path TEXT,payload TEXT,PRIMARY KEY(source,path));'
                     'CREATE TEMP TABLE grants(item TEXT PRIMARY KEY,payload TEXT);'
                     'CREATE TEMP TABLE seen(source TEXT,path TEXT,PRIMARY KEY(source,path));')
    try:
        groups = holdout_groups(inputs['references'], base_reg) | holdout_groups(inputs['references'], registry)
        _grants(db, registry, spec, inputs['approved_plan'], root, groups)
        counts = Counter(); per_source = {}; position = 0
        total = sum(sum(done['counts'].values()) for _, _, _, done in caches)
        stored = db.execute('SELECT count(*) FROM decisions').fetchone()[0]
        need(limit is None or limit >= stored, 'Limit cannot shrink resumed evidence')
        for directory, source, inventory, done in caches:
            sc = Counter(); rows = 0
            for row in read_plan(inventory):
                rows += 1
                db.execute('INSERT INTO seen VALUES (?,?)', (source['source_id'], row['path']))
                if limit is not None and position >= limit:
                    continue
                result = _decision(row, source, directory, done, base_spec, groups, db, request['identity'])
                encoded = canonical(result).decode()
                old = db.execute('SELECT payload FROM decisions WHERE position=?', (position,)).fetchone()
                if old:
                    need(old[0] == encoded, 'Committed evidence changed/corrupt')
                else:
                    db.execute('INSERT INTO decisions VALUES (?,?)', (position, encoded)); db.commit()
                position += 1; counts[result['status']] += 1; sc[result['status']] += 1
            need(rows == sum(done['counts'].values()), 'Cache count mismatch')
            per_source[source['source_id']] = {k: sc[k] for k in STATUSES}
        need(stored <= position, 'Unexpected checkpoint tail')
        need(db.execute('SELECT count(*) FROM decisions').fetchone()[0] == position,
             'Unexpected checkpoint positions')
        digest = hashlib.sha256()
        for payload, in db.execute('SELECT payload FROM decisions ORDER BY position'):
            digest.update(payload.encode()+b'\n')
        result = seal(dict(policy_version=POLICY, request_identity=request['identity'],
                           counts={k: counts[k] for k in STATUSES}, per_source=per_source,
                           reviewed_files=position, inventory_files=total, complete=position == total,
                           evidence_sha256='sha256:'+digest.hexdigest(), training_allowed=False))
        if not dry_run:
            temp = output/'evidence.part'
            with temp.open('wb') as stream:
                for payload, in db.execute('SELECT payload FROM decisions ORDER BY position'):
                    stream.write(payload.encode()+b'\n')
            # Invalidate the old publication before replacing its payload. An
            # interruption here resumes from checked transactional decisions.
            (output/'complete.json').unlink(missing_ok=True)
            atomic_replace(temp, output/'evidence.jsonl')
            atomic_json(output/'complete.json', result)
        return result
    finally:
        db.close()


def verify_bundle(directory, root):
    """Validate retained evidence before admission; never replace grant validation."""
    directory = Path(directory)
    request = json.loads((directory/'request.json').read_bytes()); verify_seal(request)
    result = json.loads((directory/'complete.json').read_bytes()); verify_seal(result)
    need(request['policy_version'] == POLICY and result['policy_version'] == POLICY, 'Unknown evidence policy')
    need(result['request_identity'] == request['identity'], 'Evidence request mismatch')
    for path, digest in request['inputs'].items():
        need(file_digest(local_path(root, path)) == digest, 'Evidence input corrupt/changed')
    need(file_digest(directory/'evidence.jsonl') == result['evidence_sha256'], 'Evidence output corrupt')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    try:
        # Check before creating even the advisory lock: cache is strictly read-only.
        _check_output(ROOT.resolve(), args.cache.resolve(), args.output.resolve(), defaults())
        if args.dry_run:
            result = build(**vars(args))
        else:
            # Sibling lock leaves an absent output absent until inputs are verified.
            with _writer_lock(args.output.with_name(args.output.name+'.writer')):
                result = build(**vars(args))
        print(json.dumps(result, sort_keys=True)); return 0
    except Exception as exc:
        print(f'Evidence stopped: {type(exc).__name__}: {exc}'); return 2


if __name__ == '__main__':
    raise SystemExit(main())
