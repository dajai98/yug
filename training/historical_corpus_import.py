"""Read-only reconstruction of the explicitly retained pre-ledger corpus.

This module does not approve an import. A separate trusted identity is required
before the reconstructed evidence can be committed to the corpus ledger.
"""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import sqlite3

from training.production_corpus import canonical, file_digest, identity
from training.production_spec import QUOTAS, TOKENIZER, need, verify_seal


SCHEMA = 'yug-historical-corpus-import-v1'
EXPECTED_DOCUMENTS = 174
EXPECTED_TRAIN_TOKENS = 417_405
EXPECTED_VALIDATION_TOKENS = 20_616


def _file(root, entry):
    need(isinstance(entry, dict) and set(entry) == {'path', 'sha256'},
         'Historical evidence locator is malformed')
    path = (root / entry['path']).resolve()
    need(path.is_relative_to(root) and path.is_file() and
         file_digest(path) == entry['sha256'], 'Historical evidence file missing or changed')
    return path


def verify_approved_manifest(root, manifest_path, approved_identity, approved_sha256):
    """Check a manifest against pins supplied outside mutable corpus storage."""
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    need(manifest_path.is_relative_to(root) and manifest_path.is_file(),
         'Approved historical manifest missing or outside root')
    need(isinstance(approved_identity, str) and isinstance(approved_sha256, str)
         and file_digest(manifest_path) == approved_sha256,
         'Historical manifest differs from external approved file SHA256')
    manifest = json.loads(manifest_path.read_bytes())
    verify_seal(manifest)
    need(manifest['identity'] == approved_identity,
         'Historical manifest differs from external approved identity')
    need(file_digest(manifest_path) == approved_sha256,
         'Historical manifest changed during verification')
    return manifest


def reconstruct(root, manifest_path, *, approved_manifest_identity=None,
                approved_manifest_sha256=None, preview=False):
    """Verify retained items; unapproved candidate inspection requires preview=True."""
    from training.production_acquisition import item_key
    from training.corpus_prep import normalize_text
    from training.experiments.representative_mini_corpus import normalize_structured_text

    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    need(manifest_path.is_relative_to(root), 'Historical import manifest escapes root')
    approved = approved_manifest_identity is not None or approved_manifest_sha256 is not None
    need(preview or approved, 'External approved historical manifest pins required')
    if approved:
        manifest = verify_approved_manifest(root, manifest_path,
                                            approved_manifest_identity,
                                            approved_manifest_sha256)
    else:
        manifest = json.loads(manifest_path.read_bytes())
        verify_seal(manifest)
    need(manifest.get('schema') == SCHEMA and isinstance(manifest.get('builds'), list)
         and manifest['builds'], 'Invalid historical import manifest')
    all_items = []
    all_usage = []
    source_scopes = {}
    locator_scopes = {}
    group_splits = {}
    build_evidence = []
    seen_states = set()
    seen_items = set()
    seen_normalized = set()
    for build in manifest['builds']:
        need(isinstance(build, dict) and set(build) == {'state', 'plan', 'spec', 'registry'},
             'Historical build evidence incomplete')
        state = _file(root, build['state'])
        need(build['state']['sha256'] not in seen_states, 'Duplicate historical build')
        seen_states.add(build['state']['sha256'])
        plan = _file(root, build['plan'])
        spec = json.loads(_file(root, build['spec']).read_bytes())
        registry = json.loads(_file(root, build['registry']).read_bytes())
        verify_seal(spec)
        verify_seal(registry)
        sources = {s['source_id']: s for s in registry['sources']}
        plan_rows = {}
        for line in plan.read_bytes().splitlines():
            record = json.loads(line)
            verify_seal(record)
            key = item_key(record)
            need(key not in plan_rows, 'Duplicate historical planned item')
            plan_rows[key] = record
        with sqlite3.connect(state.as_uri() + '?mode=ro', uri=True) as db:
            need(db.execute('PRAGMA quick_check').fetchone()[0] == 'ok',
                 'Corrupt historical production database')
            meta = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM meta')}
            request = meta['request']
            need(request['diagnostic'] is False and request['tokenizer'] == TOKENIZER
                 and request['spec'] == spec['identity']
                 and request['registry'] == registry['identity']
                 and meta.get('plan_identity') == build['plan']['sha256']
                 and meta.get('state') in ('COMPLETE', 'TARGET_REACHED', 'TARGET_BOUNDARY')
                 and meta.get('dedup_settled') is True,
                 'Historical build identity or completion evidence changed')
            accepted = list(db.execute("SELECT item,record,normalized_sha,token_sha,token_count,split,owner,source "
                "FROM items WHERE stage='ACCEPTED' ORDER BY item"))
            for item, encoded, nsha, tsha, count, split, owner, source_id in accepted:
                record = json.loads(encoded)
                verify_seal(record)
                need(record == plan_rows.get(item) and item_key(record) == item,
                     'Historical accepted item differs from pinned plan')
                need(item not in seen_items and nsha not in seen_normalized,
                     'Duplicate historical accepted item/content')
                seen_items.add(item)
                seen_normalized.add(nsha)
                need(source_id == record['source_id'] and split == record['split']
                     and split in ('train', 'validation') and owner == record['owner']
                     and owner in QUOTAS and type(count) is int and count > 0,
                     'Historical item source, owner, split or count changed')
                source = sources.get(source_id)
                need(source is not None and source['revision'] == record['revision']
                     and owner in source['allowed_categories']
                     and record['split_group'] == source['split_group']
                     and split == source['split']
                     and group_splits.setdefault(source['split_group'], split) == split,
                     'Historical source/revision/split evidence changed')
                license_evidence = record.get('license_evidence')
                need(isinstance(license_evidence, list) and license_evidence,
                     'Historical item license evidence missing')
                for reference in license_evidence:
                    _file(root, reference)
                base = state.parent
                suffix = item[7:]
                clearance = base / 'clearances' / (suffix + '.json')
                raw = base / 'raw' / suffix
                normalized = base / 'normalized' / suffix
                tokens = base / 'tokens' / (suffix + '.bin')
                need(json.loads(clearance.read_bytes()) == record
                     and file_digest(raw) == record['sha256']
                     and file_digest(normalized) == nsha
                     and file_digest(tokens) == tsha
                     and tokens.stat().st_size == count * 4,
                     'Historical clearance or payload evidence changed')
                text = raw.read_bytes().decode('utf-8-sig')
                text = normalize_structured_text(text) if record['content_type'] in ('code', 'markdown') else normalize_text(text)
                need(normalized.read_bytes() == text.encode('utf-8'),
                     'Historical normalization changed')
                scope = (source['repository_group'], source['domain_group'], source['split_group'])
                locator = (source.get('locator') or source_id).casefold().rstrip('/')
                need(source_scopes.setdefault(source_id, scope) == scope
                     and locator_scopes.setdefault(locator, scope) == scope,
                     'Historical source scope changed across builds')
                all_items.append(dict(item=item, record=record, normalized_sha=nsha,
                    token_sha=tsha, tokens=count, source_id=source_id, owner=owner,
                    split=split, repository_group=scope[0], domain_group=scope[1],
                    split_group=scope[2], source_locator=source.get('locator') or source_id,
                    three_d_topic=record.get('three_d_topic'),
                    three_d_form=record.get('three_d_form'),
                    historical_state_sha256=build['state']['sha256']))
                all_usage.extend((item, kind, key, tokens, cap) for kind, key, tokens, cap in
                                 usage_for(all_items[-1], spec, source))
            local_totals = {(kind, key): tokens for kind, key, tokens in
                            db.execute('SELECT kind,key,tokens FROM totals')}
            computed = Counter()
            for evidence in all_items:
                if evidence['historical_state_sha256'] != build['state']['sha256']:
                    continue
                for kind, key, tokens, _ in usage_for(evidence, spec, sources[evidence['source_id']]):
                    computed[(kind, key)] += tokens
            need(dict(computed) == local_totals,
                 'Historical aggregate totals differ from accepted item evidence')
        need(file_digest(state) == build['state']['sha256'],
             'Historical production database changed during verification')
        build_evidence.append({kind: build[kind]['sha256'] for kind in
                               ('state', 'plan', 'spec', 'registry')})
    all_items.sort(key=lambda row: row['item'])
    all_usage.sort()
    need(len(all_items) == EXPECTED_DOCUMENTS, 'Retained historical item count incomplete')
    train = sum(x['tokens'] for x in all_items if x['split'] == 'train')
    validation = sum(x['tokens'] for x in all_items if x['split'] == 'validation')
    need((train, validation) == (EXPECTED_TRAIN_TOKENS, EXPECTED_VALIDATION_TOKENS),
         'Retained historical token totals differ')
    cumulative = Counter()
    caps = {}
    for _, kind, key, tokens, cap in all_usage:
        previous = caps.setdefault((kind, key), cap)
        need(previous == cap, 'Historical cap changed across retained builds')
        cumulative[(kind, key)] += tokens
        need(cumulative[(kind, key)] <= cap, 'Historical cumulative usage exceeds frozen cap')
    # The ordered item evidence and each source's exact retained file identities
    # are part of the approval candidate; filesystem paths are only locators.
    proposal = dict(schema=SCHEMA, policy='yug-cumulative-accounting-v1',
        builds=sorted(build_evidence, key=lambda x: x['state']),
        accepted_documents=len(all_items), training_tokens=train,
        validation_tokens=validation, item_identity=identity(all_items),
        usage_identity=identity(all_usage))
    if approved:
        need(file_digest(manifest_path) == approved_manifest_sha256,
             'Historical manifest changed during reconstruction')
    return dict(identity=identity(proposal), proposal=proposal, items=all_items,
                usage=all_usage, manifest_identity=manifest['identity'])


def usage_for(evidence, spec, source):
    count = evidence['tokens']
    owner = evidence['owner']
    source_id = evidence['source_id']
    rows = [('source', source_id, source['target_cap_tokens']),
            ('source_owner', source_id + ':' + owner, source['category_caps_tokens'][owner]),
            ('repository_group', source['repository_group'],
             spec['group_caps']['repository_group'][source['repository_group']]),
            ('domain_group', source['domain_group'],
             spec['group_caps']['domain_group'][source['domain_group']])]
    if evidence['split'] == 'train':
        rows.append(('category', owner, QUOTAS[owner]))
        if evidence['three_d_topic']:
            topics = {t['id']: t['target_tokens'] for t in spec['three_d']['topics']}
            rows.extend((('three_d_topic', evidence['three_d_topic'],
                         topics[evidence['three_d_topic']]),
                         ('three_d_form', evidence['three_d_form'],
                          spec['three_d']['form_targets'][evidence['three_d_form']])))
    else:
        rows.append(('validation_category', owner, 2_000_000_000))
    return [(kind, key, count, cap) for kind, key, cap in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preview', action='store_true')
    action.add_argument('--commit', action='store_true')
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--approved-head')
    parser.add_argument('--approved-import-identity')
    parser.add_argument('--approved-manifest-identity')
    parser.add_argument('--approved-manifest-sha256')
    parser.add_argument('--governance-log', type=Path)
    parser.add_argument('--approved-governance-sha256')
    args = parser.parse_args()
    if args.preview:
        candidate = reconstruct(args.root, args.manifest, preview=True)
        print(canonical(dict(identity=candidate['identity'],
                             proposal=candidate['proposal'])).decode())
        return
    from training.cumulative_accounting import CorpusLedger
    need(args.ledger is not None and args.approved_head is not None
         and args.approved_import_identity is not None
         and args.approved_manifest_identity is not None
         and args.approved_manifest_sha256 is not None,
         'Commit requires ledger and independent approved head/import/manifest pins')
    with CorpusLedger(args.ledger, args.root, args.approved_head,
                      approved_historical_import_identity=args.approved_import_identity,
                      approved_historical_manifest_identity=args.approved_manifest_identity,
                      approved_historical_manifest_sha256=args.approved_manifest_sha256,
                      governance_log=args.governance_log,
                      expected_governance_sha256=args.approved_governance_sha256) as ledger:
        print(canonical(ledger.import_historical(
            args.manifest, args.approved_import_identity)).decode())


if __name__ == '__main__':
    main()
