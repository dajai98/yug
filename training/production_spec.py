"""Offline production corpus specification validation. Never acquires or trains.

This is an admission/planning boundary, not a replacement for production_corpus
or token_shards. Missing acquisition evidence fails closed. Runtime obligations
are explicitly reported, not represented as completed by a valid JSON document.
"""
from __future__ import annotations
import argparse
from collections import Counter
import fnmatch
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
from contextlib import closing
from training.production_corpus import identity, file_digest, atomic_json

TOKENIZER = 'sha256:0b9b2bb9efe58902e3cf3dcb359cf174a25f3d7bf603b14bd3fe8e2d7fec0507'
QUOTAS = dict(code=440_000_000, csharp_gamedev=260_000_000,
    software_engineering=240_000_000, general_english=240_000_000,
    educational_reference=160_000_000, nepali=160_000_000,
    software_docs=120_000_000, graphics=100_000_000, math_science=80_000_000,
    algorithms=60_000_000, linux_systems=40_000_000, security=20_000_000,
    instruction=30_000_000, reasoning=20_000_000, tool_use=10_000_000,
    conversation=20_000_000)
STATUSES = {'ACCEPT', 'REVIEW', 'REJECT', 'CONDITIONAL'}
CHECKPOINTS = [100_000_000, 250_000_000, 500_000_000, 1_000_000_000, 1_500_000_000, 2_000_000_000]

class SpecificationError(ValueError):
    pass

def need(condition, message):
    if not condition:
        raise SpecificationError(message)

def positive(value):
    return type(value) is int and value > 0

def seal(data):
    data = dict(data)
    data.pop('identity', None)
    return dict(data, identity=identity(data))

def verify_seal(data):
    need(data.get('identity') == seal(data)['identity'], 'Specification identity mismatch')

def local_path(root, name):
    need(isinstance(name, str) and name, 'Missing evidence path')
    path = (Path(root) / name).resolve()
    need(path.is_relative_to(Path(root).resolve()), 'Evidence path escapes repository')
    return path

def path_allowed(path, allow, deny):
    """Case-insensitive conservative deny matching; POSIX repository paths only."""
    if not isinstance(path, str) or not path or '\\' in path or ':' in path:
        return False
    if path.startswith('/') or any(p in ('', '.', '..') for p in path.split('/')):
        return False
    def matches(pattern):
        pattern = pattern.casefold()
        value = path.casefold()
        return fnmatch.fnmatchcase(value, pattern) or (pattern.startswith('**/') and fnmatch.fnmatchcase(value, pattern[3:]))
    return any(matches(p) for p in allow) and not any(matches(p) for p in deny)

def source_manifest(root, source):
    path=local_path(root,source['file_manifest'])
    need(path.stat().st_size <= 16*1024*1024, 'Manifest metadata exceeds 16 MiB; use files_jsonl')
    manifest=json.loads(path.read_bytes());verify_seal(manifest)
    return manifest

def iter_manifest_files(root, manifest):
    """Backward-compatible small JSON lists or hash-pinned streamed JSONL."""
    if 'files_jsonl' in manifest:
        need('files' not in manifest, 'Ambiguous manifest file representation')
        path=local_path(root,manifest['files_jsonl'])
        need(file_digest(path)==manifest['files_sha256'], 'File inventory hash mismatch')
        count=0
        with path.open('rb') as handle:
            while line:=handle.readline(1_048_577):
                need(len(line)<=1_048_576, 'Oversized inventory record')
                need(bool(line.strip()), 'Blank inventory record')
                count+=1;yield json.loads(line)
        need(type(manifest['file_count']) is int and count==manifest['file_count'] and count>0,'Inventory count mismatch')
    else:
        need(isinstance(manifest.get('files'),list) and bool(manifest['files']),'Empty acquisition manifest')
        yield from manifest['files']


def acquisition_blockers(source):
    reasons = []
    if source['status'] != 'ACCEPT':
        reasons.append('status_' + source['status'].lower())
    for key in ('locator', 'revision', 'license_evidence', 'clearance_record', 'file_manifest'):
        if not source.get(key):
            reasons.append('missing_' + key)
    if not source['allow_paths']:
        reasons.append('missing_path_allowlist')
    if source.get('origin_verified') is not True:
        reasons.append('unverified_canonical_origin')
    if source['split'] == 'train' and not source.get('training_allowed_after_clearance'):
        reasons.append('validation_only_source')
    if source['split'] not in ('train', 'validation'):
        reasons.append('unassigned_source_split')
    return reasons


TRANSITION_CLEARANCE_SCHEMAS = frozenset({
    'reviewed-revision-file-clearance-v1',
    'verified-revision-final-file-clearance-v1',
})


def require_activated_transition_clearance(clearance):
    """F1 evidence cannot be used as an active production source grant."""
    need(clearance.get('schema') not in TRANSITION_CLEARANCE_SCHEMAS and
         'old_revision' not in clearance and
         'comparison_identity' not in clearance and
         'provisional_clearance_identity' not in clearance,
         'R2 transition clearance requires later verified activation')

def validate(spec, registry, root):
    with closing(sqlite3.connect('')) as owners:
        owners.execute('CREATE TABLE files(repository TEXT,path TEXT,PRIMARY KEY(repository,path))')
        return _validate(spec,registry,root,owners)

def _validate(spec,registry,root,owners):
    verify_seal(spec); verify_seal(registry)
    need(spec['schema'] == 'yug-production-corpus-spec-v1', 'Wrong spec schema')
    need(registry['schema'] == 'yug-production-source-registry-v1', 'Wrong registry schema')
    need(spec['registry_identity'] == registry['identity'], 'Registry identity mismatch')
    need(spec['tokenizer_identity'] == TOKENIZER and spec['vocab_size'] == 4096, 'Frozen tokenizer changed')
    cats = spec['categories']
    need(len(cats) == 16 and len({c['id'] for c in cats}) == 16, 'Exactly 16 unique categories required')
    need({c['id']:c['target_tokens'] for c in cats} == QUOTAS, 'Frozen category quotas changed')
    need(all(positive(c['target_tokens']) for c in cats), 'Quotas must be integers')
    need(spec['total_clean_train_tokens'] == sum(QUOTAS.values()) == 2_000_000_000, 'Total must be 2B')
    need(spec['checkpoints'] == CHECKPOINTS, 'Build checkpoints changed')
    need(spec['accounting']['one_budget_owner'] is True, 'Multiple budget ownership forbidden')
    need(spec['accounting']['validation_counts_toward_2b'] is False, 'Validation must not fill training quota')
    need(spec['fallback']['cross_category_transfer'] == 'forbidden', 'Cross-category rebalance forbidden')
    topics = spec['three_d']['topics']
    need(len(topics) == 8 and len({t['id'] for t in topics}) == 8, 'Eight unique 3D topics required')
    need(sum(t['target_tokens'] for t in topics) == 100_000_000, '3D total must be internal 100M')
    used = Counter()
    for t in topics:
        need(t['owner'] in QUOTAS and positive(t['target_tokens']), 'Invalid 3D owner/quota')
        used[t['owner']] += t['target_tokens']
    need(all(used[c] <= QUOTAS[c] for c in used), '3D exceeds owner category')
    for c in cats:
        need(c['non_3d_target_tokens'] == QUOTAS[c['id']]-used[c['id']], '3D/non-3D residual mismatch')
        need(bool(c['subcategories']) and len(set(c['subcategories']))==len(c['subcategories']), 'Invalid subcategory labels')
    forms = spec['three_d']['form_targets']
    need(forms == dict(authoritative=30_000_000, implementation=25_000_000,
        workflow=20_000_000, troubleshooting=10_000_000, procedural_tasks=10_000_000,
        instruction_qa=5_000_000), '3D quality mix changed')
    ids = set(); split_groups = {}; repositories = {}; locators = {}
    for s in registry['sources']:
        sid = s['source_id']
        need(re.fullmatch(r'[a-z0-9][a-z0-9-]*', sid) is not None and sid not in ids, 'Invalid/duplicate source ID')
        ids.add(sid)
        need(s['status'] in STATUSES, 'Unknown status')
        need(s['category'] in QUOTAS and s['subcategory'], 'Unknown category/subcategory')
        need(s['category'] in s['allowed_categories'] and set(s['allowed_categories']) <= set(QUOTAS), 'Invalid allowed owners')
        need(set(s['category_caps_tokens']) == set(s['allowed_categories']), 'Missing per-owner caps')
        need(all(type(n) is int and 0 <= n <= QUOTAS[c] for c,n in s['category_caps_tokens'].items()), 'Invalid per-owner cap')
        cap = s['target_cap_tokens']
        need(type(cap) is int and 0 <= cap <= QUOTAS[s['category']] and (cap > 0 or s['status'] == 'REJECT'), 'Invalid source cap')
        need(s['repository_group'] and s['domain_group'] and s['split_group'], 'Missing concentration groups')
        if s.get('locator'):
            key = s['locator'].casefold().rstrip('/')
            groups = (s['repository_group'], s['domain_group'], s['split_group'])
            need(locators.setdefault(key, groups) == groups, 'Locator aliases must share concentration/split groups')
        need(s['split'] in ('unassigned', 'train', 'validation'), 'Invalid split')
        for field in ('allow_paths', 'deny_paths'):
            need(isinstance(s[field], list) and all(isinstance(x,str) and x and not x.startswith('/') and '..' not in PurePosixPath(x).parts for x in s[field]), 'Invalid path patterns')
        for field in ('repository_group', 'domain_group'):
            need(s[field] in spec['group_caps'][field], 'Missing group cap')
        if s['split'] != 'unassigned':
            old = split_groups.setdefault(s['split_group'], s['split'])
            need(old == s['split'], 'Source group crosses train/validation')
        old = repositories.setdefault(s['repository_group'], s['split_group'])
        need(old == s['split_group'], 'Repository aliases must share split group')
        if s['status'] == 'ACCEPT':
            need(s['license_id'] and s['license_evidence'], 'ACCEPT without historical license evidence')
        if not acquisition_blockers(s):
            need(re.fullmatch(r'(?:[0-9a-f]{40}|(?:sha256:)?[0-9a-f]{64})', s['revision']) is not None, 'Revision must be immutable content hash')
            manifest = source_manifest(root,s)
            verify_seal(manifest)
            need(manifest['source_id'] == sid and manifest['revision'] == s['revision'], 'File manifest source/revision mismatch')
            for f in iter_manifest_files(root,manifest):
                need(f['owner'] in s['allowed_categories'], 'Manifest owner is not allowed')
                need(f['content_type'] in ('code','markdown','prose'), 'Manifest normalization route missing')
                try:
                    owners.execute('INSERT INTO files VALUES (?,?)',(s['repository_group'],f['path'].casefold()))
                except sqlite3.IntegrityError as exc:
                    raise SpecificationError('Duplicate manifest file / multiple budget owners') from exc
                need(path_allowed(f['path'], s['allow_paths'], s['deny_paths']), 'Manifest path excluded')
                need(re.fullmatch(r'sha256:[0-9a-f]{64}', f['sha256']) is not None and positive(f['bytes']), 'Unbounded or unhashed acquisition')
            clearance = json.loads(local_path(root, s['clearance_record']).read_text(encoding='utf-8'))
            verify_seal(clearance)
            require_activated_transition_clearance(clearance)
            need(clearance['source_id'] == sid and clearance['revision'] == s['revision'] and clearance['file_manifest_identity'] == manifest['identity'], 'Clearance does not cover manifest')
            need(clearance['decision'] == 'ACCEPT' and clearance['evidence'] and clearance['attribution'], 'Incomplete clearance')
            if (s.get('approval_composition') or
                    clearance.get('schema') == 'yug-composite-source-clearance-v1' or
                    'components' in clearance):
                from training.composite_lineage import verify
                verify(root, s, manifest, clearance)
            for evidence in clearance['evidence']:
                need(file_digest(local_path(root,evidence['path'])) == evidence['sha256'], 'Clearance evidence hash mismatch')
    for values in spec['group_caps'].values():
        need(all(type(v) is int and 0 <= v <= 2_000_000_000 for v in values.values()), 'Invalid group cap')
    need(bool(spec['evidence']), 'Missing evidence manifest')
    for entry in spec['evidence']:
        need(file_digest(local_path(root, entry['path'])) == entry['sha256'], 'Changed evidence: ' + entry['path'])
    return True

def audit_checkpoint(spec, registry, receipts, *, tokenizer_identity):
    """Audit already-produced per-document accounting receipts, never acquire.

    Receipts are claims from a build, not proof of encoded payload contents.
    The report explicitly requires independent shard/payload verification before
    corpus release. Hash sets here are O(documents); not a 2B-scale index.
    One receipt per whole document; no segment-level quota double counting.
    """
    verify_seal(spec); verify_seal(registry)
    need(spec['registry_identity'] == registry['identity'], 'Registry identity mismatch')
    need(tokenizer_identity == TOKENIZER, 'Wrong accounting tokenizer')
    sources = {s['source_id']:s for s in registry['sources']}
    categories = Counter(); source_counts = Counter(); owner_counts = Counter(); groups = Counter()
    topic_counts = Counter(); form_counts = Counter(); reasons = Counter()
    exact = set(); files = set(); clusters = {}; source_splits = {}
    accepted_docs = 0; validation_tokens = 0; raw_tokens = 0; rejected_tokens = 0
    topics = {t['id']:t for t in spec['three_d']['topics']}
    for r in receipts:
        need(r['source_id'] in sources, 'Unknown receipt source')
        source = sources[r['source_id']]
        need(r['owner'] in source['allowed_categories'], 'Wrong budget owner')
        need(type(r['raw_tokens']) is int and r['raw_tokens'] >= 0, 'Invalid raw token count')
        raw_tokens += r['raw_tokens']
        if r['disposition'] != 'accepted':
            need(r['disposition'] in ('rejected','review') and r['reason'], 'Invalid rejection')
            reasons[r['reason']] += 1
            rejected_tokens += r['raw_tokens']
            continue
        need(not acquisition_blockers(source), 'Receipt source is not cleared')
        need(positive(r['clean_tokens']), 'Invalid clean token count')
        need(r['split'] == source['split'], 'Receipt split differs from pinned assignment')
        need(r['normalization'] in ('structured-v1','prose-v1'), 'Unknown normalization route')
        need(path_allowed(r['path'],source['allow_paths'],source['deny_paths']), 'Receipt path excluded')
        need(re.fullmatch(r'sha256:[0-9a-f]{64}',r['normalized_sha256']) is not None, 'Invalid content hash')
        key = (source['repository_group'],r['path'])
        need(key not in files, 'File counted more than once')
        need(r['normalized_sha256'] not in exact, 'Global exact duplicate counted')
        files.add(key); exact.add(r['normalized_sha256'])
        # A near-duplicate cluster has ONE canonical accepted member; others
        # must be rejected/reviewed, regardless of category or split.
        need(r['near_duplicate_review'] == 'cleared' and r['duplicate_cluster'], 'Near-duplicate review missing')
        need(r['duplicate_cluster'] not in clusters, 'Near-duplicate cluster counted more than once')
        clusters[r['duplicate_cluster']] = r['split']
        prior = source_splits.setdefault(source['split_group'],r['split'])
        need(prior == r['split'], 'Source group crosses train/validation')
        n = r['clean_tokens']; accepted_docs += 1
        source_counts[source['source_id']] += n
        owner_key=(source['source_id'],r['owner']); owner_counts[owner_key] += n
        need(owner_counts[owner_key] <= source['category_caps_tokens'][r['owner']], 'Source per-owner cap exceeded')
        need(source_counts[source['source_id']] <= source['target_cap_tokens'], 'Source cap exceeded')
        for field in ('repository_group','domain_group'):
            key = (field,source[field]); groups[key] += n
            need(groups[key] <= spec['group_caps'][field][source[field]], 'Concentration cap exceeded')
        if r['split']=='validation':
            validation_tokens += n
            continue
        categories[r['owner']] += n
        need(categories[r['owner']] <= QUOTAS[r['owner']], 'Category quota exceeded')
        topic, form = r.get('three_d_topic'),r.get('three_d_form')
        need(bool(topic) == bool(form), '3D topic/form must be assigned together')
        if topic:
            need(topic in topics and topics[topic]['owner']==r['owner'], 'Wrong 3D budget owner')
            need(form in spec['three_d']['form_targets'], 'Unknown 3D form')
            topic_counts[topic] += n; form_counts[form] += n
            need(topic_counts[topic] <= topics[topic]['target_tokens'], '3D topic cap exceeded')
            need(form_counts[form] <= spec['three_d']['form_targets'][form], '3D form cap exceeded')
    total=sum(categories.values())
    return seal(dict(schema='yug-production-checkpoint-audit-v1',spec_identity=spec['identity'],
        tokenizer_identity=tokenizer_identity,accepted_clean_train_tokens=total,
        validation_tokens=validation_tokens,raw_tokens=raw_tokens,rejected_or_review_raw_tokens=rejected_tokens,
        accepted_documents=accepted_docs,filter_counts=dict(reasons),
        category_progress={c:dict(tokens=categories[c],deficit=QUOTAS[c]-categories[c]) for c in QUOTAS},
        source_progress={s['source_id']:dict(tokens=source_counts[s['source_id']],remaining_cap=s['target_cap_tokens']-source_counts[s['source_id']]) for s in registry['sources']},
        three_d_topic_progress=dict(topic_counts),three_d_form_progress=dict(form_counts),
        source_diversity=sum(n>0 for n in source_counts.values()),
        crossed_checkpoints=[n for n in CHECKPOINTS if total>=n],
        actual_corpus_complete=False,
        evidence_limitation='Receipt accounting only; payload/shard hashes, tokenization and global near-duplicate detector results must be independently verified before release. No 2B completeness certification.'))


def dry_run(spec, registry, root):
    validate(spec, registry, root)
    eligible = []; blocked = {}; capacity = Counter(); planned_capacity = Counter()
    for s in registry['sources']:
        if s['status'] != 'REJECT' and s['training_allowed_after_clearance']:
            for owner in s['allowed_categories']:
                planned_capacity[owner] += min(s['target_cap_tokens'],s['category_caps_tokens'][owner])
        reasons = acquisition_blockers(s)
        if reasons:
            blocked[s['source_id']] = reasons
        else:
            m = source_manifest(root,s)
            total_bytes=sum(f['bytes'] for f in iter_manifest_files(root,m))
            details={'files':m['files']} if 'files' in m else {'files_jsonl':m['files_jsonl'],'files_sha256':m['files_sha256'],'file_count':m['file_count']}
            eligible.append(dict(source_id=s['source_id'],locator=s['locator'],revision=s['revision'],max_download_bytes=total_bytes,**details))
            if s['split'] == 'train':
                for owner in s['allowed_categories']:
                    capacity[owner] += min(s['target_cap_tokens'],s['category_caps_tokens'][owner])
    return seal(dict(schema='yug-production-dry-run-v1', spec_identity=spec['identity'],
        registry_identity=registry['identity'], status_counts=dict(sorted(Counter(s['status'] for s in registry['sources']).items())),
        configured_clean_train_tokens=sum(QUOTAS.values()), accepted_clean_train_tokens=0,
        actual_corpus_complete=False, category_deficits=dict(QUOTAS),
        source_progress={s['source_id']:dict(accepted_clean_tokens=0, target_tokens=None, deficit_tokens=None, cap_tokens=s['target_cap_tokens'], remaining_cap_tokens=s['target_cap_tokens']) for s in registry['sources']},
        source_target_note='Source targets/yield are unknown; remaining cap is not a measured source deficit.',
        acquisition_plan=eligible, acquisition_blockers=blocked,
        eligible_cap_upper_bounds={c:min(capacity[c], QUOTAS[c]) for c in QUOTAS},
        candidate_cap_upper_bounds={c:min(planned_capacity[c],QUOTAS[c]) for c in QUOTAS},
        minimum_structural_deficits={c:max(0,QUOTAS[c]-planned_capacity[c]) for c in QUOTAS},
        upper_bound_note='Caps are ceilings, not available yield; owner bounds are not additive across source views and group caps can reduce these further.',
        runtime_obligations=spec['runtime_obligations'],
        three_d_progress=dict(accepted_clean_tokens=0, topic_deficits={t['id']:t['target_tokens'] for t in spec['three_d']['topics']},form_deficits=spec['three_d']['form_targets']),
        ready_for_controlled_acquisition=bool(eligible) and not spec['runtime_obligations'],
        storage=dict(train_shard_bytes=8_000_000_000, train_spool_plus_shard_bytes=16_000_000_000,
            excludes='validation, raw/normalized/rejected text, indices, provenance, temp files and safety margin'),
        metrics=dict(raw_tokens=None,rejected_tokens=None,duplication_rate=None,
            bad_document_counts={},source_diversity=0,tokenizer_statistics=None,validation_split_status='not_built')))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--receipts',type=Path,help='Audit JSONL checkpoint receipts; no payload certification')
    parser.add_argument('--require-ready',action='store_true',help='Exit 2 if acquisition readiness is blocked')
    args=parser.parse_args()
    folder=args.root/'training/specifications'
    spec=json.loads((folder/'production_corpus_v1.json').read_text())
    registry=json.loads((folder/'source_registry_v1.json').read_text())
    report=dry_run(spec,registry,args.root)
    if args.receipts:
        with args.receipts.open(encoding='utf-8') as stream:
            report=audit_checkpoint(spec,registry,(json.loads(line) for line in stream if line.strip()),tokenizer_identity=spec['tokenizer_identity'])
    if args.output:
        need(not args.output.exists(), 'Refuse to overwrite an existing dry-run artifact')
        atomic_json(args.output,report)
    print(json.dumps(report,indent=2))
    if args.require_ready and not report.get('ready_for_controlled_acquisition',False):
        raise SystemExit(2)

if __name__ == '__main__':
    main()
