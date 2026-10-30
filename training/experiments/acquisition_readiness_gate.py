"""Offline acquisition-readiness evidence. Status capacities are cap ceilings,
NOT measured rights-cleared clean-token yield. No acquisition or training."""
import argparse
from collections import Counter, deque, defaultdict
import json
from pathlib import Path
from training.production_spec import QUOTAS, acquisition_blockers, validate, dry_run, seal
from training.production_corpus import atomic_json, file_digest
from training.production_dedup import POLICY

ROOT=Path(__file__).resolve().parents[2]


def capacity(spec,registry,category,statuses,*,cleared=False):
    """Single-category max flow respects shared source/repository/domain caps.

    Different category/status columns are independent upper bounds, not a
    jointly attainable allocation. Source yield is not supplied by max flow.
    """
    edges=defaultdict(dict)
    def add(a,b,n):
        edges[a][b]=edges[a].get(b,0)+n;edges[b].setdefault(a,0)
    repo_domains={};domains=set()
    for s in registry['sources']:
        if s['status'] not in statuses or not s['training_allowed_after_clearance'] or category not in s['allowed_categories']:continue
        if s['split']=='validation':continue
        if cleared and acquisition_blockers(s):continue
        repo=s['repository_group'];domain=s['domain_group']
        # Registry currently has a consistent repository->origin hierarchy.
        # Unknown/inconsistent aliases must never get an optimistic capacity.
        if repo in repo_domains and repo_domains[repo]!=domain:raise ValueError('Repository has conflicting origin domains')
        repo_domains[repo]=domain;domains.add(domain)
        add('start','source:'+s['source_id'],min(s['target_cap_tokens'],s['category_caps_tokens'][category]))
        add('source:'+s['source_id'],'repo:'+repo,2_000_000_000)
    for repo,domain in repo_domains.items():add('repo:'+repo,'domain:'+domain,spec['group_caps']['repository_group'][repo])
    for domain in domains:add('domain:'+domain,'end',spec['group_caps']['domain_group'][domain])
    flow=0
    while flow<QUOTAS[category]:
        previous={'start':None};queue=deque(['start'])
        while queue and 'end' not in previous:
            node=queue.popleft()
            for other in sorted(edges[node]):
                if edges[node][other]>0 and other not in previous:previous[other]=node;queue.append(other)
        if 'end' not in previous:break
        amount=QUOTAS[category]-flow;node='end'
        while previous[node] is not None:amount=min(amount,edges[previous[node]][node]);node=previous[node]
        node='end'
        while previous[node] is not None:
            parent=previous[node];edges[parent][node]-=amount;edges[node][parent]+=amount;node=parent
        flow+=amount
    return flow


def evaluate(root=ROOT):
    root=Path(root)
    spec=json.loads((root/'training/specifications/production_corpus_v1.json').read_bytes())
    registry=json.loads((root/'training/specifications/source_registry_v1.json').read_bytes())
    validate(spec,registry,root)
    dry=dry_run(spec,registry,root)
    table=[]
    for c,target in QUOTAS.items():
        possible=capacity(spec,registry,c,{'ACCEPT','CONDITIONAL','REVIEW'})
        approved_ceiling=capacity(spec,registry,c,{'ACCEPT'},cleared=True)
        candidates=sorted([s for s in registry['sources'] if c in s['allowed_categories'] and s['status'] in ('REVIEW','CONDITIONAL') and s['training_allowed_after_clearance']],key=lambda s:(s['category']!=c,s['source_id']))
        table.append(dict(category=c,target_tokens=target,
            accept_cap_ceiling=capacity(spec,registry,c,{'ACCEPT'}),conditional_cap_ceiling=capacity(spec,registry,c,{'CONDITIONAL'}),review_cap_ceiling=capacity(spec,registry,c,{'REVIEW'}),
            cleared_cap_ceiling=approved_ceiling,current_approved_clean_capacity=0,
            clean_yield_status='UNMEASURED',deficit_tokens=target,
            all_status_cap_upper_bound=possible,structural_cap_deficit=max(0,target-possible),
            candidates=[dict(source_id=s['source_id'],name=s['name'],status=s['status']) for s in candidates],
            issues=['missing per-item clearance','unverified clean yield']+(['licensing/component review'] if candidates else [])+(['current source/group cap or inventory size'] if possible<target else []),
            recommendation='Clear pinned permissive component/file scopes first, measure unique clean yield, then extend qualified sources in this category. Review cap changes explicitly only if measured yield justifies them; no automatic rebalance.'))
    topic_candidates={
        'modeling_workflows':['candidate-blender-manual','candidate-godot-3d-animation-skeleton-docs'],
        'topology_mesh_retopology':['candidate-libigl','candidate-opensubdiv'],
        'rigging_skeleton_skin_weights':['candidate-blender-manual','candidate-godot-3d-animation-skeleton-docs'],
        'animation_ik_fk_retarget':['candidate-blender-manual','candidate-godot-3d-animation-skeleton-docs'],
        'uv_material_pbr':['candidate-blender-manual','candidate-gltf-blender-io'],
        'blender_python_procedural':['candidate-blender-python-api','candidate-gltf-blender-io'],
        'asset_interchange':['candidate-khronos-gltf-specification','candidate-assimp-code-docs','candidate-openassetio'],
        'lod_optimization':['candidate-meshoptimizer','candidate-libigl']}
    three_d=[dict(**t,current_cleared_capacity=0,sufficient=False,candidates=topic_candidates[t['id']],note='Discovery candidates only; subtopic/form yield not measured or rights-cleared') for t in spec['three_d']['topics']]
    return seal(dict(schema='yug-acquisition-readiness-gate-v1',gate='NOT_READY',spec_identity=spec['identity'],registry_identity=registry['identity'],
        quota_tokens=sum(QUOTAS.values()),three_d_tokens=spec['three_d']['total_tokens'],registry_status_counts=dict(Counter(s['status'] for s in registry['sources'])),
        category_capacity=table,three_d_capacity=three_d,
        capacity_interpretation='Independent cap upper bounds, respecting repository/domain bottlenecks. Columns/categories cannot be added as independent available supply. Zero approved clean yield; no source was promoted.',
        dry_run=dict(valid=True,planned_download_files=sum(len(p['files']) if 'files' in p else p['file_count'] for p in dry['acquisition_plan']),planned_download_bytes=sum(p['max_download_bytes'] for p in dry['acquisition_plan']),eligible_sources=len(dry['acquisition_plan']),blockers=dry['acquisition_blockers']),
        assessment_scope='Offline inventory/capacity assessment. This command cannot authorize acquisition; completed implementation tests are recorded separately in verification.json.',
        pipeline_implementation=dict(near_dedup='training.production_dedup',clearance_split_resume='training.production_acquisition',replay='existing training.token_shards'),
        blockers=['No current source has all immutable per-item production clearance and split/origin assignments; meaningful controlled acquisition scope is still empty.',
            'Clean source/subtopic yield is unmeasured; several categories have structural cap/inventory deficits (see table).',
            '2B-scale throughput/disk use and lexical precision/recall have only bounded diagnostic evidence; size and quality-check the first cleared real-source batch before scaling.',
            'Known historical held-out membership must be supplied to the global dedup universe before production accounting; disjointness from every older experiment is not established.'],
        production_tokens_built=0,external_downloads=0,pretraining_authorized=False,
        storage=dict(known_minimum_training_token_bytes=24_000_000_000,
            basis='Per-item token cache + retained shard-build spool + final uint32 shards: 12 bytes per accepted token.',
            validation_token_bytes='12 * validation_token_count; validation budget not yet assigned',
            formulas=dict(raw_compressed='unknown upstream compressed bytes C',raw_extracted='approved request sum R (known only for concrete inventories)',normalized='N retained normalized UTF-8 bytes',dedup_features='16 * S stored unique lexical shingle hashes, plus SQLite row/indices',dedup_anchors='at most 128 per feature-bearing document; exact copies reuse anchors',clearance_manifests='O(document count), multiple copies of bounded provenance metadata',temporary='SQLite journals/sorts + at most 1 MiB per acquired item + bounded shard temp files'),
            provisional_working_capacity_gb=[250,400],
            assumptions='Planning allowance only: ~4 UTF-8 bytes/clean BPE token, up to 2x candidate text overhead, 0.2-2 million canonical documents plus rejected candidates, validation additional, SSD index/journal headroom. Compressed inputs and actual shingle/metadata density remain unknown; this is not a guaranteed sufficiency bound.',
            storage_release_rule='Use pinned inventory sizes and measured first-batch bytes/doc, features/doc and rejection rate to reserve disk before large acquisition.'),dedup_policy=POLICY))


def markdown(report):
    lines=['# YUG-100M Production Acquisition Readiness Gate v1','','Date: 2026-10-09 (Asia/Riyadh).','','**Final gate: NOT_READY.** No production corpus or pretraining was started.','',
        '## Category capacity','',
        'All values below are millions of tokens. ACCEPT/CONDITIONAL/REVIEW are independent cap ceilings, not observed clean yield. Shared repository/domain caps are respected per column; columns and categories are not additive available supply. Current approved clean capacity is zero.','',
        '| Category | Target | ACCEPT ceiling | CONDITIONAL ceiling | REVIEW ceiling | Approved clean capacity | Deficit |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in report['category_capacity']:
        lines.append('| '+r['category']+' | '+' | '.join(f'{r[k]/1e6:g}' for k in ('target_tokens','accept_cap_ceiling','conditional_cap_ceiling','review_cap_ceiling','current_approved_clean_capacity','deficit_tokens'))+' |')
    lines.extend(['','## Category-specific deficits and least-risk next steps',''])
    for r in report['category_capacity']:
        names=', '.join(x['name']+' ('+x['status']+')' for x in r['candidates'][:5]) or 'No reviewed source subset yet.'
        lines.append(f"- **{r['category']}**: clean deficit {r['deficit_tokens']/1e6:g}M; all-status structural cap deficit {r['structural_cap_deficit']/1e6:g}M. Issues: {', '.join(r['issues'])}. Candidates: {names}. {r['recommendation']}")
    lines.extend(['','## 100M internal 3D capacity','','| Topic | Owner | Target M | Cleared M | Sufficient |','|---|---|---:|---:|---|'])
    for t in report['three_d_capacity']:lines.append(f"| {t['id']} | {t['owner']} | {t['target_tokens']/1e6:g} | 0 | No |")
    lines.extend(['','Targets/form mix and 16 top-level allocations remain unchanged. Topic candidate IDs are retained in gate.json; no title-based capacity or rights clearance is inferred.','',
        '## Remaining blockers',''])
    lines.extend('- '+s for s in report['blockers'])
    return '\n'.join(lines)+'\n'



def retained_heldout_inventory(root=ROOT):
    """Known local comparison holdouts only; no assertion of historical completeness.

    Output is exclusion evidence, never source clearance or training admission.
    """
    from training.production_acquisition import read_plan, digest
    from training.protected_references import make_reference, validate_records
    sources=[('artifacts/tokenizer_fit_sample_gate_v1/validation.jsonl','fit_validation'),
             ('artifacts/tokenizer_transfer_gate_v1/documents.jsonl','transfer_evaluation')]
    records=[];inputs=[]
    for relative,kind in sources:
        path=Path(root)/relative;sha=file_digest(path);inputs.append(dict(path=relative,sha256=sha))
        for row in read_plan(path):
            text=row['text'];expected=row['content_hash'] if kind=='fit_validation' else row['text_sha256']
            if digest(text.encode('utf-8'))!='sha256:'+expected:raise ValueError('Retained held-out text hash mismatch')
            if kind=='fit_validation':
                if row['partition']!='validation':raise ValueError('Not a held-out record')
                source_path=row['provenance'].get('path','');domain=row['primary_domain']
                content_type='code' if domain in ('code','csharp_gamedev') else 'markdown' if domain=='software_docs' else 'prose'
                rid=row['document_id'];language=row['language']
            else:
                source_path=row['path'];rid=row['id'];language=''
                content_type='markdown' if Path(source_path).suffix.lower() in ('.md','.rst','.txt') else 'code'
            if Path(source_path).suffix.lower()=='.py':language='python'
            origin=row.get('provenance',row)
            work=origin.get('source') or row.get('source_id') or row.get('repo') or kind
            version=origin.get('revision') or row.get('commit') or sha
            records.append(make_reference(work_id=work,work_version=version,item_id=rid,
                reference_id=kind+':'+rid,role='reference',
                protected_split='validation' if kind=='fit_validation' else 'evaluation',
                text=text,content_type=content_type,language=language or 'und',
                provenance=dict(artifact=relative,artifact_sha256=sha,
                                original={k:v for k,v in row.items() if k!='text'})))
    return list(validate_records(records)),inputs

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    report=evaluate()
    for name in ('gate.json','capacity.md'):
        if (args.output/name).exists():raise ValueError('Refuse to overwrite prior gate evidence')
    atomic_json(args.output/'gate.json',report)
    (args.output/'capacity.md').write_text(markdown(report),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('gate','quota_tokens','three_d_tokens','registry_status_counts','production_tokens_built')},indent=2))

if __name__=='__main__':main()
