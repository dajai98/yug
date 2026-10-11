"""Offline cache-to-admission compiler. Exact existing grants only; no license inference."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit
from training.production_spec import (seal,verify_seal,validate,need,local_path,source_manifest,iter_manifest_files)
from training.production_corpus import atomic_json,canonical,file_digest,atomic_replace
from training.production_acquisition import read_plan,validate_clearance,item_key,ProductionBuild
from training.source_prepare import screening,ROOT
from training.token_shards import _writer_lock

POLICY='exact-evidenced-cache-admission-v1'


def holdout_groups(references,registry):
    ids=set();repos=set()
    def visit(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ('source_id','split_group','repository_group') and isinstance(item,str):ids.add(item.casefold())
                if key in ('repo','repository') and isinstance(item,str):repos.add(item.casefold().rstrip('/').removesuffix('.git'))
                if key in ('url','locator') and isinstance(item,str):
                    u=urlsplit(item);parts=u.path.strip('/').split('/')
                    if u.netloc.lower() in ('github.com','raw.githubusercontent.com') and len(parts)>=2:repos.add('/'.join(parts[:2]).casefold().removesuffix('.git'))
                visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    for row in read_plan(references):visit(row['provenance'])
    groups=set()
    for s in registry['sources']:
        repo=urlsplit(s.get('locator') or '').path.strip('/').casefold().removesuffix('.git')
        if repo in repos or any(str(s.get(k,'')).casefold() in ids for k in ('source_id','split_group','repository_group')):groups.add(s['split_group'])
    return groups


def cached_sources(cache,registry,spec,root):
    """Verify retained receipts and payloads without invoking source preparation or Git."""
    sources={s['source_id']:s for s in registry['sources']}
    for receipt in sorted(Path(cache).glob('*/complete.json')):
        directory=receipt.parent;done=json.loads(receipt.read_bytes());verify_seal(done)
        request=json.loads((directory/'request.json').read_bytes());verify_seal(request)
        source=request['source'];sid=source['source_id']
        need(sid in sources and source==sources[sid],'Cache source differs from registry')
        need(request['filters']==spec['filters'],'Cache screening policy changed')
        need(done['request_identity']==request['identity'] and done['revision']==source['revision'],'Cache revision/receipt mismatch')
        path=directory/'inventory.jsonl';need(file_digest(path)==done['inventory_sha256'],'Cache inventory corrupt')
        yield directory,source,path,done


def build(cache,root_registry,root_spec,approved_registry,approved_spec,approved_plan,references,output,*,root=ROOT,evidence=None):
    root=Path(root).resolve();output=Path(output).resolve();cache=Path(cache).resolve()
    need(output.is_relative_to(root) and not output.is_relative_to(cache) and not cache.is_relative_to(output),'Output must be separate from cache under repository')
    for protected in (root_registry,root_spec,approved_registry,approved_spec,approved_plan,references):
        need(not Path(protected).resolve().is_relative_to(output),'Output overlaps immutable admission inputs')
    output.mkdir(parents=True,exist_ok=True)
    base_reg=json.loads(Path(root_registry).read_bytes());base_spec=json.loads(Path(root_spec).read_bytes())
    verify_seal(base_reg);verify_seal(base_spec)
    registry=json.loads(Path(approved_registry).read_bytes());spec=json.loads(Path(approved_spec).read_bytes())
    validate(spec,registry,root)
    # Only authority changes represented by existing grants are allowed. Allocation is frozen.
    need(spec['categories']==base_spec['categories'] and spec['three_d']==base_spec['three_d'],'Approved overlay changed allocation')
    inputs={str(Path(p).resolve().relative_to(root)).replace('\\','/'):file_digest(p) for p in (root_registry,root_spec,approved_registry,approved_spec,approved_plan,references)}
    caches=list(cached_sources(cache,base_reg,base_spec,root));need(caches,'No completed cache receipts found')
    for directory,source,path,done in caches:
        for p in (path,directory/'request.json',directory/'complete.json'):inputs[p.relative_to(root).as_posix()]=file_digest(p)
    if evidence is not None:
        from training.clearance_evidence import verify_bundle
        evidence=Path(evidence).resolve()
        need(not evidence.is_relative_to(output) and not output.is_relative_to(evidence),'Admission output overlaps evidence')
        bundle=verify_bundle(evidence,root)
        need(bundle['complete'],'Partial evidence cannot be admitted')
        evidence_request=json.loads((evidence/'request.json').read_bytes())
        need(evidence_request['inputs']==inputs,'Evidence does not cover these admission inputs')
        for name in ('request.json','complete.json','evidence.jsonl'):
            p=evidence/name;inputs[p.relative_to(root).as_posix()]=file_digest(p)
    request=seal(dict(policy=POLICY,inputs=inputs))
    old_request=output/'request.json'
    if old_request.exists():need(json.loads(old_request.read_bytes())==request,'Admission inputs changed; use a new output version')
    else:atomic_json(old_request,request)
    existing=output/'complete.json'
    if existing.exists():
        complete=json.loads(existing.read_bytes());verify_seal(complete)
        need(complete['request_identity']==request['identity'],'Admission receipt changed')
        for name,sha in complete['artifacts'].items():need(file_digest(output/name)==sha,'Admission output corrupt')
    db=sqlite3.connect('');db.execute('PRAGMA cache_size=-8192')
    db.executescript('CREATE TABLE grants (item TEXT PRIMARY KEY,source TEXT,path TEXT,record TEXT); CREATE TABLE files(source TEXT,path TEXT,record TEXT,PRIMARY KEY(source,path)); CREATE TABLE seen(source TEXT,path TEXT,PRIMARY KEY(source,path));')
    try:
        db.execute('CREATE TABLE evidence(source TEXT,path TEXT,payload TEXT,PRIMARY KEY(source,path))')
        if evidence is not None:
            for e in read_plan(evidence/'evidence.jsonl'):
                verify_seal(e)
                need(e['request_identity']==bundle['request_identity'],'Evidence row request mismatch')
                db.execute('INSERT INTO evidence VALUES (?,?,?)',(e['source_id'],e['path'],canonical(e).decode()))
        sources={s['source_id']:s for s in registry['sources']}
        grant_sources={r['source_id'] for r in read_plan(approved_plan)}
        for sid in sorted(grant_sources):
            for f in iter_manifest_files(root,source_manifest(root,sources[sid])):db.execute('INSERT INTO files VALUES (?,?,?)',(sid,f['path'],canonical(f).decode()))
        groups=holdout_groups(references,base_reg)|holdout_groups(references,registry)
        for r in read_plan(approved_plan):
            f=db.execute('SELECT record FROM files WHERE source=? AND path=?',(r['source_id'],r['path'])).fetchone()
            need(f is not None,'Grant absent from approved manifest')
            validate_clearance(r,sources[r['source_id']],spec,root,json.loads(f[0]))
            if r.get('input_path'):
                payload=local_path(root,r['input_path'])
                need(payload.is_file() and payload.stat().st_size==r['bytes'] and file_digest(payload)==r['sha256'],'Approved carry-forward payload corrupt or missing')
            need(r['split_group'] not in groups,'Existing approved plan conflicts with retained holdout group')
            db.execute('INSERT INTO grants VALUES (?,?,?,?)',(item_key(r),r['source_id'],r['path'],canonical(r).decode()))
        counts=Counter();reasons=Counter();per_source={};matched=0
        temp=output/'decisions.part'
        with temp.open('wb') as decisions:
            for directory,source,path,done in caches:
                sc=Counter()
                for row in read_plan(path):
                    sid=source['source_id'];need(row['source_id']==sid and row['revision']==source['revision'] and row['locator']==source['locator'],'Cache row provenance mismatch')
                    db.execute('INSERT INTO seen VALUES (?,?)',(sid,row['path']))
                    blob=None;data=b''
                    if row['local_blob']:
                        blob=local_path(directory,row['local_blob'])
                        need(blob.is_relative_to(directory/'files') and blob.is_file(),'Invalid cached payload location')
                        need(blob.stat().st_size==row['bytes'] and file_digest(blob)==row['sha256'],'Cached payload corrupt')
                        if row['bytes']<=1_048_576:data=blob.read_bytes()
                    need(row['screening'] in ('UNRESOLVED','EXCLUDED'),'Unknown cache screening state')
                    status,reason=screening(row['path'],row['git_mode'],row['bytes'] or 0,data,source,base_spec)
                    status='EXCLUDED' if status=='EXCLUDED' or row['screening']=='EXCLUDED' else 'REVIEW'
                    grant=None
                    original_clearance=None
                    if status!='EXCLUDED' and source['split_group'] in groups:status,reason='EXCLUDED','retained evaluation source group'
                    if status=='REVIEW':
                        key=item_key(dict(source_id=sid,revision=row['revision'],path=row['path'],sha256=row['sha256']))
                        match=db.execute('SELECT record FROM grants WHERE item=?',(key,)).fetchone()
                        if match:
                            grant=json.loads(match[0]);need(grant['bytes']==row['bytes'],'Grant byte size mismatch')
                            original_clearance=grant['identity']
                            need(sources[sid]['locator'].casefold().rstrip('/')==source['locator'].casefold().rstrip('/'),'Grant repository origin mismatch')
                            status,reason='CLEARED','exact existing immutable per-file clearance'
                            # Same approved payload; route acquisition to verified local cache.
                            grant=seal(dict(grant,input_path=blob.relative_to(root).as_posix()))
                            db.execute('UPDATE grants SET record=? WHERE item=?',(canonical(grant).decode(),key));matched+=1
                        else:reason='no exact evidenced per-file clearance; source/root license is insufficient'
                    if evidence is not None:
                        found=db.execute('SELECT payload FROM evidence WHERE source=? AND path=?',(sid,row['path'])).fetchone()
                        need(found is not None,'Missing file evidence')
                        e=json.loads(found[0])
                        need(all(e[k]==row[k] for k in ('source_id','revision','path','sha256','bytes')) and e['cache_inventory_sha256']==done['inventory_sha256'],'Evidence file binding mismatch')
                        need(e['status']==('EVIDENCED' if status=='CLEARED' else status),'Evidence/admission classification mismatch')
                        if grant:need(e['scope']['clearance_identity']==original_clearance and e['license_id']==grant['license_id'],'Evidence clearance binding mismatch')
                    result=dict(source_id=sid,revision=row['revision'],path=row['path'],sha256=row['sha256'],bytes=row['bytes'],status=status,reason=reason,category=row.get('category'),subcategory=row.get('subcategory'),cache_inventory_sha256=done['inventory_sha256'],clearance_identity=grant['identity'] if grant else None,production_already_in_approved_scope=grant is not None)
                    decisions.write(canonical(result)+b'\n');counts[status]+=1;reasons[reason]+=1;sc[status]+=1
                need(sum(sc.values())==sum(done['counts'].values()),'Cache inventory count mismatch')
                per_source[sid]=dict(sc)
        if evidence is not None:need(db.execute('SELECT count(*) FROM evidence').fetchone()[0]==sum(counts.values()),'Unexpected evidence rows')
        if existing.exists():
            need(file_digest(temp)==file_digest(output/'decisions.jsonl'),'Admission decisions changed');temp.unlink()
            return complete
        atomic_replace(temp,output/'decisions.jsonl')
        atomic_json(output/'registry.json',registry);atomic_json(output/'spec.json',spec)
        with (output/'plan.jsonl').open('wb') as stream:
            for encoded, in db.execute('SELECT record FROM grants ORDER BY item'):stream.write(encoded.encode()+b'\n')
        # Read-only production preflight: no payload copying, normalization, tokenization or accounting.
        with ProductionBuild(output/'dry_run',spec,registry,repository_root=root,reference_plan=references) as coordinator:
            preflight=coordinator.dry_run(output/'plan.jsonl')
        atomic_json(output/'dry_run.json',preflight)
        result=seal(dict(policy=POLICY,request_identity=request['identity'],counts={k:counts[k] for k in ('CLEARED','EXCLUDED','REVIEW')},reasons=dict(reasons),per_source=per_source,cached_sources=len(caches),reviewed_files=sum(counts.values()),cleared_cache_matches=matched,newly_authorized_files=0,production_plan_files=db.execute('SELECT count(*) FROM grants').fetchone()[0],holdout_source_groups=sorted(groups),new_production_tokens=0,note='Plan preserves entire approved universe. Use a fresh build directory; never append this plan to an existing production database.',artifacts={name:file_digest(output/name) for name in ('decisions.jsonl','registry.json','spec.json','plan.jsonl','dry_run.json')}))
        atomic_json(existing,result);return result
    finally:db.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'artifacts/source_admission_gate_v1')
    p.add_argument('--evidence',type=Path,help='Complete clearance_evidence output; exact grants remain mandatory')
    p.add_argument('--registry',type=Path,default=ROOT/'training/specifications/source_registry_v1.json');p.add_argument('--spec',type=Path,default=ROOT/'training/specifications/production_corpus_v1.json')
    p.add_argument('--approved-registry',type=Path,default=ROOT/'artifacts/production_batch_1_v1/admission/registry.json');p.add_argument('--approved-spec',type=Path,default=ROOT/'artifacts/production_batch_1_v1/admission/spec.json');p.add_argument('--approved-plan',type=Path,default=ROOT/'artifacts/production_batch_1_v1/admission/plan.jsonl');p.add_argument('--references',type=Path,default=ROOT/'artifacts/acquisition_readiness_gate_v1/heldout_references.jsonl')
    a=p.parse_args()
    try:
        need(a.output.resolve().is_relative_to(ROOT/'artifacts'),'Output must remain under artifacts')
        with _writer_lock(a.output/'compiler'):
            r=build(a.cache,a.registry,a.spec,a.approved_registry,a.approved_spec,a.approved_plan,a.references,a.output,evidence=a.evidence)
        print(json.dumps({k:r[k] for k in ('counts','reviewed_files','production_plan_files','newly_authorized_files','new_production_tokens')},sort_keys=True));return 0
    except Exception as exc:print(f'Admission stopped: {type(exc).__name__}: {exc}');return 2

if __name__=='__main__':raise SystemExit(main())
