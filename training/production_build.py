"""Autonomous execution of approved immutable corpus inventories.

No license inference, source discovery or training. A new inventory/version needs
new approval; target increases within the same version do not. Global dedup must
settle across that version before token accounting starts.
"""
from __future__ import annotations
import argparse
import json
import hashlib
import os
from pathlib import Path
import shutil
import sqlite3
from contextlib import closing
import time
import urllib.error
import urllib.parse
import urllib.request
from training.production_spec import (TOKENIZER,QUOTAS,validate,need,seal,acquisition_blockers,
 source_manifest,iter_manifest_files,local_path)
from training.production_acquisition import ProductionBuild,read_plan,validate_clearance
from training.production_corpus import canonical,file_digest,atomic_json,atomic_replace
from training.token_shards import _writer_lock
from yug.tokenizer.bpe import ByteLevelBpeTokenizer

ROOT=Path(__file__).resolve().parents[1]
SAFE_LICENSES=frozenset(('MIT','Apache-2.0','BSD-2-Clause','BSD-3-Clause','CC0-1.0','CC-BY-4.0'))
POLICY='approved-inventory-runner-v1'


def compile_inventory(spec,registry,root,directory,plan=None):
    """Stream already-approved per-file grants; unresolved files never scheduled."""
    validate(spec,registry,root)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    request=dict(policy=POLICY,spec=spec['identity'],registry=registry['identity'],
                 input_plan=file_digest(plan) if plan else None)
    final=directory/'plan.jsonl';receipt=directory/'inventory.json';skipped=directory/'unresolved.jsonl'
    if receipt.exists():
        old=json.loads(receipt.read_bytes())
        need(old['request']==request,'Runner inventory changed; create a new corpus version')
        need(file_digest(final)==old['plan_sha256'] and file_digest(skipped)==old['unresolved_sha256'],'Runner inventory corrupt')
        return final,old
    sources={s['source_id']:s for s in registry['sources']}
    counts=dict(scheduled=0,unresolved=0,raw_bytes=0)
    temp=final.with_suffix('.part');stemp=skipped.with_suffix('.part')
    with closing(sqlite3.connect('')) as lookup,temp.open('wb') as out,stemp.open('wb') as log:
        lookup.execute('PRAGMA cache_size=-4096')
        lookup.execute('CREATE TABLE approved (source TEXT,path TEXT,record TEXT,PRIMARY KEY(source,path))')
        for source in registry['sources']:
            if not acquisition_blockers(source):
                for entry in iter_manifest_files(root,source_manifest(root,source)):
                    lookup.execute('INSERT INTO approved VALUES (?,?,?)',(source['source_id'],entry['path'],canonical(entry).decode()))
        def skip(sid,path,reason):
            counts['unresolved']+=1;log.write(canonical(dict(source_id=sid,path=path,status='UNRESOLVED',reason=reason))+b'\n')
        def accept(r):
            source=sources.get(r['source_id'])
            if source is None:skip(r['source_id'],r.get('path'),'unknown source');return
            if acquisition_blockers(source):skip(r['source_id'],r.get('path'),'source has no approved immutable acquisition scope');return
            if r.get('license_id') not in SAFE_LICENSES or r.get('clearance_status')!='ACCEPT':skip(r['source_id'],r.get('path'),'unsupported/ambiguous license or uncleared item');return
            origin=urllib.parse.urlsplit(source['locator'])
            github=(origin.scheme=='https' and origin.netloc=='github.com' and not origin.query and not origin.fragment
                    and len(origin.path.strip('/').split('/'))==2 and '..' not in origin.path)
            if not github and not (r.get('input_path') and local_path(root,r['input_path']).is_file()):
                skip(r['source_id'],r.get('path'),'unsupported transport; requires verified local payload');return
            row=lookup.execute('SELECT record FROM approved WHERE source=? AND path=?',(r['source_id'],r['path'])).fetchone()
            need(row is not None,'Plan path has no immutable approval')
            validate_clearance(r,source,spec,root,approved_file=json.loads(row[0]))
            counts['scheduled']+=1;counts['raw_bytes']+=r['bytes'];out.write(canonical(r)+b'\n')
        if plan:
            for r in read_plan(plan):accept(r)
        else:
            for source in registry['sources']:
                sid=source['source_id']
                if acquisition_blockers(source):skip(sid,None,'source has no approved immutable acquisition scope');continue
                grant=json.loads(local_path(root,source['clearance_record']).read_bytes())
                for f in iter_manifest_files(root,source_manifest(root,source)):
                    if f.get('license_id') not in SAFE_LICENSES:skip(sid,f['path'],'unsupported/ambiguous per-file license');continue
                    owner=f['owner'];category=next(c for c in spec['categories'] if c['id']==owner)
                    sub=f.get('subcategory',source['subcategory'])
                    if sub not in category['subcategories']:skip(sid,f['path'],'subcategory needs approved routing');continue
                    from training.production_spec import path_allowed
                    rules=[rule for rule in source['allow_paths'] if path_allowed(f['path'],[rule],source['deny_paths'])]
                    if not rules:skip(sid,f['path'],'path excluded');continue
                    r=dict(source_id=sid,revision=source['revision'],path=f['path'],sha256=f['sha256'],bytes=f['bytes'],owner=owner,
                           content_type=f['content_type'],license_id=f['license_id'],clearance_status='ACCEPT',quality_review='ACCEPT',
                           subcategory=sub,split=source['split'],split_group=source['split_group'],split_unit_kind='source',
                           allow_rule=rules[0],deny_rules_checked=source['deny_paths'],language=f.get('language',sub),
                           license_evidence=f.get('license_evidence',grant['evidence']),attribution=grant['attribution'])
                    for key in ('input_path','three_d_topic','three_d_form'):
                        if key in f:r[key]=f[key]
                    accept(seal(r))
        out.flush();os.fsync(out.fileno());log.flush();os.fsync(log.fileno())
    atomic_replace(temp,final);atomic_replace(stemp,skipped)
    result=dict(request=request,**counts,plan_sha256=file_digest(final),unresolved_sha256=file_digest(skipped))
    atomic_json(receipt,result)
    return final,result


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*a,**k):raise ValueError('Acquisition redirects are not approved')


class Fetcher:
    """One in-flight item, bounded 64KiB reads, rate-limited pinned GitHub URLs."""
    def __init__(self,root,registry,interval=0.25,timeout=30):
        self.root=Path(root);self.sources={s['source_id']:s for s in registry['sources']}
        self.interval=interval;self.timeout=timeout;self.last=0.;self.network_files=0;self.network_bytes=0
    def __call__(self,r):
        if r.get('input_path'):
            path=local_path(self.root,r['input_path'])
            if path.exists():
                need(file_digest(path)==r['sha256'] and path.stat().st_size==r['bytes'],'Local approved payload corrupt')
                with path.open('rb') as stream:
                    while block:=stream.read(65536):yield block
                return
        source=self.sources[r['source_id']];url=urllib.parse.urlsplit(source['locator'])
        need(url.scheme=='https' and url.netloc.lower()=='github.com' and not url.query and not url.fragment,'No approved transport for this source; provide a verified local input_path')
        repo=url.path.strip('/');need(len(repo.split('/'))==2 and '..' not in repo,'Invalid GitHub origin')
        target='https://raw.githubusercontent.com/'+repo+'/'+r['revision']+'/'+urllib.parse.quote(r['path'],safe='/')
        delay=self.interval-(time.monotonic()-self.last)
        if delay>0:time.sleep(delay)
        self.last=time.monotonic();size=0
        request=urllib.request.Request(target,headers={'User-Agent':'YUG-approved-production-runner','Accept-Encoding':'identity'})
        with urllib.request.build_opener(NoRedirect()).open(request,timeout=self.timeout) as stream:
            need(stream.status==200 and stream.geturl()==target,'Unexpected acquisition response');self.network_files+=1
            while block:=stream.read(min(65536,r['bytes']-size+1)):
                size+=len(block);self.network_bytes+=len(block)
                need(size<=r['bytes'],'Source exceeds approved byte ceiling');yield block


def disk_preflight(directory,required):
    free=shutil.disk_usage(directory).free
    need(free>=required,f'Insufficient free disk: need {required} bytes, available {free}. No payload acquisition started for this chunk.')
    return free


def execute(args):
    root=Path(args.root).resolve();directory=Path(args.directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    spec=json.loads(Path(args.spec).read_bytes());registry=json.loads(Path(args.registry).read_bytes())
    need(0<args.target_tokens<=sum(QUOTAS.values()),'Target must be between 1 and 2,000,000,000')
    need(args.chunk_items>0 and args.reserve_mib>=0 and args.request_interval>=0,'Invalid resource limits')
    need(args.resume or not (directory/'state.sqlite').exists(),'Existing build: use --resume')
    with _writer_lock(directory/'runner'):
        plan,inventory=compile_inventory(spec,registry,root,directory/'runner',args.plan)
        need(inventory['scheduled']>0,'No cleared files scheduled. See runner/unresolved.jsonl; no licensing guesses or downloads were made.')
        tokenizer=ByteLevelBpeTokenizer.load(args.tokenizer);need(tokenizer.identity==TOKENIZER,'Frozen tokenizer mismatch')
        fetch=Fetcher(root,registry,args.request_interval)
        with ProductionBuild(directory,spec,registry,repository_root=root,reference_plan=args.references) as build:
            current=build.db.execute("SELECT coalesce(sum(tokens),0) FROM totals WHERE kind='category'").fetchone()[0]
            need(args.target_tokens>=current,'Target is below already committed tokens; accounting cannot be rolled back')
            if args.dry_run:
                result=build.dry_run(plan);atomic_json(directory/'runner/progress.json',result);return 0
            # Conservative estimate, not an OS reservation or a guarantee for all source distributions.
            pending_bytes=sum(json.loads(r[0])['bytes'] for r in build.db.execute("SELECT record FROM items WHERE stage='PENDING'")) if build.get('plan_registered') else inventory['raw_bytes']
            pending_count=build.db.execute("SELECT count(*) FROM items WHERE stage='PENDING'").fetchone()[0] if build.get('plan_registered') else inventory['scheduled']
            reserve=args.reserve_mib*1024*1024
            disk_preflight(directory,reserve+pending_bytes*32+pending_count*65536)
            retry=0
            while True:
                disk_preflight(directory,reserve)
                try:progress=build.run(plan,tokenizer,fetch=fetch,max_items=args.chunk_items,target_tokens=args.target_tokens)
                except (urllib.error.URLError,TimeoutError,ConnectionError):
                    retry+=1
                    if retry>2:raise
                    time.sleep(retry);continue
                retry=0;total=progress['accepted_clean_train_tokens']
                result=dict(progress,target_tokens=args.target_tokens,network_files_this_process=fetch.network_files,network_bytes_this_process=fetch.network_bytes,
                            note='All inventory payloads settle global dedup before accounting; token target is not a download-byte limit.')
                atomic_json(directory/'runner/progress.json',result)
                print(json.dumps(dict(state=progress['state'],train_tokens=total,target=args.target_tokens,stages=progress['stages'])),flush=True)
                if progress['state']!='PARTIAL':break
            ledger=hashlib.sha256()
            for row in build.db.execute('SELECT item,stage,reason,normalized_sha,token_sha,token_count FROM items ORDER BY item'):
                ledger.update(canonical(list(row))+b'\n')
            destination=directory/'exports'/f'tokens-{total}-{ledger.hexdigest()[:16]}'
            shards=build.export_shards(destination)
            result.update(export_directory=str(destination),shard_manifest_identity=shards['manifest_identity'],
                          target_satisfied=total>=args.target_tokens,inventory_exhausted=progress['state']=='COMPLETE' and total<args.target_tokens)
            atomic_json(directory/'runner/progress.json',result)
            return 0 if result['target_satisfied'] else 3


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(ROOT));p.add_argument('--spec',default=str(ROOT/'training/specifications/production_corpus_v1.json'))
    p.add_argument('--registry',default=str(ROOT/'training/specifications/source_registry_v1.json'))
    p.add_argument('--plan',type=Path);p.add_argument('--directory',default=str(ROOT/'artifacts/autonomous_production_v1'))
    p.add_argument('--references',default=str(ROOT/'artifacts/acquisition_readiness_gate_v1/heldout_references.jsonl'))
    p.add_argument('--tokenizer',default=str(ROOT/'artifacts/bpe_fitter_optimization_gate_v1/comparison/run-4096-0/tokenizer.json'))
    p.add_argument('--target-tokens',type=int,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--chunk-items',type=int,default=50);p.add_argument('--reserve-mib',type=int,default=512);p.add_argument('--request-interval',type=float,default=.25)
    args=p.parse_args()
    try:return execute(args)
    except KeyboardInterrupt:
        print('Interrupted. Committed stages remain valid; rerun the same command with --resume.',flush=True);return 130
    except Exception as exc:
        print(f'Production runner stopped: {type(exc).__name__}: {exc}',flush=True);return 2

if __name__=='__main__':raise SystemExit(main())
