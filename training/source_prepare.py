"""Pinned GitHub review caches. Downloaded never means cleared for training."""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.parse import urlsplit
from training.production_corpus import atomic_json,atomic_replace,canonical,file_digest
from training.production_spec import need,path_allowed,seal,verify_seal
from training.production_acquisition import MAX_ITEM_BYTES
from training.token_shards import _writer_lock
from training.conditional_license import CONDITIONAL

ROOT=Path(__file__).resolve().parents[1]
POLICY='local-source-review-cache-v1'
LICENSES={'MIT','Apache-2.0','BSD-2-Clause','BSD-3-Clause','CC0-1.0','CC-BY-3.0','CC-BY-4.0'}
EXTRA_DENY=['**/.git/**','**/__pycache__/**','**/.venv/**','**/venv/**','**/.cache/**','**/out/**','**/obj/**','**/bin/**']
NON_TEXT={'.zip','.gz','.tar','.7z','.xz','.pdf','.png','.jpg','.jpeg','.gif','.webp','.svg','.ico','.mp3','.wav','.ogg','.mp4','.mov','.woff','.ttf','.dll','.exe','.so','.a','.o','.class','.pyc','.bin','.blend','.fbx','.glb'}

def classify(source):
    if source.get('status')=='REJECT':return 'REJECTED'
    locator=source.get('locator')
    if not locator:return 'BLOCKED_NO_LOCATOR'
    u=urlsplit(locator)
    if u.scheme!='https' or u.netloc!='github.com' or u.query or u.fragment or not re.fullmatch(r'/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?',u.path):return 'UNRESOLVED'
    if not re.fullmatch('[0-9a-f]{40}',source.get('revision') or ''):return 'BLOCKED_NO_REVISION'
    if source.get('license_id') in CONDITIONAL:
        if source.get('status') not in ('CONDITIONAL','ACCEPT') or not source.get('license_evidence'):return 'BLOCKED_LICENSE'
        return 'CACHE_READY'  # Review cache only; explicit approval is enforced at clearance.
    if source.get('status')!='ACCEPT' or source.get('license_id') not in LICENSES or not source.get('license_evidence'):return 'BLOCKED_LICENSE'
    return 'CACHE_READY'


def git(repo,*args,check=True,stdout=subprocess.PIPE):
    env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL=os.devnull,GIT_TERMINAL_PROMPT='0',GIT_LFS_SKIP_SMUDGE='1')
    command=['git','-c','core.hooksPath='+os.devnull,'-c','protocol.file.allow=never','-c','http.followRedirects=false','-c','credential.helper=']
    if repo is not None:command+=['--git-dir',str(repo)]
    return subprocess.run(command+list(args),stdin=subprocess.DEVNULL,stdout=stdout,stderr=subprocess.PIPE,env=env,check=check,timeout=180)


def fetch(repo,source):
    git(repo,'fetch','--depth=1','--no-tags','--no-recurse-submodules',source['locator'],source['revision'])


def entries(repo,revision,directory):
    listing=directory/'tree.part'
    with listing.open('wb') as out:git(repo,'ls-tree','-r','-z','--long',revision,stdout=out)
    with listing.open('rb') as stream:
        pending=b''
        while block:=stream.read(65536):
            pending+=block
            while b'\0' in pending:
                row,pending=pending.split(b'\0',1)
                meta,path=row.split(b'\t',1);mode,kind,oid,size=meta.split()
                yield mode.decode(),kind.decode(),oid.decode(),None if size==b'-' else int(size),path.decode('utf-8','surrogateescape')
        need(not pending,'Incomplete Git tree listing')
    listing.unlink()


def screening(path,mode,size,data,source,spec):
    if mode not in ('100644','100755'):return 'EXCLUDED','symlink/submodule/nonregular entry'
    deny=spec['filters']['deny_paths']+source.get('deny_paths',[])+EXTRA_DENY
    if not path_allowed(path,['**'],deny):return 'EXCLUDED','path denied by corpus/cache policy'
    if Path(path).suffix.lower() in NON_TEXT:return 'EXCLUDED','binary/media/archive extension'
    if size>MAX_ITEM_BYTES:return 'EXCLUDED','exceeds supported production document size'
    if b'\0' in data:return 'EXCLUDED','binary NUL'
    try:text=data.decode('utf-8-sig')
    except UnicodeError:return 'EXCLUDED','non-UTF8/binary'
    if any(m.casefold() in text.casefold() for m in spec['filters']['generated_markers']):return 'EXCLUDED','generated marker'
    if not source.get('allow_paths') or not path_allowed(path,source['allow_paths'],[]):return 'UNRESOLVED','no approved path routing'
    return 'UNRESOLVED','requires immutable file-level clearance and retained-holdout review'


def inventory(repo,source,spec,directory):
    """Read exact pinned Git blobs; no filters, LFS smudge, hooks or submodule fetch."""
    temp=directory/'inventory.part';counts=Counter();materialized=directory/'files';materialized.mkdir(exist_ok=True)
    # Content-addressed files avoid OS path aliases, symlinks and traversal on Windows.
    with temp.open('wb') as out:
        for mode,kind,oid,size,path in entries(repo,source['revision'],directory):
            data=b'';sha=None;blob_path=None
            if kind=='blob':
                blob=materialized/(oid+'.blob')
                # git cat-file writes raw object bytes, equivalent to a checkout without transforms.
                part=blob.with_suffix('.part')
                with part.open('wb') as target:git(repo,'cat-file','blob',oid,stdout=target)
                need(part.stat().st_size==size,'Git blob size mismatch')
                sha=file_digest(part)
                if size<=MAX_ITEM_BYTES:data=part.read_bytes()
                if blob.exists():
                    need(file_digest(blob)==sha,'Corrupt materialized cache blob')
                    part.unlink()
                else:atomic_replace(part,blob)
                blob_path=str(blob.relative_to(directory)).replace('\\','/')
            status,reason=screening(path,mode,size or 0,data,source,spec)
            counts[status]+=1
            detected='text'
            if kind!='blob':detected='gitlink'
            elif mode=='120000':detected='symlink'
            elif size>MAX_ITEM_BYTES:detected='oversized'
            elif b'\0' in data:detected='binary'
            else:
                try:data.decode('utf-8-sig')
                except UnicodeError:detected='non-UTF8/binary'
            row=dict(source_id=source['source_id'],locator=source['locator'],revision=source['revision'],path=path,bytes=size,sha256=sha,git_object=oid,git_mode=mode,extension=Path(path).suffix.lower(),detected_type=detected,category=source.get('category'),subcategory=source.get('subcategory'),source_license_id=source.get('license_id'),source_license_evidence=source.get('license_evidence'),screening=status,reason=reason,local_blob=blob_path,training_allowed=False)
            out.write(canonical(row)+b'\n')
        out.flush();os.fsync(out.fileno())
    atomic_replace(temp,directory/'inventory.jsonl')
    return dict(counts)


def _verify_review_license(directory, source):
    """Bind retained license metadata to the pinned Git bytes, without resolving rights."""
    reference=source['license_evidence']
    license_row=json.loads(Path(reference['path']).read_bytes())
    body=license_row['body']
    data=base64.b64decode(''.join(body['content'].split()),validate=True)
    need(body['path']==reference['upstream_path'] and
         len(data)==body['size'] and
         file_digest(Path(reference['path']))==reference['sha256'] and
         'sha256:'+hashlib.sha256(data).hexdigest()==reference['text_sha256'] and
         hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==body['sha'],
         'Retained license bytes or Git identity changed')
    with (directory/'inventory.jsonl').open('rb') as inventory_rows:
        matches=[row for line in inventory_rows
                 if (row:=json.loads(line))['path']==body['path']]
    need(len(matches)==1 and matches[0]['sha256']==reference['text_sha256'] and
         matches[0]['git_object']==body['sha'] and matches[0]['bytes']==len(data),
         'Pinned license is absent or differs from the Git inventory')


def prepare_registered_review(registry, source_id, revision, spec, references,
                              output, *, expected_registry_identity,
                              expected_spec_identity, expected_references_sha256,
                              root=ROOT, resume=False, fetcher=None):
    """Cache an already registered REVIEW source for inspection only."""
    from training.source_admission import holdout_groups
    from training.source_grant_proposals import _source_evidence

    root=Path(root).resolve()
    output=Path(output).resolve()
    references=Path(references).resolve()
    verify_seal(registry)
    verify_seal(spec)
    need(registry.get('schema')=='yug-production-source-registry-v1' and
         registry['identity']==expected_registry_identity and
         spec['identity']==expected_spec_identity and
         references.is_file() and references.is_relative_to(root) and
         file_digest(references)==expected_references_sha256 and
         output.is_relative_to(root/'artifacts') and
         not references.is_relative_to(output),
         'Registered review inputs or isolated output mismatch')
    matches=[s for s in registry['sources'] if s['source_id']==source_id]
    need(len(matches)==1,'Exact registered source required')
    source=matches[0]
    locator=source.get('locator')
    parsed=urlsplit(locator or '')
    parts=parsed.path.strip('/').split('/')
    need(source['status']=='REVIEW' and source.get('clearance_record') is None and
         source.get('file_manifest') is None and source['revision']==revision and
         re.fullmatch('[0-9a-f]{40}',revision) and
         parsed.scheme=='https' and parsed.netloc=='github.com' and
         not parsed.query and not parsed.fragment and len(parts)==2 and
         all(re.fullmatch('[A-Za-z0-9_.-]+',part) for part in parts) and
         source_id=='gh-'+'-'.join(parts).lower() and
         source['split_group'] not in holdout_groups(references,registry) and
         fetcher is not None,
         'Registered REVIEW identity, revision, protection, or fetcher invalid')
    need(_source_evidence(source,root) is not None,
         'Pinned candidate license metadata required')
    reference=source['license_evidence']
    evidence_path=(root/reference['path']).resolve()
    need(evidence_path.is_relative_to(root) and
         not evidence_path.is_relative_to(output) and
         all(key in reference for key in ('sha256','text_sha256','upstream_path')),
         'Candidate license evidence incomplete or output-local')
    context=dict(review_only=True,authority='NONE',training_allowed=False,
                 registry_identity=registry['identity'],
                 references_sha256=expected_references_sha256)
    return _prepare(source,spec,output,resume=resume,fetcher=fetcher,
                    review_context=context,review_license=evidence_path)


def prepare(source,spec,output,*,resume=False,retries=2,fetcher=fetch):
    need(classify(source)=='CACHE_READY','Source is not cache eligible')
    return _prepare(source,spec,output,resume=resume,retries=retries,fetcher=fetcher)


def _prepare(source,spec,output,*,resume=False,retries=2,fetcher=fetch,
             review_context=None,review_license=None):
    sid=source['source_id'];need(re.fullmatch('[A-Za-z0-9_.-]+',sid) and sid not in ('.','..'),'Unsafe source ID')
    directory=Path(output)/sid;directory.mkdir(parents=True,exist_ok=True)
    request_fields=dict(policy=POLICY,source=source,filters=spec['filters'],max_bytes=MAX_ITEM_BYTES)
    if review_context is not None:
        request_fields['review_context']=review_context
    request=seal(request_fields)
    with _writer_lock(directory/'lock'):
        request_path=directory/'request.json';receipt=directory/'complete.json';repo=directory/'git'
        if request_path.exists():
            need(resume,'Existing source cache: use --resume')
            need(json.loads(request_path.read_bytes())==request,'Cache source/revision/policy changed')
        else:
            need(not repo.exists(),'Unowned cache directory')
            atomic_json(request_path,request)
        if not repo.exists():git(None,'init','--bare','--template=',str(repo))
        config=repo/'config';config_receipt=directory/'git_config.json'
        if config_receipt.exists():need(file_digest(config)==json.loads(config_receipt.read_bytes())['sha256'],'Git cache configuration changed')
        else:atomic_json(config_receipt,dict(sha256=file_digest(config)))
        present=git(repo,'cat-file','-e',source['revision']+'^{commit}',check=False).returncode==0
        if receipt.exists():need(present,'Completed cache missing pinned commit')
        if not present:
            need(shutil.disk_usage(directory).free>=512*1024*1024,'Insufficient disk for source cache')
            for attempt in range(retries+1):
                try:fetcher(repo,source);break
                except (subprocess.SubprocessError,OSError):
                    if attempt==retries:raise
                    time.sleep(min(attempt+1,3))
        need(git(repo,'rev-parse',source['revision']+'^{commit}').stdout.decode().strip()==source['revision'],'Wrong revision fetched')
        git(repo,'fsck','--full','--strict','--no-reflogs')
        if receipt.exists():
            old=json.loads(receipt.read_bytes());verify_seal(old)
            need(old['request_identity']==request['identity'],'Cache receipt request mismatch')
            need(git(repo,'rev-parse','HEAD').stdout.decode().strip()==source['revision'],'Wrong checked-out revision')
            need(file_digest(directory/'inventory.jsonl')==old['inventory_sha256'],'Corrupt inventory')
            with (directory/'inventory.jsonl').open('rb') as rows:
                for line in rows:
                    row=json.loads(line)
                    if row['local_blob']:
                        blob=directory/row['local_blob']
                        need(blob.is_file() and blob.stat().st_size==row['bytes'] and file_digest(blob)==row['sha256'],'Corrupt/incomplete materialized cache')
            if review_license is not None:
                checked_source=dict(source,license_evidence=dict(
                    source['license_evidence'],path=str(review_license)))
                _verify_review_license(directory,checked_source)
            return seal(dict(old,reused=True))
        # Detached exact revision, not a branch. Files are raw content-addressed checkout blobs.
        git(repo,'update-ref','--no-deref','HEAD',source['revision'])
        counts=inventory(repo,source,spec,directory)
        if review_license is not None:
            checked_source=dict(source,license_evidence=dict(
                source['license_evidence'],path=str(review_license)))
            _verify_review_license(directory,checked_source)
        result=seal(dict(request_identity=request['identity'],revision=source['revision'],inventory_sha256=file_digest(directory/'inventory.jsonl'),counts=counts,training_allowed=False))
        atomic_json(receipt,result)
        return seal(dict(result,reused=False))


def run(registry,spec,output,*,dry_run=False,resume=False,limit=None,retries=2,fetcher=fetch,
        registered_registry=None,registered_source_ids=(),references=None,root=ROOT):
    verify_seal(registry);verify_seal(spec)
    need(limit is None or limit>0,'Source limit must be positive');need(0<=retries<=5,'Retries must be 0..5')
    registered_source_ids=set(registered_source_ids)
    if registered_source_ids:
        need(registered_registry is not None and references is not None,
             'Registered review requires original registry and protected references')
        verify_seal(registered_registry)
        original={s['source_id']:s for s in registered_registry['sources']}
        need(len(original)==len(registered_registry['sources']) and
             registered_source_ids <= set(original) and
             all(s['source_id'] not in registered_source_ids or
                 s==original[s['source_id']] for s in registry['sources']) and
             {s['source_id'] for s in registry['sources'] if s['source_id'] in registered_source_ids}
             == registered_source_ids,
             'Registered cache source differs from original policy')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];processed=0
    for source in sorted(registry['sources'],key=lambda s:s['source_id']):
        registered=source['source_id'] in registered_source_ids
        classification='REGISTERED_REVIEW' if registered else classify(source)
        row=dict(source_id=source['source_id'],classification=classification,status=classification)
        if classification in ('CACHE_READY','REGISTERED_REVIEW') and not dry_run and (limit is None or processed<limit):
            processed+=1
            try:
                if registered:
                    result=prepare_registered_review(registered_registry,source['source_id'],
                        source['revision'],spec,references,output,
                        expected_registry_identity=registered_registry['identity'],
                        expected_spec_identity=spec['identity'],
                        expected_references_sha256=file_digest(references),root=root,
                        resume=resume,fetcher=fetcher)
                else:
                    result=prepare(source,spec,output,resume=resume,retries=retries,fetcher=fetcher)
                row.update(status='CACHED',result=result)
            except Exception as exc:row.update(status='FAILED',error=type(exc).__name__+': '+str(exc))
        rows.append(row)
        atomic_json(output/'status.json',seal(dict(policy=POLICY,registry_identity=registry['identity'],spec_identity=spec['identity'],dry_run=dry_run,classification_counts=dict(Counter(r['classification'] for r in rows)),sources=rows,training_allowed=False)))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,default=ROOT/'training/specifications/source_registry_v1.json')
    p.add_argument('--spec',type=Path,default=ROOT/'training/specifications/production_corpus_v1.json')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/source_review_cache_v1')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--retries',type=int,default=2)
    args=p.parse_args()
    try:
        need(args.output.resolve().is_relative_to(ROOT/'artifacts'),'Review cache must stay under artifacts/')
        with _writer_lock(args.output/'batch_lock'):
            rows=run(json.loads(args.registry.read_bytes()),json.loads(args.spec.read_bytes()),args.output,dry_run=args.dry_run,resume=args.resume,limit=args.limit,retries=args.retries)
        print(json.dumps(dict(classifications=dict(Counter(r['classification'] for r in rows)),statuses=dict(Counter(r['status'] for r in rows))),sort_keys=True))
        return 1 if any(r['status']=='FAILED' for r in rows) else 0
    except KeyboardInterrupt:
        print('Interrupted; rerun with --resume. Nothing is cleared for training.');return 130
    except Exception as exc:
        print(f'Source preparation failed: {exc}');return 2

if __name__=='__main__':raise SystemExit(main())
