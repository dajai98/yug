"""Bounded pinned HF Parquet review caches. No production admission."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import urllib.request
from urllib.parse import urlsplit

from training.production_corpus import atomic_json, atomic_replace, canonical, file_digest
from training.production_spec import need, seal, verify_seal, local_path
from training.source_prepare import ROOT, LICENSES
from training.token_shards import _writer_lock
from training.dataset_payload import download as download_chunks, payload_digest, ChunkReader, CHUNK_BYTES, MAX_PAYLOAD
from training.conditional_license import CONDITIONAL

VERSION='pinned-dataset-review-cache-v1'
MAX_DOC=1048576
PILOT={'HuggingFaceH4/ultrachat_200k':'data/train_sft-00000-of-00003-a3ecf92756993583.parquet',
       'openai/gsm8k':'main/train-00000-of-00001.parquet'}


class Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        u=urlsplit(newurl)
        need(u.scheme=='https' and not u.username and not u.password and u.port in (None,443)
             and (u.hostname in ('huggingface.co','us.aws.cdn.hf.co') or (u.hostname or '').endswith('.xethub.hf.co')),
             'Unapproved payload redirect')
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def download(url,path,size,sha,*,opener=None):
    """Resume byte ranges; hash the complete file before atomic publication."""
    need(re.fullmatch(r'https://huggingface\.co/datasets/[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+/resolve/[0-9a-f]{40}/[A-Za-z0-9_./-]+\.(?:parquet|jsonl)',url), 'Unpinned payload URL')
    need('..' not in urlsplit(url).path.split('/'),'Unsafe payload path')
    if path.exists():
        need(path.stat().st_size==size and file_digest(path)=='sha256:'+sha,'Cached payload corrupt')
        return
    temp=path.with_suffix('.part'); offset=temp.stat().st_size if temp.exists() else 0
    need(offset<=size,'Partial payload exceeds size')
    if offset<size:
        headers={'Accept-Encoding':'identity','User-Agent':VERSION}
        if offset: headers['Range']=f'bytes={offset}-'
        open_url=opener or urllib.request.build_opener(Redirect).open
        with open_url(urllib.request.Request(url,headers=headers),timeout=60) as response:
            if response.status==206:
                need(response.headers.get('Content-Range')==f'bytes {offset}-{size-1}/{size}','Invalid resume range')
            else:
                need(response.status==200,'Unexpected download status'); offset=0
            with temp.open('ab' if offset else 'wb') as out:
                while block:=response.read(min(65536,size-offset+1)):
                    offset+=len(block); need(offset<=size,'Payload exceeds pinned size'); out.write(block)
    need(temp.stat().st_size==size,'Incomplete download; resume required')
    if file_digest(temp)!='sha256:'+sha:
        temp.unlink()  # Only our exact temporary payload, never recursive deletion.
        raise ValueError('Payload hash mismatch; partial discarded')
    atomic_replace(temp,path)


def tree_metadata(source,path):
    from training.source_discovery import NoRedirect
    parent=PurePosixPath(path).parent.as_posix()
    parent='' if parent=='.' else '/'+parent
    url=f"https://huggingface.co/api/datasets/{source['dataset_id']}/tree/{source['revision']}{parent}"
    with urllib.request.build_opener(NoRedirect).open(url,timeout=30) as response:
        data=response.read(MAX_DOC+1)
    need(len(data)<=MAX_DOC,'Tree metadata too large')
    return dict(url=url,body=json.loads(data))


def parquet_rows(path,limit):
    import pyarrow.parquet as pq
    parquet=pq.ParquetFile(path)
    # Bounded records/batches; Arrow may decode compressed pages larger than a row.
    count=0
    for batch in parquet.iter_batches(batch_size=1,use_threads=False):
        for row in batch.to_pylist():
            if count>=limit:return
            yield row;count+=1
        if count>=limit:return


def payload_rows(path, limit, payload_path):
    if payload_path.endswith('.parquet'):
        yield from parquet_rows(path, limit)
    else:
        with (io.BufferedReader(ChunkReader(path)) if path.is_dir() else path.open('rb')) as stream:
            for _ in range(limit):
                line=stream.readline(MAX_DOC+1)
                if not line:return
                need(len(line)<=MAX_DOC,'JSONL row exceeds bound')
                row=json.loads(line)
                need(isinstance(row,dict),'JSONL record must be an object')
                yield row


def selected_path(source, selection=None):
    if selection is None:
        need(source['dataset_id'] in PILOT,'Unsupported dataset')
        return PILOT[source['dataset_id']]
    entry=selection.get(source['dataset_id'])
    need(isinstance(entry,dict) and set(entry)=={'revision','path'},'Invalid payload selection')
    need(entry['revision']==source['revision'],'Selection revision mismatch')
    path=entry['path']
    need(isinstance(path,str) and re.fullmatch(r'[A-Za-z0-9_./-]+\.(?:parquet|jsonl)',path)
         and not path.startswith('/') and all(p not in ('','.','..') for p in path.split('/')),'Unsafe selection path')
    return path


def build(catalog,spec,output,*,root=ROOT,limit=32,budget_mib=384,resume=False,
          fetch=False,metadata_fetch=tree_metadata,payload_fetch=download,rows=None,selection=None):
    root=Path(root).resolve(); output=Path(output).resolve()
    need(output.is_relative_to(root),'Output escapes root')
    need(all(not Path(p).resolve().is_relative_to(output) for p in (catalog,spec)),'Output overlaps inputs')
    need(type(limit) is int and 1<=limit<=1000 and type(budget_mib) is int and 1<=budget_mib<=1024,'Invalid bounds')
    reg=json.loads(Path(catalog).read_bytes()); policy=json.loads(Path(spec).read_bytes())
    verify_seal(reg);verify_seal(policy)
    sources=sorted(reg['sources'],key=lambda s:s['source_id'])
    if isinstance(selection,Path):selection=json.loads(selection.read_bytes())
    if selection is not None:need(isinstance(selection,dict) and set(selection)=={s['dataset_id'] for s in sources},'Selection source scope mismatch')
    need(0<len(sources)<=2 and len({s['source_id'] for s in sources})==len(sources),'Unexpected source scope')
    for s in sources:
        did=s.get('dataset_id'); path=selected_path(s,selection)
        need(s['locator']=='https://huggingface.co/datasets/'+did and re.fullmatch('[0-9a-f]{40}',s['revision']),'Source/pin mismatch')
        need(s['license_id'] in LICENSES | CONDITIONAL.keys() and s['status']=='REVIEW' and not s['allow_paths'] and not s['training_allowed_after_clearance'],'Not an unapproved metadata source')
        e=s['license_evidence']; ep=local_path(root,e['path'])
        need(file_digest(ep)==e['sha256'],'License evidence corrupt')
        evidence=json.loads(ep.read_bytes());verify_seal(evidence)
        b=evidence['body']; need(b['id']==did and b['sha']==s['revision'],'License source identity mismatch')
        if selection is not None:need(path in [x.get('rfilename') for x in b.get('siblings',[])],'Selected path absent from pinned metadata')
        need(evidence['url']==f"https://huggingface.co/api/datasets/{did}/revision/{s['revision']}",'License URL mismatch')
        declared=b.get('cardData',{}).get('license')
        if isinstance(declared,list) and len(declared)==1:declared=declared[0]
        need(isinstance(declared,str) and declared.casefold()==s['license_id'].casefold() and b.get('private') is False and b.get('gated') is False,'License/access mismatch')
    request_data=dict(version=VERSION,catalog=file_digest(catalog),spec=file_digest(spec),limit=limit,budget_mib=budget_mib,paths=PILOT)
    if selection is not None:request_data['selection']=selection
    request=seal(request_data)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'request.json').exists():
        need(resume and json.loads((output/'request.json').read_bytes())==request,'Resume inputs changed')
    else:
        need(not any(output.iterdir()),'Unknown output');atomic_json(output/'request.json',request)
    total=0;plans=[]
    for source in sources:
        directory=output/source['source_id'];directory.mkdir(exist_ok=True)
        path=selected_path(source,selection); meta=directory/'payload_metadata.json'
        if not meta.exists():
            need(fetch,'Missing metadata; --fetch required');atomic_json(meta,seal(metadata_fetch(source,path)))
        m=json.loads(meta.read_bytes());verify_seal(m)
        parent=PurePosixPath(path).parent.as_posix()
        expected=f"https://huggingface.co/api/datasets/{source['dataset_id']}/tree/{source['revision']}"+('' if parent=='.' else '/'+parent)
        need(m['url']==expected,'Tree source/revision mismatch')
        matches=[x for x in m['body'] if x.get('path')==path];need(len(matches)==1,'Missing/ambiguous payload')
        f=matches[0]; lfs=f.get('lfs',{})
        need(f.get('type')=='file' and type(f.get('size')) is int and 0<f['size']<=MAX_PAYLOAD,'Payload bound exceeded')
        need(f['size']<=256*1024*1024 or path.endswith('.jsonl'),'Oversized payload requires chunked JSONL transport')
        need(lfs.get('size')==f['size'] and isinstance(lfs.get('oid'),str) and re.fullmatch('[0-9a-f]{64}',lfs['oid']),'Missing pinned LFS SHA256')
        total+=f['size']; plans.append((source,directory,path,f))
    # Reserve raw files + per-document maximum + conservative manifest workspace.
    required=total+len(sources)*limit*(MAX_DOC+8192)+8*MAX_DOC
    if any(f['size']>256*1024*1024 for _,_,_,f in plans):required+=CHUNK_BYTES
    need(required<=budget_mib*1024*1024,'Configured storage budget insufficient')
    used=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    need(used<=budget_mib*1024*1024 and shutil.disk_usage(output).free>=required,'Insufficient storage')
    documents=0
    for source,directory,path,f in plans:
        chunked=f['size']>256*1024*1024
        raw=directory/('payload_chunks' if chunked else 'payload.parquet')
        url=f"{source['locator']}/resolve/{source['revision']}/{path}"
        if chunked:
            if fetch:download_chunks(url,raw,f['size'],f['lfs']['oid'])
            from training.dataset_payload import verify
            receipt,_=verify(raw)
            need((receipt['url'],receipt['size'],receipt['sha256'])==(url,f['size'],f['lfs']['oid']),'Chunk source binding mismatch')
        else:
            if not raw.exists():
                need(fetch,'Missing payload; --fetch required')
                payload_fetch(url,raw,f['size'],f['lfs']['oid'])
            need(raw.stat().st_size==f['size'],'Payload size failure')
        need(payload_digest(directory)=='sha256:'+f['lfs']['oid'],'Payload integrity failure')
        req=seal(dict(source=source,filters=policy['filters'],transport_request=request['identity'],payload_sha256=payload_digest(directory)))
        if (directory/'request.json').exists():need(json.loads((directory/'request.json').read_bytes())==req,'Cache request changed')
        else:atomic_json(directory/'request.json',req)
        (directory/'files').mkdir(exist_ok=True); inventory=directory/'inventory.jsonl';complete=directory/'complete.json'
        if complete.exists():
            done=json.loads(complete.read_bytes());verify_seal(done)
            need(done['request_identity']==req['identity'] and file_digest(inventory)==done['inventory_sha256'],'Inventory corrupt')
            records=[json.loads(line) for line in inventory.read_bytes().splitlines()]
            need(len(records)==done['counts']['UNRESOLVED'],'Inventory count mismatch')
            for r in records:need(file_digest(local_path(directory,r['local_blob']))==r['sha256'],'Document corrupt')
            documents+=len(records);continue
        records=[]
        for i,row in enumerate(rows(raw,limit) if rows else payload_rows(raw,limit,path)):
            need(i<limit,'Extractor exceeded bound')
            data=canonical(row);need(0<len(data)<=MAX_DOC,'Document exceeds bound; no silent truncation')
            name=f'files/{i:08d}.json';target=directory/name
            if target.exists():need(target.read_bytes()==data,'Partial document differs')
            else:target.write_bytes(data)
            records.append(dict(source_id=source['source_id'],locator=source['locator'],revision=source['revision'],
                path=f'{path}/row-{i:08d}.json',bytes=len(data),sha256=file_digest(target),
                git_object=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest(),git_mode='100644',local_blob=name,
                category=source['category'],subcategory=source['subcategory'],screening='UNRESOLVED',training_allowed=False,
                provenance=dict(dataset_id=source['dataset_id'],revision=source['revision'],payload_path=path,
                    payload_sha256=req['payload_sha256'],row_index=i,upstream_split=('unspecified' if selection is not None else ('train_sft' if 'train_sft' in path else 'train')),
                    license_evidence=source['license_evidence'],serialization='canonical-json-row-v1')))
        temp=directory/'inventory.part';temp.write_bytes(b''.join(canonical(r)+b'\n' for r in records));atomic_replace(temp,inventory)
        done=seal(dict(request_identity=req['identity'],revision=source['revision'],inventory_sha256=file_digest(inventory),counts={'UNRESOLVED':len(records)}))
        atomic_json(complete,done);documents+=len(records)
    result=seal(dict(request_identity=request['identity'],documents=documents,payload_bytes=total,
        inventory_hashes={s['source_id']:file_digest(d/'inventory.jsonl') for s,d,_,_ in plans},
        production_admitted_documents=0,training_allowed=False))
    if (output/'complete.json').exists():need(json.loads((output/'complete.json').read_bytes())==result,'Replay differs')
    else:atomic_json(output/'complete.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--spec',type=Path,default=ROOT/'training/specifications/production_corpus_v1.json')
    p.add_argument('--limit',type=int,default=32);p.add_argument('--budget-mib',type=int,default=384)
    p.add_argument('--fetch',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--selection',type=Path)
    a=p.parse_args()
    try:
        need(a.output.resolve().is_relative_to(ROOT/'artifacts'),'Output must stay under artifacts')
        with _writer_lock(a.output.with_name(a.output.name+'.transport')): result=build(**vars(a))
        print(json.dumps(result,sort_keys=True));return 0
    except Exception as exc: print(f'Dataset transport stopped: {exc}');return 2


if __name__=='__main__':raise SystemExit(main())
