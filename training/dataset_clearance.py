"""Offline document-bound dataset clearance decisions; never creates grants."""
import argparse
from collections import Counter
import json
from pathlib import Path

from training.clearance_evidence import verify_bundle
from training.dataset_transport import build as verify_transport, payload_rows, selected_path, PILOT
from training.dataset_payload import payload_digest
from training.production_acquisition import read_plan
from training.production_corpus import atomic_json, atomic_replace, canonical, file_digest, identity
from training.production_spec import need, seal, verify_seal, local_path
from training.source_admission import cached_sources
from training.source_prepare import ROOT
from training.token_shards import _writer_lock

VERSION='dataset-document-clearance-v1'


def decision(record, source, raw_row, evidence, payload_sha, request_identity, selection=None):
    """Require both reconstructed origin and existing exact-file clearance."""
    p=record['provenance'];path=selected_path(source,selection)
    need(record['source_id']==source['source_id'] and record['locator']==source['locator']
         and record['revision']==source['revision'],'Document source mismatch')
    need(type(p['row_index']) is int and p['row_index']>=0,'Invalid row index')
    need(p['dataset_id']==source['dataset_id'] and p['revision']==source['revision']
         and p['payload_path']==path and p['payload_sha256']==payload_sha
         and p['license_evidence']==source['license_evidence']
         and p['serialization']=='canonical-json-row-v1','Document provenance mismatch')
    need(record['path']==f"{path}/row-{p['row_index']:08d}.json",'Document path mismatch')
    expected_split='unspecified' if selection is not None else ('train_sft' if 'train_sft' in path else 'train')
    need(p['upstream_split']==expected_split,'Upstream split mismatch')
    data=canonical(raw_row)
    need(record['bytes']==len(data) and record['sha256']==identity_bytes(data),'Row reconstruction differs')
    verify_seal(evidence)
    need(all(evidence[k]==record[k] for k in ('source_id','revision','path','sha256','bytes')),'Clearance record mismatch')
    statuses={'EVIDENCED':'CLEAR','EXCLUDED':'EXCLUDE','REVIEW':'REVIEW'}
    need(evidence['status'] in statuses,'Unknown existing clearance decision')
    status=statuses[evidence['status']]
    if status=='CLEAR':
        need(isinstance(evidence.get('scope'),dict) and evidence['scope'].get('clearance_identity') and evidence['license_id']==source['license_id']
             and evidence.get('license_evidence'),'CLEAR requires existing exact-file authority')
    document_identity=identity(dict(dataset_id=p['dataset_id'],revision=p['revision'],
        payload_path=path,payload_sha256=payload_sha,row_index=p['row_index'],sha256=record['sha256']))
    return seal(dict(policy=VERSION,request_identity=request_identity,document_identity=document_identity,
        source_id=record['source_id'],revision=record['revision'],path=record['path'],sha256=record['sha256'],
        bytes=record['bytes'],provenance=p,source_license_claim=source['license_id'],
        license_evidence=source['license_evidence'],existing_clearance_identity=evidence['identity'],
        status=status,reason=evidence['reason'],training_allowed=False,production_admitted=False))


def identity_bytes(data):
    import hashlib
    return 'sha256:'+hashlib.sha256(data).hexdigest()


def build(cache,catalog,spec,evidence,output,*,root=ROOT,resume=False):
    root=Path(root).resolve();cache=Path(cache).resolve();output=Path(output).resolve();evidence=Path(evidence).resolve()
    need(output.is_relative_to(root) and not output.is_relative_to(cache) and not cache.is_relative_to(output),'Unsafe output/cache overlap')
    inputs=[Path(catalog).resolve(),Path(spec).resolve(),evidence/'request.json',evidence/'complete.json',evidence/'evidence.jsonl',cache/'request.json',cache/'complete.json']
    need(all(not p.is_relative_to(output) for p in inputs),'Output overlaps input evidence')
    bundle=verify_bundle(evidence,root);need(bundle['complete'],'Existing clearance evidence incomplete')
    er=json.loads((evidence/'request.json').read_bytes())
    for p in inputs[:2]:need(er['inputs'].get(p.relative_to(root).as_posix())==file_digest(p),'Clearance input binding differs')
    transport_request=json.loads((cache/'request.json').read_bytes());verify_seal(transport_request)
    # Completed offline transport replay verifies pins, access/license evidence,
    # raw LFS hash, inventory and every cached payload, without network permission.
    verify_transport(catalog,spec,cache,root=root,limit=transport_request['limit'],
                     budget_mib=transport_request['budget_mib'],resume=True,fetch=False,selection=transport_request.get('selection'))
    reg=json.loads(Path(catalog).read_bytes());policy=json.loads(Path(spec).read_bytes())
    records={}
    for e in read_plan(evidence/'evidence.jsonl'):
        key=(e['source_id'],e['path']);need(key not in records,'Duplicate clearance evidence');records[key]=e
    request=seal(dict(policy=VERSION,inputs={p.relative_to(root).as_posix():file_digest(p) for p in inputs}))
    output.mkdir(parents=True,exist_ok=True)
    if (output/'request.json').exists():need(resume and json.loads((output/'request.json').read_bytes())==request,'Resume inputs changed')
    else:
        need(not any(output.iterdir()),'Unknown output');atomic_json(output/'request.json',request)
    counts=Counter();per_source={};seen=set();temp=output/'decisions.part'
    with temp.open('wb') as out:
        for directory,source,inventory,done in cached_sources(cache,reg,policy,root):
            n=sum(done['counts'].values());raw=directory/('payload_chunks' if (directory/'payload_chunks').exists() else 'payload.parquet');raw_sha=payload_digest(directory)
            original=iter(payload_rows(raw,n,selected_path(source,transport_request.get('selection'))));sc=Counter()
            for i,r in enumerate(read_plan(inventory)):
                need(r['provenance']['row_index']==i,'Missing/reordered row')
                original_row=next(original,None);need(original_row is not None,'Row absent from pinned payload')
                need(local_path(directory,r['local_blob']).read_bytes()==canonical(original_row),'Cached document differs from source row')
                key=(source['source_id'],r['path']);need(key in records,'Missing document clearance evidence')
                e=records.pop(key);need(e['cache_inventory_sha256']==done['inventory_sha256'],'Clearance inventory mismatch')
                result=decision(r,source,original_row,e,raw_sha,request['identity'],transport_request.get('selection'))
                need(result['document_identity'] not in seen,'Duplicate document identity');seen.add(result['document_identity'])
                out.write(canonical(result)+b'\n');counts[result['status']]+=1;sc[result['status']]+=1
            need(sum(sc.values())==n,'Incomplete inventory');per_source[source['dataset_id']]=dict(sc)
    need(not records,'Extra existing clearance rows')
    result=seal(dict(request_identity=request['identity'],counts={s:counts[s] for s in ('CLEAR','EXCLUDE','REVIEW')},
        per_source=per_source,documents=len(seen),decisions_sha256=file_digest(temp),production_admitted_documents=0))
    if (output/'complete.json').exists():
        old=json.loads((output/'complete.json').read_bytes());verify_seal(old)
        need(old==result and file_digest(output/'decisions.jsonl')==result['decisions_sha256'],'Clearance replay differs')
        temp.unlink()
    else:
        atomic_replace(temp,output/'decisions.jsonl');atomic_json(output/'complete.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','catalog','spec','evidence','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:
        need(a.output.resolve().is_relative_to(ROOT/'artifacts'),'Output must stay under artifacts')
        with _writer_lock(a.output.with_name(a.output.name+'.clearance')):result=build(**vars(a))
        print(json.dumps(result,sort_keys=True));return 0
    except Exception as exc:print(f'Dataset clearance stopped: {exc}');return 2


if __name__=='__main__':raise SystemExit(main())
