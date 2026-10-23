"""Bounded HF dataset metadata discovery; no payload transport or clearance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import urllib.request

from training.production_corpus import atomic_json, canonical, file_digest
from training.production_spec import need, seal, verify_seal
from training.source_prepare import ROOT, LICENSES
from training.source_discovery import NoRedirect
from training.token_shards import _writer_lock
from training.conditional_license import CONDITIONAL, pending

VERSION = 'hf-dataset-metadata-v1'
MAX_BYTES = 524288
ID = r'[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+'
URL = rf'https://huggingface\.co/api/datasets/{ID}/revision/(?:main|[0-9a-f]{{40}})'


def fetch_json(url):
    need(re.fullmatch(URL, url), 'Only dataset metadata endpoints permitted')
    request = urllib.request.Request(url, headers={'User-Agent': VERSION, 'Accept': 'application/json'})
    with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
        need(response.status == 200 and response.geturl() == url, 'Unexpected metadata response')
        data = response.read(MAX_BYTES + 1)
    need(len(data) <= MAX_BYTES, 'Metadata exceeds bound')
    return json.loads(data)


def response(path, url, fetcher):
    need(re.fullmatch(URL, url), 'Invalid metadata URL')
    if path.exists():
        retained = json.loads(path.read_bytes()); verify_seal(retained)
        need(retained['url'] == url, 'Response binding changed')
        body = retained['body']
    else:
        need(fetcher is not None, 'Metadata missing; use --fetch explicitly')
        body = fetcher(url)
        need(isinstance(body, dict) and len(canonical(body)) <= MAX_BYTES, 'Invalid/oversized metadata')
        atomic_json(path, seal(dict(url=url, body=body)))
    need(isinstance(body, dict) and len(canonical(body)) <= MAX_BYTES, 'Invalid/oversized metadata')
    return body


def build(registry, spec, seeds, output, *, root=ROOT, resume=False, fetcher=None):
    root=Path(root).resolve(); output=Path(output).resolve()
    need(output.is_relative_to(root), 'Output escapes repository')
    need(all(not Path(p).resolve().is_relative_to(output) for p in (registry,spec)), 'Output overlaps input')
    catalog=json.loads(Path(registry).read_bytes()); policy=json.loads(Path(spec).read_bytes())
    verify_seal(catalog); verify_seal(policy)
    need(catalog['schema']=='yug-production-source-registry-v1','Unknown registry schema')
    need(isinstance(seeds,list) and 0<len(seeds)<=10,'Require 1..10 explicit dataset seeds')
    categories={c['id']:c['subcategories'] for c in policy['categories']}
    seen=set()
    for s in seeds:
        need(set(s)=={'dataset_id','category','subcategory'}, 'Unknown seed fields')
        need(isinstance(s['dataset_id'],str) and re.fullmatch(ID,s['dataset_id']), 'Invalid dataset ID')
        need(s['dataset_id'].casefold() not in seen,'Duplicate dataset'); seen.add(s['dataset_id'].casefold())
        need(s['subcategory'] in categories.get(s['category'],[]),'Invalid category route')
    seeds=sorted(seeds,key=lambda s:s['dataset_id'])
    request=seal(dict(version=VERSION,seeds=seeds,registry=file_digest(registry),spec=file_digest(spec),licenses=sorted(LICENSES)))
    output.mkdir(parents=True,exist_ok=True)
    if (output/'request.json').exists():
        need(resume and json.loads((output/'request.json').read_bytes())==request,'Resume inputs differ')
    else:
        need(not any(output.iterdir()),'Unknown output directory'); atomic_json(output/'request.json',request)
    prior=None
    if (output/'complete.json').exists():
        prior=json.loads((output/'complete.json').read_bytes()); verify_seal(prior)
        need(prior['request_identity']==request['identity'],'Receipt mismatch')
        for name,digest in prior['artifacts'].items():
            p=(output/name).resolve()
            need(p.is_relative_to(output) and file_digest(p)==digest,'Retained artifact corrupt')
    decisions=[]; sources=[]
    known={(s.get('locator') or '').rstrip('/').casefold() for s in catalog['sources']}
    aliases={s.casefold():s for s in LICENSES | CONDITIONAL.keys()}
    for i,seed in enumerate(seeds):
        did=seed['dataset_id']; locator='https://huggingface.co/datasets/'+did
        if locator.casefold() in known:
            decisions.append(dict(seed,status='REJECTED',reason='already registered dataset')); continue
        directory=output/f'source-{i:02d}'; directory.mkdir(exist_ok=True)
        endpoint='https://huggingface.co/api/datasets/'+did+'/revision/'
        head=response(directory/'head.json',endpoint+'main',None if prior else fetcher)
        revision=head.get('sha')
        need(head.get('id')==did and isinstance(revision,str) and re.fullmatch('[0-9a-f]{40}',revision),'Dataset identity/pin mismatch')
        pinned=response(directory/'metadata.json',endpoint+revision,None if prior else fetcher)
        need(pinned.get('id')==did and pinned.get('sha')==revision,'Pinned dataset identity mismatch')
        card=pinned.get('cardData')
        declared=card.get('license') if isinstance(card,dict) else None
        if isinstance(declared,list) and len(declared)==1: declared=declared[0]
        license_id=aliases.get(declared.casefold()) if isinstance(declared,str) else None
        tags=pinned.get('tags',[])
        tags=tags if isinstance(tags,list) else []
        tag_licenses=[t[8:] for t in tags if isinstance(t,str) and t.startswith('license:')]
        status,reason='REVIEW','missing/ambiguous license declaration'
        if pinned.get('private') is not False or pinned.get('gated') is not False:
            reason='private/gated/unknown access conditions'
        elif any(not isinstance(declared,str) or t.casefold()!=declared.casefold() for t in tag_licenses):
            reason='conflicting license metadata'
        elif license_id in CONDITIONAL:
            status,reason='REVIEW','CONDITIONAL license; explicit source-specific approval and obligations required'
        elif license_id:
            status,reason='ACCEPTED','allowlisted metadata declaration only; document clearance still required'
        elif isinstance(declared,str) and declared.casefold() not in ('','unknown','other','noassertion'):
            status,reason='REJECTED','license outside existing allowlist'
        evidence=dict(path=(directory/'metadata.json').relative_to(root).as_posix(),sha256=file_digest(directory/'metadata.json'),
                      url=endpoint+revision,field='cardData.license',scope='dataset-level declaration only; not document clearance')
        decisions.append(dict(seed,locator=locator,revision=revision,license_id=license_id,declared_license=declared,
                              license_evidence=evidence,status=status,reason=reason))
        if status=='ACCEPTED' or (license_id in CONDITIONAL and reason.startswith('CONDITIONAL')):
            sid='hf-dataset-'+hashlib.sha256(did.encode()).hexdigest()[:24]
            sources.append(dict(source_id=sid,name=did,dataset_id=did,locator=locator,revision=revision,
                category=seed['category'],subcategory=seed['subcategory'],license_id=license_id,license_evidence=evidence,
                status='REVIEW',status_authority=VERSION,status_reason=reason,allowed_categories=[seed['category']],
                target_cap_tokens=0,category_caps_tokens={seed['category']:0},cap_basis='No production scope authorized',
                repository_group=sid,domain_group=sid,split_group=sid,split='unassigned',allow_paths=[],
                deny_paths=policy['filters']['deny_paths'],file_manifest=None,clearance_record=None,origin_verified=False,
                training_allowed_after_clearance=False,provenance=dict(discovery_version=VERSION,metadata_only=True)))
            if license_id in CONDITIONAL:sources[-1]['conditional_license']=pending(license_id)
    outputs={'catalog.json':seal(dict(catalog,sources=sources)),
             'decisions.json':seal(dict(version=VERSION,sources=decisions,training_allowed=False))}
    for name,value in outputs.items():
        if prior: need(json.loads((output/name).read_bytes())==value,'Deterministic replay mismatch')
        else: atomic_json(output/name,value)
    counts=Counter(d['status'] for d in decisions)
    result=seal(dict(request_identity=request['identity'],candidate_count=len(seeds),
        counts={k:counts[k] for k in ('ACCEPTED','REVIEW','REJECTED')},payload_download_bytes=0,
        production_cleared_files=0,artifacts={p.relative_to(output).as_posix():file_digest(p)
        for p in sorted(output.rglob('*.json')) if p.name!='complete.json'}))
    if prior: need(result==prior,'Receipt replay mismatch')
    else: atomic_json(output/'complete.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,default=ROOT/'training/specifications/source_registry_v1.json')
    p.add_argument('--spec',type=Path,default=ROOT/'training/specifications/production_corpus_v1.json')
    p.add_argument('--seeds',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fetch',action='store_true'); p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    try:
        need(a.output.resolve().is_relative_to(ROOT/'artifacts'),'Output must stay under artifacts')
        need(not a.seeds.resolve().is_relative_to(a.output.resolve()),'Output overlaps seeds')
        with _writer_lock(a.output.with_name(a.output.name+'.discovery')):
            result=build(a.registry,a.spec,json.loads(a.seeds.read_bytes()),a.output,resume=a.resume,fetcher=fetch_json if a.fetch else None)
        print(json.dumps(result,sort_keys=True)); return 0
    except Exception as exc:
        print(f'Dataset discovery stopped: {type(exc).__name__}: {exc}'); return 2


if __name__=='__main__': raise SystemExit(main())
