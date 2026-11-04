"""Materialize one reviewed, exact-file revision supersession without admission.

The decision file is an external governance input. A prior grant supplies lineage
and a conservative cap/split, never clearance for the new revision. This compiler
does no network access, tokenization, or production accounting.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from training.category_ownership import decision as ownership_decision
from training.clearance_evidence import verify_bundle
from training.experiments.representative_mini_corpus import normalize_structured_text
from training.production_acquisition import read_plan, validate_clearance, item_key
from training.production_corpus import atomic_json, canonical, file_digest
from training.production_spec import need, seal, verify_seal, validate, local_path, path_allowed
from training.source_admission import cached_sources, holdout_groups
from training.source_approval import package_scope, file_reason, payload
from training.source_prepare import ROOT, screening

POLICY = 'reviewed-revision-supersession-v1'
SOURCE = 'gh-python-attrs-attrs'
REVISION = 'c8bb9bfaa2843e7844b5bb13bed4370b51feec4f'
OLD_REVISION = '644b4e165bfbeee7e127de6fcbda08b64014316f'
LOCATOR = 'https://github.com/python-attrs/attrs'
OLD_REGISTRY_SHA256 = 'sha256:335aeeed7f3a6d1bd08832039e763abacbdf8ba74f7363210f7ec204372ebf68'


def verify_decision_scope(d, old_files, candidate_rows):
    """Check the reviewed file set, including independent coverage of changed bytes."""
    need(d.get('schema')==POLICY and d.get('source_id')==SOURCE and
         d.get('locator')==LOCATOR and d.get('revision')==REVISION and
         d.get('prior_revision')==OLD_REVISION and d.get('owner')=='software_engineering' and
         d.get('subcategory')=='curated_subset' and d.get('license_id')=='MIT' and
         d.get('decision')=='APPROVE_EXACT_FILES' and d.get('reviewer') and
         d.get('attribution') and d.get('notice_obligation')=='preserve_required_mit_attribution',
         'Incomplete or wrong revision-supersession decision')
    approved=d.get('files')
    need(type(approved) is list and len(approved)==16 and
         approved==sorted(approved,key=lambda x:x['path']) and
         all(type(x) is dict and set(x)=={'path','sha256'} for x in approved) and
         len({x['path'] for x in approved})==16,'Decision must pin exactly 16 unique files')
    prior={f['path']:f for f in old_files}
    need(len(prior)==16 and set(prior)=={x['path'] for x in approved} and
         len(candidate_rows)==16 and set(candidate_rows)==set(prior),
         'Supersession scope differs from reviewed historical scope')
    changed={x['path'] for x in approved if x['sha256']!=prior[x['path']]['sha256']}
    need(changed=={'src/attr/_funcs.py'},'Unexpected revision content change')
    for x in approved:
        candidate=candidate_rows[x['path']]
        need(candidate.get('source_id')==SOURCE and candidate.get('revision')==REVISION and
             candidate.get('path')==x['path'] and candidate.get('sha256')==x['sha256'] and
             candidate.get('owner_claim')==d['owner'] and
             candidate.get('subcategory_claim')==d['subcategory'] and
             candidate.get('license_id')==d['license_id'],
             'Candidate source/revision/path/content/ownership mismatch')
    return changed


def _read(path):
    value = json.loads(Path(path).read_bytes())
    verify_seal(value)
    return value


def build(decision, output, *, root=ROOT,
          cache='artifacts/phase3_github_prepare_batch01_v1',
          catalog='artifacts/phase3_github_discovery_batch01_v3/catalog.json',
          proposals='artifacts/phase3_github_grant_proposals_batch01_v1',
          review='artifacts/phase3_github_clearance_batch01_v1',
          references='artifacts/phase3_protected_references_v1/references.jsonl',
          old_clearance='artifacts/source_approval_v1/gh-python-attrs-attrs-clearance.json',
          old_manifest='artifacts/source_approval_v1/gh-python-attrs-attrs-files.json',
          old_registry='artifacts/source_approval_v1/registry.json',
          base_registry='training/specifications/source_registry_v1.json',
          base_spec='training/specifications/production_corpus_v1.json',
          expected_decision_sha256=None):
    root = Path(root).resolve(); output = Path(output).resolve()
    resolve = lambda p: local_path(root, str(p)) if not Path(p).is_absolute() else Path(p).resolve()
    cache, catalog, proposals, review, references = map(resolve, (cache,catalog,proposals,review,references))
    old_clearance, old_manifest, old_registry, base_registry, base_spec, decision = map(
        resolve,(old_clearance,old_manifest,old_registry,base_registry,base_spec,decision))
    inputs = [decision,catalog,references,old_clearance,old_manifest,old_registry,base_registry,base_spec,
              proposals/'request.json',proposals/'complete.json',proposals/'proposals.jsonl',
              review/'request.json',review/'complete.json',review/'evidence.jsonl',cache/'status.json']
    need(output.is_relative_to(root) and not output.is_relative_to(cache) and
         all(not p.is_relative_to(output) for p in inputs),'Output overlaps input or escapes repository')
    need(expected_decision_sha256 is not None and file_digest(decision)==expected_decision_sha256,
         'Externally pinned governance decision missing or changed')
    d=_read(decision)
    approved=d['files']
    need(d.get('prior_clearance_sha256')==file_digest(old_clearance) and
         d.get('prior_manifest_sha256')==file_digest(old_manifest),
         'Historical lineage changed')
    need(file_digest(old_registry)==OLD_REGISTRY_SHA256,
         'Historical registry cap/split evidence changed')
    old=_read(old_clearance);old_files=_read(old_manifest);old_reg=_read(old_registry)
    need(old['source_id']==SOURCE and old['revision']==OLD_REVISION and
         old['file_manifest_identity']==old_files['identity'] and
         old_files['source_id']==SOURCE and old_files['revision']==OLD_REVISION,
         'Historical source clearance/manifest mismatch')
    old_source=next(s for s in old_reg['sources'] if s['source_id']==SOURCE)
    need(old_source['revision']==OLD_REVISION and old_source['locator']==LOCATOR and
         old_source['split']=='train' and old_source['target_cap_tokens']>0,
         'Historical source lineage/cap/split mismatch')
    reg=_read(base_registry);spec=_read(base_spec);validate(spec,reg,root)
    need(not any(s['source_id']==SOURCE or s.get('locator')==LOCATOR for s in reg['sources']),
         'Current registry already contains this source/locator')
    cat=_read(catalog);source=next(s for s in cat['sources'] if s['source_id']==SOURCE)
    need(source['locator']==LOCATOR and source['revision']==REVISION and
         source['category']==d['owner'] and source['subcategory']==d['subcategory'] and
         source['license_id']=='MIT' and source['status']=='ACCEPT',
         'Discovery source does not match decision')
    need(file_digest(local_path(root,source['license_evidence']['path']))==source['license_evidence']['sha256'],
         'Pinned license evidence changed')
    reviewed=verify_bundle(review,root)
    need(reviewed['complete'] and reviewed['counts']['EVIDENCED']==0,
         'Expected complete review-only evidence')
    pr=_read(proposals/'request.json');pc=_read(proposals/'complete.json')
    need(pc['request_identity']==pr['identity'] and pc['file_scope_candidates']==16 and
         file_digest(proposals/'proposals.jsonl')==pc['proposals_sha256'],
         'Proposal bundle incomplete or changed')
    for name,sha in pr['inputs'].items():
        need(file_digest(local_path(root,name))==sha,'Proposal input changed: '+name)
    cache_rows=list(cached_sources(cache,cat,spec,root))
    matched=[x for x in cache_rows if x[1]['source_id']==SOURCE]
    need(len(matched)==1,'Missing/ambiguous attrs cache')
    directory,_,inventory,done=matched[0]
    rows={r['path']:r for r in read_plan(inventory)}
    need(len(rows)==sum(done['counts'].values()),'Duplicate/incomplete cache inventory')
    scope,reason=package_scope(source,rows,directory,root)
    need(scope is not None and scope['packages']==['src/attr','src/attrs'],
         'Pinned MIT package scope unavailable: '+str(reason))
    proposal_rows=[r for r in read_plan(proposals/'proposals.jsonl')
                   if r['source_id']==SOURCE and r.get('file_scope_profile')]
    proposal={r['path']:r for r in proposal_rows}
    need(len(proposal_rows)==16 and len(proposal)==16,'Duplicate/extra proposal candidates')
    changed=verify_decision_scope(d,old_files['files'],proposal)
    group_conflicts=holdout_groups(references,reg)|holdout_groups(references,cat)
    need(source['split_group'] not in group_conflicts,'Retained holdout source-group conflict')
    files=[];plan=[]; evidence=[]
    evidence.extend(scope['evidence'])
    for p in (decision,inventory,proposals/'complete.json',proposals/'proposals.jsonl'):
        evidence.append(dict(path=p.relative_to(root).as_posix(),sha256=file_digest(p)))
    # Content/revision-specific ownership evidence is separate from old structural decisions.
    for approved_file in approved:
        path=approved_file['path'];row=rows.get(path);p=proposal[path]
        need(row is not None and row['revision']==REVISION and row['source_id']==SOURCE and
             row['locator']==LOCATOR and row['sha256']==approved_file['sha256'] and
             p['sha256']==row['sha256'] and p['revision']==REVISION and
             p['owner_claim']==d['owner'] and p['subcategory_claim']==d['subcategory'] and
             p['file_scope_profile']=='package' and p['grantable'] is False,
             'Candidate revision/path/hash/owner mismatch')
        raw=payload(directory,row)
        screened,_=screening(path,row['git_mode'],row['bytes'],raw,source,spec)
        need(screened!='EXCLUDED' and file_reason(row,raw,source,scope,rows) is None and
             path_allowed(path,[path],source['deny_paths']),'File rights/path/quality evidence failed')
        normalized='sha256:'+hashlib.sha256(normalize_structured_text(raw.decode('utf-8-sig')).encode()).hexdigest()
        own=seal(dict(source_id=SOURCE,revision=REVISION,path=path,sha256=row['sha256'],
                      owner=d['owner'],subcategory=d['subcategory'],decision_type='manual_review',
                      rationale_ref=d['identity'],reviewer=d['reviewer']))
        own_path=output/'ownership'/ (hashlib.sha256(path.encode()).hexdigest()+'.json')
        files.append(dict(path=path,sha256=row['sha256'],bytes=row['bytes'],owner=d['owner'],
                          subcategory=d['subcategory'],content_type='code',license_id='MIT',tags=[],
                          ownership_decision=ownership_decision(source_id=SOURCE,revision=REVISION,
                              path=path,sha256=row['sha256'],owner=d['owner'],subcategory=d['subcategory'],
                              decision_type='manual_review',rationale_ref=d['identity'],normalized_sha256=normalized,
                              evidence_ref=dict(path=own_path.relative_to(root).as_posix(),sha256='PENDING'))))
        plan.append((row,own_path,own))
    # Publish only after all immutable upstream checks pass; on replay compare byte-for-byte.
    output.mkdir(parents=True,exist_ok=True);(output/'ownership').mkdir(exist_ok=True)
    def publish(path,value):
        encoded=canonical(value)
        if path.exists(): need(path.read_bytes()==encoded,'Supersession replay differs: '+str(path))
        else: atomic_json(path,value)
    for _,own_path,own in plan: publish(own_path,own)
    for f,(_,own_path,_) in zip(files,plan):
        f['ownership_decision']=ownership_decision(source_id=SOURCE,revision=REVISION,path=f['path'],
            sha256=f['sha256'],owner=d['owner'],subcategory=d['subcategory'],decision_type='manual_review',
            rationale_ref=d['identity'],normalized_sha256=f['ownership_decision']['normalized_sha256'],
            evidence_ref=dict(path=own_path.relative_to(root).as_posix(),sha256=file_digest(own_path)))
    manifest=seal(dict(source_id=SOURCE,revision=REVISION,files=files));publish(output/'files.json',manifest)
    attribution=d['attribution']
    clearance=seal(dict(source_id=SOURCE,revision=REVISION,file_manifest_identity=manifest['identity'],
        decision='ACCEPT',evidence=evidence,attribution=attribution,
        governance_decision_identity=d['identity'],prior_clearance_identity=old['identity']))
    publish(output/'clearance.json',clearance)
    cap=min(old_source['target_cap_tokens'],sum(r['bytes']+1 for r,_,_ in plan))
    s=copy.deepcopy(source);s.update(target_cap_tokens=cap,category_caps_tokens={d['owner']:cap},
        cap_basis='Bounded by historical attrs cap; not training authorization',split='train',
        allow_paths=[x['path'] for x in approved],origin_verified=True,
        training_allowed_after_clearance=True,
        file_manifest=(output/'files.json').relative_to(root).as_posix(),
        clearance_record=(output/'clearance.json').relative_to(root).as_posix())
    reg=copy.deepcopy(reg);reg['sources'].append(s);reg=seal(reg)
    spec=copy.deepcopy(spec)
    for kind in ('repository_group','domain_group'):
        need(s[kind] not in spec['group_caps'][kind],'Concentration cap already exists')
        spec['group_caps'][kind][s[kind]]=cap
    spec=seal(dict(spec,registry_identity=reg['identity']))
    validate(spec,reg,root)
    records=[]
    for row,f in zip((x[0] for x in plan),files):
        r=seal(dict(source_id=SOURCE,revision=REVISION,**f,clearance_status='ACCEPT',
            split='train',split_group=s['split_group'],split_unit_kind='repository',
            allow_rule=row['path'],deny_rules_checked=s['deny_paths'],language='python',
            license_evidence=evidence,attribution=attribution,
            source_clearance_identity=clearance['identity'],quality_review='ACCEPT',
            input_path=(directory/row['local_blob']).relative_to(root).as_posix()))
        validate_clearance(r,s,spec,root,f);records.append(r)
    publish(output/'registry.json',reg);publish(output/'spec.json',spec)
    plan_bytes=b''.join(canonical(r)+b'\n' for r in sorted(records,key=item_key))
    plan_path=output/'plan.jsonl'
    if plan_path.exists(): need(plan_path.read_bytes()==plan_bytes,'Supersession plan replay differs')
    else: plan_path.write_bytes(plan_bytes)
    result=seal(dict(policy=POLICY,decision_identity=d['identity'],revision=REVISION,
        prior_clearance_identity=old['identity'],files=16,changed_from_prior=sorted(changed),
        production_tokens=0,training_authorized=False,
        artifacts={p.relative_to(output).as_posix():file_digest(p) for p in
                   [output/'files.json',output/'clearance.json',output/'registry.json',
                    output/'spec.json',plan_path,*sorted((output/'ownership').glob('*.json'))]}))
    publish(output/'complete.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--decision',type=Path,required=True)
    p.add_argument('--approved-decision-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:
        result=build(args.decision,args.output,expected_decision_sha256=args.approved_decision_sha256)
        print(json.dumps(result,sort_keys=True));return 0
    except Exception as exc:
        print(f'Supersession stopped: {type(exc).__name__}: {exc}');return 2


if __name__=='__main__':raise SystemExit(main())
