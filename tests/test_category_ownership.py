"""Synthetic file-level category evidence; no real source is approved here."""
import json

import pytest

from training.category_ownership import (decision as ownership_decision,
    approval as ownership_approval, verify_approval, verify_structure)
from training.production_acquisition import ProductionBuild, validate_clearance
from training.production_corpus import canonical, file_digest
from training.production_spec import QUOTAS, SpecificationError, seal
from tests.test_production_acquisition import (BASE, Tokenizer, fixture_plan,
                                               fixture_reference, prior_inventory, approval_kwargs)


def case(root, text=BASE, *, owner='code', subcategory='python', tags=(),
         decision_type='manual_review', normalized_sha256=None):
    spec, registry, plan, records=fixture_plan(root,texts=[text])
    source=registry['sources'][0]
    source['allowed_categories']=sorted(set(source['allowed_categories'])|{owner})
    source['category_caps_tokens'][owner]=min(source['target_cap_tokens'],QUOTAS[owner])
    record=dict(records[0]);record['owner']=owner;record['subcategory']=subcategory
    record['tags']=sorted(tags)
    approved=json.loads((root/'files0.json').read_bytes())['files'][0]
    approved.update(owner=owner,subcategory=subcategory,tags=sorted(tags))
    rationale='Explicit diagnostic file-level category decision'
    evidence=seal(dict(source_id=source['source_id'],revision=source['revision'],
        path=record['path'],sha256=record['sha256'],owner=owner,subcategory=subcategory,
        decision_type=decision_type,rationale_ref=rationale,reviewer='fixture reviewer'))
    (root/'owner-review.json').write_bytes(canonical(evidence))
    d=ownership_decision(source_id=source['source_id'],revision=source['revision'],
        path=record['path'],sha256=record['sha256'],owner=owner,subcategory=subcategory,
        decision_type=decision_type,rationale_ref=rationale,tags=sorted(tags),
        normalized_sha256=normalized_sha256,
        evidence_ref=dict(path='owner-review.json',sha256=file_digest(root/'owner-review.json')))
    approved['ownership_decision']=d;record['ownership_decision']=d
    manifest=seal(dict(source_id=source['source_id'],revision=source['revision'],files=[approved]))
    (root/'files0.json').write_bytes(canonical(manifest))
    clearance=json.loads((root/'clear0.json').read_bytes())
    clearance=seal(dict(clearance,file_manifest_identity=manifest['identity']))
    (root/'clear0.json').write_bytes(canonical(clearance))
    record['source_clearance_identity']=clearance['identity']
    record=seal(record)
    plan.write_bytes(canonical(record)+b'\n')
    registry=seal(registry);spec=seal(dict(spec,registry_identity=registry['identity']))
    return spec,registry,plan,record,approved


@pytest.mark.parametrize('value',[None,'',[],['code','security'],'unsupported'])
def test_missing_empty_multiple_unsupported_owner(tmp_path,value):
    spec,registry,_,record,approved=case(tmp_path)
    row=dict(record)
    if value is None:row.pop('owner')
    else:row['owner']=value
    with pytest.raises(SpecificationError,match='budget owner'):
        validate_clearance(seal(row),registry['sources'][0],spec,tmp_path,approved)


@pytest.mark.parametrize('alias', ['category','budget_owner','budget_category',
    'accounting_owner','accounting_category','allocation_owner','categories'])
def test_owner_aliases_forbidden_even_if_equal(tmp_path,alias):
    spec,registry,_,record,approved=case(tmp_path)
    with pytest.raises(SpecificationError,match='alias'):
        validate_clearance(seal(dict(record,**{alias:'code'})),registry['sources'][0],spec,tmp_path,approved)


def test_owner_aliases_rejected_in_approved_decision_and_evidence(tmp_path):
    spec,registry,_,record,approved=case(tmp_path)
    altered=seal(dict(approved['ownership_decision'],accounting_owner='security'))
    with pytest.raises(SpecificationError,match='alias'):
        validate_clearance(seal(dict(record,ownership_decision=altered)),
                           registry['sources'][0],spec,tmp_path,
                           dict(approved,ownership_decision=altered))
    evidence=json.loads((tmp_path/'owner-review.json').read_bytes())
    (tmp_path/'owner-review.json').write_bytes(canonical(seal(dict(evidence,budget_category='security'))))
    changed=ownership_decision(source_id=record['source_id'],revision=record['revision'],
        path=record['path'],sha256=record['sha256'],owner=record['owner'],
        subcategory=record['subcategory'],decision_type='manual_review',
        rationale_ref='Explicit diagnostic file-level category decision',
        evidence_ref=dict(path='owner-review.json',sha256=file_digest(tmp_path/'owner-review.json')))
    with pytest.raises(SpecificationError,match='alias'):
        validate_clearance(seal(dict(record,ownership_decision=changed)),
                           registry['sources'][0],spec,tmp_path,
                           dict(approved,ownership_decision=changed))


def test_external_approval_pin_blocks_self_supplied_plan_and_review(tmp_path):
    _,_,plan,_,_=case(tmp_path)
    p1=file_digest(plan)
    kwargs=approval_kwargs(tmp_path,p1)
    assert verify_approval(kwargs['ownership_approval'],
        kwargs['expected_ownership_approval_sha256'],p1,tmp_path/'build')
    forged=seal(dict(json.loads(plan.read_bytes()),tags=['forged'],
                     ownership_decision=seal(dict(json.loads(plan.read_bytes())['ownership_decision'],
                                                  tags=['forged']))))
    plan.write_bytes(canonical(forged)+b'\n')
    p2=file_digest(plan)
    assert p2!=p1
    with pytest.raises(SpecificationError,match='authorize'):
        verify_approval(kwargs['ownership_approval'],
            kwargs['expected_ownership_approval_sha256'],p2,tmp_path/'build')
    local=tmp_path/'local-approval.json'
    local.write_bytes(canonical(ownership_approval(plan_sha256=p2,
        approver='fabricated reviewer',approved_at='2026-10-26')))
    with pytest.raises(SpecificationError,match='changed'):
        verify_approval(local,kwargs['expected_ownership_approval_sha256'],p2,tmp_path/'build')
    with pytest.raises(SpecificationError,match='Missing'):
        verify_approval(None,None,p2,tmp_path/'build')


def test_fabricated_plan_cannot_register_with_original_external_approval(tmp_path):
    spec,registry,plan,record,approved=case(tmp_path)
    original=file_digest(plan)
    pinned=approval_kwargs(tmp_path,original)
    altered=seal(dict(record,tags=['security']))
    plan.write_bytes(canonical(altered)+b'\n')
    refs=tmp_path/'references.jsonl'
    refs.write_bytes(canonical(fixture_reference('Distant stars and spectral analysis.'))+b'\n')
    with pytest.raises(SpecificationError,match='authorize'):
        ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,
            reference_plan=refs,prior_production=prior_inventory(tmp_path),
            expected_plan_sha256=file_digest(plan),**pinned)
    assert not (tmp_path/'out'/'prepared.json').exists()


def test_structural_disguise_cannot_account_or_export(tmp_path):
    dialogue='User: Hello\nAssistant: Please help me\n'
    spec,registry,plan,record,approved=case(tmp_path,dialogue)
    source=registry['sources'][0]
    assert source['category']=='code' and source['subcategory']=='python'
    record=dict(record,path='src/dialogue.py',content_type='code',language='python')
    approved=dict(approved,path=record['path'],content_type='code')
    d=ownership_decision(source_id=source['source_id'],revision=source['revision'],
        path=record['path'],sha256=record['sha256'],owner='code',subcategory='python',
        decision_type='structural',rationale_ref='approved-source-and-extension-v1')
    record=seal(dict(record,ownership_decision=d));approved['ownership_decision']=d
    manifest=seal(dict(source_id=source['source_id'],revision=source['revision'],files=[approved]))
    (tmp_path/'files0.json').write_bytes(canonical(manifest))
    clearance=json.loads((tmp_path/'clear0.json').read_bytes())
    clearance=seal(dict(clearance,file_manifest_identity=manifest['identity']))
    (tmp_path/'clear0.json').write_bytes(canonical(clearance))
    record=seal(dict(record,source_clearance_identity=clearance['identity']))
    plan.write_bytes(canonical(record)+b'\n')
    refs=tmp_path/'references.jsonl'
    refs.write_bytes(canonical(fixture_reference('Distant stars and spectral analysis.'))+b'\n')
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,
                         reference_plan=refs,prior_production=prior_inventory(tmp_path),
                         expected_plan_sha256=file_digest(plan),
                         **approval_kwargs(tmp_path,file_digest(plan))) as build:
        with pytest.raises(SpecificationError,match='structural Python'):
            build.run(plan,Tokenizer())
        assert build.db.execute('SELECT count(*) FROM totals').fetchone()[0]==0
        assert not (build.directory/'prepared.json').exists()


@pytest.mark.parametrize('text,accepted',[('def add(a,b):\n    return a+b\n',True),
    ('User: Hello\nAssistant: Please help me\n',False),
    ('This is plain English dialogue.\n',False),
    ('"Only a docstring"\n',False)])
def test_structural_python_checks_actual_payload(text,accepted):
    record=dict(subcategory='python',language='python',
        ownership_decision=dict(decision_type='structural'))
    if accepted:verify_structure(text.encode(),record)
    else:
        with pytest.raises(SpecificationError):verify_structure(text.encode(),record)


@pytest.mark.parametrize('subcategory',('rust','go','cpp'))
def test_other_languages_require_nonstructural_evidence(subcategory):
    record=dict(subcategory=subcategory,language=subcategory,
        ownership_decision=dict(decision_type='structural'))
    with pytest.raises(SpecificationError,match='Unsupported structural'):
        verify_structure(b'fn main() {}',record)


@pytest.mark.parametrize('change',[('owner','security'),('subcategory','cpp'),
    ('tags',['security']),('path','src/another.txt'),('revision','f'*40),
    ('sha256','sha256:'+'f'*64)])
def test_decision_bound_to_approved_file_and_identity(tmp_path,change):
    spec,registry,_,record,approved=case(tmp_path,tags=('python',))
    field,value=change
    with pytest.raises((SpecificationError,KeyError)):
        validate_clearance(seal(dict(record,**{field:value})),registry['sources'][0],spec,tmp_path,approved)


@pytest.mark.parametrize('field,value',[('identity','sha256:'+'0'*64),
    ('rationale_ref','changed'),('decision_type','source_evidence'),
    ('policy_identity','sha256:'+'0'*64)])
def test_decision_metadata_tamper(tmp_path,field,value):
    spec,registry,_,record,approved=case(tmp_path)
    changed=dict(record['ownership_decision'],**{field:value})
    with pytest.raises(SpecificationError):
        validate_clearance(seal(dict(record,ownership_decision=changed)),
                           registry['sources'][0],spec,tmp_path,approved)


def test_subcategory_needs_file_evidence_and_manual_override(tmp_path):
    spec,registry,_,record,approved=case(tmp_path,subcategory='cpp')
    assert validate_clearance(record,registry['sources'][0],spec,tmp_path,approved)
    with pytest.raises(SpecificationError,match='subcategory'):
        validate_clearance(seal(dict(record,subcategory='python')),
                           registry['sources'][0],spec,tmp_path,approved)


@pytest.mark.parametrize('text,wrong_owner,wrong_subcategory',[
    ('def add(a, b):\n    return a + b\n','conversation','dialogue'),
    ('User: Hello.\nAssistant: Hello, how may I help?','security','secure_development'),
    ('A fragment shader computes a color for each pixel in the rendered scene.','general_english','explanatory'),
    ('Linux process signals and shell pipelines coordinate command execution.','csharp_gamedev','engine_architecture'),
])
def test_contradictions_without_independent_decision_fail(tmp_path,text,wrong_owner,wrong_subcategory):
    spec,registry,plan,record,approved=case(tmp_path,text,owner=wrong_owner,subcategory=wrong_subcategory)
    approved.pop('ownership_decision');record=seal(dict(record,ownership_decision=None))
    with pytest.raises(SpecificationError,match='ownership decision'):
        validate_clearance(record,registry['sources'][0],spec,tmp_path,approved)


@pytest.mark.parametrize('owner,subcategory,tags',[
    ('csharp_gamedev','unity_authored',('code','unity')),
    ('graphics','shaders',('shader','unity')),
    ('software_engineering','debugging',('debugging','unity')),
    ('algorithms','algorithms',('algorithm','python')),
    ('linux_systems','linux',('linux','programming')),
    ('software_docs','api',('api','security')),
    ('nepali','technical_transfer',('nepali','unity')),
])
def test_overlaps_need_explicit_single_owner_decision(tmp_path,owner,subcategory,tags):
    spec,registry,_,record,approved=case(tmp_path,owner=owner,subcategory=subcategory,tags=tags)
    assert validate_clearance(record,registry['sources'][0],spec,tmp_path,approved)
    assert record['owner']==record['ownership_decision']['owner']==approved['owner']
    assert record['tags']==sorted(tags)


def test_accounting_and_export_bind_decision_but_tags_charge_no_tokens(tmp_path):
    spec,registry,plan,record,_=case(tmp_path,tags=('implementation','python'))
    refs=tmp_path/'references.jsonl'
    refs.write_bytes(canonical(fixture_reference('Distant stars and spectral analysis.'))+b'\n')
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,
                         reference_plan=refs,prior_production=prior_inventory(tmp_path),
                         expected_plan_sha256=file_digest(plan),**approval_kwargs(tmp_path,file_digest(plan))) as build:
        result=build.run(plan,Tokenizer())
        assert result['stages']=={'ACCEPTED':1}
        assert build.db.execute("SELECT kind,key,tokens FROM totals WHERE kind='category'").fetchall()==[
            ('category','code',len(BASE.encode()))]
        exported=build.export_shards()
        assert exported['total_token_count']==len(BASE.encode())
        admission=json.loads((build.directory/'admission.jsonl').read_text().splitlines()[0])
        assert admission['clearance']['ownership_decision']==record['ownership_decision']
        assert admission['clearance']['tags']==['implementation','python']
        build.verify()


def test_historical_plan_without_decision_cannot_register(tmp_path):
    spec,registry,plan,record,approved=case(tmp_path)
    approved.pop('ownership_decision')
    manifest=seal(dict(source_id=record['source_id'],revision=record['revision'],files=[approved]))
    (tmp_path/'files0.json').write_bytes(canonical(manifest))
    clearance=json.loads((tmp_path/'clear0.json').read_bytes())
    clearance=seal(dict(clearance,file_manifest_identity=manifest['identity']))
    (tmp_path/'clear0.json').write_bytes(canonical(clearance))
    plan.write_bytes(canonical(seal(dict(record,ownership_decision=None,
                                        source_clearance_identity=clearance['identity'])))+b'\n')
    refs=tmp_path/'references.jsonl'
    refs.write_bytes(canonical(fixture_reference('Distant stars and spectral analysis.'))+b'\n')
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,
                         reference_plan=refs,prior_production=prior_inventory(tmp_path),
                         expected_plan_sha256=file_digest(plan),**approval_kwargs(tmp_path,file_digest(plan))) as build:
        with pytest.raises(SpecificationError,match='ownership decision'):
            build.run(plan,Tokenizer())
        assert not (build.directory/'shards'/'prepared.json').exists()


def test_decision_normalized_identity_blocks_accounting(tmp_path):
    spec,registry,plan,_,_=case(tmp_path,normalized_sha256='sha256:'+'0'*64)
    refs=tmp_path/'references.jsonl'
    refs.write_bytes(canonical(fixture_reference('Distant stars and spectral analysis.'))+b'\n')
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,
                         reference_plan=refs,prior_production=prior_inventory(tmp_path),
                         expected_plan_sha256=file_digest(plan),**approval_kwargs(tmp_path,file_digest(plan))) as build:
        with pytest.raises(SpecificationError,match='ownership normalization'):
            build.run(plan,Tokenizer())
        assert build.db.execute("SELECT count(*) FROM items WHERE stage='ACCEPTED'").fetchone()[0]==0


def test_resealed_local_owner_change_cannot_verify_resume_or_export(tmp_path):
    spec,registry,plan,record,_=case(tmp_path)
    refs=tmp_path/'references.jsonl'
    refs.write_bytes(canonical(fixture_reference('Distant stars and spectral analysis.'))+b'\n')
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,
                         reference_plan=refs,prior_production=prior_inventory(tmp_path),
                         expected_plan_sha256=file_digest(plan),**approval_kwargs(tmp_path,file_digest(plan))) as build:
        assert build.run(plan,Tokenizer())['stages']=={'ACCEPTED':1}
        item=build.db.execute('SELECT item FROM items').fetchone()[0]
        altered=seal(dict(record,owner='security'))
        build.db.execute('UPDATE items SET record=?,owner=? WHERE item=?',
                         (canonical(altered).decode(),'security',item))
        build.db.commit()
        build.path('clearances',item,'.json').write_bytes(canonical(altered))
        with pytest.raises(SpecificationError):build.verify()
        with pytest.raises(SpecificationError):build.run(plan,Tokenizer())
        with pytest.raises(SpecificationError):build.export_shards()
        assert not (build.directory/'prepared.json').exists()
