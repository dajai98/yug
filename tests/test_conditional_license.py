import copy
import json
import pytest
from training.conditional_license import CONDITIONAL, USES, VERSION, validate_approval
from training.production_corpus import atomic_json, file_digest
from training.production_spec import seal, SpecificationError
from training.source_prepare import LICENSES, classify
from training.production_acquisition import validate_clearance
from training.dataset_rights_scope import scope_from_card


@pytest.fixture
def approved(tmp_path):
    evidence=tmp_path/'review.txt';evidence.write_text('Explicit reviewer fixture; never a real approval.')
    source=dict(source_id='sample',locator='https://github.com/org/sample',revision='a'*40,license_id='MPL-2.0',status='ACCEPT',license_evidence={'path':'review.txt'})
    record=dict(source_id='sample',revision='a'*40,path='src/a.py',sha256='sha256:'+'b'*64,license_id='MPL-2.0')
    approval=dict(policy=VERSION,decision='APPROVED',**{k:source[k] for k in ('source_id','locator','revision','license_id')},
        approved_by='fixture reviewer',approved_at='2026-10-13',
        intended_use={u:dict(decision='APPROVED' if u=='production_preparation' else 'FORBIDDEN',rationale='fixture scoped decision') for u in USES},
        obligations={k:'Retained explicit reviewer plan' for k in ('attribution','notices','modifications','reciprocity','release_controls')},
        evidence=[dict(role=k,path='review.txt',sha256=file_digest(evidence)) for k in ('rights_scope','provenance','obligations','approval')],
        files=[dict(path=record['path'],sha256=record['sha256'])])
    def publish(value):
        value=seal(value);atomic_json(tmp_path/'approval.json',value)
        source['conditional_approval']=dict(path='approval.json',sha256=file_digest(tmp_path/'approval.json'))
        record['conditional_approval_identity']=value['identity']
        return value
    publish(approval)
    return tmp_path,source,record,approval,publish


def test_allowed_list_unchanged():
    assert LICENSES=={'MIT','Apache-2.0','BSD-2-Clause','BSD-3-Clause','CC0-1.0','CC-BY-3.0','CC-BY-4.0'}


@pytest.mark.parametrize('license',sorted(CONDITIONAL))
def test_exact_conditional_license_approval(approved,license):
    root,s,r,a,publish=approved
    s['license_id']=r['license_id']=a['license_id']=license
    expected=publish(a)
    assert validate_approval(r,s,root)==expected
    assert classify(dict(s,status='CONDITIONAL'))=='CACHE_READY'
    assert scope_from_card('---\nlicense: '+license+'\n---\n',dict(s,dataset_id='org/sample'),'card')['status']=='REVIEW'


@pytest.mark.parametrize('field',['approved_by','approved_at','obligations','evidence','files','intended_use'])
def test_incomplete_never_approves(approved,field):
    root,s,r,a,publish=approved;a.pop(field);publish(a)
    with pytest.raises((SpecificationError,KeyError)):validate_approval(r,s,root)


@pytest.mark.parametrize('field',['source_id','locator','revision','license_id'])
def test_wrong_source_binding(approved,field):
    root,s,r,a,publish=approved;a[field]='wrong';publish(a)
    with pytest.raises(SpecificationError):validate_approval(r,s,root)


def test_missing_approval_rejected_at_acquisition_boundary(approved):
    root,s,r,a,publish=approved;s.pop('conditional_approval')
    with pytest.raises(SpecificationError,match='Missing source-specific'):validate_clearance(seal(r),s,{},root)


@pytest.mark.parametrize('mutation',['hash','scope','use','reciprocity','evidence'])
def test_changed_or_incomplete_record(approved,mutation):
    root,s,r,a,publish=approved
    if mutation=='hash':r['sha256']='sha256:'+'c'*64
    if mutation=='scope':a['files'][0]['path']='src/*';publish(a)
    if mutation=='use':a['intended_use']['production_preparation']['decision']='FORBIDDEN';publish(a)
    if mutation=='reciprocity':a['obligations']['reciprocity']='';publish(a)
    if mutation=='evidence':(root/'review.txt').write_text('changed')
    with pytest.raises(SpecificationError):validate_approval(r,s,root)


def test_full_clearance_boundary_with_explicit_approval(approved):
    from tests.test_production_acquisition import fixture_plan
    root,_,_,a,_=approved
    spec,reg,plan,records=fixture_plan(root)
    s=reg['sources'][0];s['license_id']='MPL-2.0'
    r=dict(records[0],license_id='MPL-2.0')
    a.update({k:s[k] for k in ('source_id','locator','revision','license_id')})
    a['files']=[dict(path=r['path'],sha256=r['sha256'])]
    a=seal(a);atomic_json(root/'approval.json',a)
    s['conditional_approval']=dict(path='approval.json',sha256=file_digest(root/'approval.json'))
    r=seal(dict(r,conditional_approval_identity=a['identity']))
    assert validate_clearance(r,s,spec,root,approved_file=r)
    bad=seal(dict(r,conditional_approval_identity='stale'))
    with pytest.raises(SpecificationError):validate_clearance(bad,s,spec,root,approved_file=bad)
