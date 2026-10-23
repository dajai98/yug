"""Production specification tests: no network, corpus acquisition or model training."""
import copy
import json
from pathlib import Path
import pytest
from training import production_spec as ps
from training.experiments.production_corpus_spec import build

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def bundle():
    return build(ROOT)

def reseal(spec,registry):
    registry=ps.seal(registry)
    spec['registry_identity']=registry['identity']
    return ps.seal(spec),registry

def test_exact_quota_and_3d_owner_budgets(bundle):
    spec,registry=bundle
    assert sum(c['target_tokens'] for c in spec['categories'])==2_000_000_000
    assert len(spec['categories'])==16
    assert sum(t['target_tokens'] for t in spec['three_d']['topics'])==100_000_000
    assert sum(spec['three_d']['form_targets'].values())==100_000_000
    assert sum(c['non_3d_target_tokens'] for c in spec['categories'])==1_900_000_000
    assert ps.validate(spec,registry,ROOT)

def test_generated_artifacts_reproduce(bundle):
    for name,value in zip(('production_corpus_v1.json','source_registry_v1.json'),bundle):
        assert json.loads((ROOT/'training/specifications'/name).read_text())==value
    assert build(ROOT)==bundle

@pytest.mark.parametrize('change', ['quota','tokenizer','category','three_d','form','checkpoint','ownership','validation','fallback'])
def test_invalid_frozen_decisions_rejected(bundle,change):
    spec,registry=copy.deepcopy(bundle)
    if change=='quota': spec['categories'][0]['target_tokens']-=1
    elif change=='tokenizer': spec['tokenizer_identity']='sha256:'+'0'*64
    elif change=='category': spec['categories'].append(spec['categories'][0])
    elif change=='three_d': spec['three_d']['topics'][0]['target_tokens']+=1
    elif change=='form': spec['three_d']['form_targets']['workflow']+=1
    elif change=='checkpoint': spec['checkpoints'][-1]-=1
    elif change=='ownership': spec['accounting']['one_budget_owner']=False
    elif change=='validation': spec['accounting']['validation_counts_toward_2b']=True
    else: spec['fallback']['cross_category_transfer']='automatic'
    with pytest.raises(ps.SpecificationError): ps.validate(ps.seal(spec),registry,ROOT)

@pytest.mark.parametrize('change',['duplicate','status','cap','license','group','split'])
def test_registry_fail_closed(bundle,change):
    spec,registry=copy.deepcopy(bundle)
    s=next(s for s in registry['sources'] if s['status']=='ACCEPT')
    if change=='duplicate': registry['sources'].append(copy.deepcopy(s))
    elif change=='status': s['status']='UNKNOWN'
    elif change=='cap': s['target_cap_tokens']=True
    elif change=='license': s['license_evidence']=None
    elif change=='group': s['repository_group']='not-registered'
    else: s['split']='both'
    spec,registry=reseal(spec,registry)
    with pytest.raises(ps.SpecificationError): ps.validate(spec,registry,ROOT)

def test_evidence_and_seal_corruption_rejected(bundle):
    spec,registry=copy.deepcopy(bundle)
    registry['sources'][0]['name']='changed'
    with pytest.raises(ps.SpecificationError,match='identity'): ps.validate(spec,registry,ROOT)
    spec,registry=copy.deepcopy(bundle)
    spec['evidence'][0]['sha256']='sha256:'+'0'*64
    with pytest.raises(ps.SpecificationError,match='Changed evidence'): ps.validate(ps.seal(spec),registry,ROOT)

def test_locator_and_source_split_aliases(bundle):
    spec,registry=copy.deepcopy(bundle)
    a,b=registry['sources'][:2]
    a['locator']=b['locator']='https://example.invalid/project'
    spec,registry=reseal(spec,registry)
    with pytest.raises(ps.SpecificationError,match='Locator aliases'): ps.validate(spec,registry,ROOT)
    spec,registry=copy.deepcopy(bundle)
    a,b=registry['sources'][:2]; a['split']='train';b['split']='validation';b['split_group']=a['split_group']
    spec,registry=reseal(spec,registry)
    with pytest.raises(ps.SpecificationError,match='crosses'): ps.validate(spec,registry,ROOT)

@pytest.mark.parametrize('path,allowed', [('src/a.py',True),('vendor/a.py',False),('src/vendor/a.py',False),('src/X.MIN.JS',False),('../secret',False),('/src/a.py',False),('src\\a.py',False),('C:/src/a.py',False),('src//a.py',False),('src/./a.py',False)])
def test_path_boundary(path,allowed):
    assert ps.path_allowed(path,['**'],['**/vendor/**','**/*.min.js']) is allowed

def test_dry_run_reports_no_invented_acquisition_or_yield(bundle):
    report=ps.dry_run(*bundle,ROOT)
    assert report['configured_clean_train_tokens']==2_000_000_000
    assert report['accepted_clean_train_tokens']==0
    assert not report['acquisition_plan'] and not report['ready_for_controlled_acquisition']
    assert sum(report['category_deficits'].values())==2_000_000_000
    assert sum(report['status_counts'].values())==len(bundle[1]['sources'])
    assert all(s['deficit_tokens'] is None for s in report['source_progress'].values())
    assert report['storage']['train_spool_plus_shard_bytes']==16_000_000_000

def test_old_sources_preserved_new_candidates_not_promoted(bundle):
    old=json.loads((ROOT/'training/resources/phase1/rep_corpus_10m_v1/01_source_policy.json').read_text())
    current={s['source_id']:s for s in bundle[1]['sources']}
    for source in old['source_policy']:
        assert current[source['source_id']]['prior_status']==source['stage1c_status']
    for sid,s in current.items():
        if sid.startswith('candidate-'):
            assert s['status'] in ('REVIEW','CONDITIONAL') and s['license_id'] is None
    assert current['cp-wikimedia-filtered']['status']=='CONDITIONAL'
    assert current['gh-monogame-monogame']['status']=='CONDITIONAL'

def cleared(bundle):
    spec,registry=copy.deepcopy(bundle)
    s=next(s for s in registry['sources'] if s['source_id']=='gh-psf-requests')
    s.update(origin_verified=True,revision='a'*40,locator='https://example.invalid/repo',
        clearance_record='clearance.json',file_manifest='files.json',allow_paths=['src/**'],split='train')
    spec,registry=reseal(spec,registry)
    return spec,registry,s

def receipt(source,**changes):
    r=dict(source_id=source['source_id'],owner=source['category'],raw_tokens=25,
        disposition='accepted',clean_tokens=20,split='train',normalization='structured-v1',
        path='src/a.py',normalized_sha256='sha256:'+'1'*64,
        near_duplicate_review='cleared',duplicate_cluster='cluster-1')
    r.update(changes);return r

def test_checkpoint_accounting_and_deficits(bundle):
    spec,registry,s=cleared(bundle)
    rows=[receipt(s),receipt(s,disposition='rejected',reason='generated',path='src/generated.py')]
    report=ps.audit_checkpoint(spec,registry,rows,tokenizer_identity=ps.TOKENIZER)
    assert report['accepted_clean_train_tokens']==20
    assert report['raw_tokens']==50 and report['rejected_or_review_raw_tokens']==25
    assert report['category_progress']['code']['deficit']==440_000_000-20
    assert report['filter_counts']=={'generated':1}
    assert report['actual_corpus_complete'] is False

@pytest.mark.parametrize('changes,match', [({'owner':'graphics'},'owner'),({'clean_tokens':True},'token count'),({'split':'validation'},'split'),({'near_duplicate_review':'pending'},'review'),({'path':'vendor/a.py'},'excluded'),({'three_d_topic':'asset_interchange','three_d_form':'workflow'},'3D budget'),({'three_d_form':'workflow'},'together')])
def test_receipt_rejects_invalid_accounting(bundle,changes,match):
    spec,registry,s=cleared(bundle)
    with pytest.raises(ps.SpecificationError,match=match):
        ps.audit_checkpoint(spec,registry,[receipt(s,**changes)],tokenizer_identity=ps.TOKENIZER)

@pytest.mark.parametrize('kind',['file','exact','near','source_cap','group_cap'])
def test_global_exclusivity_and_caps(bundle,kind):
    spec,registry,s=cleared(bundle)
    a=receipt(s);b=receipt(s,path='src/b.py',normalized_sha256='sha256:'+'2'*64,duplicate_cluster='cluster-2')
    if kind=='file': b['path']=a['path']
    elif kind=='exact': b['normalized_sha256']=a['normalized_sha256']
    elif kind=='near': b['duplicate_cluster']=a['duplicate_cluster']
    elif kind=='source_cap':
        next(x for x in registry['sources'] if x['source_id']==s['source_id'])['target_cap_tokens']=30
    else: spec['group_caps']['repository_group'][s['repository_group']]=30
    spec,registry=reseal(spec,registry)
    with pytest.raises(ps.SpecificationError): ps.audit_checkpoint(spec,registry,[a,b],tokenizer_identity=ps.TOKENIZER)

def test_validation_not_counted_toward_2b(bundle):
    spec,registry,s=cleared(bundle)
    next(x for x in registry['sources'] if x['source_id']==s['source_id'])['split']='validation'
    spec,registry=reseal(spec,registry)
    report=ps.audit_checkpoint(spec,registry,[receipt(s,split='validation')],tokenizer_identity=ps.TOKENIZER)
    assert report['accepted_clean_train_tokens']==0 and report['validation_tokens']==20

def test_bfcl_train_blocked_even_after_clearance(bundle):
    source=copy.deepcopy(next(s for s in bundle[1]['sources'] if 'bfcl' in s['source_id']))
    source.update(status='ACCEPT',origin_verified=True,split='train')
    assert 'validation_only_source' in ps.acquisition_blockers(source)

def test_evidence_path_escape_rejected(tmp_path):
    with pytest.raises(ps.SpecificationError,match='escapes'): ps.local_path(tmp_path,'../outside.json')

def acquisition_fixture(bundle,tmp_path):
    spec,registry,s=cleared(bundle)
    evidence=tmp_path/'license.txt'; evidence.write_text('Fixture evidence; NOT a real source clearance.')
    entries=[dict(path='license.txt',sha256=ps.file_digest(evidence))]
    manifest=ps.seal(dict(source_id=s['source_id'],revision=s['revision'],files=[dict(path='src/a.py',owner='code',content_type='code',sha256='sha256:'+'a'*64,bytes=123)]))
    (tmp_path/'files.json').write_text(json.dumps(manifest))
    clearance=ps.seal(dict(source_id=s['source_id'],revision=s['revision'],file_manifest_identity=manifest['identity'],decision='ACCEPT',evidence=entries,attribution='Fixture author'))
    (tmp_path/'clearance.json').write_text(json.dumps(clearance))
    spec['evidence']=entries
    return ps.seal(spec),registry,s,manifest,clearance

def test_manifest_bound_plan_and_actual_download_ceiling(bundle,tmp_path):
    spec,registry,s,manifest,clearance=acquisition_fixture(bundle,tmp_path)
    report=ps.dry_run(spec,registry,tmp_path)
    assert len(report['acquisition_plan'])==1
    assert report['acquisition_plan'][0]['max_download_bytes']==123
    assert report['acquisition_plan'][0]['revision']=='a'*40
    assert report['acquisition_plan'][0]['files'][0]['owner']=='code'
    # File readiness cannot bypass pending system-level production obligations.
    assert not report['ready_for_controlled_acquisition']

@pytest.mark.parametrize('change',['revision','checksum','path','duplicate','owner','normalization','clearance','license_hash'])
def test_acquisition_manifest_and_clearance_fail_closed(bundle,tmp_path,change):
    spec,registry,s,m,c=acquisition_fixture(bundle,tmp_path)
    if change=='revision': m['revision']='b'*40
    elif change=='checksum': m['files'][0]['sha256']='sha256:broken'
    elif change=='path': m['files'][0]['path']='../escape'
    elif change=='duplicate': m['files'].append(dict(m['files'][0]))
    elif change=='owner': m['files'][0]['owner']='security'
    elif change=='normalization': m['files'][0]['content_type']='guess'
    elif change=='clearance': c['decision']='REVIEW'
    else: c['evidence'][0]['sha256']='sha256:'+'0'*64
    m=ps.seal(m);c['file_manifest_identity']=m['identity'];c=ps.seal(c)
    (tmp_path/'files.json').write_text(json.dumps(m));(tmp_path/'clearance.json').write_text(json.dumps(c))
    with pytest.raises(ps.SpecificationError): ps.validate(spec,registry,tmp_path)

def test_global_exact_duplicate_cannot_move_to_other_budget_owner(bundle):
    spec,registry,s=cleared(bundle)
    a=receipt(s)
    b=receipt(s,owner='software_engineering',path='src/b.py')
    with pytest.raises(ps.SpecificationError,match='Global exact duplicate'):
        ps.audit_checkpoint(spec,registry,[a,b],tokenizer_identity=ps.TOKENIZER)

def test_3d_accounting_is_one_budget_two_views(bundle):
    spec,registry=copy.deepcopy(bundle)
    source=next(s for s in registry['sources'] if s['source_id']=='candidate-gltf-blender-io')
    source.update(status='ACCEPT',license_id='fixture',license_evidence='fixture',origin_verified=True,
        revision='a'*40,locator='https://example.invalid/repo',clearance_record='fixture.json',file_manifest='fixture.json',allow_paths=['src/**'],split='train')
    spec,registry=reseal(spec,registry)
    row=receipt(source,three_d_topic='blender_python_procedural',three_d_form='procedural_tasks')
    report=ps.audit_checkpoint(spec,registry,[row],tokenizer_identity=ps.TOKENIZER)
    assert report['accepted_clean_train_tokens']==20
    assert sum(report['three_d_topic_progress'].values())==20
    assert sum(report['three_d_form_progress'].values())==20
    assert report['category_progress']['code']['tokens']==20
