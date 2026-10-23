"""Offline admission scope tests. No downloads or production tokenization."""
import copy
import json
from pathlib import Path
import pytest
from tests.test_production_acquisition import fixture_plan,Tokenizer,prior_inventory
from training.production_spec import seal,SpecificationError
from training.production_corpus import canonical,file_digest
from training.source_admission import build
from training.production_build import compile_inventory
from training.production_acquisition import ProductionBuild,read_plan


def write(path,value):path.write_bytes(canonical(value))


@pytest.fixture
def inputs(tmp_path):
    spec,reg,plan,records=fixture_plan(tmp_path)
    write(tmp_path/'spec.json',spec);write(tmp_path/'reg.json',reg)
    refs=tmp_path/'refs.jsonl';refs.write_bytes(canonical(dict(reference_id='retained',text='A genuinely separate text on stars and astronomy observations.',content_type='prose',language='en',provenance={'source_id':'not-in-registry'}))+b'\n')
    cache=tmp_path/'cache';d=cache/records[0]['source_id'];(d/'files').mkdir(parents=True)
    source=reg['sources'][0];rows=[]
    for name,payload in [(records[0]['path'],(tmp_path/records[0]['input_path']).read_bytes()),('src/unknown.py',b'# SPDX-License-Identifier: MIT\nprint("Not automatically authorized")\n'),('vendor/copied.py',b'copied code')]:
        blob=d/'files'/str(len(rows));blob.write_bytes(payload)
        rows.append(dict(source_id=source['source_id'],locator=source['locator'],revision=source['revision'],path=name,bytes=len(payload),sha256=file_digest(blob),git_mode='100644',local_blob=blob.relative_to(d).as_posix(),screening='UNRESOLVED',category=source['category'],subcategory=source['subcategory']))
    inventory=d/'inventory.jsonl';inventory.write_bytes(b''.join(canonical(r)+b'\n' for r in rows))
    request=seal(dict(source=source,filters=spec['filters']));write(d/'request.json',request)
    write(d/'complete.json',seal(dict(request_identity=request['identity'],revision=source['revision'],inventory_sha256=file_digest(inventory),counts={'UNRESOLVED':3})))
    return dict(cache=cache,root_registry=tmp_path/'reg.json',root_spec=tmp_path/'spec.json',approved_registry=tmp_path/'reg.json',approved_spec=tmp_path/'spec.json',approved_plan=plan,references=refs,output=tmp_path/'out',root=tmp_path)


def test_exact_grants_only_and_production_compatibility(inputs):
    result=build(**inputs)
    assert result['counts']=={'CLEARED':1,'EXCLUDED':1,'REVIEW':1}
    assert result['newly_authorized_files']==0 and result['production_plan_files']==2
    out=inputs['output'];spec=json.loads((out/'spec.json').read_bytes());reg=json.loads((out/'registry.json').read_bytes())
    plan,receipt=compile_inventory(spec,reg,inputs['root'],out/'runner',out/'plan.jsonl')
    assert receipt['scheduled']==2
    with ProductionBuild(out/'fixture_production',spec,reg,repository_root=inputs['root'],reference_plan=inputs['references'],prior_production=prior_inventory(inputs['root'])) as b:
        assert b.run(plan,Tokenizer())['stages']=={'ACCEPTED':2}


def test_replay_determinism_and_cache_unchanged(inputs):
    hashes={str(p):file_digest(p) for p in inputs['cache'].rglob('*') if p.is_file()}
    a=build(**inputs);b=build(**inputs)
    assert a==b
    assert hashes=={str(p):file_digest(p) for p in inputs['cache'].rglob('*') if p.is_file()}


@pytest.mark.parametrize('damage',['blob','inventory','receipt','output'])
def test_corruption_rejected(inputs,damage):
    build(**inputs)
    d=next(inputs['cache'].iterdir())
    if damage=='blob':target=next((d/'files').iterdir())
    elif damage=='inventory':target=d/'inventory.jsonl'
    elif damage=='receipt':target=d/'complete.json'
    else:target=inputs['output']/'plan.jsonl'
    target.write_bytes(b'broken')
    with pytest.raises((ValueError,SpecificationError)):build(**inputs)


def test_source_holdout_overrides_grant(inputs):
    r=next(read_plan(inputs['approved_plan']))
    inputs['references'].write_bytes(canonical(dict(reference_id='retained',text='retained',content_type='prose',language='en',provenance={'original':{'source_id':r['source_id']}}))+b'\n')
    with pytest.raises(SpecificationError,match='holdout group'):build(**inputs)


def test_changed_revision_not_promoted(inputs):
    d=next(inputs['cache'].iterdir());r=json.loads((d/'request.json').read_bytes());r['source']['revision']='a'*40
    write(d/'request.json',seal(r))
    with pytest.raises(SpecificationError,match='registry'):build(**inputs)


def test_interrupted_output_resumes(inputs,monkeypatch):
    import training.source_admission as module
    original=module.atomic_json
    def interrupted(path,value):
        original(path,value)
        if Path(path).name=='dry_run.json':raise KeyboardInterrupt()
    monkeypatch.setattr(module,'atomic_json',interrupted)
    with pytest.raises(KeyboardInterrupt):build(**inputs)
    monkeypatch.setattr(module,'atomic_json',original)
    assert build(**inputs)['counts']['CLEARED']==1


def test_holdout_project_excluded_without_grant(inputs):
    records=list(read_plan(inputs['approved_plan']));sid=records[0]['source_id']
    inputs['approved_plan'].write_bytes(canonical(records[1])+b'\n')
    inputs['references'].write_bytes(canonical(dict(reference_id='retained',text='retained',content_type='prose',language='en',provenance={'original':{'source_id':sid}}))+b'\n')
    result=build(**inputs)
    assert result['counts']=={'CLEARED':0,'EXCLUDED':3,'REVIEW':0}
    assert result['production_plan_files']==1


def test_changed_content_requires_new_grant(inputs):
    d=next(inputs['cache'].iterdir());path=d/'inventory.jsonl';rows=list(read_plan(path))
    blob=d/rows[0]['local_blob'];blob.write_bytes(b'Changed source text without new per-file authorization.')
    rows[0].update(sha256=file_digest(blob),bytes=blob.stat().st_size)
    path.write_bytes(b''.join(canonical(r)+b'\n' for r in rows))
    done=json.loads((d/'complete.json').read_bytes());done['inventory_sha256']=file_digest(path);write(d/'complete.json',seal(done))
    result=build(**inputs)
    assert result['counts']=={'CLEARED':0,'EXCLUDED':1,'REVIEW':2}


def test_cannot_overwrite_approval_inputs(inputs):
    protected=inputs['root']/'protected';protected.mkdir()
    target=protected/'registry.json';target.write_bytes(inputs['approved_registry'].read_bytes())
    inputs['approved_registry']=target;inputs['output']=protected
    with pytest.raises(SpecificationError,match='overlaps immutable'):build(**inputs)
