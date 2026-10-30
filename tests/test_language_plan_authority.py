"""Adversarial plan-authority and language-specific dedup-route checks."""
import hashlib
import json

import pytest

from tests.test_language_validation import ENGLISH, NEPALI, BALANCED, PYTHON, RUST, GO, CSHARP
from tests.test_production_acquisition import Tokenizer, fixture_reference, prior_inventory, approval_kwargs
from tests.test_quality_filtering import production_case
from training.production_acquisition import ProductionBuild, item_key
from training.production_corpus import canonical, file_digest
from training.production_dedup import features
from training.production_spec import SpecificationError, seal


@pytest.mark.parametrize('original,replacement,text,kind', [
    ('en','unknown',ENGLISH,'prose'),
    ('python','unknown',PYTHON,'code'),
    ('python','ne',PYTHON,'code'),
    ('en','python',ENGLISH,'prose'),
])
def test_external_plan_pin_blocks_resealed_local_universe(tmp_path,original,replacement,text,kind):
    spec,registry,plan=production_case(tmp_path,text,kind)
    first=seal(dict(json.loads(plan.read_bytes()),language=original))
    plan.write_bytes(canonical(first)+b'\n')
    approved=file_digest(plan)  # Kept outside the mutable build state.
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,
                         diagnostic=True,expected_plan_sha256=approved) as build:
        assert build.run(plan,Tokenizer())['stages']=={'ACCEPTED':1}
        item=item_key(first)
        tampered=seal(dict(first,language=replacement))
        plan.write_bytes(canonical(tampered)+b'\n')
        build.db.execute('UPDATE items SET record=? WHERE item=?',(canonical(tampered).decode(),item))
        build.db.execute('UPDATE dd SET provenance=? WHERE item=?',(canonical(tampered).decode(),item))
        build.path('clearances',item,'.json').write_bytes(canonical(tampered))
        build.put('plan_identity',file_digest(plan))
        request=build.get('request');request['plan_authority']=file_digest(plan)
        build.put('request',request);build.db.commit()
        for operation in (build.verify,lambda:build.export_shards(tmp_path/'out'/'fresh'),
                          lambda:build.run(plan,Tokenizer())):
            with pytest.raises(SpecificationError,match='external.*(approved|authority)'):
                operation()
        assert not (tmp_path/'out'/'fresh'/'prepared.json').exists()
    with pytest.raises(SpecificationError,match='Incompatible production build resume'):
        ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,
                        diagnostic=True,expected_plan_sha256=approved)


def test_stale_route_features_rejected_even_with_unchanged_plan(tmp_path):
    spec,registry,plan=production_case(tmp_path,PYTHON,'code')
    original=seal(dict(json.loads(plan.read_bytes()),language='python'))
    plan.write_bytes(canonical(original)+b'\n')
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,
                         diagnostic=True,expected_plan_sha256=file_digest(plan)) as build:
        build.run(plan,Tokenizer())
        item=item_key(original)
        normalized=build.path('normalized',item).read_text(encoding='utf-8')
        old=build.db.execute('SELECT shingles FROM dd WHERE item=?',(item,)).fetchone()[0]
        different=b''.join(features(normalized,'code','unknown'))
        assert old!=different
        node=build.db.execute('SELECT id FROM dd WHERE item=?',(item,)).fetchone()[0]
        build.db.execute('UPDATE dd SET shingles=?,checksum=? WHERE item=?',
                         (different,hashlib.sha256(different).hexdigest(),item))
        build.db.execute('DELETE FROM anchors WHERE doc=?',(node,))
        build.db.executemany('INSERT INTO anchors VALUES (?,?)',
                             ((h,node) for h in features(normalized,'code','unknown')[:128]))
        build.db.commit()
        with pytest.raises(SpecificationError,match='Dedup features differ'):
            build.verify()


@pytest.mark.parametrize('label,text,kind',[
    ('en',ENGLISH,'prose'),('ne',NEPALI,'prose'),('mixed',BALANCED,'prose'),
    ('unknown',ENGLISH,'prose'),('python',PYTHON,'code'),('rust',RUST,'code'),
    ('go',GO,'code'),('csharp',CSHARP,'code'),
])
def test_honest_external_plan_and_replay(tmp_path,label,text,kind):
    spec,registry,plan=production_case(tmp_path,text,kind)
    record=seal(dict(json.loads(plan.read_bytes()),language=label))
    plan.write_bytes(canonical(record)+b'\n')
    approved=file_digest(plan)
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,
                         diagnostic=True,expected_plan_sha256=approved) as build:
        assert build.run(plan,Tokenizer())['stages']=={'ACCEPTED':1}
        assert build.export_shards()==build.export_shards()
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,
                         diagnostic=True,expected_plan_sha256=approved) as build:
        assert build.run(plan,Tokenizer())['stages']=={'ACCEPTED':1}


def test_old_mutable_plan_state_cannot_resume_without_external_pin(tmp_path):
    spec,registry,plan=production_case(tmp_path,ENGLISH)
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,
                         diagnostic=True,expected_plan_sha256=file_digest(plan)) as build:
        build.run(plan,Tokenizer())
        request=build.get('request');request.pop('plan_authority')
        build.put('request',request);build.db.commit()
    with pytest.raises(SpecificationError,match='Incompatible production build resume'):
        ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,
                        diagnostic=True,expected_plan_sha256=file_digest(plan))


def test_new_authoritative_route_rebuilds_features_in_new_build(tmp_path):
    spec,registry,plan=production_case(tmp_path,PYTHON,'code')
    python_record=seal(dict(json.loads(plan.read_bytes()),language='python'))
    plan.write_bytes(canonical(python_record)+b'\n')
    with ProductionBuild(tmp_path/'python_build',spec,registry,repository_root=tmp_path,
                         diagnostic=True,expected_plan_sha256=file_digest(plan)) as build:
        build.run(plan,Tokenizer())
        old=build.db.execute('SELECT shingles FROM dd WHERE item=?',
                             (item_key(python_record),)).fetchone()[0]
    unknown_record=seal(dict(python_record,language='unknown'))
    plan.write_bytes(canonical(unknown_record)+b'\n')
    with ProductionBuild(tmp_path/'new_approved_build',spec,registry,repository_root=tmp_path,
                         diagnostic=True,expected_plan_sha256=file_digest(plan)) as build:
        assert build.run(plan,Tokenizer())['stages']=={'ACCEPTED':1}
        new=build.db.execute('SELECT shingles FROM dd WHERE item=?',
                             (item_key(unknown_record),)).fetchone()[0]
        assert new!=old
        build.verify()


def test_real_production_refuses_missing_external_plan_identity(tmp_path):
    spec,registry,plan=production_case(tmp_path,ENGLISH)
    references=tmp_path/'references.jsonl'
    references.write_bytes(canonical(fixture_reference('Independent holdout text about ocean currents.'))+b'\n')
    with pytest.raises(SpecificationError,match='external approved plan identity'):
        ProductionBuild(tmp_path/'without_pin',spec,registry,repository_root=tmp_path,
                        reference_plan=references,prior_production=prior_inventory(tmp_path),
                        **approval_kwargs(tmp_path,file_digest(plan)))
    with ProductionBuild(tmp_path/'with_pin',spec,registry,repository_root=tmp_path,
                         reference_plan=references,prior_production=prior_inventory(tmp_path),
                         expected_plan_sha256=file_digest(plan),**approval_kwargs(tmp_path,file_digest(plan))) as build:
        assert build.run(plan,Tokenizer())['stages']=={'ACCEPTED':1}


def test_real_production_rejects_local_rebaseline_after_acceptance(tmp_path):
    spec,registry,plan=production_case(tmp_path,ENGLISH)
    approved=file_digest(plan)
    references=tmp_path/'references.jsonl'
    references.write_bytes(canonical(fixture_reference('Independent holdout text about ocean currents.'))+b'\n')
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,
                         reference_plan=references,prior_production=prior_inventory(tmp_path),
                         expected_plan_sha256=approved,**approval_kwargs(tmp_path,approved)) as build:
        assert build.run(plan,Tokenizer())['stages']=={'ACCEPTED':1}
        original=json.loads(plan.read_bytes());item=item_key(original)
        changed=seal(dict(original,language='unknown'))
        plan.write_bytes(canonical(changed)+b'\n')
        build.db.execute('UPDATE items SET record=? WHERE item=?',(canonical(changed).decode(),item))
        build.db.execute('UPDATE dd SET provenance=? WHERE item=?',(canonical(changed).decode(),item))
        build.path('clearances',item,'.json').write_bytes(canonical(changed))
        build.put('plan_identity',file_digest(plan))
        request=build.get('request');request['plan_authority']=file_digest(plan)
        build.put('request',request);build.db.commit()
        with pytest.raises(SpecificationError,match='external production plan authority'):
            build.verify()
        with pytest.raises(SpecificationError,match='external production plan authority'):
            build.export_shards(tmp_path/'out'/'new_export')
        assert not (tmp_path/'out'/'new_export'/'prepared.json').exists()


@pytest.mark.parametrize('registration', [False, None, 'true', 1, []])
def test_completed_production_requires_valid_registration_for_every_boundary(tmp_path,registration):
    spec,registry,plan=production_case(tmp_path,ENGLISH)
    approved=file_digest(plan)
    references=tmp_path/'references.jsonl'
    references.write_bytes(canonical(fixture_reference('Independent ocean current measurements.'))+b'\n')
    kwargs=dict(repository_root=tmp_path,reference_plan=references,
                prior_production=prior_inventory(tmp_path),expected_plan_sha256=approved,
                **approval_kwargs(tmp_path,approved))
    with ProductionBuild(tmp_path/'out',spec,registry,**kwargs) as build:
        assert build.run(plan,Tokenizer())['stages']=={'ACCEPTED':1}
        if registration is None:
            build.db.execute("DELETE FROM meta WHERE key='plan_registered'")
        else:
            build.put('plan_registered',registration)
        build.db.commit()
    # Opening existing state must not silently restore missing registration.
    with ProductionBuild(tmp_path/'out',spec,registry,**kwargs) as build:
        assert build.get('plan_registered')==registration
        with pytest.raises(SpecificationError,match='registration'):
            build.verify()
        with pytest.raises(SpecificationError,match='registration'):
            build.export_shards(tmp_path/'out'/'fresh')
        assert not (tmp_path/'out'/'fresh'/'prepared.json').exists()
        with pytest.raises(SpecificationError,match='registration'):
            build.run(plan,Tokenizer())
        assert build.get('plan_registered')==registration


@pytest.mark.parametrize('missing', ['plan_path','plan_identity','items'])
def test_registered_production_rejects_missing_plan_evidence(tmp_path,missing):
    spec,registry,plan=production_case(tmp_path,ENGLISH)
    references=tmp_path/'references.jsonl'
    references.write_bytes(canonical(fixture_reference('Independent ocean current measurements.'))+b'\n')
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,
                         reference_plan=references,prior_production=prior_inventory(tmp_path),
                         expected_plan_sha256=file_digest(plan),**approval_kwargs(tmp_path,file_digest(plan))) as build:
        assert build.run(plan,Tokenizer())['stages']=={'ACCEPTED':1}
        if missing=='items':
            build.db.execute('DELETE FROM items')
        else:
            build.db.execute('DELETE FROM meta WHERE key=?',(missing,))
        build.db.commit()
        with pytest.raises(SpecificationError):
            build.verify()
        with pytest.raises(SpecificationError):
            build.export_shards(tmp_path/'out'/'fresh')
        assert not (tmp_path/'out'/'fresh'/'prepared.json').exists()


def test_resealed_plan_cannot_hide_behind_false_registration(tmp_path):
    spec,registry,plan=production_case(tmp_path,ENGLISH)
    approved=file_digest(plan)
    references=tmp_path/'references.jsonl'
    references.write_bytes(canonical(fixture_reference('Independent ocean current measurements.'))+b'\n')
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,
                         reference_plan=references,prior_production=prior_inventory(tmp_path),
                         expected_plan_sha256=approved,**approval_kwargs(tmp_path,approved)) as build:
        assert build.run(plan,Tokenizer())['stages']=={'ACCEPTED':1}
        original=json.loads(plan.read_bytes());item=item_key(original)
        changed=seal(dict(original,language='unknown'))
        plan.write_bytes(canonical(changed)+b'\n')
        build.db.execute('UPDATE items SET record=? WHERE item=?',(canonical(changed).decode(),item))
        build.db.execute('UPDATE dd SET provenance=? WHERE item=?',(canonical(changed).decode(),item))
        build.path('clearances',item,'.json').write_bytes(canonical(changed))
        build.put('plan_identity',file_digest(plan))
        build.put('plan_registered',False)
        build.db.commit()
        with pytest.raises(SpecificationError,match='registration'):
            build.verify()
        with pytest.raises(SpecificationError,match='registration'):
            build.export_shards(tmp_path/'out'/'fresh')
        assert not (tmp_path/'out'/'fresh'/'prepared.json').exists()
