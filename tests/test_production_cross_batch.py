"""Production history boundaries, using the existing dedup engine and fixtures."""
import json
import sqlite3
import pytest
from tests.test_production_acquisition import fixture_plan, fixture_reference, prior_inventory, Tokenizer, BASE, approval_kwargs
from training.production_acquisition import ProductionBuild
from training.production_corpus import canonical, file_digest
from training.production_spec import SpecificationError, seal


def setup(root, texts=None, splits=None):
    spec,reg,_,rows=fixture_plan(root,texts or [BASE,BASE],splits)
    refs=root/'references.jsonl'
    refs.write_bytes(canonical(fixture_reference('Independent astronomical spectra of distant stars.',
        'separate', {'source':'test'}))+b'\n')
    plans=[]
    for i,r in enumerate(rows):
        p=root/f'plan-{i}.jsonl';p.write_bytes(canonical(r)+b'\n');plans.append(p)
    return spec,reg,refs,plans


def run(root, data, name, plan, history, tokenizer=None, **historical_approval):
    spec,reg,refs,_=data
    approval=approval_kwargs(root,file_digest(plan))
    approval.update(historical_approval)
    with ProductionBuild(root/name,spec,reg,repository_root=root,reference_plan=refs,prior_production=history,
                         expected_plan_sha256=file_digest(plan),**approval) as b:
        return b.run(plan,tokenizer or Tokenizer())


@pytest.mark.parametrize('splits,second,reason',[
    (['train','train'],BASE,'prior_production_duplicate'),
    (['validation','train'],BASE,'prior_production_split_conflict'),
    (['train','validation'],BASE,'prior_production_split_conflict'),
    (['train','train'],BASE.replace('word40','edited40'),'prior_production_duplicate'),
])
def test_cross_build_duplicates_and_split_conflicts(tmp_path,splits,second,reason):
    data=setup(tmp_path,[BASE,second],splits);plans=data[3]
    first=run(tmp_path,data,'a',plans[0],prior_inventory(tmp_path))
    assert first['stages']=={'ACCEPTED':1}
    db=tmp_path/'a/state.sqlite';before=file_digest(db)
    result=run(tmp_path,data,'b',plans[1],prior_inventory(tmp_path,[db],name='next.json'))
    assert result['stages']=={'DUPLICATE':1}
    assert result['accepted_clean_train_tokens']==sum(result['category_validation_tokens'].values())==0
    with sqlite3.connect(tmp_path/'b/state.sqlite') as conn:
        assert conn.execute('SELECT reason FROM items').fetchone()[0]==reason
    assert file_digest(db)==before


def test_distinct_content_and_replay_without_fetch_encode(tmp_path):
    data=setup(tmp_path,[BASE,'Ocean tides respond to lunar gravity and coastal geography. Sea level measurements vary by season.'])
    plans=data[3];run(tmp_path,data,'a',plans[0],prior_inventory(tmp_path))
    history=prior_inventory(tmp_path,[tmp_path/'a/state.sqlite'],name='next.json')
    assert run(tmp_path,data,'b',plans[1],history)['stages']=={'ACCEPTED':1}
    class NoEncode(Tokenizer):
        def encode(self,*args):raise AssertionError('Retokenization')
    def no_fetch(*args):raise AssertionError('Fetch')
    spec,reg,refs,_=data
    with ProductionBuild(tmp_path/'b',spec,reg,repository_root=tmp_path,reference_plan=refs,prior_production=history,
                         expected_plan_sha256=file_digest(plans[1]),**approval_kwargs(tmp_path,file_digest(plans[1]))) as b:
        before=list(b.db.execute('SELECT * FROM totals'));shards=b.export_shards()
        assert b.run(plans[1],NoEncode(),fetch=no_fetch)['stages']=={'ACCEPTED':1}
        assert list(b.db.execute('SELECT * FROM totals'))==before
        assert b.export_shards()==shards


@pytest.mark.parametrize('damage',['missing','file_missing','hash','payload','incompatible','incomplete','implicit_genesis'])
def test_missing_corrupt_incompatible_history_fails_closed(tmp_path,damage):
    data=setup(tmp_path);plans=data[3]
    run(tmp_path,data,'a',plans[0],prior_inventory(tmp_path))
    db=tmp_path/'a/state.sqlite';history=prior_inventory(tmp_path,[db],name='next.json')
    if damage=='missing':history=None
    elif damage=='file_missing':history.unlink()
    elif damage=='hash':
        with sqlite3.connect(db) as c:c.execute("UPDATE meta SET value='null' WHERE key='state'")
    elif damage=='payload':next((tmp_path/'a/normalized').iterdir()).write_text('corruption')
    elif damage=='incompatible':
        with sqlite3.connect(db) as c:
            req=json.loads(c.execute("SELECT value FROM meta WHERE key='request'").fetchone()[0]);req['tokenizer']='wrong'
            c.execute("UPDATE meta SET value=? WHERE key='request'",(json.dumps(req),))
        history=prior_inventory(tmp_path,[db],name='next.json')
    else:
        r=json.loads(history.read_bytes())
        if damage=='incomplete':r['complete']=False
        else:r['builds']=[]  # Must explicitly attest genesis, not just omit history.
        history.write_bytes(canonical(seal(r)))
    with pytest.raises((SpecificationError,FileNotFoundError)):
        run(tmp_path,data,'b',plans[1],history)
    with sqlite3.connect(tmp_path/'b/state.sqlite') as c:
        assert c.execute('SELECT count(*) FROM totals').fetchone()[0]==0


def test_bound_inventory_cannot_be_removed_or_replaced(tmp_path):
    data=setup(tmp_path);plans=data[3];history=prior_inventory(tmp_path)
    run(tmp_path,data,'a',plans[0],history)
    with pytest.raises(SpecificationError,match='required on resume'):run(tmp_path,data,'a',plans[0],None)
    r=json.loads(history.read_bytes());r['note']='changed';history.write_bytes(canonical(seal(r)))
    with pytest.raises(SpecificationError,match='changed on resume'):run(tmp_path,data,'a',plans[0],history)


def test_legacy_completed_replay_but_no_unbound_admission(tmp_path):
    data=setup(tmp_path);plans=data[3]
    run(tmp_path,data,'a',plans[0],prior_inventory(tmp_path))
    # Simulate the pre-boundary metadata of a completed legacy build.
    with sqlite3.connect(tmp_path/'a/state.sqlite') as c:c.execute("DELETE FROM meta WHERE key='prior_production'")
    class NoEncode(Tokenizer):
        def encode(self,*args):raise AssertionError('Retokenization')
    assert run(tmp_path,data,'a',plans[0],None,NoEncode())['stages']=={'ACCEPTED':1}
    with pytest.raises(SpecificationError,match='requires explicit'):run(tmp_path,data,'b',plans[1],None)
