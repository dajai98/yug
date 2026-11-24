"""Small synthetic quota and lineage probes; no production corpus is acquired."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sqlite3

import pytest

import training.production_acquisition as acquisition
from training.category_ownership import decision as ownership_decision
from training.cumulative_accounting import CorpusLedger, GENESIS
from training.production_corpus import canonical, file_digest, identity
from training.production_spec import SpecificationError, seal
from tests.test_production_acquisition import BASE, Tokenizer, approval_kwargs, prior_inventory
from tests.test_production_cross_batch import setup, run


ORBIT = ' '.join('orbit' + str(i) for i in range(90))
OTHER = ' '.join('other' + str(i) for i in range(29))
PLANET = ' '.join('planet' + str(i) for i in range(50))
PLANET_LONG = ' '.join('planet' + str(i) for i in range(90))
NOVA = ' '.join('nova' + str(i) for i in range(90))


def head(root):
    path = root / 'corpus-ledger.sqlite'
    if not path.exists():
        return GENESIS
    with sqlite3.connect(path) as db:
        return db.execute('SELECT value FROM meta WHERE key=?', ('head',)).fetchone()[0]


def same_owner(root, texts):
    spec, registry, refs, plans = setup(root, texts)
    source = registry['sources'][1]
    path = root / 'files1.json'
    f = json.loads(path.read_bytes())['files'][0]
    f['owner'] = 'code'
    f['subcategory'] = 'python'
    rationale = 'Synthetic fixture ownership; no production clearance'
    evidence = seal(dict(source_id=source['source_id'], revision=source['revision'],
        path=f['path'], sha256=f['sha256'], owner='code', subcategory='python',
        decision_type='manual_review', rationale_ref=rationale, reviewer='test fixture'))
    epath = root / 'ownership1.json'
    epath.write_bytes(canonical(evidence))
    f['ownership_decision'] = ownership_decision(
        source_id=source['source_id'], revision=source['revision'], path=f['path'],
        sha256=f['sha256'], owner='code', subcategory='python',
        decision_type='manual_review', rationale_ref=rationale,
        evidence_ref=dict(path=epath.name, sha256=file_digest(epath)))
    manifest = seal(dict(source_id=source['source_id'], revision=source['revision'], files=[f]))
    path.write_bytes(canonical(manifest))
    cpath = root / 'clear1.json'
    clearance = json.loads(cpath.read_bytes())
    clearance = seal(dict(clearance, file_manifest_identity=manifest['identity']))
    cpath.write_bytes(canonical(clearance))
    row = json.loads(plans[1].read_bytes())
    plans[1].write_bytes(canonical(seal(dict(row, owner='code', subcategory='python',
        ownership_decision=f['ownership_decision'],
        source_clearance_identity=clearance['identity']))) + b'\n')
    return spec, registry, refs, plans


def same_source(root, texts, cap):
    spec, registry, refs, plans = same_owner(root, texts)
    source = registry['sources'][0]
    source['target_cap_tokens'] = cap
    source['category_caps_tokens']['code'] = cap
    for field in ('repository_group', 'domain_group'):
        spec['group_caps'][field][source[field]] = cap
    f0 = json.loads((root / 'files0.json').read_bytes())['files'][0]
    f1 = json.loads((root / 'files1.json').read_bytes())['files'][0]
    evidence = seal(dict(source_id=source['source_id'], revision=source['revision'],
        path=f1['path'], sha256=f1['sha256'], owner='code', subcategory='python',
        decision_type='manual_review', rationale_ref='Synthetic fixture ownership; no production clearance',
        reviewer='test fixture'))
    epath = root / 'ownership1.json'
    epath.write_bytes(canonical(evidence))
    f1['ownership_decision'] = ownership_decision(
        source_id=source['source_id'], revision=source['revision'], path=f1['path'],
        sha256=f1['sha256'], owner='code', subcategory='python',
        decision_type='manual_review', rationale_ref=evidence['rationale_ref'],
        evidence_ref=dict(path=epath.name, sha256=file_digest(epath)))
    manifest = seal(dict(source_id=source['source_id'], revision=source['revision'],
                         files=[f0, f1]))
    (root / 'files0.json').write_bytes(canonical(manifest))
    cpath = root / 'clear0.json'
    clearance = json.loads(cpath.read_bytes())
    clearance = seal(dict(clearance, file_manifest_identity=manifest['identity']))
    cpath.write_bytes(canonical(clearance))
    first = json.loads(plans[0].read_bytes())
    plans[0].write_bytes(canonical(seal(dict(first,
        source_clearance_identity=clearance['identity']))) + b'\n')
    row = json.loads(plans[1].read_bytes())
    plans[1].write_bytes(canonical(seal(dict(row, source_id=source['source_id'],
        revision=source['revision'], split_group=source['split_group'],
        ownership_decision=f1['ownership_decision'],
        source_clearance_identity=clearance['identity']))) + b'\n')
    registry['sources'] = [source]
    registry = seal(registry)
    spec = seal(dict(spec, registry_identity=registry['identity']))
    return spec, registry, refs, plans


def pair(root, data, *, prior=None):
    plans = data[3]
    first = run(root, data, 'a', plans[0], prior_inventory(root))
    history = prior or prior_inventory(root, [root / 'a/state.sqlite'], name='next.json')
    second = run(root, data, 'b', plans[1], history)
    return first, second


@pytest.mark.parametrize('a,b,cap,accepted', [
    (BASE, NOVA, 1550, True),           # roughly 40% + 40%
    (ORBIT, OTHER, 1000, True),        # roughly 70% + 20%
    (ORBIT, PLANET, 1000, False),      # roughly 70% + 40%
    (ORBIT, OTHER, len(ORBIT), False), # A fills cap
    (BASE, PLANET_LONG, len(BASE) + len(PLANET_LONG), True),
    (BASE, PLANET_LONG, len(BASE) + len(PLANET_LONG) - 1, False),
])
def test_category_cumulative_boundary(tmp_path, monkeypatch, a, b, cap, accepted):
    data = same_owner(tmp_path, [a, b])
    monkeypatch.setattr(acquisition, 'QUOTAS', dict(acquisition.QUOTAS, code=cap))
    first, second = pair(tmp_path, data)
    assert first['stages'] == {'ACCEPTED': 1}
    assert second['stages'] == ({'ACCEPTED': 1} if accepted else {'QUOTA_REJECTED': 1})
    with CorpusLedger(tmp_path / 'corpus-ledger.sqlite', tmp_path, head(tmp_path)) as ledger:
        totals = ledger.reconstruction()
    assert totals['category:code'] == len(a) + (len(b) if accepted else 0)


@pytest.mark.parametrize('cap,accepted', [(1000, False), (1200, True)])
def test_source_and_repository_group_cumulative(tmp_path, cap, accepted):
    data = same_source(tmp_path, [ORBIT, PLANET], cap)
    first, second = pair(tmp_path, data)
    assert first['stages'] == {'ACCEPTED': 1}
    assert second['stages'] == ({'ACCEPTED': 1} if accepted else {'QUOTA_REJECTED': 1})
    with CorpusLedger(tmp_path / 'corpus-ledger.sqlite', tmp_path, head(tmp_path)) as ledger:
        totals = ledger.reconstruction()
    assert totals['source:gh-psf-requests'] == len(ORBIT) + (len(PLANET) if accepted else 0)
    assert totals['repository_group:gh-psf-requests'] == totals['source:gh-psf-requests']


def test_three_d_topic_and_form_caps_cannot_be_overridden(tmp_path, monkeypatch):
    data = same_owner(tmp_path, [ORBIT, PLANET])
    plans = data[3]
    for path in plans:
        row = json.loads(path.read_bytes())
        path.write_bytes(canonical(seal(dict(row,
            three_d_topic='blender_python_procedural', three_d_form='implementation'))) + b'\n')
    original = acquisition.ProductionBuild._limits
    def bounded(self, record):
        return [(kind, key, 1000 if kind in ('three_d_topic', 'three_d_form') else cap)
                for kind, key, cap in original(self, record)]
    monkeypatch.setattr(acquisition.ProductionBuild, '_limits', bounded)
    with pytest.raises(SpecificationError, match='reservation caps'):
        pair(tmp_path, data)


def test_second_genesis_stale_parent_and_omitted_history_fail(tmp_path):
    data = same_owner(tmp_path, [BASE, NOVA]);plans = data[3]
    run(tmp_path, data, 'a', plans[0], prior_inventory(tmp_path))
    with pytest.raises(SpecificationError, match='omits'):
        run(tmp_path, data, 'b', plans[1], prior_inventory(tmp_path, name='false-genesis.json'))
    history = prior_inventory(tmp_path, [tmp_path / 'a/state.sqlite'], name='next.json')
    kwargs = approval_kwargs(tmp_path, file_digest(plans[1]))
    kwargs['expected_corpus_head'] = GENESIS
    from training.production_acquisition import ProductionBuild
    with ProductionBuild(tmp_path / 'stale', data[0], data[1], repository_root=tmp_path,
        reference_plan=data[2], prior_production=history,
        expected_plan_sha256=file_digest(plans[1]), **kwargs) as build:
        with pytest.raises(SpecificationError, match='Stale'):
            build.run(plans[1], Tokenizer())
    assert run(tmp_path, data, 'c', plans[1], history)['stages'] == {'ACCEPTED': 1}


def test_prior_noncategory_tamper_and_copied_build_identity(tmp_path):
    data = same_owner(tmp_path, [BASE, NOVA]);plans=data[3]
    run(tmp_path, data, 'a', plans[0], prior_inventory(tmp_path))
    original_head = head(tmp_path)
    shutil.copytree(tmp_path / 'a', tmp_path / 'copy-a')
    assert run(tmp_path, data, 'copy-a', plans[0], prior_inventory(tmp_path))['stages'] == {'ACCEPTED': 1}
    assert head(tmp_path) == original_head
    with sqlite3.connect(tmp_path / 'a/state.sqlite') as db:
        db.execute('DELETE FROM totals WHERE kind=?', ('source',))
    history = prior_inventory(tmp_path, [tmp_path / 'a/state.sqlite'], name='tampered.json')
    with pytest.raises(SpecificationError, match='accounting differs'):
        run(tmp_path, data, 'b', plans[1], history)


def _direct_item(name, tokens):
    return dict(item=identity(dict(item=name)),
        record=dict(source_id='s',owner='code',split='train',split_group='s'),normalized_sha=None,
        token_sha=None,source_id='s',owner='code',split='train',
        repository_group='r',domain_group='d',source_locator='https://example.invalid/s',
        split_group='s',three_d_topic=None,
        three_d_form=None,tokens=tokens)


def _direct_limits(cap):
    return [('source','s',cap),('source_owner','s:code',cap),
            ('repository_group','r',cap),('domain_group','d',cap),
            ('category','code',cap)]


def test_atomic_competing_reservations_abort_and_commit_idempotency(tmp_path):
    from tests.test_prospective_ledger_authority import governed_ledger_case
    path, _, authority, evidence, limits, inventory=governed_ledger_case(tmp_path)
    plan=authority['prospective_authority']['expected_plan_sha256']
    with CorpusLedger(path,tmp_path,GENESIS):pass
    def compete(name):
        with CorpusLedger(path,tmp_path,GENESIS,**authority) as ledger:
            bid=identity(dict(build=name))
            try:
                ledger.claim(bid,plan,name,inventory)
                return name,ledger.reserve(bid,evidence,limits)
            except (SpecificationError, ValueError):
                return name,'BLOCKED'
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(compete,('a','b')))
    assert sum(value is None for _, value in results) == 1
    assert sum(value == 'BLOCKED' for _, value in results) == 1
    winner=next(name for name,value in results if value is None)
    bid=identity(dict(build=winner))
    with CorpusLedger(path,tmp_path,GENESIS,**authority) as ledger:
        ledger.abort(bid)
        with pytest.raises(SpecificationError,match='reservation'):
            ledger.commit(bid,plan,[evidence])
        other='b' if winner=='a' else 'a'
        obid=identity(dict(build=other))
        ledger.claim(obid,plan,other,inventory)
        assert ledger.reserve(obid,evidence,limits) is None
        committed=ledger.commit(obid,plan,[evidence])
        assert ledger.commit(obid,plan,[evidence])==committed
        with pytest.raises(SpecificationError,match='Only an active'):
            ledger.abort(obid)
    with CorpusLedger(path,tmp_path,committed) as ledger:
        assert ledger.head==committed


def test_freshness_failure_inside_reservation_and_commit_rolls_back(tmp_path):
    from tests.test_prospective_ledger_authority import governed_ledger_case
    path, _, authority, evidence, limits, inventory=governed_ledger_case(tmp_path)
    bid=identity(dict(build='freshness'));plan=authority['prospective_authority']['expected_plan_sha256']
    def revoked():
        raise SpecificationError('Approval revoked before effect')
    with CorpusLedger(path,tmp_path,GENESIS,**authority) as ledger:
        ledger.claim(bid,plan,'freshness',inventory)
        with pytest.raises(SpecificationError,match='revoked'):
            ledger.reserve(bid,evidence,limits,before_commit=revoked)
        assert ledger.db.execute('SELECT count(*) FROM usage').fetchone()[0]==0
        assert ledger.db.execute('SELECT count(*) FROM item_evidence').fetchone()[0]==0
        assert ledger.reserve(bid,evidence,limits) is None
        with pytest.raises(SpecificationError,match='revoked'):
            ledger.commit(bid,plan,[evidence],before_commit=revoked)
        assert ledger.db.execute('SELECT count(*) FROM commits').fetchone()[0]==0
        assert ledger.head==GENESIS
        committed=ledger.commit(bid,plan,[evidence])
        assert ledger.commit(bid,plan,[evidence])==committed


def test_ledger_entry_tamper_and_deletion_detected(tmp_path):
    from tests.test_prospective_ledger_authority import governed_ledger_case
    path, _, authority, evidence, limits, inventory=governed_ledger_case(tmp_path)
    bid=identity(dict(build='a'));plan=authority['prospective_authority']['expected_plan_sha256']
    with CorpusLedger(path,tmp_path,GENESIS,**authority) as ledger:
        ledger.claim(bid,plan,'a',inventory)
        ledger.reserve(bid,evidence,limits)
        trusted=ledger.commit(bid,plan,[evidence])
    with sqlite3.connect(path) as db:
        db.execute('UPDATE usage SET tokens=1 WHERE build_id=?',(bid,))
    with pytest.raises(SpecificationError,match='usage/evidence'):
        CorpusLedger(path,tmp_path,trusted)
    with sqlite3.connect(path) as db:
        db.execute('DELETE FROM commits')
    with pytest.raises(SpecificationError):CorpusLedger(path,tmp_path,trusted)


def test_committed_reuse_requires_current_external_head(tmp_path):
    from tests.test_prospective_ledger_authority import governed_ledger_case
    path, _, authority, evidence, limits, inventory=governed_ledger_case(tmp_path)
    bid=identity(dict(build='a'));plan=authority['prospective_authority']['expected_plan_sha256']
    with CorpusLedger(path,tmp_path,GENESIS,**authority) as ledger:
        ledger.claim(bid,plan,'a',inventory)
        ledger.reserve(bid,evidence,limits)
        ledger.commit(bid,plan,[evidence])
    with CorpusLedger(path,tmp_path,GENESIS,**authority) as ledger:
        with pytest.raises(SpecificationError,match='Stale'):
            ledger.claim(bid,plan,'a',inventory)
        with pytest.raises(SpecificationError,match='Stale'):
            ledger.require_committed(bid,plan,[evidence])
