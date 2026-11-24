"""Synthetic retained-corpus import controls; no ignored artifact dependency."""
from collections import Counter
import json
from pathlib import Path
import shutil
import sqlite3

import pytest

from training.cumulative_accounting import CorpusLedger, GENESIS
from training.historical_corpus_import import reconstruct, usage_for
from training.production_acquisition import item_key
from training.production_corpus import canonical, file_digest, identity
from training.production_spec import SpecificationError, TOKENIZER, seal


def history(root, *, count=174, train_tokens=417405, validation_tokens=20616,
            duplicate_content=False, source_override=None):
    base = root / 'historical'
    base.mkdir()
    sources = []
    for split in ('train', 'validation'):
        sid = 'source-' + split
        cap = train_tokens + 100 if split == 'train' else validation_tokens + 100
        sources.append(dict(source_id=sid, revision='r1', split_group=sid,
            split=split,
            repository_group=sid, domain_group=sid, locator='https://example.invalid/' + sid,
            target_cap_tokens=cap, category_caps_tokens={'code': cap},
            allowed_categories=['code']))
    if source_override is not None:
        original=sources[0]
        sources[0]={**original, **{k: source_override[k] for k in (
            'source_id','revision','split','split_group','repository_group','domain_group',
            'target_cap_tokens','category_caps_tokens','allowed_categories')}}
        sources[0]['locator']=source_override.get('locator') or sources[0]['source_id']
    spec = seal(dict(group_caps={k: {s[k]: s['target_cap_tokens'] for s in sources}
                        for k in ('repository_group', 'domain_group')},
                     three_d={'topics': [], 'form_targets': {}}))
    registry = seal(dict(sources=sources))
    spec_path = base / 'spec.json';spec_path.write_bytes(canonical(spec))
    reg_path = base / 'registry.json';reg_path.write_bytes(canonical(registry))
    license_path=base/'LICENSE';license_path.write_text('Synthetic fixture license evidence',encoding='utf-8')
    license_ref=dict(path=license_path.relative_to(root).as_posix(),sha256=file_digest(license_path))
    build = base / 'build';build.mkdir()
    for sub in ('clearances', 'raw', 'normalized', 'tokens'):(build / sub).mkdir()
    records = []
    usages = Counter()
    train_count = count - 1
    for i in range(count):
        split = 'validation' if i == count - 1 else 'train'
        source = sources[1 if split == 'validation' else 0]
        raw = (('same historical content' if duplicate_content else
                'historical item ' + str(i) + '.')).encode()
        n = validation_tokens if split == 'validation' else train_tokens // train_count
        if i == 0:n += train_tokens % train_count
        payload = root / ('raw-' + str(i));payload.write_bytes(raw)
        record = seal(dict(source_id=source['source_id'], revision=source['revision'], path='doc/' + str(i),
            sha256=file_digest(payload), owner='code', split=split,
            split_group=source['split_group'], content_type='prose',language='en',
            license_evidence=[license_ref]))
        item = item_key(record);suffix=item[7:]
        (build/'clearances'/(suffix+'.json')).write_bytes(canonical(record))
        (build/'raw'/suffix).write_bytes(raw)
        (build/'normalized'/suffix).write_bytes(raw)
        tok = build/'tokens'/(suffix+'.bin');tok.write_bytes(b'\0\0\0\0'*n)
        records.append((item,record,file_digest(build/'normalized'/suffix),file_digest(tok),n))
        evidence=dict(item=item,record=record,normalized_sha=records[-1][2],
            token_sha=records[-1][3],tokens=n,source_id=source['source_id'],
            owner='code',split=split,repository_group=source['repository_group'],
            domain_group=source['domain_group'],split_group=source['split_group'],
            source_locator=source['locator'],three_d_topic=None,three_d_form=None)
        for kind,key,tokens,_ in usage_for(evidence,spec,source):usages[(kind,key)]+=tokens
        payload.unlink()
    plan = base/'plan.jsonl';plan.write_bytes(b''.join(canonical(r)+b'\n' for _,r,_,_,_ in records))
    state = build/'state.sqlite'
    with sqlite3.connect(state) as db:
        db.executescript('CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT);'
                         'CREATE TABLE items(item TEXT PRIMARY KEY,record TEXT,stage TEXT,'
                         'normalized_sha TEXT,token_sha TEXT,token_count INTEGER,split TEXT,'
                         'owner TEXT,source TEXT);'
                         'CREATE TABLE totals(kind TEXT,key TEXT,tokens INTEGER,PRIMARY KEY(kind,key));')
        from training.production_dedup import POLICY as DEDUP_POLICY
        req=dict(diagnostic=False,tokenizer=TOKENIZER,spec=spec['identity'],
                 registry=registry['identity'],dedup=identity(DEDUP_POLICY))
        for k,v in dict(request=req,plan_identity=file_digest(plan),state='COMPLETE',
                        dedup_settled=True).items():
            db.execute('INSERT INTO meta VALUES (?,?)',(k,json.dumps(v)))
        for item,r,nsha,tsha,n in records:
            db.execute('INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?)',
                       (item,canonical(r).decode(),'ACCEPTED',nsha,tsha,n,r['split'],r['owner'],r['source_id']))
        for (kind,key),n in usages.items():db.execute('INSERT INTO totals VALUES (?,?,?)',(kind,key,n))
    entries = {k: dict(path=p.relative_to(root).as_posix(),sha256=file_digest(p))
               for k,p in [('state',state),('plan',plan),('spec',spec_path),('registry',reg_path)]}
    manifest = root/'import.json'
    manifest.write_bytes(canonical(seal(dict(schema='yug-historical-corpus-import-v1',builds=[entries]))))
    return manifest,state,records


def approval_pins(manifest, candidate):
    return dict(approved_historical_import_identity=candidate['identity'],
        approved_historical_manifest_identity=json.loads(manifest.read_bytes())['identity'],
        approved_historical_manifest_sha256=file_digest(manifest))


def governance_pins(manifest, candidate):
    from training.approval_governance import event
    record=event(approval_type='historical_import',
        approval_identity=candidate['identity'],
        scope=dict(manifest_identity=json.loads(manifest.read_bytes())['identity'],
                   manifest_sha256=file_digest(manifest)),
        previous_status='NONE',new_status='ACTIVE',
        transition_at='2026-10-27T00:00:00Z',reason_ref='synthetic import approval')
    path=manifest.parent/'historical-governance.jsonl'
    path.write_bytes(canonical(record)+b'\n')
    return dict(governance_log=path,expected_governance_sha256=file_digest(path))


def small_approved_history(tmp_path, monkeypatch):
    import training.historical_corpus_import as historical
    monkeypatch.setattr(historical,'EXPECTED_DOCUMENTS',2)
    monkeypatch.setattr(historical,'EXPECTED_TRAIN_TOKENS',80)
    monkeypatch.setattr(historical,'EXPECTED_VALIDATION_TOKENS',20)
    manifest,_,_=history(tmp_path,count=2,train_tokens=80,validation_tokens=20)
    candidate=reconstruct(tmp_path,manifest,preview=True)
    pins=approval_pins(manifest,candidate)
    path=tmp_path/'ledger.sqlite'
    with CorpusLedger(path,tmp_path,GENESIS,**pins,
                      **governance_pins(manifest,candidate)) as ledger:
        ledger.import_historical(manifest,candidate['identity'])
        head=ledger.head
        assert ledger.reconstruction()['category:code']==80
    return manifest,path,head,pins


def test_uncommitted_historical_genesis_requires_current_governance(tmp_path,monkeypatch):
    import training.historical_corpus_import as historical
    monkeypatch.setattr(historical,'EXPECTED_DOCUMENTS',2)
    monkeypatch.setattr(historical,'EXPECTED_TRAIN_TOKENS',80)
    monkeypatch.setattr(historical,'EXPECTED_VALIDATION_TOKENS',20)
    manifest,_,_=history(tmp_path,count=2,train_tokens=80,validation_tokens=20)
    candidate=reconstruct(tmp_path,manifest,preview=True)
    pins=approval_pins(manifest,candidate)
    with CorpusLedger(tmp_path/'new-ledger.sqlite',tmp_path,GENESIS,**pins) as ledger:
        with pytest.raises(SpecificationError,match='governance pin'):
            ledger.import_historical(manifest,candidate['identity'])
        assert ledger.db.execute('SELECT count(*) FROM commits').fetchone()[0]==0


def test_historical_governance_revoked_during_import_rolls_back(tmp_path,monkeypatch):
    import training.historical_corpus_import as historical
    import training.approval_governance as governance
    monkeypatch.setattr(historical,'EXPECTED_DOCUMENTS',2)
    monkeypatch.setattr(historical,'EXPECTED_TRAIN_TOKENS',80)
    monkeypatch.setattr(historical,'EXPECTED_VALIDATION_TOKENS',20)
    manifest,_,_=history(tmp_path,count=2,train_tokens=80,validation_tokens=20)
    candidate=reconstruct(tmp_path,manifest,preview=True)
    pins=approval_pins(manifest,candidate)
    status=governance_pins(manifest,candidate)
    original=governance.Governance.require_current
    calls=[]
    def revoke_on_final_check(self,*args,**kwargs):
        calls.append(True)
        if len(calls)==2:
            raise SpecificationError('Approval revoked before genesis commit')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(governance.Governance,'require_current',revoke_on_final_check)
    with CorpusLedger(tmp_path/'ledger.sqlite',tmp_path,GENESIS,**pins,**status) as ledger:
        with pytest.raises(SpecificationError,match='revoked'):
            ledger.import_historical(manifest,candidate['identity'])
        assert len(calls)==2
        assert ledger.db.execute('SELECT count(*) FROM commits').fetchone()[0]==0
        assert ledger.db.execute('SELECT count(*) FROM item_evidence').fetchone()[0]==0
        assert ledger.head==GENESIS


def test_unpinned_reconstruction_requires_explicit_preview(tmp_path,monkeypatch):
    manifest,_,_,pins=small_approved_history(tmp_path,monkeypatch)
    with pytest.raises(SpecificationError,match='pins required'):
        reconstruct(tmp_path,manifest)
    assert reconstruct(tmp_path,manifest,preview=True)['identity']==pins['approved_historical_import_identity']


@pytest.mark.parametrize('damaged',('missing','missing_one','import','manifest_identity',
                                    'manifest_sha256'))
def test_historical_reconstruction_requires_external_manifest_pins(tmp_path,monkeypatch,damaged):
    manifest,path,head,pins=small_approved_history(tmp_path,monkeypatch)
    pins=dict(pins)
    if damaged=='missing':pins.clear()
    elif damaged=='missing_one':pins.pop('approved_historical_manifest_sha256')
    else:pins['approved_historical_'+damaged+('_identity' if damaged=='import' else '')]=GENESIS
    with pytest.raises(SpecificationError):
        CorpusLedger(path,tmp_path,head,**pins)


@pytest.mark.parametrize('change',('resealed','whitespace','added_build','omitted_build',
                                   'substituted_copy'))
def test_historical_manifest_substitution_rejected(tmp_path,monkeypatch,change):
    manifest,path,head,pins=small_approved_history(tmp_path,monkeypatch)
    data=json.loads(manifest.read_bytes())
    target=manifest
    if change=='resealed':target.write_bytes(canonical(seal(dict(data,note='unapproved'))))
    elif change=='whitespace':target.write_text(json.dumps(data,indent=2),encoding='utf-8')
    elif change=='added_build':
        target.write_bytes(canonical(seal(dict(data,builds=data['builds']*2))))
    elif change=='omitted_build':target.write_bytes(canonical(seal(dict(data,builds=[]))))
    else:
        target=tmp_path/'substituted.json'
        target.write_bytes(canonical(seal(dict(data,note='copied and resealed'))))
        with sqlite3.connect(path) as db:
            db.execute('UPDATE builds SET path_hint=?',(target.name,))
    with pytest.raises(SpecificationError,match='manifest'):
        CorpusLedger(path,tmp_path,head,**pins)


@pytest.mark.parametrize('change',('bytes','resealed','identity','wrong_sha','missing_pin',
                                   'wrong_import','omitted_build','added_build'))
def test_post_claim_manifest_change_cannot_reserve(tmp_path,monkeypatch,change):
    from tests.test_prospective_ledger_authority import governed_ledger_case
    manifest,path,head,pins=small_approved_history(tmp_path,monkeypatch)
    _,_,authority,evidence,limits,_=governed_ledger_case(tmp_path)
    state=tmp_path/'historical'/'build'/'state.sqlite'
    candidate=reconstruct(tmp_path,manifest,
        approved_manifest_identity=pins['approved_historical_manifest_identity'],
        approved_manifest_sha256=pins['approved_historical_manifest_sha256'])
    inventory=seal(dict(schema='yug-prior-production-v1',complete=True,genesis=False,
        builds=[dict(path=state.relative_to(tmp_path).as_posix(),sha256=file_digest(state))]))
    bid=identity(dict(build='post-claim'))
    plan=authority['prospective_authority']['expected_plan_sha256']
    original=manifest.read_bytes()
    with CorpusLedger(path,tmp_path,head,**pins,**authority) as ledger:
        assert ledger.claim(bid,plan,'new',inventory)=='ACTIVE'
        before_usage=list(ledger.db.execute('SELECT * FROM usage'))
        before_totals=list(ledger.db.execute('SELECT * FROM totals'))
        before_items=list(ledger.db.execute('SELECT * FROM item_evidence'))
        if change=='bytes':manifest.write_bytes(original+b'\n')
        elif change=='wrong_sha':ledger.approved_historical_manifest_sha256=GENESIS
        elif change=='missing_pin':ledger.approved_historical_manifest_identity=None
        elif change=='wrong_import':ledger.approved_historical_import_identity=GENESIS
        else:
            data=json.loads(original)
            if change=='resealed':data=seal(dict(data,note='changed after claim'))
            elif change=='identity':data['identity']=GENESIS
            elif change=='omitted_build':data=seal(dict(data,builds=[]))
            elif change=='added_build':data=seal(dict(data,builds=data['builds']*2))
            manifest.write_bytes(canonical(data))
        with pytest.raises(SpecificationError):ledger.reserve(bid,evidence,limits)
        assert list(ledger.db.execute('SELECT * FROM usage'))==before_usage
        assert list(ledger.db.execute('SELECT * FROM totals'))==before_totals
        assert list(ledger.db.execute('SELECT * FROM item_evidence'))==before_items
        assert ledger.db.execute("SELECT count(*) FROM builds WHERE state='ACTIVE'").fetchone()[0]==1
        assert ledger.head==head
        manifest.write_bytes(original)
        ledger.approved_historical_import_identity=pins['approved_historical_import_identity']
        ledger.approved_historical_manifest_identity=pins['approved_historical_manifest_identity']
        ledger.approved_historical_manifest_sha256=pins['approved_historical_manifest_sha256']
        assert ledger.reserve(bid,evidence,limits) is None
        after_retry=list(ledger.db.execute('SELECT * FROM usage'))
        assert ledger.reserve(bid,evidence,limits) is None
        assert list(ledger.db.execute('SELECT * FROM usage'))==after_retry


def test_manifest_change_during_reservation_rolls_back(tmp_path,monkeypatch):
    from tests.test_prospective_ledger_authority import governed_ledger_case
    manifest,path,head,pins=small_approved_history(tmp_path,monkeypatch)
    _,_,authority,evidence,limits,_=governed_ledger_case(tmp_path)
    state=tmp_path/'historical'/'build'/'state.sqlite'
    candidate=reconstruct(tmp_path,manifest,
        approved_manifest_identity=pins['approved_historical_manifest_identity'],
        approved_manifest_sha256=pins['approved_historical_manifest_sha256'])
    inventory=seal(dict(schema='yug-prior-production-v1',complete=True,genesis=False,
        builds=[dict(path=state.relative_to(tmp_path).as_posix(),sha256=file_digest(state))]))
    bid=identity(dict(build='during-reserve'))
    plan=authority['prospective_authority']['expected_plan_sha256']
    with CorpusLedger(path,tmp_path,head,**pins,**authority) as ledger:
        assert ledger.claim(bid,plan,'new',inventory)=='ACTIVE'
        before_usage=list(ledger.db.execute('SELECT * FROM usage'))
        original_check=ledger._verify_historical_manifest_binding
        calls=[]
        def change_before_second_check(*args):
            calls.append(args)
            if len(calls)==2:
                manifest.write_bytes(canonical(seal(dict(json.loads(manifest.read_bytes()),
                                                         note='changed during reserve'))))
            return original_check(*args)
        monkeypatch.setattr(ledger,'_verify_historical_manifest_binding',change_before_second_check)
        with pytest.raises(SpecificationError,match='manifest'):
            ledger.reserve(bid,evidence,limits)
        assert len(calls)==2
        assert list(ledger.db.execute('SELECT * FROM usage'))==before_usage
        assert not ledger.db.execute('SELECT 1 FROM item_evidence WHERE build_id=?',(bid,)).fetchone()
        assert ledger.head==head


def test_wrong_manifest_approval_cannot_create_genesis(tmp_path,monkeypatch):
    import training.historical_corpus_import as historical
    monkeypatch.setattr(historical,'EXPECTED_DOCUMENTS',2)
    monkeypatch.setattr(historical,'EXPECTED_TRAIN_TOKENS',80)
    monkeypatch.setattr(historical,'EXPECTED_VALIDATION_TOKENS',20)
    manifest,_,_=history(tmp_path,count=2,train_tokens=80,validation_tokens=20)
    candidate=reconstruct(tmp_path,manifest,preview=True)
    pins=approval_pins(manifest,candidate)
    pins['approved_historical_manifest_sha256']=GENESIS
    path=tmp_path/'ledger.sqlite'
    with CorpusLedger(path,tmp_path,GENESIS,**pins,
                      **governance_pins(manifest,candidate)) as ledger:
        with pytest.raises(SpecificationError,match='manifest'):
            ledger.import_historical(manifest,candidate['identity'])
        assert ledger.head==GENESIS
        assert ledger.db.execute('SELECT count(*) FROM commits').fetchone()[0]==0


def test_complete_174_import_preserves_history_and_rejects_unapproved_continuation(tmp_path):
    manifest,state,records=history(tmp_path)
    candidate=reconstruct(tmp_path,manifest,preview=True)
    assert len(candidate['items'])==174
    assert (candidate['proposal']['training_tokens'],candidate['proposal']['validation_tokens'])==(417405,20616)
    path=tmp_path/'ledger.sqlite'
    with CorpusLedger(path,tmp_path,GENESIS,**approval_pins(manifest,candidate),
                      **governance_pins(manifest,candidate)) as ledger:
        with pytest.raises(SpecificationError,match='approved'):
            ledger.import_historical(manifest,GENESIS)
        result=ledger.import_historical(manifest,candidate['identity'])
        assert result['accepted_documents']==174
        approved_head=result['head']
        with pytest.raises(SpecificationError,match='genesis'):
            ledger.import_historical(manifest,candidate['identity'])
        copied=tmp_path/'copied-import.json';shutil.copyfile(manifest,copied)
        with pytest.raises(SpecificationError,match='genesis'):
            ledger.import_historical(copied,candidate['identity'])
        assert ledger.reconstruction()['source:source-train']==417405
        bid=identity(dict(build='continuation'));plan=identity(dict(plan='continuation'))
        from training.production_spec import seal as sealed
        inventory=sealed(dict(schema='yug-prior-production-v1',complete=True,genesis=False,
                              builds=[dict(path=state.relative_to(tmp_path).as_posix(),sha256=file_digest(state))]))
        with pytest.raises(SpecificationError, match='external authority'):
            ledger.claim(bid,plan,'new',inventory)
        assert ledger.head==approved_head


def test_historical_manifest_and_aggregate_tamper_fail_closed(tmp_path,monkeypatch):
    import training.historical_corpus_import as historical
    monkeypatch.setattr(historical,'EXPECTED_DOCUMENTS',2)
    monkeypatch.setattr(historical,'EXPECTED_TRAIN_TOKENS',80)
    monkeypatch.setattr(historical,'EXPECTED_VALIDATION_TOKENS',20)
    manifest,state,records=history(tmp_path,count=2,train_tokens=80,validation_tokens=20)
    candidate=reconstruct(tmp_path,manifest,preview=True)
    assert candidate['proposal']['accepted_documents']==2
    with sqlite3.connect(state) as db:
        db.execute("DELETE FROM totals WHERE kind='source'")
    data=json.loads(manifest.read_bytes());data['builds'][0]['state']['sha256']=file_digest(state)
    manifest.write_bytes(canonical(seal(data)))
    with pytest.raises(SpecificationError,match='aggregate totals'):
        reconstruct(tmp_path,manifest,preview=True)


@pytest.mark.parametrize('kind',('source','repository_group','domain_group'))
def test_historical_quota_dimension_cannot_be_deleted(tmp_path,monkeypatch,kind):
    import training.historical_corpus_import as historical
    monkeypatch.setattr(historical,'EXPECTED_DOCUMENTS',2)
    monkeypatch.setattr(historical,'EXPECTED_TRAIN_TOKENS',80)
    monkeypatch.setattr(historical,'EXPECTED_VALIDATION_TOKENS',20)
    manifest,state,_=history(tmp_path,count=2,train_tokens=80,validation_tokens=20)
    with sqlite3.connect(state) as db:db.execute('DELETE FROM totals WHERE kind=?',(kind,))
    data=json.loads(manifest.read_bytes());data['builds'][0]['state']['sha256']=file_digest(state)
    manifest.write_bytes(canonical(seal(data)))
    with pytest.raises(SpecificationError,match='aggregate totals'):reconstruct(tmp_path,manifest,preview=True)


def test_missing_extra_and_duplicate_historical_items(tmp_path,monkeypatch):
    import training.historical_corpus_import as historical
    monkeypatch.setattr(historical,'EXPECTED_DOCUMENTS',2)
    monkeypatch.setattr(historical,'EXPECTED_TRAIN_TOKENS',80)
    monkeypatch.setattr(historical,'EXPECTED_VALIDATION_TOKENS',20)
    missing=tmp_path/'missing';missing.mkdir()
    manifest,state,records=history(missing,count=2,train_tokens=80,validation_tokens=20)
    with sqlite3.connect(state) as db:
        db.execute('DELETE FROM items WHERE item=?',(records[0][0],))
    data=json.loads(manifest.read_bytes());data['builds'][0]['state']['sha256']=file_digest(state)
    manifest.write_bytes(canonical(seal(data)))
    with pytest.raises(SpecificationError):reconstruct(missing,manifest,preview=True)
    extra=tmp_path/'extra';extra.mkdir()
    manifest,_,_=history(extra,count=3,train_tokens=80,validation_tokens=20)
    with pytest.raises(SpecificationError,match='count incomplete'):reconstruct(extra,manifest,preview=True)
    duplicated=tmp_path/'duplicate';duplicated.mkdir()
    manifest,_,_=history(duplicated,count=2,train_tokens=80,validation_tokens=20,
                         duplicate_content=True)
    with pytest.raises(SpecificationError,match='Duplicate historical accepted'):
        reconstruct(duplicated,manifest,preview=True)


@pytest.mark.parametrize('field',('token_count','owner','split'))
def test_historical_item_tamper_fails(tmp_path,monkeypatch,field):
    import training.historical_corpus_import as historical
    monkeypatch.setattr(historical,'EXPECTED_DOCUMENTS',2)
    monkeypatch.setattr(historical,'EXPECTED_TRAIN_TOKENS',80)
    monkeypatch.setattr(historical,'EXPECTED_VALIDATION_TOKENS',20)
    manifest,state,records=history(tmp_path,count=2,train_tokens=80,validation_tokens=20)
    with sqlite3.connect(state) as db:
        value=81 if field=='token_count' else ('security' if field=='owner' else 'validation')
        db.execute(f'UPDATE items SET {field}=? WHERE item=?',(value,records[0][0]))
    data=json.loads(manifest.read_bytes());data['builds'][0]['state']['sha256']=file_digest(state)
    manifest.write_bytes(canonical(seal(data)))
    with pytest.raises(SpecificationError):reconstruct(tmp_path,manifest,preview=True)


def test_production_build_extends_imported_usage(tmp_path,monkeypatch):
    import training.historical_corpus_import as historical
    from tests.test_cumulative_accounting import ORBIT, OTHER, same_source, head
    from tests.test_production_acquisition import prior_inventory
    from tests.test_production_cross_batch import run

    monkeypatch.setattr(historical,'EXPECTED_DOCUMENTS',2)
    monkeypatch.setattr(historical,'EXPECTED_TRAIN_TOKENS',80)
    monkeypatch.setattr(historical,'EXPECTED_VALIDATION_TOKENS',20)
    cap=80+len(ORBIT)
    data=same_source(tmp_path,[ORBIT,OTHER],cap)
    manifest,state,_=history(tmp_path,count=2,train_tokens=80,validation_tokens=20,
                             source_override=data[1]['sources'][0])
    candidate=reconstruct(tmp_path,manifest,preview=True)
    pins=approval_pins(manifest,candidate)
    with CorpusLedger(tmp_path/'corpus-ledger.sqlite',tmp_path,GENESIS,**pins,
                      **governance_pins(manifest,candidate)) as ledger:
        ledger.import_historical(manifest,candidate['identity'])
    first=run(tmp_path,data,'new-a',data[3][0],
              prior_inventory(tmp_path,[state],name='prior-historical.json'),**pins)
    assert first['stages']=={'ACCEPTED':1}
    assert first['accepted_clean_train_tokens']==len(ORBIT)
    assert head(tmp_path)!=GENESIS
    second=run(tmp_path,data,'new-b',data[3][1],
               prior_inventory(tmp_path,[state,tmp_path/'new-a/state.sqlite'],
                               name='prior-historical-and-new.json'),**pins)
    assert second['stages']=={'QUOTA_REJECTED':1}
