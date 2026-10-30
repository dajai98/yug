"""Disposable approval-status history; no real approval or corpus is created."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import training.corpus_gate as corpus_gate
import training.production_acquisition as acquisition
from training.approval_governance import Governance, event, append_event
from training.cumulative_accounting import CorpusLedger
from training.production_corpus import canonical, file_digest
from training.production_spec import SpecificationError, seal
from tests.test_corpus_gate import setup, bundle, builder
from tests.test_production_acquisition import Tokenizer, approval_kwargs, prior_inventory

A = 'sha256:' + 'a' * 64
B = 'sha256:' + 'b' * 64
C = 'sha256:' + 'c' * 64
SCOPE = dict(source_id='source-one', revision='1' * 40)
OTHER = dict(source_id='source-two', revision='2' * 40)
WHEN = '2026-10-27T00:00:00Z'


def row(previous=None, *, aid=A, kind='source_clearance', scope=SCOPE,
        before='NONE', after='ACTIVE', replacement=None, replacement_scope=None,
        when=WHEN):
    return event(approval_identity=aid, approval_type=kind, scope=scope,
                 previous_status=before, new_status=after,
                 replaced_by=replacement, replacement_scope=replacement_scope,
                 transition_at=when, reason_ref='fixture decision',
                 previous_event_identity=previous['identity'] if previous else None)


def log(tmp_path, *events):
    path = tmp_path / 'governance.jsonl'
    path.write_bytes(b''.join(canonical(e) + b'\n' for e in events))
    return path, file_digest(path)


def test_active_revoked_and_historical_trace(tmp_path):
    first = row()
    path, pin = log(tmp_path, first)
    assert Governance(path, pin).require_current('source_clearance', A, SCOPE)['status'] == 'ACTIVE'
    revoked = row(first, before='ACTIVE', after='REVOKED')
    with path.open('ab') as output: output.write(canonical(revoked) + b'\n')
    history = Governance(path, file_digest(path))
    assert [e['identity'] for e in history.events] == [first['identity'], revoked['identity']]
    assert history.states[A]['status'] == 'REVOKED'
    with pytest.raises(SpecificationError): history.require_current('source_clearance', A, SCOPE)
    with pytest.raises(SpecificationError, match='stale or changed'):
        Governance(path, pin)


def test_append_requires_current_pin_and_rejects_duplicate_transition(tmp_path):
    path = tmp_path / 'governance.jsonl'
    first = row()
    pin = append_event(path, None, first)
    revoked = row(first, before='ACTIVE', after='REVOKED')
    current = append_event(path, pin, revoked)
    assert Governance(path, current).states[A]['status'] == 'REVOKED'
    before = path.read_bytes()
    with pytest.raises(SpecificationError): append_event(path, pin, revoked)
    with pytest.raises(SpecificationError): append_event(path, current, revoked)
    assert path.read_bytes() == before


def test_supersession_replacement_and_exclusive_scope(tmp_path):
    first = row()
    replacement = row(first, before='ACTIVE', after='SUPERSEDED', replacement=B,
                      replacement_scope=SCOPE)
    path, pin = log(tmp_path, first, replacement)
    history = Governance(path, pin)
    assert history.states[A] == dict(type='source_clearance', scope=SCOPE,
                                     status='SUPERSEDED', replaced_by=B)
    assert history.require_current('source_clearance', B, SCOPE)['status'] == 'ACTIVE'
    with pytest.raises(SpecificationError): history.require_current('source_clearance', A, SCOPE)
    with pytest.raises(SpecificationError): history.require_current('source_clearance', B, OTHER)
    with pytest.raises(SpecificationError): event(approval_identity=A,
        approval_type='source_clearance', scope=SCOPE, previous_status='ACTIVE',
        new_status='SUPERSEDED', replaced_by=C, replacement_scope=OTHER,
        transition_at=WHEN, reason_ref='wrong scope',
        previous_event_identity=first['identity'])


@pytest.mark.parametrize('bad', ['two_active', 'unknown', 'double_revoke',
                                 'double_supersede', 'order', 'duplicate', 'forged',
                                 'malformed', 'wrong_policy'])
def test_conflicts_and_forgery_fail_closed(tmp_path, bad):
    first = row()
    second = row(first, aid=B)
    revoke = row(first, before='ACTIVE', after='REVOKED')
    supersede = row(first, before='ACTIVE', after='SUPERSEDED', replacement=B,
                    replacement_scope=SCOPE)
    if bad == 'two_active': events = [first, second]
    elif bad == 'unknown': events = [row(None, aid=B, before='ACTIVE', after='REVOKED')]
    elif bad == 'double_revoke': events = [first, revoke, row(revoke, before='ACTIVE', after='REVOKED')]
    elif bad == 'double_supersede':
        events = [first, supersede, row(supersede, before='ACTIVE', after='SUPERSEDED',
                                       replacement=C, replacement_scope=SCOPE)]
    elif bad == 'order': events = [second, first]
    elif bad == 'duplicate': events = [first, first]
    elif bad == 'forged': events = [first, dict(revoke, reason_ref='forged')]
    elif bad == 'wrong_policy': events = [first, seal(dict(revoke, policy='other'))]
    else:
        events = [first]
    path, pin = log(tmp_path, *events)
    if bad == 'malformed':
        path.write_bytes(path.read_bytes().replace(b'"approval_type":',
                     b'"approval_type":"wrong","approval_type":', 1))
        pin = file_digest(path)
    with pytest.raises((SpecificationError, ValueError)):
        Governance(path, pin)


def test_revocation_blocks_future_admission_without_changing_approved_plan(tmp_path):
    data = setup(tmp_path)
    with builder(tmp_path, data) as build:
        history = Governance(build.governance_log, build.expected_governance_sha256)
        plan_id = file_digest(data[3][0])
        old = history.current[('production_plan', canonical(dict(plan_sha256=plan_id)))]
        transition = event(approval_identity=old, approval_type='production_plan',
            scope=dict(plan_sha256=plan_id), previous_status='ACTIVE',
            new_status='REVOKED', transition_at=WHEN, reason_ref='fixture revocation',
            previous_event_identity=history.head)
        with build.governance_log.open('ab') as output:
            output.write(canonical(transition) + b'\n')
        build.expected_governance_sha256 = file_digest(build.governance_log)
        with pytest.raises(SpecificationError, match='revoked'):
            build.run(data[3][0], Tokenizer())
        assert build.db.execute('SELECT count(*) FROM items').fetchone()[0] == 0


def test_historical_verification_survives_receipt_revocation(tmp_path):
    data = setup(tmp_path)
    decisions = bundle(tmp_path, data)
    with builder(tmp_path, data) as build:
        build.run(data[3][0], Tokenizer(), intended_use='commercial_training',
                  decision_receipts=decisions)
        out = tmp_path / 'build' / 'export'
        build.export_shards(out, intended_use='commercial_training',
                            decision_receipts=decisions)
        assert list(build.replay_for_use(out, split='train',
                    intended_use='commercial_training', decision_receipts=decisions))
        receipt = json.loads(decisions.read_bytes())['receipts'][0]
        history = Governance(build.governance_log, build.expected_governance_sha256)
        transition = event(approval_identity=receipt['identity'],
            approval_type='decision_receipt',
            scope=dict(clearance_identity=receipt['binding']['clearance_identity']),
            previous_status='ACTIVE', new_status='REVOKED', transition_at=WHEN,
            reason_ref='fixture revocation', previous_event_identity=history.head)
        with build.governance_log.open('ab') as output:
            output.write(canonical(transition) + b'\n')
        build.expected_governance_sha256 = file_digest(build.governance_log)
        build.verify()  # Historical evidence remains valid.
        with pytest.raises(SpecificationError, match='revoked'):
            build.replay_for_use(out, split='train',
                intended_use='commercial_training', decision_receipts=decisions)


@pytest.mark.parametrize('kind', ['ownership_approval', 'source_clearance',
                                  'file_clearance'])
def test_other_approval_revocation_blocks_fresh_export(tmp_path, kind):
    data = setup(tmp_path)
    with builder(tmp_path, data) as build:
        build.run(data[3][0], Tokenizer())
        history = Governance(build.governance_log, build.expected_governance_sha256)
        (approval_type, encoded_scope), aid = next(
            (key, value) for key, value in history.current.items() if key[0] == kind)
        scope = json.loads(encoded_scope)
        transition = event(approval_identity=aid, approval_type=approval_type,
            scope=scope, previous_status='ACTIVE', new_status='REVOKED',
            transition_at=WHEN, reason_ref='fixture revocation',
            previous_event_identity=history.head)
        with build.governance_log.open('ab') as output:
            output.write(canonical(transition) + b'\n')
        build.expected_governance_sha256 = file_digest(build.governance_log)
        build.verify()
        destination = tmp_path / 'build' / 'fresh-export'
        with pytest.raises(SpecificationError, match='revoked'):
            build.export_shards(destination)
        assert not (destination / 'prepared.json').exists()


def _revoke(build, kind, aid, scope, *, update_pin=True):
    history=Governance(build.governance_log,build.expected_governance_sha256)
    transition=event(approval_identity=aid,approval_type=kind,scope=scope,
        previous_status='ACTIVE',new_status='REVOKED',transition_at=WHEN,
        reason_ref='disposable race probe',previous_event_identity=history.head)
    with build.governance_log.open('ab') as output:
        output.write(canonical(transition)+b'\n')
    if update_pin:build.expected_governance_sha256=file_digest(build.governance_log)


def test_governance_hash_and_parser_use_one_byte_snapshot(tmp_path,monkeypatch):
    first=row()
    revoked=row(first,before='ACTIVE',after='REVOKED')
    path,pin=log(tmp_path,first,revoked)
    active=canonical(first)+b'\n'
    original_open=Path.open
    reads=[]

    class ChangingReader:
        def __init__(self,handle):self.handle=handle
        def __enter__(self):self.handle.__enter__();return self
        def __exit__(self,*args):return self.handle.__exit__(*args)
        def read(self,*args):
            snapshot=self.handle.read(*args)
            reads.append(snapshot)
            with original_open(path,'wb') as output:output.write(active)
            return snapshot

    def changed_open(self,*args,**kwargs):
        handle=original_open(self,*args,**kwargs)
        return ChangingReader(handle) if self==path and args and args[0]=='rb' else handle

    monkeypatch.setattr(Path,'open',changed_open)
    history=Governance(path,pin)
    assert reads==[canonical(first)+b'\n'+canonical(revoked)+b'\n']
    assert history.byte_sha256==pin and history.states[A]['status']=='REVOKED'
    with pytest.raises(SpecificationError,match='revoked'):
        history.require_current('source_clearance',A,SCOPE)


def test_revocation_during_fetch_blocks_accounting_then_retries_once(tmp_path):
    data=setup(tmp_path)
    with builder(tmp_path,data) as build:
        plan=data[3][0]
        plan_id=file_digest(plan)
        active=build.governance_log.read_bytes()
        def fetch(record):
            _revoke(build,'production_plan',plan_id,dict(plan_sha256=plan_id))
            yield (tmp_path/record['input_path']).read_bytes()
        with pytest.raises(SpecificationError,match='revoked'):
            build.run(plan,Tokenizer(),fetch=fetch)
        assert build.db.execute("SELECT count(*) FROM items WHERE stage='ACCEPTED'").fetchone()[0]==0
        assert build.db.execute('SELECT coalesce(sum(tokens),0) FROM totals').fetchone()[0]==0
        if build.corpus_ledger is not None:
            assert build.corpus_ledger.db.execute('SELECT count(*) FROM usage').fetchone()[0]==0
        # Restore only disposable test state; no production status history is rewritten.
        build.governance_log.write_bytes(active)
        build.expected_governance_sha256=file_digest(build.governance_log)
        result=build.run(plan,Tokenizer(),fetch=lambda _:pytest.fail('unexpected refetch'))
        assert result['accepted_clean_train_tokens']==619
        assert build.db.execute("SELECT count(*) FROM items WHERE stage='ACCEPTED'").fetchone()[0]==1
        assert build.run(plan,Tokenizer())['accepted_clean_train_tokens']==619


def test_receipt_revoked_during_replay_validation_returns_no_tokens(tmp_path,monkeypatch):
    data=setup(tmp_path)
    decisions=bundle(tmp_path,data)
    with builder(tmp_path,data) as build:
        build.run(data[3][0],Tokenizer(),intended_use='commercial_training',decision_receipts=decisions)
        destination=tmp_path/'build'/'export'
        build.export_shards(destination,intended_use='commercial_training',decision_receipts=decisions)
        receipt=json.loads(decisions.read_bytes())['receipts'][0]
        original=corpus_gate.verify_receipt
        changed=[]
        def revoke_during_check(*args,**kwargs):
            if not changed:
                _revoke(build,'decision_receipt',receipt['identity'],
                        dict(clearance_identity=receipt['binding']['clearance_identity']))
                changed.append(True)
            return original(*args,**kwargs)
        monkeypatch.setattr(corpus_gate,'verify_receipt',revoke_during_check)
        with pytest.raises(SpecificationError,match='revoked'):
            list(build.replay_for_use(destination,split='train',
                 intended_use='commercial_training',decision_receipts=decisions))
        assert changed
        build.verify()  # Revocation does not invalidate historical integrity.


def test_replay_rechecks_before_first_token(tmp_path):
    data=setup(tmp_path)
    decisions=bundle(tmp_path,data)
    with builder(tmp_path,data) as build:
        build.run(data[3][0],Tokenizer(),intended_use='commercial_training',decision_receipts=decisions)
        destination=tmp_path/'build'/'export'
        build.export_shards(destination,intended_use='commercial_training',decision_receipts=decisions)
        stream=build.replay_for_use(destination,split='train',
            intended_use='commercial_training',decision_receipts=decisions)
        receipt=json.loads(decisions.read_bytes())['receipts'][0]
        _revoke(build,'decision_receipt',receipt['identity'],
                dict(clearance_identity=receipt['binding']['clearance_identity']))
        with pytest.raises(SpecificationError,match='stale or changed'):
            list(stream)


def test_export_publication_rechecks_after_shard_build(tmp_path,monkeypatch):
    data=setup(tmp_path)
    with builder(tmp_path,data) as build:
        build.run(data[3][0],Tokenizer())
        plan_id=file_digest(data[3][0])
        original=acquisition.build_token_shards
        def revoke_after_build(*args,**kwargs):
            result=original(*args,**kwargs)
            _revoke(build,'production_plan',plan_id,dict(plan_sha256=plan_id))
            return result
        monkeypatch.setattr(acquisition,'build_token_shards',revoke_after_build)
        destination=tmp_path/'build'/'export'
        with pytest.raises(SpecificationError,match='revoked'):
            build.export_shards(destination)
        assert not (destination/'prepared.json').exists()
        assert build.db.execute("SELECT count(*) FROM items WHERE stage='ACCEPTED'").fetchone()[0]==1


def test_export_reuse_rechecks_after_shard_validation(tmp_path,monkeypatch):
    data=setup(tmp_path)
    with builder(tmp_path,data) as build:
        build.run(data[3][0],Tokenizer())
        destination=tmp_path/'build'/'export'
        build.export_shards(destination)
        original_bytes=(destination/'prepared.json').read_bytes()
        plan_id=file_digest(data[3][0])
        original=acquisition.TokenShardStream
        changed=[]
        def revoke_during_reuse(*args,**kwargs):
            result=original(*args,**kwargs)
            if not changed:
                _revoke(build,'production_plan',plan_id,dict(plan_sha256=plan_id))
                changed.append(True)
            return result
        monkeypatch.setattr(acquisition,'TokenShardStream',revoke_during_reuse)
        with pytest.raises(SpecificationError,match='revoked'):
            build.export_shards(destination)
        assert changed and (destination/'prepared.json').read_bytes()==original_bytes


def test_artifact_check_revocation_is_not_hidden_by_earlier_status_snapshot(tmp_path,monkeypatch):
    data=setup(tmp_path)
    with builder(tmp_path,data) as build:
        plan=data[3][0]
        plan_id=file_digest(plan)
        original=acquisition.source_clearance
        changed=[]
        def revoke_during_artifact_check(*args,**kwargs):
            result=original(*args,**kwargs)
            if not changed:
                _revoke(build,'production_plan',plan_id,dict(plan_sha256=plan_id))
                changed.append(True)
            return result
        monkeypatch.setattr(acquisition,'source_clearance',revoke_during_artifact_check)
        with pytest.raises(SpecificationError,match='revoked'):
            build.require_current_approvals(plan)
        assert changed


def test_replay_revoked_inside_underlying_iterator_before_first_token(tmp_path,monkeypatch):
    data=setup(tmp_path);decisions=bundle(tmp_path,data)
    with builder(tmp_path,data) as build:
        build.run(data[3][0],Tokenizer(),intended_use='commercial_training',decision_receipts=decisions)
        destination=tmp_path/'build'/'export'
        build.export_shards(destination,intended_use='commercial_training',decision_receipts=decisions)
        replay=build.replay_for_use(destination,split='train',intended_use='commercial_training',
                                    decision_receipts=decisions)
        old_log=build.governance_log.read_bytes()
        receipt=json.loads(decisions.read_bytes())['receipts'][0]
        original=acquisition.TokenShardStream.__iter__
        changed=[]
        def revoke_before_first(self):
            if not changed:
                _revoke(build,'decision_receipt',receipt['identity'],
                        dict(clearance_identity=receipt['binding']['clearance_identity']))
                changed.append(True)
            return original(self)
        monkeypatch.setattr(acquisition.TokenShardStream,'__iter__',revoke_before_first)
        exposed=[]
        with pytest.raises(SpecificationError,match='stale or changed'):
            for token in replay:exposed.append(token)
        assert changed and exposed==[]
        monkeypatch.setattr(acquisition.TokenShardStream,'__iter__',original)
        build.governance_log.write_bytes(old_log)
        build.expected_governance_sha256=file_digest(build.governance_log)
        fresh=build.replay_for_use(destination,split='train',intended_use='commercial_training',
                                   decision_receipts=decisions)
        assert len(list(fresh))==619


def test_replay_revoked_between_emissions_stops_next_token(tmp_path,monkeypatch):
    data=setup(tmp_path);decisions=bundle(tmp_path,data)
    with builder(tmp_path,data) as build:
        build.run(data[3][0],Tokenizer(),intended_use='commercial_training',decision_receipts=decisions)
        destination=tmp_path/'build'/'export'
        build.export_shards(destination,intended_use='commercial_training',decision_receipts=decisions)
        replay=build.replay_for_use(destination,split='train',intended_use='commercial_training',
                                    decision_receipts=decisions)
        receipt=json.loads(decisions.read_bytes())['receipts'][0]
        original=acquisition.TokenShardStream.__iter__
        def revoke_after_three(self):
            for index,token in enumerate(original(self)):
                if index==3:
                    _revoke(build,'decision_receipt',receipt['identity'],
                            dict(clearance_identity=receipt['binding']['clearance_identity']))
                yield token
        monkeypatch.setattr(acquisition.TokenShardStream,'__iter__',revoke_after_three)
        exposed=[]
        with pytest.raises(SpecificationError,match='stale or changed'):
            for token in replay:exposed.append(token)
        assert len(exposed)==3


def test_commit_time_revocation_cleans_provisional_state_and_allows_one_retry(tmp_path,monkeypatch):
    spec,registry,refs,plans=setup(tmp_path)
    plan=plans[0];plan_pin=file_digest(plan)
    prior=prior_inventory(tmp_path)
    with acquisition.ProductionBuild(tmp_path/'governed-build',spec,registry,
            repository_root=tmp_path,reference_plan=refs,prior_production=prior,
            expected_plan_sha256=plan_pin,**approval_kwargs(tmp_path,plan_pin)) as build:
        original_log=build.governance_log.read_bytes()
        original_commit=CorpusLedger.commit
        calls=[]
        def revoke_after_reservation(ledger,*args,**kwargs):
            calls.append(True)
            _revoke(build,'production_plan',plan_pin,dict(plan_sha256=plan_pin))
            return original_commit(ledger,*args,**kwargs)
        monkeypatch.setattr(CorpusLedger,'commit',revoke_after_reservation)
        with pytest.raises(SpecificationError,match='revoked'):
            build.run(plan,Tokenizer())
        monkeypatch.setattr(CorpusLedger,'commit',original_commit)
        assert calls==[True]
        assert build.get('state')=='FAILED'
        assert build.db.execute("SELECT count(*) FROM items WHERE stage='ACCEPTED'").fetchone()[0]==0
        assert build.db.execute('SELECT coalesce(sum(tokens),0) FROM totals').fetchone()[0]==0
        ledger=build.corpus_ledger
        assert ledger.db.execute('SELECT count(*) FROM builds').fetchone()[0]==0
        assert ledger.db.execute('SELECT count(*) FROM usage').fetchone()[0]==0
        assert ledger.db.execute('SELECT count(*) FROM item_evidence').fetchone()[0]==0
        assert ledger.db.execute('SELECT count(*) FROM commits').fetchone()[0]==0
        build._rollback_failed_governance()
        assert ledger.db.execute('SELECT count(*) FROM totals').fetchone()[0]==0
        with pytest.raises(SpecificationError):build.export_shards(tmp_path/'governed-build'/'failed-export')
        build.governance_log.write_bytes(original_log)
        build.expected_governance_sha256=file_digest(build.governance_log)
        class NoEncode(Tokenizer):
            def encode(self,*args):pytest.fail('Unexpected retokenization on retry')
        result=build.run(plan,NoEncode(),fetch=lambda _:pytest.fail('Unexpected refetch on retry'))
        assert result['accepted_clean_train_tokens']==619
        assert build.run(plan,NoEncode())['accepted_clean_train_tokens']==619
        assert ledger.db.execute('SELECT count(*) FROM item_evidence').fetchone()[0]==1
        assert ledger.db.execute("SELECT tokens FROM totals WHERE kind='source'").fetchone()[0]==619
        assert ledger.db.execute('SELECT count(*) FROM commits').fetchone()[0]==1
