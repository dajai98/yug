"""Synthetic F4 authority tests; fixture approvals do not confer real rights."""
import json
import multiprocessing

import pytest

from tests.test_production_acquisition import BASE, approval_kwargs, fixture_plan
from training.approval_governance import Governance, append_event, event, revision_scope
from training.cumulative_accounting import CorpusLedger, GENESIS
from training.production_acquisition import item_key
from training.production_corpus import canonical, file_digest, identity
from training.production_spec import QUOTAS, SpecificationError, seal


def _attempt_synthetic_selection(log, pin, selected, result):
    try:
        append_event(log, pin, selected)
        result.put('COMMITTED')
    except (SpecificationError, ValueError):
        result.put('DRAIN_BLOCKED')


def _interruptible_direct_reservation(path, root, kwargs, evidence, limits,
                                      inventory, ready, release):
    plan = kwargs['prospective_authority']['expected_plan_sha256']
    bid = identity(dict(build='interruptible-direct'))
    with CorpusLedger(path, root, GENESIS, **kwargs) as ledger:
        ledger.claim(bid, plan, 'build', inventory)
        def interrupted():
            ready.set()
            if not release.wait(15):
                raise SpecificationError('Synthetic writer timed out')
            raise SpecificationError('Synthetic writer interrupted before commit')
        try:
            ledger.reserve(bid, evidence, limits, before_commit=interrupted)
        except SpecificationError:
            pass


def governed_ledger_case(root):
    spec, registry, plan, records = fixture_plan(root, [BASE])
    pins = approval_kwargs(root, file_digest(plan))
    source = registry['sources'][0]
    row = records[0]
    authority = dict(build_directory=root/'build', plan_path=plan,
        expected_plan_sha256=file_digest(plan),
        ownership_approval=pins['ownership_approval'],
        expected_ownership_approval_sha256=pins['expected_ownership_approval_sha256'],
        spec=spec, registry=registry, intended_use='production_preparation',
        decision_receipts=None)
    evidence = dict(item=item_key(row), record=row,
        normalized_sha=None, token_sha=None, source_id=source['source_id'],
        owner=row['owner'], split=row['split'],
        repository_group=source['repository_group'],
        domain_group=source['domain_group'], source_locator=source['locator'],
        split_group=source['split_group'], three_d_topic=None,
        three_d_form=None, tokens=8)
    limits = [('source', source['source_id'], source['target_cap_tokens']),
        ('source_owner', source['source_id']+':'+row['owner'],
         source['category_caps_tokens'][row['owner']]),
        ('repository_group', source['repository_group'],
         spec['group_caps']['repository_group'][source['repository_group']]),
        ('domain_group', source['domain_group'],
         spec['group_caps']['domain_group'][source['domain_group']]),
        ('category', row['owner'], QUOTAS[row['owner']])]
    inventory = seal(dict(schema='yug-prior-production-v1', complete=True,
                          genesis=True, builds=[]))
    kwargs = dict(governance_log=pins['governance_log'],
        expected_governance_sha256=pins['expected_governance_sha256'],
        prospective_authority=authority)
    return root/'ledger.sqlite', root, kwargs, evidence, limits, inventory


@pytest.fixture
def ledger_case(tmp_path):
    return governed_ledger_case(tmp_path)


def test_direct_ledger_requires_complete_governed_authority(ledger_case):
    path, root, kwargs, evidence, limits, inventory = ledger_case
    bid = identity(dict(build='synthetic-direct'))
    with CorpusLedger(path, root, GENESIS) as ledger:
        with pytest.raises(SpecificationError, match='external authority'):
            ledger.claim(bid, kwargs['prospective_authority']['expected_plan_sha256'],
                         'build', inventory)
    with CorpusLedger(path, root, GENESIS, **kwargs) as ledger:
        assert ledger.claim(bid, kwargs['prospective_authority']['expected_plan_sha256'],
                            'build', inventory) == 'ACTIVE'
        assert ledger.reserve(bid, evidence, limits) is None
        assert ledger.commit(bid, kwargs['prospective_authority']['expected_plan_sha256'],
                             [evidence]) != GENESIS


@pytest.mark.parametrize('damage', ['governance', 'plan', 'approval', 'record',
                                     'use', 'missing'])
def test_direct_ledger_forged_or_incomplete_authority_fails(ledger_case, damage):
    path, root, kwargs, evidence, limits, inventory = ledger_case
    kwargs = dict(kwargs)
    kwargs['prospective_authority'] = dict(kwargs['prospective_authority'])
    if damage == 'governance':
        kwargs['expected_governance_sha256'] = 'sha256:'+'0'*64
    elif damage == 'plan':
        kwargs['prospective_authority']['expected_plan_sha256'] = 'sha256:'+'0'*64
    elif damage == 'approval':
        kwargs['prospective_authority']['expected_ownership_approval_sha256'] = 'sha256:'+'0'*64
    elif damage == 'record':
        evidence['record'] = seal(dict(evidence['record'], path='src/forged.txt'))
    elif damage == 'use':
        kwargs['prospective_authority']['intended_use'] = 'commercial_training'
    else:
        kwargs['prospective_authority'] = None
    with CorpusLedger(path, root, GENESIS, **kwargs) as ledger:
        bid = identity(dict(build='synthetic-direct'))
        if damage == 'record':
            ledger.claim(bid, kwargs['prospective_authority']['expected_plan_sha256'],
                         'build', inventory)
            with pytest.raises(SpecificationError):
                ledger.reserve(bid, evidence, limits)
        else:
            with pytest.raises((SpecificationError, ValueError, TypeError)):
                ledger.claim(bid, kwargs['prospective_authority']['expected_plan_sha256']
                             if kwargs['prospective_authority'] else 'sha256:'+'1'*64,
                             'build', inventory)


def test_stale_r1_object_cannot_reserve_or_commit_after_synthetic_selection(ledger_case):
    path, root, kwargs, evidence, limits, inventory = ledger_case
    plan = kwargs['prospective_authority']['expected_plan_sha256']
    source = kwargs['prospective_authority']['registry']['sources'][0]
    bid = identity(dict(build='before-selection'))
    with CorpusLedger(path, root, GENESIS, **kwargs) as ledger:
        ledger.claim(bid, plan, 'build', inventory)
        ledger.reserve(bid, evidence, limits)
        log = kwargs['governance_log']
        prior = Governance(log, file_digest(log))
        selected = event(approval_type='source_revision',
            approval_identity='sha256:'+'3'*64,
            scope=revision_scope(source['source_id'], source['locator']),
            previous_status='NONE', new_status='ACTIVE',
            transition_at='2026-11-22T00:00:00Z',
            reason_ref='isolated synthetic selection',
            previous_event_identity=prior.head)
        new_pin = append_event(log, file_digest(log), selected)
        with pytest.raises(SpecificationError):
            ledger.commit(bid, plan, [evidence])
        ledger.expected_governance_sha256 = new_pin
        with pytest.raises(SpecificationError, match='revision activation'):
            ledger.commit(bid, plan, [evidence])
        with pytest.raises(SpecificationError, match='revision activation'):
            ledger.reserve(bid, evidence, limits)
        with pytest.raises(SpecificationError, match='revision activation'):
            ledger.claim(identity(dict(build='after-selection')), plan, 'new', inventory)
        with pytest.raises(SpecificationError, match='revision activation'):
            ledger.abort(bid)
        with pytest.raises(SpecificationError, match='revision activation'):
            ledger.rollback_failed_governance(bid)
        assert ledger.head == GENESIS


def test_direct_ledger_rejects_revoked_plan_and_forged_caps(ledger_case):
    path, root, kwargs, evidence, limits, inventory = ledger_case
    plan = kwargs['prospective_authority']['expected_plan_sha256']
    bid = identity(dict(build='revoke'))
    with CorpusLedger(path, root, GENESIS, **kwargs) as ledger:
        ledger.claim(bid, plan, 'build', inventory)
        with pytest.raises(SpecificationError, match='reservation caps'):
            ledger.reserve(bid, evidence,
                [(kind, key, cap+1) for kind, key, cap in limits])
        log = kwargs['governance_log']
        current = Governance(log, file_digest(log))
        revoked = event(approval_type='production_plan',
            approval_identity=plan, scope=dict(plan_sha256=plan),
            previous_status='ACTIVE', new_status='REVOKED',
            transition_at='2026-11-22T00:00:00Z',
            reason_ref='isolated synthetic revocation',
            previous_event_identity=current.head)
        new_pin = append_event(log, file_digest(log), revoked)
        ledger.expected_governance_sha256 = new_pin
        with pytest.raises(SpecificationError):
            ledger.reserve(bid, evidence, limits)
        assert ledger.head == GENESIS


def test_independent_selection_cannot_overtake_active_direct_reservation(ledger_case):
    path, root, kwargs, evidence, limits, inventory = ledger_case
    plan = kwargs['prospective_authority']['expected_plan_sha256']
    source = kwargs['prospective_authority']['registry']['sources'][0]
    log = kwargs['governance_log']
    pin = file_digest(log)
    selected = event(approval_type='source_revision',
        approval_identity='sha256:'+'4'*64,
        scope=revision_scope(source['source_id'], source['locator']),
        previous_status='NONE', new_status='ACTIVE',
        transition_at='2026-11-22T00:00:00Z',
        reason_ref='isolated synthetic selection',
        previous_event_identity=Governance(log, pin).head)
    bid = identity(dict(build='in-flight-direct'))
    with CorpusLedger(path, root, GENESIS, **kwargs) as ledger:
        ledger.claim(bid, plan, 'build', inventory)
        def before_commit():
            result = multiprocessing.Queue()
            process = multiprocessing.Process(target=_attempt_synthetic_selection,
                args=(log, pin, selected, result))
            process.start()
            try:
                assert result.get(timeout=15) == 'DRAIN_BLOCKED'
            finally:
                process.join(10)
            assert process.exitcode == 0
            assert file_digest(log) == pin
        assert ledger.reserve(bid, evidence, limits, before_commit=before_commit) is None
        new_pin = append_event(log, pin, selected)
        ledger.expected_governance_sha256 = new_pin
        with pytest.raises(SpecificationError, match='revision activation'):
            ledger.commit(bid, plan, [evidence])
        assert ledger.head == GENESIS


def test_interrupted_direct_writer_does_not_commit_after_selection(ledger_case):
    path, root, kwargs, evidence, limits, inventory = ledger_case
    log = kwargs['governance_log']
    pin = file_digest(log)
    source = kwargs['prospective_authority']['registry']['sources'][0]
    selected = event(approval_type='source_revision',
        approval_identity='sha256:'+'5'*64,
        scope=revision_scope(source['source_id'], source['locator']),
        previous_status='NONE', new_status='ACTIVE',
        transition_at='2026-11-22T00:00:00Z',
        reason_ref='isolated interrupted writer test',
        previous_event_identity=Governance(log, pin).head)
    ready, release = multiprocessing.Event(), multiprocessing.Event()
    process = multiprocessing.Process(target=_interruptible_direct_reservation,
        args=(path, root, kwargs, evidence, limits, inventory, ready, release))
    process.start()
    try:
        assert ready.wait(15)
        with pytest.raises(ValueError, match='not drained'):
            append_event(log, pin, selected)
        assert file_digest(log) == pin
    finally:
        release.set()
        process.join(10)
    assert not process.is_alive()
    selected_pin = append_event(log, pin, selected)
    with CorpusLedger(path, root, GENESIS, **kwargs) as ledger:
        assert ledger.head == GENESIS
        assert ledger.db.execute('SELECT count(*) FROM item_evidence').fetchone()[0] == 0
        ledger.expected_governance_sha256 = selected_pin
        with pytest.raises(SpecificationError, match='revision activation'):
            ledger.commit(identity(dict(build='interruptible-direct')),
                kwargs['prospective_authority']['expected_plan_sha256'], [evidence])
