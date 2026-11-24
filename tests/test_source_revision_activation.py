"""Isolated F3 activation; synthetic decisions are never real human authority."""
import hashlib
import json
import multiprocessing
from pathlib import Path

import pytest

from tests.test_source_revision_candidate_plan import f2_case, run as build_f2
from tests.test_source_revision_final_clearance import f1_case
from tests.test_source_revision_transition import transition_case
from tests.test_source_revision_clearance import r2_case
from tests.test_registered_source_continuation import registered
from training.approval_governance import (
    Governance, governance_lock, event, revision_scope, append_event,
    REVISION_SCHEMA_V2,
)
from training.production_corpus import canonical, file_digest
from training.production_spec import SpecificationError, seal
from training.source_revision_activation import (
    _next_event, _commit_isolated as commit, commit as live_commit,
    preflight, recover, selected_revision,
)
from training import approval_governance as governance_module
from training import source_revision_activation as activation_module
from training.cumulative_accounting import CorpusLedger, GENESIS
from training.notice_attribution import source_clearance
from training.production_corpus import identity
from tests.test_cumulative_accounting import _direct_item, _direct_limits


def _hold_governance_lock(path, ready, release):
    with governance_lock(path):
        ready.set()
        release.wait(10)


@pytest.fixture
def activation_case(f2_case):
    case, prepared, final_dir, candidate_dir, kwargs = f2_case
    complete = build_f2(f2_case)
    root = kwargs['root']
    prior = case['old_source']
    prepared_registry = json.loads((prepared/'registry.json').read_bytes())
    new = next(s for s in prepared_registry['sources'] if
               s['source_id'] == prior['source_id'])
    old_clearance = json.loads((case['active']/'clearance.json').read_bytes())
    rights = json.loads((candidate_dir/'rights.json').read_bytes())
    current = Governance(kwargs['governance_log'], kwargs['expected_governance_sha256'])
    decision = seal(dict(schema=REVISION_SCHEMA_V2,
        decision='ACTIVATE_REVISION', source_id=prior['source_id'],
        locator=prior['locator'], prior_revision=prior['revision'],
        revision=new['revision'], prior_clearance_identity=old_clearance['identity'],
        clearance_identity=complete['final_clearance_identity'],
        rights_dependencies=rights['dependencies'], reviewer='synthetic fixture reviewer',
        decided_at='2026-11-21T00:00:00Z',
        candidate_complete_identity=complete['identity'],
        candidate_complete_sha256=file_digest(candidate_dir/'complete.json'),
        candidate_plan_sha256=complete['plan_sha256'],
        expected_governance_sha256=current.byte_sha256,
        expected_governance_head=current.head))
    decision_path = root/'decisions/activation.json'
    decision_path.write_bytes(canonical(decision))
    proposed = kwargs['governance_log'].read_bytes() + canonical(
        _next_event(current, prior, decision)) + b'\n'
    next_sha = 'sha256:' + hashlib.sha256(proposed).hexdigest()
    arguments = dict(f2_kwargs=kwargs,
        expected_candidate_sha256=file_digest(candidate_dir/'complete.json'),
        decision_path=decision_path,
        expected_decision_sha256=file_digest(decision_path),
        expected_new_governance_sha256=next_sha)
    return case, prepared, final_dir, candidate_dir, arguments


def _paths(value):
    case, prepared, final_dir, candidate_dir, arguments = value
    return case['output'], prepared, final_dir, candidate_dir, arguments


def test_preflight_is_read_only_and_binds_exact_f2(activation_case):
    provisional, prepared, final_dir, candidate, arguments = _paths(activation_case)
    log = arguments['f2_kwargs']['governance_log']
    old = log.read_bytes()
    result = preflight(provisional, prepared, final_dir, candidate, **arguments)
    assert log.read_bytes() == old
    assert result['training_authorized'] is False
    assert result['candidate_plan_sha256'] == file_digest(candidate/'plan.jsonl')
    assert recover(log, trusted_old_sha256=file_digest(log),
        trusted_new_sha256=arguments['expected_new_governance_sha256'],
        expected_current_sha256=file_digest(log)) == 'OLD'


def test_live_activation_entry_point_remains_disabled(activation_case):
    provisional, prepared, final_dir, candidate, arguments = _paths(activation_case)
    log = arguments['f2_kwargs']['governance_log']
    old = log.read_bytes()
    with pytest.raises(SpecificationError, match='requires F4'):
        live_commit(provisional, prepared, final_dir, candidate, **arguments)
    assert log.read_bytes() == old


def test_commit_selects_only_r2_and_keeps_production_gate_closed(activation_case):
    provisional, prepared, final_dir, candidate, arguments = _paths(activation_case)
    log = arguments['f2_kwargs']['governance_log']
    old_pin = file_digest(log)
    result = commit(provisional, prepared, final_dir, candidate, **arguments)
    assert result['selected_revision'] == result['revision']
    assert file_digest(log) == arguments['expected_new_governance_sha256']
    assert recover(log, trusted_old_sha256=old_pin,
        trusted_new_sha256=file_digest(log),
        expected_current_sha256=file_digest(log)) == 'NEW'
    selected = selected_revision(log, file_digest(log),
        trusted_old_sha256=old_pin,
        decision_path=arguments['decision_path'],
        expected_decision_sha256=arguments['expected_decision_sha256'],
        candidate_dir=candidate,
        expected_candidate_sha256=arguments['expected_candidate_sha256'],
        prepared=prepared, final_dir=final_dir,
        expected_final_sha256=arguments['f2_kwargs']['expected_final_sha256'],
        expected_final_clearance_sha256=arguments['f2_kwargs']['expected_final_clearance_sha256'],
        root=arguments['f2_kwargs']['root'])
    assert selected['revision'] == result['revision']
    assert selected['training_authorized'] is False
    current = Governance(log, file_digest(log))
    assert (source_clearance(activation_case[0]['old_source'],
                             arguments['f2_kwargs']['root'])['identity'] ==
            json.loads((activation_case[0]['active']/'clearance.json').read_bytes())['identity'])
    with pytest.raises(SpecificationError):
        current.require_prospective_revision(activation_case[0]['old_source'])
    with pytest.raises(SpecificationError):
        current.require_prospective_revision(json.loads(
            (prepared/'registry.json').read_bytes())['sources'][0])
    with pytest.raises((SpecificationError, ValueError)):
        commit(provisional, prepared, final_dir, candidate, **arguments)


@pytest.mark.parametrize('damage', [
    'missing_final', 'candidate_sha', 'decision_sha', 'wrong_rights',
    'stale_head', 'revoked_right', 'wrong_next', 'forged_plan',
])
def test_bad_preflight_never_retires_r1(activation_case, damage):
    provisional, prepared, final_dir, candidate, arguments = _paths(activation_case)
    log = arguments['f2_kwargs']['governance_log']
    old = log.read_bytes()
    if damage == 'missing_final':
        (final_dir/'clearance.json').unlink()
    elif damage == 'candidate_sha':
        arguments['expected_candidate_sha256'] = 'sha256:' + '0'*64
    elif damage == 'decision_sha':
        arguments['expected_decision_sha256'] = 'sha256:' + '0'*64
    elif damage == 'wrong_rights':
        d = json.loads(arguments['decision_path'].read_bytes())
        d['rights_dependencies'] = d['rights_dependencies'][1:]
        arguments['decision_path'].write_bytes(canonical(seal({
            k:v for k,v in d.items() if k != 'identity'})))
        arguments['expected_decision_sha256'] = file_digest(arguments['decision_path'])
    elif damage == 'stale_head':
        arguments['f2_kwargs']['expected_governance_sha256'] = 'sha256:'+'0'*64
    elif damage == 'revoked_right':
        pin = file_digest(log)
        current = Governance(log, pin)
        ref = arguments['f2_kwargs']['expected_rights_dependencies'][0]
        append_event(log, pin, event(approval_identity=ref['sha256'],
            approval_type='rights_evidence_status',
            scope=revision_scope(activation_case[0]['old_source']['source_id'],
                activation_case[0]['old_source']['locator']),
            previous_status='ACTIVE', new_status='REVOKED',
            transition_at='2026-11-20T00:00:00Z',
            reason_ref='synthetic fixture revocation',
            previous_event_identity=current.head))
        old = log.read_bytes()
    elif damage == 'wrong_next':
        arguments['expected_new_governance_sha256'] = 'sha256:'+'0'*64
    else:
        with (candidate/'plan.jsonl').open('ab') as stream:
            stream.write(b'forged')
    with pytest.raises((SpecificationError, ValueError, FileNotFoundError)):
        commit(provisional, prepared, final_dir, candidate, **arguments)
    assert log.read_bytes() == old


def test_active_consumer_blocks_activation_until_drain(activation_case):
    provisional, prepared, final_dir, candidate, arguments = _paths(activation_case)
    log = arguments['f2_kwargs']['governance_log']
    old = file_digest(log)
    with governance_lock(log):
        with pytest.raises((SpecificationError, ValueError)):
            commit(provisional, prepared, final_dir, candidate, **arguments)
        assert file_digest(log) == old
    commit(provisional, prepared, final_dir, candidate, **arguments)


def test_old_log_substitution_and_unknown_recovery_fail(activation_case):
    provisional, prepared, final_dir, candidate, arguments = _paths(activation_case)
    log = arguments['f2_kwargs']['governance_log']
    old = log.read_bytes()
    old_pin = file_digest(log)
    commit(provisional, prepared, final_dir, candidate, **arguments)
    log.write_bytes(old)
    with pytest.raises((SpecificationError, ValueError)):
        recover(log, trusted_old_sha256=old_pin,
            trusted_new_sha256=arguments['expected_new_governance_sha256'],
            expected_current_sha256=arguments['expected_new_governance_sha256'])
    log.write_bytes(b'forged')
    with pytest.raises((SpecificationError, ValueError)):
        recover(log, trusted_old_sha256=old_pin,
            trusted_new_sha256=arguments['expected_new_governance_sha256'],
            expected_current_sha256=arguments['expected_new_governance_sha256'])


def test_interrupted_stage_recovers_without_duplicate_event(activation_case, monkeypatch):
    provisional, prepared, final_dir, candidate, arguments = _paths(activation_case)
    log = arguments['f2_kwargs']['governance_log']
    before = log.read_bytes()
    old_pin = file_digest(log)
    before_events = len(Governance(log, old_pin).events)
    original = governance_module._replace_governance
    def interrupt(path, data):
        stage = path.with_name(path.name + '.stage')
        stage.write_bytes(data)
        raise OSError('synthetic process stop before replacement')
    monkeypatch.setattr(governance_module, '_replace_governance', interrupt)
    with pytest.raises(OSError, match='before replacement'):
        commit(provisional, prepared, final_dir, candidate, **arguments)
    assert log.read_bytes() == before
    assert recover(log, trusted_old_sha256=old_pin,
        trusted_new_sha256=arguments['expected_new_governance_sha256'],
        expected_current_sha256=old_pin) == 'OLD'
    monkeypatch.setattr(governance_module, '_replace_governance', original)
    commit(provisional, prepared, final_dir, candidate, **arguments)
    assert file_digest(log) == arguments['expected_new_governance_sha256']
    assert len(Governance(log, file_digest(log)).events) == before_events + 1


def test_interruption_after_real_replace_recovers_new(activation_case, monkeypatch):
    provisional, prepared, final_dir, candidate, arguments = _paths(activation_case)
    log = arguments['f2_kwargs']['governance_log']
    old_pin = file_digest(log)
    original = activation_module._append_event_locked
    def interrupt_after(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError('synthetic process stop after replacement')
    monkeypatch.setattr(activation_module, '_append_event_locked', interrupt_after)
    with pytest.raises(OSError, match='after replacement'):
        commit(provisional, prepared, final_dir, candidate, **arguments)
    assert recover(log, trusted_old_sha256=old_pin,
        trusted_new_sha256=arguments['expected_new_governance_sha256'],
        expected_current_sha256=arguments['expected_new_governance_sha256']) == 'NEW'
    with pytest.raises((SpecificationError, ValueError)):
        commit(provisional, prepared, final_dir, candidate, **arguments)


def test_cross_process_consumer_blocks_governance_writer(tmp_path):
    log = tmp_path/'governance.jsonl'
    first = event(approval_identity='sha256:'+'1'*64,
        approval_type='production_plan', scope=dict(plan_sha256='sha256:'+'2'*64),
        previous_status='NONE', new_status='ACTIVE',
        transition_at='2026-11-21T00:00:00Z',
        reason_ref='synthetic genesis', previous_event_identity=None)
    pin = append_event(log, None, first)
    ready, release = multiprocessing.Event(), multiprocessing.Event()
    process = multiprocessing.Process(target=_hold_governance_lock,
                                      args=(log, ready, release))
    process.start()
    try:
        assert ready.wait(10)
        second = event(approval_identity='sha256:'+'1'*64,
            approval_type='production_plan', scope=dict(plan_sha256='sha256:'+'2'*64),
            previous_status='ACTIVE', new_status='REVOKED',
            transition_at='2026-11-21T00:00:01Z',
            reason_ref='synthetic revocation', previous_event_identity=first['identity'])
        with pytest.raises(ValueError, match='not drained'):
            append_event(log, pin, second)
        assert file_digest(log) == pin
    finally:
        release.set()
        process.join(10)
    assert process.exitcode == 0
    assert append_event(log, pin, second) != pin


def test_direct_ledger_without_complete_authority_is_rejected(tmp_path):
    """A governance log and head alone cannot grant prospective ledger writes."""
    log, ledger_path = tmp_path/'governance.jsonl', tmp_path/'ledger.sqlite'
    genesis = event(approval_identity='sha256:'+'1'*64,
        approval_type='production_plan', scope=dict(plan_sha256='sha256:'+'2'*64),
        previous_status='NONE', new_status='ACTIVE',
        transition_at='2026-11-21T00:00:00Z',
        reason_ref='synthetic genesis', previous_event_identity=None)
    pin = append_event(log, None, genesis)
    inventory = seal(dict(schema='yug-prior-production-v1',
                          complete=True, genesis=True, builds=[]))
    build_id, plan = identity(dict(build='direct')), identity(dict(plan='direct'))
    evidence = _direct_item('direct', 8)
    with CorpusLedger(ledger_path, tmp_path, GENESIS,
                      governance_log=log, expected_governance_sha256=pin) as ledger:
        with pytest.raises(SpecificationError, match='external authority'):
            ledger.claim(build_id, plan, 'direct', inventory)
    activation = event(approval_identity='sha256:'+'3'*64,
        approval_type='source_revision',
        scope=revision_scope('s', 'https://example.invalid/s'),
        previous_status='NONE', new_status='ACTIVE',
        transition_at='2026-11-21T00:00:01Z',
        reason_ref='synthetic selection only',
        previous_event_identity=genesis['identity'])
    selected_pin = append_event(log, pin, activation)
    with CorpusLedger(ledger_path, tmp_path, GENESIS,
                      governance_log=log,
                      expected_governance_sha256=selected_pin) as guarded:
        with pytest.raises(SpecificationError, match='external authority'):
            guarded.commit(build_id, plan, [evidence])
    with CorpusLedger(ledger_path, tmp_path, GENESIS) as unguarded:
        with pytest.raises(SpecificationError, match='external authority'):
            unguarded.commit(build_id, plan, [evidence])
    with CorpusLedger(ledger_path, tmp_path, GENESIS) as audit:
        assert audit.verify_chain() == GENESIS
