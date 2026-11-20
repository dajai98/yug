"""Synthetic preparation only; no fixture is real activation authority."""
import json

import pytest

from tests.test_source_revision_clearance import r2_case
from tests.test_registered_source_continuation import registered
from training.approval_governance import Governance, append_event, event
from training.production_corpus import file_digest
from training.production_spec import SpecificationError, acquisition_blockers, validate
from training.source_revision_clearance import build as clear_r2
from training.source_revision_transition import prepare
from training import source_revision_transition


@pytest.fixture
def transition_case(r2_case):
    case = r2_case
    clear_r2(case['decision'], case['output'], **case['kwargs'])
    root = case['r']['root']
    old = case['old_source']
    grant = json.loads((case['active']/'clearance.json').read_bytes())
    log = root/'decisions/governance.jsonl'
    value = event(approval_identity=grant['identity'],
        approval_type='source_clearance',
        scope=dict(source_id=old['source_id'], revision=old['revision']),
        previous_status='NONE', new_status='ACTIVE',
        transition_at='2026-11-20T00:00:00Z',
        reason_ref='synthetic historical fixture')
    pin = append_event(log, None, value)
    args = dict(expected_provisional_sha256=file_digest(case['output']/'complete.json'),
        r2_replay_kwargs=dict(decision_path=case['decision'], **case['kwargs']),
        prior_plan=case['active']/'plan.jsonl',
        expected_prior_plan_sha256=file_digest(case['active']/'plan.jsonl'),
        governance_log=log, expected_governance_sha256=pin, root=root)
    return case, args, root/'artifacts/transition-candidate'


def test_preparation_keeps_r1_and_review_only_r2(transition_case):
    case, args, output = transition_case
    before = {p: file_digest(p) for p in case['historical']}
    result = prepare(case['output'], output, **args)
    registry = json.loads((output/'registry.json').read_bytes())
    spec = json.loads((output/'spec.json').read_bytes())
    validate(spec, registry, case['r']['root'])
    old = case['old_source']
    new = next(s for s in registry['sources'] if s['source_id'] == old['source_id'])
    assert new['revision'] == case['decision_value']['revision']
    assert new['status'] == 'REVIEW' and acquisition_blockers(new)
    assert result['review_only'] and not result['training_authorized']
    assert result['production_tokens'] == 0
    assert [f['path'] for f in result['r2_files']] == [
        'LICENSE', 'README.md', 'docs/added.md', 'docs/guide.md']
    assert case['comparison']['counts'] == dict(
        ADDED=1, CHANGED=1, REMOVED=1, UNCHANGED=6)
    assert all(not (output/name).exists() for name in
               ('plan.jsonl', 'clearance.json', 'activation.json'))
    assert before == {p: file_digest(p) for p in before}


def test_preparation_replays_and_rejects_conflicting_output(transition_case):
    case, args, output = transition_case
    first = prepare(case['output'], output, **args)
    assert prepare(case['output'], output, **args) == first
    (output/'spec.json').write_bytes(b'altered')
    with pytest.raises(SpecificationError):
        prepare(case['output'], output, **args)


def test_interrupted_preparation_preserves_r1_and_recovers(transition_case, monkeypatch):
    case, args, output = transition_case
    before = {p: file_digest(p) for p in case['historical']}
    original = source_revision_transition.atomic_json
    calls = []
    def interrupt(path, value):
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError('synthetic interruption')
        return original(path, value)
    monkeypatch.setattr(source_revision_transition, 'atomic_json', interrupt)
    with pytest.raises(RuntimeError, match='synthetic interruption'):
        prepare(case['output'], output, **args)
    monkeypatch.setattr(source_revision_transition, 'atomic_json', original)
    assert prepare(case['output'], output, **args)['review_only']
    assert before == {p: file_digest(p) for p in before}


@pytest.mark.parametrize('damage', ['prior_plan_pin', 'prior_plan_missing',
    'governance_pin', 'provisional_pin', 'decision_pin', 'source_revoked',
    'input_overlap'])
def test_preparation_fails_closed(transition_case, damage):
    case, args, output = transition_case
    if damage == 'prior_plan_pin':
        args['expected_prior_plan_sha256'] = 'sha256:' + '0'*64
    elif damage == 'prior_plan_missing':
        args['prior_plan'].write_bytes(b'')
        args['expected_prior_plan_sha256'] = file_digest(args['prior_plan'])
    elif damage == 'governance_pin':
        args['expected_governance_sha256'] = 'sha256:' + '0'*64
    elif damage == 'provisional_pin':
        args['expected_provisional_sha256'] = 'sha256:' + '0'*64
    elif damage == 'decision_pin':
        args['r2_replay_kwargs']['expected_decision_sha256'] = 'sha256:' + '0'*64
    elif damage == 'input_overlap':
        output = case['active']
    elif damage == 'source_revoked':
        old = case['old_source']
        grant = json.loads((case['active']/'clearance.json').read_bytes())
        history = Governance(args['governance_log'], args['expected_governance_sha256'])
        value = event(approval_identity=grant['identity'],
            approval_type='source_clearance',
            scope=dict(source_id=old['source_id'], revision=old['revision']),
            previous_status='ACTIVE', new_status='REVOKED',
            previous_event_identity=history.head,
            transition_at='2026-11-20T00:00:00Z',
            reason_ref='synthetic R1 revocation')
        args['expected_governance_sha256'] = append_event(
            args['governance_log'], args['expected_governance_sha256'], value)
        assert prepare(case['output'], output, **args)['review_only']
        return
    with pytest.raises((SpecificationError, ValueError)):
        prepare(case['output'], output, **args)
    if damage != 'input_overlap':
        assert not output.exists()
