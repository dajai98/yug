"""Synthetic Git-backed R1→R2→PREPARE→F1 evidence, never real authority."""
import copy
import json

import pytest

from tests.test_source_revision_transition import transition_case
from tests.test_source_revision_clearance import r2_case
from tests.test_registered_source_continuation import registered
from training import source_revision_final_clearance as final
from training import source_revision_transition
from training.notice_attribution import source_clearance
from training.production_corpus import canonical, file_digest
from training.production_spec import SpecificationError, seal, validate
from training.source_revision_clearance import build as provisional_build


@pytest.fixture
def f1_case(transition_case):
    case, args, prepared = transition_case
    source_revision_transition.prepare(case['output'], prepared, **args)
    return case, args, prepared, case['r']['root']/'artifacts/r2-final'


def _build(value):
    case, args, prepared, output = value
    return final.build(case['output'], prepared, output,
        expected_provisional_sha256=args['expected_provisional_sha256'],
        expected_prepare_sha256=file_digest(prepared/'complete.json'),
        prepare_kwargs=args, root=case['r']['root'])


def _verify(value):
    case, args, prepared, output = value
    return final.verify_final(case['output'], prepared, output,
        expected_final_sha256=file_digest(output/'complete.json'),
        expected_provisional_sha256=args['expected_provisional_sha256'],
        expected_prepare_sha256=file_digest(prepared/'complete.json'),
        prepare_kwargs=args, root=case['r']['root'])


def test_final_clearance_independent_lineage_and_preservation(f1_case):
    case, args, prepared, output = f1_case
    historical = {p: file_digest(p) for p in case['historical']}
    provisional = {p: file_digest(p) for p in case['output'].rglob('*') if p.is_file()}
    before = {p: file_digest(p) for p in prepared.rglob('*') if p.is_file()}
    receipt = _build(f1_case)
    grant = _verify(f1_case)
    provisional_grant = json.loads((case['output']/'clearance.json').read_bytes())
    assert grant['schema'] == final.POLICY and grant['decision'] == 'ACCEPT'
    assert grant['identity'] != provisional_grant['identity']
    assert grant['provisional_clearance_identity'] == provisional_grant['identity']
    assert grant['provisional_complete_identity'] == receipt['provisional_identity']
    assert grant['prepare_identity'] == receipt['prepare_identity']
    assert grant['governance_decision_identity'] == receipt['decision_identity']
    assert grant['file_manifest_identity'] == receipt['manifest_identity']
    assert grant['comparison_identity'] == case['comparison']['identity']
    assert all(ref in grant['rights_evidence'] for ref in
               case['decision_value']['source_scope_evidence'])
    assert all(f['rights_ref'] in grant['rights_evidence'] for f in
               case['decision_value']['files'])
    assert any(ref['path'].endswith('evidence/support.json') for ref in
               grant['rights_evidence'])
    assert any(ref['path'].endswith('inventory.jsonl') for ref in grant['evidence'])
    assert not grant['training_authorized'] and grant['production_tokens'] == 0
    assert receipt['review_only'] and not receipt['training_authorized']
    assert historical == {p: file_digest(p) for p in historical}
    assert provisional == {p: file_digest(p) for p in provisional}
    assert before == {p: file_digest(p) for p in before}
    assert not any((output/name).exists() for name in
                   ('registry.json', 'spec.json', 'plan.jsonl', 'activation.json'))


def test_replay_is_deterministic_and_conflict_fails(f1_case):
    first = _build(f1_case)
    assert _build(f1_case) == first
    f1_case[3].joinpath('clearance.json').write_bytes(b'altered')
    with pytest.raises(SpecificationError):
        _build(f1_case)


def test_interrupted_publication_recovers(f1_case, monkeypatch):
    original = final.atomic_json
    calls = []
    def interrupt(path, value):
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError('synthetic interruption')
        return original(path, value)
    monkeypatch.setattr(final, 'atomic_json', interrupt)
    with pytest.raises(RuntimeError, match='synthetic interruption'):
        _build(f1_case)
    monkeypatch.setattr(final, 'atomic_json', original)
    assert _build(f1_case)['policy'] == final.POLICY
    assert _verify(f1_case)['decision'] == 'ACCEPT'


@pytest.mark.parametrize('damage', [
    'provisional_pin', 'provisional_missing', 'prepare_pin', 'prepare_missing',
    'decision_pin', 'decision_missing', 'decision_reused', 'comparison',
    'new_inventory', 'rights', 'ownership', 'license', 'protected_references',
    'governance', 'old_clearance',
])
def test_finalization_rejects_stale_or_missing_lineage(f1_case, damage):
    case, args, prepared, output = f1_case
    altered = copy.deepcopy(args)
    if damage == 'provisional_pin':
        altered['expected_provisional_sha256'] = 'sha256:'+'0'*64
    elif damage == 'provisional_missing':
        (case['output']/'complete.json').unlink()
    elif damage == 'prepare_pin':
        expected_prepare = 'sha256:'+'0'*64
    elif damage == 'prepare_missing':
        (prepared/'complete.json').unlink()
    elif damage == 'decision_pin':
        altered['r2_replay_kwargs']['expected_decision_sha256'] = 'sha256:'+'0'*64
    elif damage == 'decision_missing':
        case['decision'].unlink()
    elif damage == 'decision_reused':
        altered['r2_replay_kwargs']['decision_path'] = next(
            p for p in case['historical'] if p.name.endswith('decision.json'))
        altered['r2_replay_kwargs']['expected_decision_sha256'] = file_digest(
            altered['r2_replay_kwargs']['decision_path'])
    elif damage == 'comparison':
        (case['r']['root']/'artifacts/r2-comparison/comparison.json').write_bytes(b'altered')
    elif damage == 'new_inventory':
        (altered['r2_replay_kwargs']['new_cache']/'inventory.jsonl').write_bytes(b'altered')
    elif damage == 'rights':
        ref = case['decision_value']['files'][0]['rights_ref']
        (case['r']['root']/ref['path']).unlink()
    elif damage == 'ownership':
        next((case['output']/'ownership').glob('*.json')).write_bytes(b'altered')
    elif damage == 'license':
        (case['r']['root']/case['decision_value']['revision_license_evidence']['path']).unlink()
    elif damage == 'protected_references':
        case['r']['references'].write_bytes(b'altered')
    elif damage == 'governance':
        altered['expected_governance_sha256'] = 'sha256:'+'0'*64
    elif damage == 'old_clearance':
        (case['active']/'clearance.json').write_bytes(b'altered')
    with pytest.raises((SpecificationError, FileNotFoundError, StopIteration)):
        final.build(case['output'], prepared, output,
            expected_provisional_sha256=args['expected_provisional_sha256'],
            expected_prepare_sha256=(expected_prepare if damage == 'prepare_pin'
                                     else file_digest(prepared/'complete.json')
                                     if (prepared/'complete.json').exists()
                                     else 'sha256:'+'0'*64),
            prepare_kwargs=altered, root=case['r']['root'])
    assert not output.exists()


def test_read_only_audit_rejects_forged_final(f1_case):
    _build(f1_case)
    output = f1_case[3]
    value = json.loads((output/'clearance.json').read_bytes())
    value['old_clearance_identity'] = 'sha256:'+'0'*64
    (output/'clearance.json').write_bytes(canonical(seal(value)))
    with pytest.raises(SpecificationError):
        _verify(f1_case)


def test_forged_status_only_accept_and_final_remain_production_ineligible(f1_case):
    case, _, prepared, output = f1_case
    _build(f1_case)
    root = case['r']['root']
    registry = json.loads((prepared/'registry.json').read_bytes())
    spec = json.loads((prepared/'spec.json').read_bytes())
    source = next(s for s in registry['sources'] if s['source_id'] ==
                  case['old_source']['source_id'])
    forged = json.loads((case['output']/'clearance.json').read_bytes())
    forged = seal(dict(forged, decision='ACCEPT'))
    forged_path = output/'forged.json'
    forged_path.write_bytes(canonical(forged))
    for path in (forged_path, output/'clearance.json'):
        trial = dict(source, status='ACCEPT', training_allowed_after_clearance=True,
            clearance_record=path.relative_to(root).as_posix())
        with pytest.raises(SpecificationError,
                           match='requires later verified activation'):
            source_clearance(trial, root)
        candidate = seal(dict(registry, sources=[trial if s['source_id'] ==
            trial['source_id'] else s for s in registry['sources']]))
        candidate_spec = seal(dict(spec, registry_identity=candidate['identity']))
        with pytest.raises(SpecificationError,
                           match='requires later verified activation'):
            validate(candidate_spec, candidate, root)


def test_historical_r1_grant_still_valid(f1_case):
    case, _, _, _ = f1_case
    assert source_clearance(case['old_source'], case['r']['root'])['decision'] == 'ACCEPT'


@pytest.mark.parametrize('damage', ['omit_file', 'empty_obligations', 'old_revision'])
def test_fresh_r2_scope_and_notice_cannot_be_inherited(r2_case, damage):
    case = r2_case
    value = copy.deepcopy(case['decision_value'])
    if damage == 'omit_file':
        value['files'] = value['files'][1:]
    elif damage == 'empty_obligations':
        value['obligations'] = []
    else:
        value['revision'] = value['old_revision']
    case['decision'].write_bytes(canonical(seal(value)))
    kwargs = dict(case['kwargs'], expected_decision_sha256=file_digest(case['decision']))
    with pytest.raises(SpecificationError):
        provisional_build(case['decision'], case['output'], **kwargs)
    assert not case['output'].exists()
