"""Offline synthetic F2 evidence; fixture decisions are not real authority."""
import json
from pathlib import Path

import pytest

from tests.test_source_revision_final_clearance import f1_case, _build as build_f1
from tests.test_source_revision_transition import transition_case
from tests.test_source_revision_clearance import r2_case
from tests.test_registered_source_continuation import registered
from training.approval_governance import Governance, append_event, event, revision_scope
from training.production_acquisition import read_plan
from training.production_corpus import canonical, file_digest
from training.production_spec import SpecificationError
from training.source_revision_candidate_plan import (
    build, verify, derive_rights_dependencies, POLICY,
)
from training import source_revision_candidate_plan as f2
from training.production_acquisition import validate_clearance
from training.production_spec import seal
from training.source_first_clearance import BOUNDED_RIGHTS, BOUNDED_SCOPE


@pytest.fixture
def f2_case(f1_case):
    case, prepare_kwargs, prepared, final_dir = f1_case
    build_f1(f1_case)
    root = case['r']['root']
    decision = json.loads(case['decision'].read_bytes())
    manifest = json.loads((case['output']/'files.json').read_bytes())
    final = json.loads((final_dir/'clearance.json').read_bytes())
    output = root/'artifacts/r2-candidate-plan'
    rights = derive_rights_dependencies(root, case['decision'], decision,
                                        manifest, final, output)
    # Copy the historical governance bytes into a separately pinned synthetic
    # current snapshot. PREPARE's original external pin must remain immutable.
    current = root/'decisions/current-governance.jsonl'
    current.write_bytes(prepare_kwargs['governance_log'].read_bytes())
    pin = file_digest(current)
    scope = revision_scope(decision['source_id'], decision['locator'])
    for ref in rights:
        history = Governance(current, pin)
        pin = append_event(current, pin, event(
            approval_identity=ref['sha256'], approval_type='rights_evidence_status',
            scope=scope, previous_status='NONE', new_status='ACTIVE',
            transition_at='2026-11-20T00:00:00Z',
            reason_ref='synthetic fixture evidence status only',
            previous_event_identity=history.head))
    kwargs = dict(expected_final_sha256=file_digest(final_dir/'complete.json'),
        expected_final_clearance_sha256=file_digest(final_dir/'clearance.json'),
        expected_provisional_sha256=prepare_kwargs['expected_provisional_sha256'],
        expected_prepare_sha256=file_digest(prepared/'complete.json'),
        prepare_kwargs=prepare_kwargs, governance_log=current,
        expected_governance_sha256=pin,
        expected_rights_dependencies=rights, root=root)
    return case, prepared, final_dir, output, kwargs


def run(value):
    case, prepared, final_dir, output, kwargs = value
    return build(case['output'], prepared, final_dir, output, **kwargs)


def test_exact_r2_plan_and_rights_closure(f2_case):
    case, prepared, final_dir, output, kwargs = f2_case
    previous = {p: file_digest(p) for p in case['historical']}
    complete = run(f2_case)
    assert complete['policy'] == POLICY
    assert complete['review_only'] and not complete['training_authorized']
    assert complete['production_tokens'] == 0
    rows = list(read_plan(output/'plan.jsonl'))
    manifest = json.loads((case['output']/'files.json').read_bytes())
    assert {r['path'] for r in rows} == {f['path'] for f in manifest['files']}
    assert all(r['clearance_status'] == 'CANDIDATE' for r in rows)
    assert all(r['source_clearance_identity'] == complete['final_clearance_identity']
               for r in rows)
    assert len(kwargs['expected_rights_dependencies']) == len({
        ref['path'] for ref in kwargs['expected_rights_dependencies']})
    assert previous == {p: file_digest(p) for p in previous}
    assert run(f2_case) == complete
    assert verify(case['output'], prepared, final_dir, output,
                  expected_complete_sha256=file_digest(output/'complete.json'),
                  **kwargs) == complete


@pytest.mark.parametrize('damage', ['missing_right', 'extra_right',
    'wrong_right', 'governance_pin', 'final_pin', 'prior_plan_pin',
    'revoked_right', 'missing_ownership', 'altered_support',
    'missing_license'])
def test_f2_fails_closed(f2_case, damage):
    case, prepared, final_dir, output, kwargs = f2_case
    if damage == 'missing_right':
        kwargs['expected_rights_dependencies'] = kwargs['expected_rights_dependencies'][1:]
    elif damage == 'extra_right':
        kwargs['expected_rights_dependencies'] = [*kwargs['expected_rights_dependencies'],
            dict(path='forged', sha256='sha256:'+'0'*64)]
    elif damage == 'wrong_right':
        kwargs['expected_rights_dependencies'][0]['sha256'] = 'sha256:'+'0'*64
    elif damage == 'governance_pin':
        kwargs['expected_governance_sha256'] = 'sha256:'+'0'*64
    elif damage == 'final_pin':
        kwargs['expected_final_clearance_sha256'] = 'sha256:'+'0'*64
    elif damage == 'prior_plan_pin':
        kwargs['prepare_kwargs']['expected_prior_plan_sha256'] = 'sha256:'+'0'*64
    elif damage == 'revoked_right':
        log, pin = kwargs['governance_log'], kwargs['expected_governance_sha256']
        ref = kwargs['expected_rights_dependencies'][0]
        history = Governance(log, pin)
        kwargs['expected_governance_sha256'] = append_event(log, pin, event(
            approval_identity=ref['sha256'], approval_type='rights_evidence_status',
            scope=revision_scope(case['old_source']['source_id'],
                                 case['old_source']['locator']),
            previous_status='ACTIVE', new_status='REVOKED',
            transition_at='2026-11-20T00:00:00Z',
            reason_ref='synthetic revocation', previous_event_identity=history.head))
    elif damage == 'altered_support':
        case['r']['supporting'].write_bytes(b'forged synthetic support')
    elif damage == 'missing_license':
        decision = json.loads(case['decision'].read_bytes())
        (kwargs['root']/decision['revision_license_evidence']['path']).unlink()
    else:
        ref = json.loads((case['output']/'files.json').read_bytes())['files'][0][
            'ownership_decision']['evidence_ref']
        (kwargs['root']/ref['path']).unlink()
    with pytest.raises((SpecificationError, ValueError, FileNotFoundError)):
        run(f2_case)
    assert not output.exists()


def test_r1_only_revocation_does_not_revoke_independent_r2_rights(f2_case):
    case, _, _, _, kwargs = f2_case
    log, pin = kwargs['governance_log'], kwargs['expected_governance_sha256']
    old_grant = json.loads((case['active']/'clearance.json').read_bytes())
    history = Governance(log, pin)
    kwargs['expected_governance_sha256'] = append_event(log, pin, event(
        approval_identity=old_grant['identity'], approval_type='source_clearance',
        scope=dict(source_id=case['old_source']['source_id'],
                   revision=case['old_source']['revision']),
        previous_status='ACTIVE', new_status='REVOKED',
        transition_at='2026-11-20T00:00:00Z',
        reason_ref='synthetic R1-only revocation',
        previous_event_identity=history.head))
    assert run(f2_case)['review_only']


def test_prospective_reuse_fails_after_r2_evidence_revocation(f2_case):
    case, prepared, final_dir, output, kwargs = f2_case
    run(f2_case)
    original = {p.name: p.read_bytes() for p in output.iterdir()}
    log, pin = kwargs['governance_log'], kwargs['expected_governance_sha256']
    ref = kwargs['expected_rights_dependencies'][0]
    history = Governance(log, pin)
    kwargs['expected_governance_sha256'] = append_event(log, pin, event(
        approval_identity=ref['sha256'], approval_type='rights_evidence_status',
        scope=revision_scope(case['old_source']['source_id'],
                             case['old_source']['locator']),
        previous_status='ACTIVE', new_status='REVOKED',
        transition_at='2026-11-20T00:00:00Z',
        reason_ref='synthetic post-publication revocation',
        previous_event_identity=history.head))
    with pytest.raises(SpecificationError, match='Approval revoked'):
        verify(case['output'], prepared, final_dir, output,
               expected_complete_sha256=file_digest(output/'complete.json'),
               **kwargs)
    assert original == {p.name: p.read_bytes() for p in output.iterdir()}


@pytest.mark.parametrize('r2_case', ['source_c'], indirect=True)
def test_source_c_record_is_preserved_byte_for_byte(f2_case):
    case, _, _, output, _ = f2_case
    old_lines = case['active'].joinpath('plan.jsonl').read_bytes().splitlines(keepends=True)
    c_line = next(line for line in old_lines if json.loads(line)['source_id'] ==
                  'gh-unrelated-cleared')
    complete = run(f2_case)
    candidate_lines = (output/'plan.jsonl').read_bytes().splitlines(keepends=True)
    assert c_line in candidate_lines
    assert complete['unrelated_records'] == [json.loads(c_line)['identity']]
    assert complete['r2_records'] == 4


@pytest.mark.parametrize('r2_case', ['source_c'], indirect=True)
@pytest.mark.parametrize('damage', ['missing', 'altered'])
def test_missing_or_altered_source_c_fails_closed(f2_case, damage):
    case, _, _, output, kwargs = f2_case
    prior = case['active']/'plan.jsonl'
    lines = prior.read_bytes().splitlines(keepends=True)
    if damage == 'missing':
        prior.write_bytes(b''.join(line for line in lines if json.loads(line)[
            'source_id'] != 'gh-unrelated-cleared'))
    else:
        prior.write_bytes(b''.join(line.replace(b'gh-unrelated-cleared',
            b'gh-unrelated-forged') if json.loads(line)['source_id'] ==
            'gh-unrelated-cleared' else line for line in lines))
    kwargs['prepare_kwargs']['expected_prior_plan_sha256'] = file_digest(prior)
    with pytest.raises((SpecificationError, ValueError)):
        run(f2_case)
    assert not output.exists()


@pytest.mark.parametrize('r2_case', ['source_c_and_d'], indirect=True)
def test_multiple_unrelated_cleared_sources_survive(f2_case):
    case, _, _, output, _ = f2_case
    prior = case['active'].joinpath('plan.jsonl').read_bytes().splitlines(keepends=True)
    unrelated = [line for line in prior if json.loads(line)['source_id'] !=
                 case['old_source']['source_id']]
    assert len(unrelated) == 2
    complete = run(f2_case)
    assert complete['unrelated_records'] == [json.loads(line)['identity']
                                              for line in unrelated]
    result = (output/'plan.jsonl').read_bytes().splitlines(keepends=True)
    assert all(line in result for line in unrelated)


def test_conflicting_replay_and_interruption_recovery(f2_case, monkeypatch):
    case, prepared, final_dir, output, kwargs = f2_case
    original = f2.atomic_json
    def interrupt(path, value):
        original(path, value)
        if path.name == 'rights.json':
            raise RuntimeError('synthetic F2 interruption')
    monkeypatch.setattr(f2, 'atomic_json', interrupt)
    with pytest.raises(RuntimeError, match='synthetic F2 interruption'):
        run(f2_case)
    monkeypatch.setattr(f2, 'atomic_json', original)
    assert run(f2_case)['review_only']
    assert verify(case['output'], prepared, final_dir, output,
                  expected_complete_sha256=file_digest(output/'complete.json'),
                  **kwargs)['review_only']
    (output/'rights.json').write_bytes(b'forged')
    with pytest.raises(SpecificationError):
        run(f2_case)


def test_candidate_cannot_enter_active_production_validator(f2_case):
    case, prepared, _, output, _ = f2_case
    run(f2_case)
    source = next(s for s in json.loads((prepared/'registry.json').read_bytes())['sources']
                  if s['source_id'] == case['old_source']['source_id'])
    spec = json.loads((prepared/'spec.json').read_bytes())
    row = next(read_plan(output/'plan.jsonl'))
    manifest = json.loads((case['output']/'files.json').read_bytes())
    file = next(f for f in manifest['files'] if f['path'] == row['path'])
    with pytest.raises(SpecificationError):
        validate_clearance(row, source, spec, case['r']['root'], file)
    assert not any((output/name).exists() for name in
                   ('registry.json', 'spec.json', 'activation.json', 'ledger.sqlite'))


def test_bounded_rights_closure_includes_notice_and_supporting_refs(tmp_path):
    root = tmp_path
    sid, revision, locator = ('gh-synthetic-bounded', '2'*40,
                              'https://github.com/synthetic/bounded')
    path, sha = 'docs/guide.md', 'sha256:'+'a'*64
    def save(name, value):
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical(seal(value)))
        return dict(path=name, sha256=file_digest(target))
    license_ref = save('evidence/license.json', dict(
        url='https://api.github.com/repos/synthetic/bounded/license?ref='+revision))
    publication = save('evidence/publication.json', dict(statement='synthetic publication'))
    contribution = save('evidence/contribution.json', dict(statement='synthetic contribution'))
    notice = save('evidence/notice.json', dict(statement='synthetic notice obligation'))
    history = save('evidence/history.json', dict(source_id=sid, revision=revision,
                                                 path=path, adequate=True))
    candidate = save('evidence/candidate.json', dict(
        policy='bounded-publication-candidates-v1', source_id=sid,
        revision=revision, path=path, sha256=sha))
    scope = save('evidence/scope.json', dict(schema=BOUNDED_SCOPE,
        source_id=sid, revision=revision, locator=locator,
        root_license_ref=license_ref, publication_ref=publication,
        contribution_ref=contribution, notice_refs=[notice]))
    evidence = sorted([license_ref, publication, contribution, notice,
                       history, candidate, scope], key=lambda r:r['path'])
    rights = save('evidence/rights.json', dict(schema=BOUNDED_RIGHTS,
        source_id=sid, revision=revision, locator=locator, path=path,
        sha256=sha, evidence=evidence, source_scope_ref=scope,
        candidate_ref=candidate, file_history_ref=history))
    decision = seal(dict(schema='reviewed-revision-file-clearance-v1',
        source_id=sid, revision=revision, locator=locator,
        source_scope_evidence=[scope],
        revision_license_evidence=dict(license_ref, upstream_path='LICENSE',
                                       text_sha256='sha256:'+'b'*64),
        files=[dict(path=path, sha256=sha, rights_ref=rights)]))
    decision_path = root/'decisions/r2.json'
    decision_path.parent.mkdir()
    decision_path.write_bytes(canonical(decision))
    own = save('evidence/ownership.json', dict(source_id=sid, revision=revision,
        path=path, sha256=sha, rationale_ref=decision['identity']))
    manifest = dict(files=[dict(path=path, sha256=sha,
        ownership_decision=dict(evidence_ref=own))])
    grant = dict(rights_evidence=[rights, scope, notice])
    derived = derive_rights_dependencies(root, decision_path, decision,
        manifest, grant, root/'artifacts/f2')
    assert all(ref in derived for ref in [license_ref, publication,
        contribution, notice, history, candidate, scope, rights, own])
    invalid = seal(dict(decision, files=[dict(path=path, sha256=sha,
        rights_ref=save('evidence/unsupported.json',
                        dict(schema='unsupported-rights-v1')))]))
    decision_path.write_bytes(canonical(invalid))
    invalid_own = save('evidence/unsupported-ownership.json', dict(
        source_id=sid, revision=revision, path=path, sha256=sha,
        rationale_ref=invalid['identity']))
    invalid_manifest = dict(files=[dict(path=path, sha256=sha,
        ownership_decision=dict(evidence_ref=invalid_own))])
    with pytest.raises(SpecificationError, match='Unsupported or mismatched R2 rights profile'):
        derive_rights_dependencies(root, decision_path, invalid, invalid_manifest,
                                   grant, root/'artifacts/f2')
    duplicate_rights = json.loads((root/rights['path']).read_bytes())
    duplicate_rights.pop('identity')
    duplicate_rights['evidence'].append(duplicate_rights['evidence'][0])
    duplicate_ref = save('evidence/duplicate-rights.json', duplicate_rights)
    duplicate_decision = seal(dict(decision, files=[dict(
        path=path, sha256=sha, rights_ref=duplicate_ref)]))
    decision_path.write_bytes(canonical(duplicate_decision))
    duplicate_own = save('evidence/duplicate-ownership.json', dict(
        source_id=sid, revision=revision, path=path, sha256=sha,
        rationale_ref=duplicate_decision['identity']))
    duplicate_manifest = dict(files=[dict(path=path, sha256=sha,
        ownership_decision=dict(evidence_ref=duplicate_own))])
    with pytest.raises(SpecificationError, match='Duplicate R2 rights evidence reference'):
        derive_rights_dependencies(root, decision_path, duplicate_decision,
                                   duplicate_manifest, dict(rights_evidence=[]),
                                   root/'artifacts/f2')


@pytest.mark.parametrize('r2_case', ['notice'], indirect=True)
def test_actual_r2_notice_payload_and_attribution(f2_case):
    case, _, final_dir, output, kwargs = f2_case
    receipt = run(f2_case)
    rows = list(read_plan(output/'plan.jsonl'))
    notice = next(row for row in rows if row['path'] == 'NOTICE')
    assert notice['attribution'] == 'Fixture copyright and MIT license'
    assert notice['clearance_status'] == 'CANDIDATE'
    assert notice['sha256'] == file_digest(kwargs['root']/notice['input_path'])
    grant = json.loads((final_dir/'clearance.json').read_bytes())
    assert 'preserve R2 NOTICE' in grant['obligations']
    assert any(ref['path'] == 'evidence/canonical-r2.json' for ref in
               kwargs['expected_rights_dependencies'])
    assert receipt['r2_records'] == 5
