"""Synthetic status fixtures only; no real source or rights approval is created."""
import pytest

from training.approval_governance import (
    Governance, append_event, event, revision_scope, verify_revision_activation,
)
from training.production_corpus import canonical, file_digest
from training.production_spec import SpecificationError, seal
from tests.test_corpus_gate import builder, bundle, setup
from tests.test_production_acquisition import Tokenizer


WHEN = '2026-11-20T00:00:00Z'
R1 = '1' * 40
R2 = '2' * 40
A = 'sha256:' + 'a' * 64
B = 'sha256:' + 'b' * 64


def source(revision):
    return dict(source_id='gh-example-project',
        locator='https://github.com/example/project', revision=revision,
        repository_group='example', domain_group='github.com',
        split_group='example', split='train', target_cap_tokens=1000,
        category_caps_tokens={'code': 1000}, allowed_categories=['code'],
        category='code', subcategory='python', license_id='MIT')


def activation(tmp_path):
    evidence = tmp_path / 'rights.json'
    evidence.write_bytes(canonical(seal(dict(schema='synthetic-rights-evidence'))))
    rights = [dict(path=evidence.name, sha256=file_digest(evidence))]
    decision = seal(dict(schema='yug-source-revision-activation-v1',
        decision='ACTIVATE_REVISION', source_id=source(R1)['source_id'],
        locator=source(R1)['locator'], prior_revision=R1, revision=R2,
        prior_clearance_identity=A, clearance_identity=B,
        rights_dependencies=rights, reviewer='synthetic fixture reviewer',
        decided_at=WHEN))
    path = tmp_path / 'activation.json'
    path.write_bytes(canonical(decision))
    return path, decision, rights


def add(path, pin, aid, kind, scope, *, previous='NONE', status='ACTIVE',
        replacement=None):
    head = Governance(path, pin).head if path.exists() else None
    return append_event(path, pin, event(approval_identity=aid,
        approval_type=kind, scope=scope, previous_status=previous,
        new_status=status, replaced_by=replacement,
        replacement_scope=scope if replacement else None,
        transition_at=WHEN, reason_ref='synthetic fixture decision',
        previous_event_identity=head))


def test_external_activation_record_binds_revision_source_and_rights(tmp_path):
    path, decision, rights = activation(tmp_path)
    pin = file_digest(path)
    assert verify_revision_activation(path, pin, source(R1), source(R2),
        A, B, tmp_path) == decision
    for wrong in (source('3' * 40), dict(source(R2), locator='https://github.com/other/project'),
                  dict(source(R2), split='validation'),
                  dict(source(R2), target_cap_tokens=1001)):
        with pytest.raises(SpecificationError):
            verify_revision_activation(path, pin, source(R1), wrong, A, B, tmp_path)
    with pytest.raises(SpecificationError):
        verify_revision_activation(path, 'sha256:' + '0' * 64,
                                   source(R1), source(R2), A, B, tmp_path)
    with pytest.raises(SpecificationError):
        verify_revision_activation(tmp_path / 'absent.json', pin,
                                   source(R1), source(R2), A, B, tmp_path)
    altered = seal(dict(decision, rights_dependencies=[]))
    path.write_bytes(canonical(altered))
    with pytest.raises(SpecificationError):
        verify_revision_activation(path, file_digest(path),
                                   source(R1), source(R2), A, B, tmp_path)
    assert rights[0]['sha256'] == file_digest(tmp_path / 'rights.json')


def test_revision_status_exclusive_replay_and_stale_head(tmp_path):
    path, decision, _ = activation(tmp_path)
    scope = revision_scope(decision['source_id'], decision['locator'])
    log = tmp_path / 'governance.jsonl'
    pin = add(log, None, decision['identity'], 'source_revision', scope)
    history = Governance(log, pin)
    assert history.revision_activation_identity(*scope.values()) == (True, decision['identity'])
    assert Governance(log, pin).current == history.current
    with pytest.raises(SpecificationError, match='verified transition'):
        history.require_prospective_revision(source(R1))
    with pytest.raises(SpecificationError, match='verified transition'):
        history.require_prospective_revision(source(R2))
    before = log.read_bytes()
    with pytest.raises(SpecificationError):
        add(log, pin, B, 'source_revision', scope)
    with pytest.raises(SpecificationError):
        add(log, pin, B, 'source_revision', dict(scope, locator='https://github.com/other/project'),
            previous='ACTIVE', status='SUPERSEDED', replacement=A)
    assert log.read_bytes() == before
    updated = add(log, pin, decision['identity'], 'source_revision', scope,
                  previous='ACTIVE', status='SUPERSEDED', replacement=B)
    assert Governance(log, updated).revision_activation_identity(*scope.values()) == (True, B)
    with pytest.raises(SpecificationError, match='stale or changed'):
        Governance(log, pin)
    assert log.read_bytes().startswith(before)


def test_revision_aliases_and_revocation_never_restore_legacy(tmp_path):
    path, decision, _ = activation(tmp_path)
    scope = revision_scope(decision['source_id'], decision['locator'])
    log = tmp_path / 'governance.jsonl'
    pin = add(log, None, decision['identity'], 'source_revision', scope)
    before = log.read_bytes()
    for alias in (dict(scope, source_id='gh-other-project'),
                  dict(scope, locator='https://github.com/other/project')):
        with pytest.raises(SpecificationError, match='Conflicting source revision'):
            add(log, pin, B, 'source_revision', alias)
    assert log.read_bytes() == before
    pin = add(log, pin, decision['identity'], 'source_revision', scope,
              previous='ACTIVE', status='REVOKED')
    history = Governance(log, pin)
    assert history.revision_activation_identity(*scope.values()) == (True, None)
    with pytest.raises(SpecificationError, match='verified transition'):
        history.require_prospective_revision(source(R1))
    with pytest.raises(SpecificationError, match='verified transition'):
        history.require_prospective_revision(source(R2))


def test_rights_status_is_independent_of_revision_clearance(tmp_path):
    _path, _decision, rights = activation(tmp_path)
    scope = revision_scope(source(R1)['source_id'], source(R1)['locator'])
    log = tmp_path / 'governance.jsonl'
    pin = add(log, None, rights[0]['sha256'], 'rights_evidence_status', scope)
    pin = add(log, pin, A, 'source_clearance',
              dict(source_id=source(R1)['source_id'], revision=R1))
    history = Governance(log, pin)
    history.require_rights_dependencies(scope['source_id'], scope['locator'], rights, tmp_path)
    with pytest.raises(SpecificationError, match='Unknown'):
        history.require_rights_dependencies(scope['source_id'], scope['locator'], [], tmp_path)
    pin = add(log, pin, A, 'source_clearance',
              dict(source_id=source(R1)['source_id'], revision=R1),
              previous='ACTIVE', status='REVOKED')
    Governance(log, pin).require_rights_dependencies(
        scope['source_id'], scope['locator'], rights, tmp_path)
    pin = add(log, pin, rights[0]['sha256'], 'rights_evidence_status', scope,
              previous='ACTIVE', status='REVOKED')
    with pytest.raises(SpecificationError, match='revoked'):
        Governance(log, pin).require_rights_dependencies(
            scope['source_id'], scope['locator'], rights, tmp_path)
    (tmp_path / rights[0]['path']).write_bytes(b'forged rights')
    with pytest.raises(SpecificationError, match='missing or changed'):
        Governance(log, pin).require_rights_dependencies(
            scope['source_id'], scope['locator'], rights, tmp_path)


def test_current_activation_requires_pinned_decision_and_current_rights(tmp_path):
    path, decision, rights = activation(tmp_path)
    scope = revision_scope(decision['source_id'], decision['locator'])
    log = tmp_path / 'governance.jsonl'
    pin = add(log, None, rights[0]['sha256'], 'rights_evidence_status', scope)
    pin = add(log, pin, decision['identity'], 'source_revision', scope)
    history = Governance(log, pin)
    assert history.require_activation_decision(path, file_digest(path),
        source(R1), source(R2), A, B, tmp_path) == decision
    with pytest.raises(SpecificationError, match='pinned activation'):
        history.require_activation_decision(path, 'sha256:' + '0' * 64,
            source(R1), source(R2), A, B, tmp_path)
    pin = add(log, pin, rights[0]['sha256'], 'rights_evidence_status', scope,
              previous='ACTIVE', status='REVOKED')
    with pytest.raises(SpecificationError, match='revoked'):
        Governance(log, pin).require_activation_decision(path, file_digest(path),
            source(R1), source(R2), A, B, tmp_path)


def test_existing_production_retired_by_revision_event_but_history_survives(tmp_path):
    data = setup(tmp_path)
    decisions = bundle(tmp_path, data)
    with builder(tmp_path, data) as build:
        plan = data[3][0]
        build.run(plan, Tokenizer(), intended_use='commercial_training',
                  decision_receipts=decisions)
        export = tmp_path / 'build' / 'export'
        build.export_shards(export, intended_use='commercial_training',
                            decision_receipts=decisions)
        assert list(build.replay_for_use(export, split='train',
            intended_use='commercial_training', decision_receipts=decisions))
        old = build.governance_log.read_bytes()
        source_row = next(iter(build.sources.values()))
        scope = revision_scope(source_row['source_id'], source_row['locator'])
        build.expected_governance_sha256 = add(build.governance_log,
            build.expected_governance_sha256, A, 'source_revision', scope)
        build.verify()
        with pytest.raises(SpecificationError, match='verified transition'):
            build.require_current_approvals(plan)
        with pytest.raises(SpecificationError, match='verified transition'):
            build.export_shards(tmp_path / 'build' / 'new-export',
                intended_use='commercial_training', decision_receipts=decisions)
        with pytest.raises(SpecificationError, match='verified transition'):
            build.replay_for_use(export, split='train',
                intended_use='commercial_training', decision_receipts=decisions)
        assert build.governance_log.read_bytes().startswith(old)


def test_legacy_without_revision_history_remains_compatible(tmp_path):
    data = setup(tmp_path)
    with builder(tmp_path, data) as build:
        source_row = next(iter(build.sources.values()))
        history = Governance(build.governance_log, build.expected_governance_sha256)
        assert history.revision_activation_identity(
            source_row['source_id'], source_row['locator']) == (False, None)
        history.require_prospective_revision(source_row)
        with pytest.raises(SpecificationError, match='verified transition'):
            history.require_prospective_revision(dict(source_row, revision_activation={}))
        build.run(data[3][0], Tokenizer())
