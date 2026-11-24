"""F3 source-revision selection over the existing authoritative governance log.

This module selects a revision after complete F1/F2 replay. It never changes a
registry, approves a production plan, admits a file, or opens the R2 use gate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.approval_governance import (
    Governance, REVISION_SCHEMA_V2, SHA, _append_event_locked, event,
    governance_lock, revision_scope, verify_revision_activation,
)
from training.production_corpus import canonical, file_digest
from training.production_spec import need, verify_seal
from training.source_revision_candidate_plan import verify as verify_candidate


def _read(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    need(canonical(value) == raw, 'Noncanonical revision activation evidence')
    verify_seal(value)
    return value


def _pin(path, expected):
    need(type(expected) is str and SHA.fullmatch(expected) and
         Path(path).is_file() and file_digest(path) == expected,
         'Missing or changed externally pinned activation evidence')


def _lineage(provisional, prepared, final_dir, candidate_dir, f2_kwargs,
             expected_candidate_sha256, decision_path, expected_decision_sha256):
    """Recompute the F2 closure, then bind the independent human decision."""
    need(type(f2_kwargs) is dict and type(expected_candidate_sha256) is str,
         'Full independently pinned F2 replay required')
    root = Path(f2_kwargs['root']).resolve()
    log = Path(f2_kwargs['governance_log']).resolve()
    for path in (provisional, prepared, final_dir, candidate_dir, log,
                 decision_path):
        need(Path(path).resolve().is_relative_to(root),
             'Activation evidence escapes repository root')
    _pin(Path(candidate_dir)/'complete.json', expected_candidate_sha256)
    complete = verify_candidate(provisional, prepared, final_dir, candidate_dir,
        expected_complete_sha256=expected_candidate_sha256, **f2_kwargs)
    need(complete['review_only'] is True and
         complete['training_authorized'] is False and
         complete['production_tokens'] == 0,
         'Candidate plan is not review only')
    rights = _read(Path(candidate_dir)/'rights.json')
    prepared_receipt = _read(Path(prepared)/'complete.json')
    old_registry = _read(Path(f2_kwargs['prepare_kwargs'][
        'r2_replay_kwargs']['registry_path']))
    new_registry = _read(Path(prepared)/'registry.json')
    old = [s for s in old_registry['sources'] if s['source_id'] == complete['source_id']]
    new = [s for s in new_registry['sources'] if s['source_id'] == complete['source_id']]
    need(len(old) == len(new) == 1 and
         old[0]['revision'] == complete['old_revision'] and
         new[0]['revision'] == complete['revision'] and
         new[0]['status'] == 'REVIEW' and
         complete['old_registry_identity'] == old_registry['identity'] and
         complete['candidate_registry_identity'] == new_registry['identity'] and
         complete['prepare_identity'] == prepared_receipt['identity'] and
         complete['rights_identity'] == rights['identity'] and
         complete['plan_sha256'] == file_digest(Path(candidate_dir)/'plan.jsonl'),
         'R1/R2 candidate source or complete plan changed')
    _pin(decision_path, expected_decision_sha256)
    decision = verify_revision_activation(decision_path, expected_decision_sha256,
        old[0], new[0], prepared_receipt['old_clearance_identity'],
        complete['final_clearance_identity'], root)
    need(decision['schema'] == REVISION_SCHEMA_V2 and
         decision['candidate_complete_identity'] == complete['identity'] and
         decision['candidate_complete_sha256'] == expected_candidate_sha256 and
         decision['candidate_plan_sha256'] == complete['plan_sha256'] and
         decision['rights_dependencies'] == rights['dependencies'] ==
             f2_kwargs['expected_rights_dependencies'] and
         decision['expected_governance_sha256'] ==
             f2_kwargs['expected_governance_sha256'] == complete['governance_sha256'],
         'Activation decision does not bind the exact complete F2 plan and rights')
    governance = Governance(log, decision['expected_governance_sha256'])
    need(decision['expected_governance_head'] == governance.head ==
         rights['governance_head'] and
         governance.revision_activation_identity(old[0]['source_id'],
                                                 old[0]['locator']) == (False, None),
         'Stale or conflicting revision activation')
    governance.require_rights_dependencies(old[0]['source_id'],
        old[0]['locator'], rights['dependencies'], root)
    return root, log, old[0], new[0], complete, rights, governance, decision


def _next_event(governance, old, decision):
    need(decision['decided_at'] >= governance.events[-1]['transition_at'],
         'Activation decision predates current governance head')
    return event(approval_identity=decision['identity'],
        approval_type='source_revision',
        scope=revision_scope(old['source_id'], old['locator']),
        previous_status='NONE', new_status='ACTIVE',
        transition_at=decision['decided_at'],
        reason_ref=decision['candidate_complete_identity'],
        previous_event_identity=governance.head)


def preflight(provisional, prepared, final_dir, candidate_dir, *, f2_kwargs,
              expected_candidate_sha256, decision_path,
              expected_decision_sha256, expected_new_governance_sha256):
    """Read-only authority validation; a transient lease proves drain is possible."""
    root, log, old, new, complete, rights, governance, decision = _lineage(
        provisional, prepared, final_dir, candidate_dir, f2_kwargs,
        expected_candidate_sha256, decision_path, expected_decision_sha256)
    proposed_event = _next_event(governance, old, decision)
    proposed = governance.path.read_bytes() + canonical(proposed_event) + b'\n'
    digest = 'sha256:' + hashlib.sha256(proposed).hexdigest()
    need(type(expected_new_governance_sha256) is str and
         digest == expected_new_governance_sha256 and
         expected_new_governance_sha256 != decision['expected_governance_sha256'],
         'Proposed activation log differs from separate trusted digest')
    with governance_lock(log):
        need(Governance(log, decision['expected_governance_sha256']).head ==
             governance.head, 'Governance head changed during activation preflight')
    return dict(source_id=old['source_id'], prior_revision=old['revision'],
                revision=new['revision'], prior_governance_sha256=governance.byte_sha256,
                proposed_governance_sha256=digest,
                candidate_complete_identity=complete['identity'],
                candidate_plan_sha256=complete['plan_sha256'],
                activation_decision_identity=decision['identity'],
                training_authorized=False)


def commit(*_args, **_kwargs):
    """Live activation stays disabled until F4 closes direct prospective paths."""
    need(False, 'Live revision activation requires F4 production-gate integration')


def _commit_isolated(provisional, prepared, final_dir, candidate_dir, *, f2_kwargs,
           expected_candidate_sha256, decision_path, expected_decision_sha256,
           expected_new_governance_sha256, drain_timeout_seconds=0):
    """Exercise the F3 commit on a separate synthetic evidence root only."""
    fixture_root = Path(f2_kwargs['root']).resolve()
    repository_root = Path(__file__).resolve().parents[1]
    need(fixture_root.is_relative_to(repository_root) and
         fixture_root != repository_root and
         Path(f2_kwargs['governance_log']).resolve().is_relative_to(fixture_root),
         'Synthetic commit cannot target the real repository governance root')
    log = Path(f2_kwargs['governance_log']).resolve()
    with governance_lock(log, timeout_seconds=drain_timeout_seconds, writer=True):
        result = preflight(provisional, prepared, final_dir, candidate_dir,
            f2_kwargs=f2_kwargs, expected_candidate_sha256=expected_candidate_sha256,
            decision_path=decision_path, expected_decision_sha256=expected_decision_sha256,
            expected_new_governance_sha256=expected_new_governance_sha256)
        root, log, old, new, complete, rights, governance, decision = _lineage(
            provisional, prepared, final_dir, candidate_dir, f2_kwargs,
            expected_candidate_sha256, decision_path, expected_decision_sha256)
        need(result['proposed_governance_sha256'] == expected_new_governance_sha256,
             'Activation preflight changed inside lock')
        _append_event_locked(log, decision['expected_governance_sha256'],
            _next_event(governance, old, decision),
            expected_new_sha256=expected_new_governance_sha256)
        selected = Governance(log, expected_new_governance_sha256)
        selected.require_current('source_revision', decision['identity'],
            revision_scope(old['source_id'], old['locator']))
        return dict(result, selected_revision=new['revision'],
                    governance_head=selected.head)


def recover(log, *, trusted_old_sha256, trusted_new_sha256,
            expected_current_sha256):
    """Select only an externally trusted old or new image; never repair by append."""
    need(all(type(x) is str and SHA.fullmatch(x) for x in
             (trusted_old_sha256, trusted_new_sha256, expected_current_sha256)) and
         trusted_old_sha256 != trusted_new_sha256 and
         expected_current_sha256 in (trusted_old_sha256, trusted_new_sha256),
         'Two distinct trusted activation log identities required')
    log = Path(log).resolve()
    with governance_lock(log):
        _pin(log, expected_current_sha256)
        current = Governance(log, expected_current_sha256)
        if current.byte_sha256 == trusted_old_sha256:
            return 'OLD'
        need(len(current.events) >= 2 and
             current.events[-1]['approval_type'] == 'source_revision' and
             'sha256:' + hashlib.sha256(b''.join(
                 canonical(x) + b'\n' for x in current.events[:-1])).hexdigest()
             == trusted_old_sha256,
             'New activation image lacks trusted old-log prefix')
        return 'NEW'


def selected_revision(log, expected_current_sha256, *,
                      trusted_old_sha256, decision_path,
                      expected_decision_sha256, candidate_dir,
                      expected_candidate_sha256, prepared,
                      final_dir, expected_final_sha256,
                      expected_final_clearance_sha256, root):
    """Verify one R2 selection; never grant production or intended use."""
    need(recover(log, trusted_old_sha256=trusted_old_sha256,
                 trusted_new_sha256=expected_current_sha256,
                 expected_current_sha256=expected_current_sha256) == 'NEW',
         'R2 has not been selected')
    root = Path(root).resolve()
    decision_path = Path(decision_path).resolve()
    _pin(decision_path, expected_decision_sha256)
    decision = _read(decision_path)
    need(decision['schema'] == REVISION_SCHEMA_V2 and
         decision['candidate_complete_sha256'] == expected_candidate_sha256 and
         decision['expected_governance_sha256'] == trusted_old_sha256,
         'Activation decision or old governance identity differs')
    _pin(Path(candidate_dir)/'complete.json', expected_candidate_sha256)
    _pin(Path(final_dir)/'complete.json', expected_final_sha256)
    _pin(Path(final_dir)/'clearance.json', expected_final_clearance_sha256)
    complete = _read(Path(candidate_dir)/'complete.json')
    rights = _read(Path(candidate_dir)/'rights.json')
    final_receipt = _read(Path(final_dir)/'complete.json')
    final_grant = _read(Path(final_dir)/'clearance.json')
    new_registry = _read(Path(prepared)/'registry.json')
    matches = [s for s in new_registry['sources'] if
               s['source_id'] == decision['source_id']]
    need(len(matches) == 1 and matches[0]['revision'] == decision['revision'] and
         complete['identity'] == decision['candidate_complete_identity'] and
         complete['plan_sha256'] == decision['candidate_plan_sha256'] ==
             file_digest(Path(candidate_dir)/'plan.jsonl') and
         complete['rights_identity'] == rights['identity'] and
         rights['dependencies'] == decision['rights_dependencies'] and
         decision['clearance_identity'] == complete['final_clearance_identity'] ==
             final_grant['identity'] and
         final_receipt['final_clearance_identity'] == final_grant['identity'] and
         complete['candidate_registry_identity'] == new_registry['identity'],
         'Selected revision lacks exact completed transition lineage')
    with governance_lock(log):
        current = Governance(log, expected_current_sha256)
        current.require_current('source_revision', decision['identity'],
            revision_scope(decision['source_id'], decision['locator']))
        need(current.events[-1]['identity'] == current.head and
             current.events[-1]['approval_identity'] == decision['identity'] and
             current.events[-1]['reason_ref'] == complete['identity'],
             'Current activation event differs from completed transition')
        current.require_rights_dependencies(decision['source_id'],
            decision['locator'], rights['dependencies'], root)
    return dict(source_id=decision['source_id'], revision=decision['revision'],
                governance_head=current.head, training_authorized=False)
