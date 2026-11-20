"""Prepare a verified R1→R2 candidate without activating or admitting it.

The existing governance log cannot atomically publish an activation with its
dependent snapshot. This compiler deliberately emits a REVIEW-only candidate;
it is not an accepted source clearance or a production plan.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from training.approval_governance import Governance, revision_scope
from training.notice_attribution import source_clearance
from training.production_acquisition import item_key, validate_clearance
from training.production_corpus import atomic_json, canonical, file_digest
from training.production_spec import (acquisition_blockers, iter_manifest_files,
    local_path, need, seal, source_manifest, validate, verify_seal)
from training.source_revision_clearance import build as replay_r2, verify_provisional

POLICY = 'verified-revision-transition-candidate-v1'
MAX_ROW = 1_048_576


def _pin(path, expected):
    need(type(expected) is str and expected.startswith('sha256:') and
         len(expected) == 71 and path.is_file() and file_digest(path) == expected,
         'Missing or stale transition input byte pin: '+str(path))


def _read(path):
    raw = path.read_bytes()
    value = json.loads(raw)
    need(canonical(value) == raw, 'Noncanonical transition input: '+str(path))
    verify_seal(value)
    return value


def complete_prior_plan(path, expected_sha256, registry, spec, root):
    """Verify exact sorted plan coverage for every cleared registered source."""
    _pin(path, expected_sha256)
    sources = {s['source_id']: s for s in registry['sources']
               if not acquisition_blockers(s)}
    expected = {}
    for sid, source in sources.items():
        source_clearance(source, root)
        expected[sid] = {f['path']: f for f in iter_manifest_files(root,
            source_manifest(root, source))}
        need(len(expected[sid]) > 0, 'Empty cleared source manifest')
    seen = {sid: set() for sid in sources}
    carried = []
    previous = None
    with path.open('rb') as stream:
        while line := stream.readline(MAX_ROW + 1):
            need(len(line) <= MAX_ROW and line.endswith(b'\n') and
                 bool(line.strip()), 'Incomplete prior approved-plan row')
            record = json.loads(line)
            need(canonical(record) + b'\n' == line,
                 'Noncanonical prior approved-plan row')
            key = item_key(record)
            sid, name = record['source_id'], record['path']
            need((previous is None or previous < key) and sid in sources and
                 name in expected[sid] and name not in seen[sid],
                 'Duplicate, unknown or missing prior approved-plan record')
            previous = key
            need(not any(p.casefold() == name.casefold() for p in seen[sid]),
                 'Ambiguous prior approved path')
            validate_clearance(record, sources[sid], spec, root, expected[sid][name])
            payload = record.get('input_path')
            if payload:
                data = local_path(root, payload)
                need(data.is_file() and data.stat().st_size == record['bytes'] and
                     file_digest(data) == record['sha256'],
                     'Prior approved payload changed')
            seen[sid].add(name)
            carried.append(record)
    need(all(seen[sid] == set(files) for sid, files in expected.items()),
         'Prior plan omits a cleared source or file')
    _pin(path, expected_sha256)
    return carried


def prepare(provisional, output, *, expected_provisional_sha256,
            r2_replay_kwargs, prior_plan, expected_prior_plan_sha256,
            governance_log, expected_governance_sha256, root):
    """Publish a deterministic, unusable R2 candidate after full input replay."""
    root = Path(root).resolve()
    provisional, output = Path(provisional).resolve(), Path(output).resolve()
    prior_plan, governance_log = Path(prior_plan).resolve(), Path(governance_log).resolve()
    need(output.is_relative_to(root/'artifacts') and
         all(p.is_relative_to(root) and not p.is_relative_to(output) and
             not output.is_relative_to(p) for p in
             (provisional, prior_plan, governance_log)),
         'Transition output overlaps inputs or escapes artifacts')
    need(type(r2_replay_kwargs) is dict and
         Path(r2_replay_kwargs.get('root', '')).resolve() == root,
         'Full pinned R2 replay arguments required')
    for key in ('decision_path', 'old_cache', 'new_cache', 'registry_path',
                'spec_path', 'manifest_path', 'references_path',
                'comparison_dir', 'review', 'catalog'):
        path = Path(r2_replay_kwargs[key]).resolve()
        need(path.is_relative_to(root) and not path.is_relative_to(output) and
             not output.is_relative_to(path),
             'Transition output overlaps pinned R1/R2 input: '+key)
    complete = verify_provisional(provisional,
        expected_complete_sha256=expected_provisional_sha256, root=root)
    decision_path = Path(r2_replay_kwargs['decision_path']).resolve()
    replay = replay_r2(decision_path, provisional,
        **{k: v for k, v in r2_replay_kwargs.items() if k != 'decision_path'})
    need(replay == complete and
         file_digest(provisional/'complete.json') == expected_provisional_sha256,
         'Provisional R2 replay differs')
    registry_path = Path(r2_replay_kwargs['registry_path']).resolve()
    spec_path = Path(r2_replay_kwargs['spec_path']).resolve()
    _pin(registry_path, r2_replay_kwargs['expected_registry_sha256'])
    _pin(spec_path, r2_replay_kwargs['expected_spec_sha256'])
    registry, spec = _read(registry_path), _read(spec_path)
    validate(spec, registry, root)
    aliases = [(s.get('locator') or '').casefold().rstrip('/') for s in registry['sources']
               if s.get('locator')]
    need(len(aliases) == len(set(aliases)),
         'Ambiguous registered locator aliases')
    decision = _read(decision_path)
    manifest = _read(provisional/'files.json')
    clearance = _read(provisional/'clearance.json')
    need(complete['decision_identity'] == decision['identity'] and
         complete['manifest_identity'] == manifest['identity'] and
         complete['clearance_identity'] == clearance['identity'] and
         complete['approved_paths'] == [f['path'] for f in manifest['files']] and
         clearance['decision'] == 'PROVISIONAL' and
         clearance['training_authorized'] is False,
         'R2 candidate is not exact provisional scope')
    matches = [s for s in registry['sources'] if s['source_id'] == complete['source_id'] or
               (s.get('locator') or '').casefold().rstrip('/') ==
               decision['locator'].casefold().rstrip('/')]
    need(len(matches) == 1 and matches[0]['source_id'] == complete['source_id'] and
         matches[0]['revision'] == complete['old_revision'],
         'Ambiguous or changed R1 source')
    old = matches[0]
    old_grant = source_clearance(old, root)
    prior = complete_prior_plan(prior_plan, expected_prior_plan_sha256,
                                registry, spec, root)
    need(any(r['source_id'] == old['source_id'] for r in prior),
         'Prior approved plan omits transitioning source')
    governance = Governance(governance_log, expected_governance_sha256)
    seen, active = governance.revision_activation_identity(old['source_id'], old['locator'])
    need(not seen and active is None, 'Conflicting source revision activation')
    scope = dict(source_id=old['source_id'], revision=old['revision'])
    state = governance.states.get(old_grant['identity'])
    need(state is not None and state['type'] == 'source_clearance' and
         state['scope'] == scope and state['status'] in ('ACTIVE', 'REVOKED'),
         'Missing historical R1 clearance governance')
    # R1-only revocation does not silently grant R2, and does not erase R1 history.
    new = copy.deepcopy(old)
    new.update(status='REVIEW', revision=complete['revision'],
        license_evidence=decision['revision_license_evidence'],
        file_manifest=(provisional/'files.json').relative_to(root).as_posix(),
        clearance_record=(provisional/'clearance.json').relative_to(root).as_posix(),
        allow_paths=complete['approved_paths'], training_allowed_after_clearance=False)
    new.pop('approval_composition', None)
    need(revision_scope(old['source_id'], old['locator']) ==
         revision_scope(new['source_id'], new['locator']) and
         all(old[key] == new[key] for key in ('repository_group', 'domain_group',
             'split_group', 'split', 'target_cap_tokens', 'category_caps_tokens',
             'allowed_categories', 'category', 'subcategory', 'license_id',
             'deny_paths')),
         'Transition changes source identity, rights policy or allocation')
    candidate = copy.deepcopy(registry)
    candidate['sources'] = [new if s['source_id'] == old['source_id'] else s
                            for s in candidate['sources']]
    candidate = seal(candidate)
    candidate_spec = seal(dict(copy.deepcopy(spec),
                               registry_identity=candidate['identity']))
    validate(candidate_spec, candidate, root)
    need(acquisition_blockers(new) and new['status'] == 'REVIEW',
         'Prepared R2 candidate must remain production-ineligible')
    unchanged = [r for r in prior if r['source_id'] != old['source_id']]
    need(all(next(s for s in candidate['sources'] if
                  s['source_id'] == r['source_id']) ==
             next(s for s in registry['sources'] if
                  s['source_id'] == r['source_id']) for r in unchanged),
         'Unrelated approved source changed')
    need(Governance(governance_log, expected_governance_sha256).head == governance.head,
         'Governance changed during transition preparation')
    receipt = seal(dict(policy=POLICY, source_id=old['source_id'],
        old_revision=old['revision'], revision=new['revision'],
        old_registry_identity=registry['identity'], old_spec_identity=spec['identity'],
        old_clearance_identity=old_grant['identity'],
        provisional_identity=complete['identity'],
        decision_identity=decision['identity'],
        prior_plan_sha256=expected_prior_plan_sha256,
        governance_sha256=expected_governance_sha256, governance_head=governance.head,
        unrelated_records=[r['identity'] for r in unchanged],
        r2_files=[dict(path=f['path'], sha256=f['sha256'])
                  for f in manifest['files']],
        registry_identity=candidate['identity'],
        spec_identity=candidate_spec['identity'],
        review_only=True, training_authorized=False, production_tokens=0))
    expected_files = {output/'registry.json', output/'spec.json', output/'complete.json'}
    if output.exists():
        need({p for p in output.rglob('*') if p.is_file()} <= expected_files,
             'Unknown candidate output artifact')
    output.mkdir(parents=True, exist_ok=True)
    for path, value in ((output/'registry.json', candidate),
                        (output/'spec.json', candidate_spec),
                        (output/'complete.json', receipt)):
        if path.exists():
            need(path.read_bytes() == canonical(value),
                 'Conflicting transition preparation replay')
        else:
            atomic_json(path, value)
    return receipt
