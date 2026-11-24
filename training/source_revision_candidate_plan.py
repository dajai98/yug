"""Compile an offline R2 candidate plan and its complete rights dependencies.

This is evidence for later revision activation, not an active production plan.
The active specification, NOTICE, admission and ledger gates remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.approval_governance import Governance
from training.category_ownership import verify as verify_ownership
from training.production_acquisition import item_key, read_plan
from training.production_corpus import atomic_json, atomic_replace, canonical, file_digest
from training.production_spec import (local_path, need, path_allowed, seal,
                                      verify_seal)
from training.source_first_clearance import (RIGHTS, BOUNDED_RIGHTS,
    PROJECT_RIGHTS, BOUNDED_SCOPE, BOUNDED_BATCH)
from training.source_revision_final_clearance import verify_final
from training.source_revision_transition import complete_prior_plan

POLICY = 'verified-revision-candidate-plan-v1'
SHA = 'sha256:'


def _pin(path, expected):
    need(type(expected) is str and len(expected) == 71 and
         expected.startswith(SHA) and path.is_file() and
         file_digest(path) == expected, 'Missing or stale F2 input: '+str(path))


def _read(path):
    raw = path.read_bytes()
    value = json.loads(raw)
    need(raw == canonical(value), 'Noncanonical F2 input: '+str(path))
    verify_seal(value)
    return value


def _unique_refs(refs):
    need(type(refs) is list and len({r['path'] for r in refs}) == len(refs),
         'Duplicate R2 rights evidence reference')
    return refs


def derive_rights_dependencies(root, decision_path, decision, manifest,
                               final_grant, output):
    """Walk R2 authority references, including profile-specific nested evidence."""
    references, hashes, processed = {}, {}, set()
    source_id, revision, locator = (decision[k] for k in
                                    ('source_id', 'revision', 'locator'))
    queue = [(dict(path=Path(decision_path).relative_to(root).as_posix(),
                   sha256=file_digest(decision_path)), 'decision', None)]
    queue += [(ref, 'support', None) for ref in
              _unique_refs(decision['source_scope_evidence'])]
    license_ref = decision['revision_license_evidence']
    queue.append((dict(path=license_ref['path'], sha256=license_ref['sha256']),
                  'license', None))
    if license_ref.get('canonical_evidence'):
        ref = license_ref['canonical_evidence']
        queue.append((dict(path=ref['path'], sha256=ref['sha256']),
                      'canonical_license', None))
    if decision.get('candidate_batch_ref'):
        queue.append((decision['candidate_batch_ref'], 'batch', None))
    approved = {f['path']: f for f in decision['files']}
    need(len(approved) == len(decision['files']) == len(manifest['files']),
         'Ambiguous R2 decision or manifest file set')
    for f in manifest['files']:
        name = f['path']
        need(name in approved and approved[name]['sha256'] == f['sha256'],
             'R2 rights decision differs from final manifest')
        queue.append((approved[name]['rights_ref'], 'rights', f))
        own = f['ownership_decision']['evidence_ref']
        queue.append((own, 'ownership', f))
    while queue:
        need(len(processed) < 8192, 'Unbounded R2 rights dependency graph')
        ref, role, file = queue.pop()
        need(type(ref) is dict and set(ref) == {'path', 'sha256'} and
             type(ref['path']) is str and type(ref['sha256']) is str and
             len(ref['sha256']) == 71 and ref['sha256'].startswith(SHA),
             'Invalid R2 rights reference')
        name, sha = ref['path'], ref['sha256']
        path = local_path(root, name)
        need(path.is_relative_to(root) and not path.is_relative_to(output) and
             name == path.relative_to(root).as_posix(),
             'Noncanonical or output-local R2 rights reference')
        _pin(path, sha)
        need(name not in references or references[name] == sha,
             'Conflicting R2 rights dependency path')
        need(sha not in hashes or hashes[sha] == name,
             'Ambiguous R2 rights dependency byte identity')
        references[name], hashes[sha] = sha, name
        key = (name, role)
        if key in processed:
            continue
        processed.add(key)
        if role == 'support':
            continue  # Existing rights validators authenticate opaque supporting bytes.
        value = _read(path)
        if role == 'decision':
            need(value == decision and value.get('schema') ==
                 'reviewed-revision-file-clearance-v1',
                 'Unsupported or changed R2 human decision')
        elif role == 'license':
            need(value.get('url') == 'https://api.github.com/repos/' +
                 decision['locator'].removeprefix('https://github.com/') +
                 '/license?ref=' + revision,
                 'R2 source license is not revision-bound')
        elif role == 'canonical_license':
            need(value.get('policy') == 'pinned-canonical-license-evidence-v1' and
                 all(value.get(k) == v for k, v in
                     (('source_id', source_id), ('locator', locator),
                      ('revision', revision))),
                 'Unsupported R2 canonical license/NOTICE profile')
        elif role == 'batch':
            need(value.get('schema') == BOUNDED_BATCH and
                 value.get('source_id') == source_id and
                 value.get('revision') == revision,
                 'Unsupported R2 publication batch')
        elif role == 'rights':
            need(value.get('schema') in (RIGHTS, BOUNDED_RIGHTS, PROJECT_RIGHTS)
                 and all(value.get(k) == v for k, v in
                         (('source_id', source_id), ('locator', locator),
                          ('revision', revision), ('path', file['path']),
                          ('sha256', file['sha256']))),
                 'Unsupported or mismatched R2 rights profile')
            queue += [(r, 'support', None) for r in
                      _unique_refs(value['evidence'])]
            if value['schema'] == BOUNDED_RIGHTS:
                queue += [(value['source_scope_ref'], 'scope', None),
                          (value['candidate_ref'], 'candidate', file),
                          (value['file_history_ref'], 'history', file)]
            elif value['schema'] == PROJECT_RIGHTS:
                queue += [(value[k], 'support', None) for k in
                          ('project_publication_ref', 'contribution_policy_ref',
                           'file_history_ref')]
        elif role == 'scope':
            need(value.get('schema') == BOUNDED_SCOPE and
                 all(value.get(k) == v for k, v in
                     (('source_id', source_id), ('locator', locator),
                      ('revision', revision))),
                 'Unsupported R2 bounded publication scope')
            queue += [(value[k], 'support', None) for k in
                      ('root_license_ref', 'publication_ref', 'contribution_ref')]
            queue += [(r, 'support', None) for r in
                      _unique_refs(value['notice_refs'])]
        elif role in ('ownership', 'candidate', 'history'):
            need(value.get('source_id') == source_id and
                 value.get('revision') == revision and
                 value.get('path') == file['path'],
                 'R2 ownership or publication evidence has wrong file scope')
            if role == 'ownership':
                need(value.get('sha256') == file['sha256'] and
                     value.get('rationale_ref') == decision['identity'],
                     'R2 ownership is not bound to exact human decision')
            elif role == 'candidate':
                need(value.get('policy') == 'bounded-publication-candidates-v1' and
                     value.get('sha256') == file['sha256'],
                     'Unsupported R2 publication candidate')
        else:
            need(False, 'Unsupported R2 evidence role')
    result = [dict(path=name, sha256=sha) for name, sha in sorted(references.items())]
    need(all(ref in result for ref in final_grant['rights_evidence']) and
         all(f['ownership_decision']['evidence_ref'] in result
             for f in manifest['files']),
         'Final grant or ownership evidence is outside derived rights closure')
    return result


def build(provisional, prepared, final_dir, output, *,
          expected_final_sha256, expected_final_clearance_sha256,
          expected_provisional_sha256, expected_prepare_sha256,
          prepare_kwargs, governance_log, expected_governance_sha256,
          expected_rights_dependencies, root, verify_only=False,
          expected_complete_sha256=None):
    """Build or read-only verify a review-only plan from complete pinned lineage."""
    root = Path(root).resolve()
    provisional, prepared, final_dir, output = (Path(p).resolve() for p in
        (provisional, prepared, final_dir, output))
    governance_log = Path(governance_log).resolve()
    need(output.is_relative_to(root/'artifacts') and
         all(p.is_relative_to(root) and not p.is_relative_to(output) and
             not output.is_relative_to(p) for p in
             (provisional, prepared, final_dir, governance_log)),
         'F2 output overlaps evidence or escapes artifacts')
    need(type(prepare_kwargs) is dict and
         Path(prepare_kwargs.get('root', '')).resolve() == root,
         'Full independent F1 replay arguments required')
    _pin(final_dir/'clearance.json', expected_final_clearance_sha256)
    grant = verify_final(provisional, prepared, final_dir,
        expected_final_sha256=expected_final_sha256,
        expected_provisional_sha256=expected_provisional_sha256,
        expected_prepare_sha256=expected_prepare_sha256,
        prepare_kwargs=prepare_kwargs, root=root)
    need(file_digest(final_dir/'clearance.json') == expected_final_clearance_sha256
         and grant['training_authorized'] is False and grant['production_tokens'] == 0,
         'F1 final clearance changed or granted production authority')
    replay = prepare_kwargs['r2_replay_kwargs']
    decision_path = Path(replay['decision_path']).resolve()
    _pin(decision_path, replay['expected_decision_sha256'])
    decision = _read(decision_path)
    manifest = _read(provisional/'files.json')
    receipt = _read(prepared/'complete.json')
    registry = _read(Path(replay['registry_path']).resolve())
    spec = _read(Path(replay['spec_path']).resolve())
    old = [s for s in registry['sources'] if s['source_id'] == receipt['source_id']]
    need(len(old) == 1 and old[0]['revision'] == receipt['old_revision'],
         'F2 R1 source identity changed')
    candidate_registry = _read(prepared/'registry.json')
    new = [s for s in candidate_registry['sources'] if s['source_id'] == receipt['source_id']]
    need(len(new) == 1 and new[0]['status'] == 'REVIEW' and
         new[0]['revision'] == receipt['revision'] and
         new[0]['allow_paths'] == [f['path'] for f in manifest['files']] and
         grant['file_manifest_identity'] == manifest['identity'] and
         grant['prepare_identity'] == receipt['identity'],
         'F2 source or manifest differs from verified F1')
    new = new[0]
    prior_plan = Path(prepare_kwargs['prior_plan']).resolve()
    prior = complete_prior_plan(prior_plan,
        prepare_kwargs['expected_prior_plan_sha256'], registry, spec, root)
    need(any(r['source_id'] == old[0]['source_id'] for r in prior),
         'Complete R1 plan lacks transitioning source')
    rights = derive_rights_dependencies(root, decision_path, decision,
                                        manifest, grant, output)
    need(type(expected_rights_dependencies) is list and
         expected_rights_dependencies == rights,
         'External R2 rights-dependency list is incomplete or conflicting')
    governance = Governance(governance_log, expected_governance_sha256)
    governance.require_rights_dependencies(new['source_id'], new['locator'],
                                            rights, root)
    inventory = list(read_plan(Path(replay['new_cache']).resolve()/'inventory.jsonl'))
    rows = {r['path']: r for r in inventory}
    need(len(rows) == len(inventory), 'Duplicate R2 cache path')
    r2_records = []
    payload_pins = []
    for f in manifest['files']:
        row = rows.get(f['path'])
        need(row is not None and row['source_id'] == new['source_id'] and
             row['revision'] == new['revision'] and row['sha256'] == f['sha256'] and
             row['bytes'] == f['bytes'] and
             path_allowed(f['path'], [f['path']], new['deny_paths']),
             'R2 file differs from verified inventory or policy')
        payload = local_path(root, (Path(replay['new_cache']).resolve()/
                                   row['local_blob']).relative_to(root).as_posix())
        need(payload.is_file() and payload.stat().st_size == f['bytes'] and
             file_digest(payload) == f['sha256'],
             'Selected R2 payload changed')
        payload_pins.append((payload, f['sha256']))
        record = seal(dict(source_id=new['source_id'], revision=new['revision'],
            **f, clearance_status='CANDIDATE', split=new['split'],
            split_group=new['split_group'], split_unit_kind='repository',
            allow_rule=f['path'], deny_rules_checked=new['deny_paths'],
            language=next(x['language'] for x in decision['files']
                          if x['path'] == f['path']),
            license_evidence=grant['evidence'], attribution=grant['attribution'],
            source_clearance_identity=grant['identity'], quality_review='ACCEPT',
            input_path=payload.relative_to(root).as_posix()))
        verify_ownership(record, f, new, root)
        r2_records.append(record)
    need(len(r2_records) == len(manifest['files']) and
         len({r['path'].casefold() for r in r2_records}) == len(r2_records),
         'R2 candidate plan does not cover exact final file set')
    prior_lines = prior_plan.read_bytes().splitlines(keepends=True)
    need(len(prior_lines) == len(prior), 'Prior plan changed during F2')
    retained = [(item_key(r), line, r) for r, line in zip(prior, prior_lines,
                 strict=True) if r['source_id'] != new['source_id']]
    need(all(line == canonical(r)+b'\n' for _, line, r in retained) and
         len(retained) == len(receipt['unrelated_records']) and
         [r['identity'] for _, _, r in retained] == receipt['unrelated_records'],
         'Source C or other approved record changed or disappeared')
    combined = retained + [(item_key(r), canonical(r)+b'\n', r)
                           for r in r2_records]
    combined.sort(key=lambda item: item[0])
    need(len({item[0] for item in combined}) == len(combined) and
         len({(r['source_id'], r['path'].casefold()) for _, _, r in combined})
         == len(combined), 'Duplicate or ambiguous candidate plan record')
    plan_bytes = b''.join(line for _, line, _ in combined)
    rights_record = seal(dict(policy=POLICY, source_id=new['source_id'],
        old_revision=old[0]['revision'], revision=new['revision'],
        final_clearance_identity=grant['identity'],
        governance_sha256=expected_governance_sha256,
        governance_head=governance.head, dependencies=rights,
        review_only=True, training_authorized=False, production_tokens=0))
    plan_sha = SHA+hashlib.sha256(plan_bytes).hexdigest()
    complete = seal(dict(policy=POLICY, source_id=new['source_id'],
        old_revision=old[0]['revision'], revision=new['revision'],
        old_registry_identity=registry['identity'], old_spec_identity=spec['identity'],
        candidate_registry_identity=candidate_registry['identity'],
        candidate_spec_identity=receipt['spec_identity'],
        prior_plan_sha256=prepare_kwargs['expected_prior_plan_sha256'],
        provisional_identity=grant['provisional_complete_identity'],
        prepare_identity=receipt['identity'], final_clearance_identity=grant['identity'],
        rights_identity=rights_record['identity'], governance_sha256=expected_governance_sha256,
        plan_sha256=plan_sha, r2_records=len(r2_records),
        unrelated_records=[r['identity'] for _, _, r in retained],
        review_only=True, training_authorized=False, production_tokens=0))
    immutable_pins = [(final_dir/'complete.json', expected_final_sha256),
        (final_dir/'clearance.json', expected_final_clearance_sha256),
        (provisional/'complete.json', expected_provisional_sha256),
        (prepared/'complete.json', expected_prepare_sha256),
        (decision_path, replay['expected_decision_sha256']),
        (Path(replay['registry_path']).resolve(), replay['expected_registry_sha256']),
        (Path(replay['spec_path']).resolve(), replay['expected_spec_sha256']),
        (prior_plan, prepare_kwargs['expected_prior_plan_sha256']),
        (governance_log, expected_governance_sha256),
        *payload_pins,
        *((local_path(root, ref['path']), ref['sha256']) for ref in rights)]
    def check_inputs_stable():
        for path, sha in immutable_pins:
            _pin(path, sha)
        need(_read(provisional/'files.json') == manifest and
             _read(prepared/'registry.json') == candidate_registry and
             _read(prepared/'complete.json') == receipt,
             'F2 verified input changed during publication')
    check_inputs_stable()
    expected = {output/'rights.json', output/'plan.jsonl', output/'complete.json'}
    if verify_only:
        _pin(output/'complete.json', expected_complete_sha256)
        need({p for p in output.rglob('*') if p.is_file()} == expected,
             'Incomplete or unrecognized F2 publication')
    elif output.exists():
        need({p for p in output.rglob('*') if p.is_file()} <=
             expected | {output/'plan.jsonl.tmp'},
             'Unknown F2 output artifact')
        need(not ((output/'plan.jsonl.tmp').exists() and
                  (output/'plan.jsonl').exists()),
             'Ambiguous interrupted F2 plan publication')
    if not verify_only:
        output.mkdir(parents=True, exist_ok=True)
    for path, value in ((output/'rights.json', rights_record),
                        (output/'plan.jsonl', plan_bytes),
                        (output/'complete.json', complete)):
        encoded = value if isinstance(value, bytes) else canonical(value)
        if path.exists():
            need(path.read_bytes() == encoded, 'Conflicting F2 publication replay')
        else:
            need(not verify_only, 'Missing F2 publication artifact')
            if path.name == 'plan.jsonl':
                temporary = output/'plan.jsonl.tmp'
                if temporary.exists():
                    need(temporary.read_bytes() == encoded,
                         'Conflicting interrupted F2 plan')
                else:
                    with temporary.open('wb') as stream:
                        stream.write(encoded)
                        stream.flush()
                atomic_replace(temporary, path)
            else:
                atomic_json(path, value)
    need(file_digest(output/'complete.json') ==
         (expected_complete_sha256 if verify_only else file_digest(output/'complete.json')),
         'F2 completion changed')
    check_inputs_stable()
    return complete


def verify(*args, expected_complete_sha256, **kwargs):
    """Read-only audit with the original independent pins and current governance."""
    return build(*args, verify_only=True,
                 expected_complete_sha256=expected_complete_sha256, **kwargs)
