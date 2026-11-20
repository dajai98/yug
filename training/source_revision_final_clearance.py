"""Finalize a reviewed R2 file grant without activating the source revision.

The caller supplies independent byte pins for the provisional and PREPARE
receipts and for every input to their replay. This artifact is evidence for a
later activation decision, not a production source clearance by itself.
"""
from __future__ import annotations

import json
from pathlib import Path

from training.production_corpus import atomic_json, canonical, file_digest
from training.production_spec import local_path, need, seal, verify_seal
from training.source_first_clearance import RIGHTS, BOUNDED_RIGHTS, BOUNDED_SCOPE
from training.source_revision_clearance import verify_provisional
from training.source_revision_transition import prepare

POLICY = 'verified-revision-final-file-clearance-v1'
PREPARE_POLICY = 'verified-revision-transition-candidate-v1'


def _read(path):
    raw = path.read_bytes()
    value = json.loads(raw)
    need(canonical(value) == raw, 'Noncanonical final-clearance input: '+str(path))
    verify_seal(value)
    return value


def _pin(path, expected):
    need(type(expected) is str and expected.startswith('sha256:') and
         len(expected) == 71 and path.is_file() and file_digest(path) == expected,
         'Missing or stale final-clearance byte pin: '+str(path))


def _verified_inputs(provisional, prepared, expected_provisional_sha256,
                     expected_prepare_sha256, prepare_kwargs, root):
    """Independently replay earlier validators against caller-supplied pins."""
    need(type(prepare_kwargs) is dict and
         prepare_kwargs.get('expected_provisional_sha256') == expected_provisional_sha256 and
         Path(prepare_kwargs.get('root', '')).resolve() == root and
         type(prepare_kwargs.get('r2_replay_kwargs')) is dict,
         'Complete independent PREPARE replay pins required')
    replay = prepare_kwargs['r2_replay_kwargs']
    need(Path(replay.get('root', '')).resolve() == root and
         type(replay.get('expected_decision_sha256')) is str,
         'Externally pinned R2 decision required')
    complete = verify_provisional(provisional,
        expected_complete_sha256=expected_provisional_sha256, root=root)
    for name in ('registry.json', 'spec.json', 'complete.json'):
        need((prepared/name).is_file(), 'Incomplete PREPARE evidence')
    _pin(prepared/'complete.json', expected_prepare_sha256)
    candidate = _read(prepared/'complete.json')
    need(candidate.get('policy') == PREPARE_POLICY and
         candidate.get('review_only') is True and
         candidate.get('training_authorized') is False and
         candidate.get('production_tokens') == 0 and
         candidate.get('provisional_identity') == complete['identity'] and
         candidate.get('decision_identity') == complete['decision_identity'] and
         candidate.get('source_id') == complete['source_id'] and
         candidate.get('old_revision') == complete['old_revision'] and
         candidate.get('revision') == complete['revision'],
         'PREPARE receipt does not bind provisional R2')
    need({p.name for p in prepared.iterdir() if p.is_file()} ==
         {'registry.json', 'spec.json', 'complete.json'},
         'Unrecognized PREPARE artifact')
    prepared_registry = _read(prepared/'registry.json')
    prepared_spec = _read(prepared/'spec.json')
    need(prepared_registry['identity'] == candidate['registry_identity'] and
         prepared_spec['identity'] == candidate['spec_identity'] and
         prepared_spec['registry_identity'] == prepared_registry['identity'],
         'PREPARE registry/specification differs from receipt')
    decision_path = Path(replay['decision_path']).resolve()
    _pin(decision_path, replay['expected_decision_sha256'])
    decision = _read(decision_path)
    provisional_grant = _read(provisional/'clearance.json')
    manifest = _read(provisional/'files.json')
    comparison_path = Path(replay['comparison_dir']).resolve()/'comparison.json'
    comparison = _read(comparison_path)
    matches = [s for s in _read(Path(replay['registry_path']).resolve())['sources']
               if s['source_id'] == complete['source_id']]
    need(len(matches) == 1 and type(matches[0].get('clearance_record')) is str,
         'Missing or ambiguous historical R1 source')
    old_grant_path = local_path(root, matches[0]['clearance_record'])
    _pin(old_grant_path, replay['expected_old_clearance_sha256'])
    old_grant = _read(old_grant_path)
    need(decision['identity'] == complete['decision_identity'] and
         decision['identity'] != old_grant.get('governance_decision_identity') and
         comparison['identity'] == complete['comparison_identity'] and
         provisional_grant['governance_decision_identity'] == decision['identity'] and
         provisional_grant['comparison_identity'] == comparison['identity'] and
         provisional_grant['file_manifest_identity'] == manifest['identity'] and
         candidate['old_clearance_identity'] == old_grant['identity'] and
         candidate['r2_files'] == [dict(path=f['path'], sha256=f['sha256'])
                                   for f in manifest['files']],
         'R2 decision, comparison, R1 or file lineage changed')
    # PREPARE replays the 038C builder, which replays the Git-backed 038B
    # comparator and rechecks every R2 file's rights, license and screening.
    # All PREPARE files exist, so a matching replay cannot publish new bytes.
    need(prepare(provisional, prepared, **prepare_kwargs) == candidate and
         file_digest(prepared/'complete.json') == expected_prepare_sha256,
         'Full pinned PREPARE replay differs')
    return complete, candidate, decision, old_grant, provisional_grant, manifest


def _rights_references(decision, root):
    """Retain canonical R2 rights dependencies for the later activation review."""
    references = {}
    queue = [(ref, 'other') for ref in decision['source_scope_evidence']]
    license_ref = decision['revision_license_evidence']
    queue.append((dict(path=license_ref['path'], sha256=license_ref['sha256']),
                  'other'))
    canonical_ref = license_ref.get('canonical_evidence')
    if canonical_ref is not None:
        queue.append((dict(path=canonical_ref['path'],
                           sha256=canonical_ref['sha256']), 'other'))
    queue.extend((f['rights_ref'], 'rights') for f in decision['files'])
    while queue:
        need(len(references) < 4096, 'Unbounded R2 rights dependencies')
        ref, kind = queue.pop()
        need(type(ref) is dict and set(ref) == {'path', 'sha256'},
             'Invalid R2 rights dependency reference')
        name, sha = ref['path'], ref['sha256']
        path = local_path(root, name)
        _pin(path, sha)
        need(name not in references or references[name] == sha,
             'Conflicting R2 rights dependency')
        if name in references:
            continue
        references[name] = sha
        if kind == 'other':
            continue
        value = _read(path)
        if kind == 'rights':
            need(value.get('schema') in (RIGHTS, BOUNDED_RIGHTS),
                 'Unknown R2 file-rights profile')
            queue.extend((item, 'other') for item in value['evidence'])
            for key in ('source_scope_ref', 'candidate_ref',
                        'file_history_ref'):
                if key in value:
                    queue.append((value[key], 'scope' if key ==
                                  'source_scope_ref' else 'other'))
        elif kind == 'scope':
            need(value.get('schema') == BOUNDED_SCOPE,
                 'Unknown R2 source-scope profile')
            for key in ('root_license_ref', 'publication_ref',
                        'contribution_ref'):
                queue.append((value[key], 'other'))
            queue.extend((item, 'other') for item in value['notice_refs'])
    return [dict(path=name, sha256=sha) for name, sha in
            sorted(references.items())]


def _expected(provisional, prepared, complete, candidate, decision,
              old_grant, provisional_grant, manifest, root, replay):
    references = {name: sha for name, sha in complete['inputs'].items()}
    for cache_key, hash_key in (('old_cache', 'expected_old_inventory_sha256'),
                                ('new_cache', 'expected_new_inventory_sha256')):
        path = Path(replay[cache_key]).resolve()/'inventory.jsonl'
        name, sha = path.relative_to(root).as_posix(), replay[hash_key]
        _pin(path, sha)
        need(name not in references or references[name] == sha,
             'Conflicting pinned R1/R2 inventory evidence')
        references[name] = sha
    for path in (provisional/'files.json', provisional/'clearance.json',
                 provisional/'complete.json', prepared/'registry.json',
                 prepared/'spec.json', prepared/'complete.json'):
        name = path.relative_to(root).as_posix()
        sha = file_digest(path)
        need(name not in references or references[name] == sha,
             'Conflicting final-clearance evidence reference')
        references[name] = sha
    evidence = [dict(path=name, sha256=sha)
                for name, sha in sorted(references.items())]
    rights_evidence = _rights_references(decision, root)
    need(old_grant['identity'] == candidate['old_clearance_identity'] and
         provisional_grant['decision'] == 'PROVISIONAL' and
         provisional_grant['training_authorized'] is False and
         manifest['source_id'] == candidate['source_id'] and
         manifest['revision'] == candidate['revision'] and
         all(ref in evidence for ref in provisional_grant['evidence']),
         'Final R2 evidence omits original source rights or file scope')
    grant = seal(dict(schema=POLICY, source_id=candidate['source_id'],
        locator=decision['locator'], old_revision=candidate['old_revision'],
        revision=candidate['revision'], decision='ACCEPT',
        file_manifest_identity=manifest['identity'],
        governance_decision_identity=decision['identity'],
        comparison_identity=complete['comparison_identity'],
        old_clearance_identity=old_grant['identity'],
        provisional_clearance_identity=provisional_grant['identity'],
        provisional_complete_identity=complete['identity'],
        prepare_identity=candidate['identity'],
        evidence=evidence, rights_evidence=rights_evidence,
        attribution=provisional_grant['attribution'],
        obligations=provisional_grant['obligations'],
        training_authorized=False, production_tokens=0))
    receipt = seal(dict(policy=POLICY, source_id=candidate['source_id'],
        old_revision=candidate['old_revision'], revision=candidate['revision'],
        final_clearance_identity=grant['identity'],
        provisional_identity=complete['identity'],
        prepare_identity=candidate['identity'],
        decision_identity=decision['identity'],
        comparison_identity=complete['comparison_identity'],
        manifest_identity=manifest['identity'],
        review_only=True, training_authorized=False, production_tokens=0))
    return grant, receipt


def _arguments(provisional, prepared, output, prepare_kwargs, root):
    root = Path(root).resolve()
    provisional, prepared, output = (Path(p).resolve() for p in
                                      (provisional, prepared, output))
    need(all(p.is_relative_to(root/'artifacts') for p in
             (provisional, prepared, output)) and
         len({provisional, prepared, output}) == 3 and
         all(not a.is_relative_to(b) and not b.is_relative_to(a)
             for a, b in ((output, provisional), (output, prepared),
                          (provisional, prepared))),
         'Final clearance output overlaps prior artifacts or escapes repository')
    need(type(prepare_kwargs) is dict and
         type(prepare_kwargs.get('r2_replay_kwargs')) is dict and
         all(not output.is_relative_to(Path(v).resolve()) and
             not Path(v).resolve().is_relative_to(output)
             for k, v in prepare_kwargs.items() if k in
             ('prior_plan', 'governance_log')) and
         all(not output.is_relative_to(Path(v).resolve()) and
             not Path(v).resolve().is_relative_to(output)
             for k, v in prepare_kwargs['r2_replay_kwargs'].items() if k in
             ('decision_path', 'old_cache', 'new_cache', 'registry_path',
              'spec_path', 'manifest_path', 'references_path',
              'comparison_dir', 'review', 'catalog')),
         'Final clearance output overlaps governance or prior plan')
    return provisional, prepared, output, root


def _disjoint_evidence(complete, output, root):
    need(all(not local_path(root, name).is_relative_to(output) and
             not output.is_relative_to(local_path(root, name))
             for name in complete['inputs']),
         'Final clearance output overlaps pinned historical evidence')


def build(provisional, prepared, output, *, expected_provisional_sha256,
          expected_prepare_sha256, prepare_kwargs, root):
    """Publish a distinct final evidence bundle; never install it in a registry."""
    provisional, prepared, output, root = _arguments(
        provisional, prepared, output, prepare_kwargs, root)
    values = _verified_inputs(provisional, prepared, expected_provisional_sha256,
                              expected_prepare_sha256, prepare_kwargs, root)
    _disjoint_evidence(values[0], output, root)
    grant, receipt = _expected(provisional, prepared, *values, root,
                               prepare_kwargs['r2_replay_kwargs'])
    allowed = {output/'clearance.json', output/'complete.json'}
    if output.exists():
        need({p for p in output.rglob('*') if p.is_file()} <= allowed,
             'Unknown final-clearance artifact')
    output.mkdir(parents=True, exist_ok=True)
    for path, value in ((output/'clearance.json', grant),
                        (output/'complete.json', receipt)):
        if path.exists():
            need(path.read_bytes() == canonical(value),
                 'Conflicting final-clearance replay')
        else:
            atomic_json(path, value)
    return receipt


def verify_final(provisional, prepared, output, *, expected_final_sha256,
                 expected_provisional_sha256, expected_prepare_sha256,
                 prepare_kwargs, root):
    """Read-only independent audit using original external pins and validators."""
    provisional, prepared, output, root = _arguments(
        provisional, prepared, output, prepare_kwargs, root)
    _pin(output/'complete.json', expected_final_sha256)
    need({p for p in output.rglob('*') if p.is_file()} ==
         {output/'clearance.json', output/'complete.json'},
         'Incomplete or unrecognized final-clearance bundle')
    published_grant = _read(output/'clearance.json')
    published_receipt = _read(output/'complete.json')
    values = _verified_inputs(provisional, prepared, expected_provisional_sha256,
                              expected_prepare_sha256, prepare_kwargs, root)
    _disjoint_evidence(values[0], output, root)
    expected_grant, expected_receipt = _expected(
        provisional, prepared, *values, root,
        prepare_kwargs['r2_replay_kwargs'])
    need(published_grant == expected_grant and
         published_receipt == expected_receipt and
         published_receipt['final_clearance_identity'] ==
         published_grant['identity'],
         'Final R2 clearance does not match independently verified lineage')
    return published_grant
