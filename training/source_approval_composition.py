"""Compose two independently reviewed exact-file grants without admission.

The composition decision and both leaf decisions require external byte pins.
This compiler creates prospective metadata only: source rights remain in the
independent leaves, and production/training require separate current authority.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from training.composite_lineage import CLEARANCE, POLICY, read_ref, verify
from training.production_acquisition import (MAX_ITEM_BYTES, item_key, read_plan,
                                             validate_clearance)
from training.production_corpus import atomic_json, atomic_replace, canonical, file_digest
from training.production_spec import (acquisition_blockers, iter_manifest_files,
                                      local_path, need, seal, source_manifest,
                                      validate, verify_seal)
from training.source_first_clearance import build as build_leaf
from training.source_prepare import ROOT


def _path(root, name):
    return local_path(root, name)


def build(decision_path, output, *, expected_decision_sha256,
          expected_component_decision_sha256, root=ROOT,
          prior_approved_plan=None, expected_prior_plan_sha256=None):
    root = Path(root).resolve()
    decision_path = Path(decision_path).resolve()
    output = Path(output).resolve()
    need(output.is_relative_to(root/'artifacts') and
         decision_path.is_relative_to(root) and
         not decision_path.is_relative_to(output),
         'Composition input/output escapes or overlaps repository artifacts')
    need(type(expected_decision_sha256) is str and
         file_digest(decision_path) == expected_decision_sha256,
         'External composition decision pin missing or changed')
    decision = json.loads(decision_path.read_bytes())
    need(canonical(decision) == decision_path.read_bytes(),
         'Noncanonical composition decision')
    verify_seal(decision)
    fields = {'schema', 'decision', 'reviewer', 'decided_at', 'source_id',
              'locator', 'revision', 'split', 'repository_group', 'domain_group',
              'split_group', 'category', 'subcategory', 'license_id',
              'target_cap_tokens', 'category_caps_tokens', 'attribution',
              'obligations', 'files', 'components', 'original_registry_ref',
              'original_spec_ref', 'active_registry_ref', 'active_spec_ref',
              'catalog_ref', 'references_ref', 'cache_path', 'review_path',
              'identity'}
    need(set(decision) == fields and decision['schema'] == POLICY and
         decision['decision'] == 'COMPOSE_EXACT_FILES' and
         decision['reviewer'] and decision['decided_at'],
         'Incomplete composition decision')
    originals = [read_ref(root, decision[k]) for k in
                 ('original_registry_ref', 'original_spec_ref',
                  'active_registry_ref', 'active_spec_ref', 'catalog_ref')]
    original_reg, original_spec, active_reg, active_spec, catalog = originals
    validate(original_spec, original_reg, root)
    validate(active_spec, active_reg, root)
    references = _path(root, decision['references_ref']['path'])
    need(file_digest(references) == decision['references_ref']['sha256'],
         'Protected references changed')
    cache = _path(root, decision['cache_path'])
    review = _path(root, decision['review_path'])
    need(not any(p.is_relative_to(output) for p in
             (cache, review, references, decision_path)) and
         not output.is_relative_to(cache) and
         not output.is_relative_to(review), 'Composition output overlaps inputs')
    original = [s for s in original_reg['sources']
                if s['source_id'] == decision['source_id']]
    active = [s for s in active_reg['sources']
              if s['source_id'] == decision['source_id']]
    need(len(original) == len(active) == 1 and
         original[0]['status'] == 'REVIEW' and
         original[0].get('file_manifest') is None and
         original[0].get('clearance_record') is None and
         active[0]['status'] == 'ACCEPT' and
         len(original_reg['sources']) == len(active_reg['sources']) and
         all(s == next(a for a in active_reg['sources']
                       if a['source_id'] == s['source_id'])
             for s in original_reg['sources'] if s['source_id'] != decision['source_id']),
         'Composition baseline changed or unrelated source differs')
    cleared_others = {s['source_id']: s for s in active_reg['sources']
                      if s['source_id'] != decision['source_id'] and
                      not acquisition_blockers(s)}
    need((prior_approved_plan is None) == (expected_prior_plan_sha256 is None),
         'Prior approved plan and external byte pin are required together')
    need(not cleared_others or prior_approved_plan is not None,
         'Existing cleared sources require a pinned prior approved plan')
    prior_path = None
    prior_sha = None
    if prior_approved_plan is not None:
        prior_path = _path(root, str(prior_approved_plan)) if not Path(prior_approved_plan).is_absolute() else Path(prior_approved_plan).resolve()
        need(prior_path.is_relative_to(root) and
             not prior_path.is_relative_to(output) and
             not output.is_relative_to(prior_path) and
             prior_path.is_file() and
             type(expected_prior_plan_sha256) is str and
             file_digest(prior_path) == expected_prior_plan_sha256,
             'External prior approved-plan pin missing or changed')
        prior_sha = expected_prior_plan_sha256
    need(len(decision['components']) == 2 and
         decision['components'] == sorted(decision['components'],
                                          key=lambda c: c['decision_identity']),
         'Two sorted independent components required')
    need(type(expected_component_decision_sha256) in (list, tuple) and
         len(expected_component_decision_sha256) == 2 and
         list(expected_component_decision_sha256) ==
         [c['decision_ref']['sha256'] for c in decision['components']] and
         len(set(expected_component_decision_sha256)) == 2,
         'Two independent external leaf-decision pins required')
    leaf_plans = []
    active_leaf = False
    active_leaf_plan = None
    for component in decision['components']:
        need(all(not _path(root, component[key]['path']).is_relative_to(output)
                 for key in ('decision_ref','manifest_ref','clearance_ref',
                             'complete_ref','plan_ref')),
             'Component authority cannot be output-local')
        complete_path = _path(root, component['complete_ref']['path'])
        leaf_dir = complete_path.parent
        need(leaf_dir.is_relative_to(root/'artifacts') and
             not leaf_dir.is_relative_to(output) and
             not output.is_relative_to(leaf_dir),
             'Leaf and composite outputs overlap')
        # Verify every published byte before replay. A missing artifact cannot
        # be regenerated and then mistaken for an existing independent grant.
        complete = read_ref(root, component['complete_ref'])
        for name, sha in complete['artifacts'].items():
            path = (leaf_dir/name).resolve()
            need(path.is_relative_to(leaf_dir) and path.is_file() and
                 file_digest(path) == sha, 'Leaf grant incomplete or changed')
        leaf_decision = read_ref(root, component['decision_ref'])
        need(leaf_decision['identity'] == component['decision_identity'],
             'Leaf decision identity differs')
        replay = build_leaf(_path(root, component['decision_ref']['path']),
            leaf_dir, expected_decision_sha256=component['decision_ref']['sha256'],
            cache=cache, catalog=_path(root, decision['catalog_ref']['path']),
            review=review, references=references, root=root,
            base_registry=_path(root, decision['original_registry_ref']['path']),
            base_spec=_path(root, decision['original_spec_ref']['path']))
        need(replay['identity'] == component['complete_identity'],
             'Leaf replay differs from completed independent grant')
        if ((leaf_dir/'registry.json').resolve() ==
                _path(root, decision['active_registry_ref']['path']) and
            (leaf_dir/'spec.json').resolve() ==
                _path(root, decision['active_spec_ref']['path'])):
            active_leaf = True
            active_leaf_plan = list(read_plan(_path(root, component['plan_ref']['path'])))
            need(active[0]['clearance_record'] ==
                 component['clearance_ref']['path'] and
                 active[0]['file_manifest'] == component['manifest_ref']['path'],
                 'Active leaf grant differs from registered source')
        leaf_plans.extend(read_plan(_path(root, component['plan_ref']['path'])))
    need(active_leaf, 'Active source is not one of the two approved leaves')
    carried = []
    if prior_path is not None:
        active_sources = {s['source_id']: s for s in active_reg['sources']}
        for unrelated in cleared_others.values():
            for field in ('file_manifest', 'clearance_record'):
                artifact = _path(root, unrelated[field])
                raw = artifact.read_bytes()
                need(raw == canonical(json.loads(raw)),
                     'Previously cleared source artifact changed or is noncanonical')
        expected_active = {item_key(r): r for r in active_leaf_plan}
        seen_active = {}
        seen_others = {sid: set() for sid in cleared_others}
        seen_items = set()
        seen_paths = set()
        previous_key = None
        with prior_path.open('rb') as stream:
            while line := stream.readline(MAX_ITEM_BYTES + 1):
                need(len(line) <= MAX_ITEM_BYTES and
                     line.endswith(b'\n') and line.strip(),
                     'Incomplete prior approved-plan row')
                record = json.loads(line)
                need(canonical(record) + b'\n' == line,
                     'Noncanonical prior approved-plan row')
                key = item_key(record)
                sid = record['source_id']
                path_key = (sid, record['path'].casefold())
                need((previous_key is None or previous_key < key) and
                     key not in seen_items and path_key not in seen_paths and
                     sid in active_sources and
                     (sid == decision['source_id'] or sid in cleared_others),
                     'Duplicate, unknown, or conflicting prior approved item')
                previous_key = key
                seen_items.add(key)
                seen_paths.add(path_key)
                source_before = active_sources[sid]
                manifest_before = list(iter_manifest_files(root,
                    source_manifest(root, source_before)))
                files_before = {f['path'].casefold(): f for f in manifest_before}
                approved_file = files_before.get(record['path'].casefold())
                need(approved_file is not None and
                     approved_file['path'] == record['path'],
                     'Prior item absent from original clearance')
                validate_clearance(record, source_before, active_spec, root,
                                   approved_file)
                if sid == decision['source_id']:
                    need(key in expected_active and
                         record == expected_active[key],
                         'Prior plan changed the active leaf grant')
                    seen_active[key] = record
                else:
                    if record.get('input_path'):
                        payload = _path(root, record['input_path'])
                        need(payload.is_file() and
                             payload.stat().st_size == record['bytes'] and
                             file_digest(payload) == record['sha256'],
                             'Carried-forward payload changed')
                    seen_others[sid].add(record['path'])
                    carried.append(record)
        need(seen_active == expected_active,
             'Prior plan omits active leaf approval')
        for sid, paths in seen_others.items():
            expected_paths = {f['path'] for f in iter_manifest_files(root,
                source_manifest(root, cleared_others[sid]))}
            need(paths == expected_paths,
                 'Prior plan omits cleared unrelated files')
        need(file_digest(prior_path) == prior_sha,
             'Prior approved plan changed during validation')
    source = copy.deepcopy(active[0])
    source['allow_paths'] = [f['path'] for f in decision['files']]
    source['approval_composition'] = dict(policy=POLICY,
                                          decision_identity=decision['identity'])
    source['file_manifest'] = (output/'files.json').relative_to(root).as_posix()
    source['clearance_record'] = (output/'clearance.json').relative_to(root).as_posix()
    manifest_files = []
    for component in decision['components']:
        manifest_files.extend(read_ref(root, component['manifest_ref'])['files'])
    manifest_files.sort(key=lambda f: f['path'])
    manifest = seal(dict(source_id=source['source_id'],
                         revision=source['revision'], files=manifest_files))
    evidence = {}
    for component in decision['components']:
        grant = read_ref(root, component['clearance_ref'])
        for ref in (component['decision_ref'], component['manifest_ref'],
                    component['clearance_ref'], component['complete_ref'],
                    component['plan_ref'], *grant['evidence']):
            old = evidence.setdefault(ref['path'], ref['sha256'])
            need(old == ref['sha256'], 'Conflicting composite evidence')
    decision_ref = dict(path=decision_path.relative_to(root).as_posix(),
                        sha256=file_digest(decision_path))
    old = evidence.setdefault(decision_ref['path'], decision_ref['sha256'])
    need(old == decision_ref['sha256'], 'Conflicting composition decision evidence')
    refs = [dict(path=p, sha256=sha) for p, sha in sorted(evidence.items())]
    clearance = seal(dict(schema=CLEARANCE, policy=POLICY,
        source_id=source['source_id'], revision=source['revision'],
        file_manifest_identity=manifest['identity'], decision='ACCEPT',
        evidence=refs, attribution=decision['attribution'],
        obligations=decision['obligations'],
        composition_decision_ref=decision_ref,
        composition_decision_identity=decision['identity'],
        components=decision['components']))
    verify(root, source, manifest, clearance)
    registry = copy.deepcopy(active_reg)
    registry['sources'] = [source if s['source_id'] == source['source_id'] else s
                           for s in registry['sources']]
    registry = seal(registry)
    spec = seal(dict(copy.deepcopy(active_spec), registry_identity=registry['identity']))
    approved = {f['path']: f for f in manifest_files}
    need(len(leaf_plans) == len(approved), 'Leaf plans do not cover exact union')
    records = list(carried)
    for record in leaf_plans:
        need(record['path'] in approved and
             record['sha256'] == approved[record['path']]['sha256'] and
             record['source_id'] == source['source_id'] and
             record['revision'] == source['revision'],
             'Leaf plan differs from approved file')
        value = dict(record)
        value.pop('identity')
        value.update(license_evidence=refs, attribution=decision['attribution'],
                     source_clearance_identity=clearance['identity'])
        records.append(seal(value))
    need(len({item_key(r) for r in records}) == len(records) and
         len({(r['source_id'], r['path'].casefold()) for r in records}) == len(records) and
         len({r['identity'] for r in records}) == len(records),
         'Duplicate prospective items')
    output.mkdir(parents=True, exist_ok=True)
    def publish(path, value):
        encoded = canonical(value)
        if path.exists():
            need(path.read_bytes() == encoded, 'Composition replay differs: '+str(path))
        else:
            atomic_json(path, value)
    publish(output/'files.json', manifest)
    publish(output/'clearance.json', clearance)
    validate(spec, registry, root)
    for record in records:
        if record['source_id'] == source['source_id']:
            validate_clearance(record, source, spec, root, approved[record['path']])
        else:
            validate_clearance(record, next(s for s in registry['sources']
                if s['source_id'] == record['source_id']), spec, root)
    publish(output/'registry.json', registry)
    publish(output/'spec.json', spec)
    plan = b''.join(canonical(r)+b'\n' for r in sorted(records, key=item_key))
    plan_path = output/'plan.jsonl'
    if plan_path.exists():
        need(plan_path.read_bytes() == plan, 'Composition plan replay differs')
    else:
        temporary = output/'plan.jsonl.tmp'
        with temporary.open('wb') as stream:
            stream.write(plan)
            stream.flush()
            os.fsync(stream.fileno())
        if plan_path.exists():
            need(plan_path.read_bytes() == plan, 'Composition plan replay differs')
            temporary.unlink()
        else:
            atomic_replace(temporary, plan_path)
    artifacts = [output/name for name in
                 ('files.json', 'clearance.json', 'registry.json', 'spec.json', 'plan.jsonl')]
    complete = seal(dict(policy=POLICY, decision_identity=decision['identity'],
        source_id=source['source_id'], revision=source['revision'],
        files=len(records), carried_forward_files=len(carried),
        prior_approved_plan_ref=(dict(path=prior_path.relative_to(root).as_posix(),
            sha256=prior_sha) if prior_path is not None else None),
        production_tokens=0, training_authorized=False,
        artifacts={p.name: file_digest(p) for p in artifacts}))
    publish(output/'complete.json', complete)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision', type=Path, required=True)
    parser.add_argument('--approved-decision-sha256', required=True)
    parser.add_argument('--approved-component-decision-sha256', action='append',
                        required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior-approved-plan', type=Path)
    parser.add_argument('--approved-prior-plan-sha256')
    args = parser.parse_args()
    try:
        print(json.dumps(build(args.decision, args.output,
            expected_decision_sha256=args.approved_decision_sha256,
            expected_component_decision_sha256=args.approved_component_decision_sha256,
            prior_approved_plan=args.prior_approved_plan,
            expected_prior_plan_sha256=args.approved_prior_plan_sha256),
            sort_keys=True))
        return 0
    except Exception as exc:
        print(f'Composition stopped: {type(exc).__name__}: {exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
