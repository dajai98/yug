"""Verify an exact-file composition; this does not authenticate human authority.

The composition decision's byte digest must be trusted outside this module.
Prospective use additionally requires current external governance and use receipts.
"""
from __future__ import annotations

import json
from pathlib import Path

from training.clearance_evidence import verify_bundle
from training.production_corpus import file_digest
from training.production_spec import (iter_manifest_files, local_path, need,
                                      verify_seal)

POLICY = 'reviewed-exact-file-composition-v1'
CLEARANCE = 'yug-composite-source-clearance-v1'
FIRST = 'reviewed-first-source-clearance-v1'


def read_ref(root, ref):
    need(type(ref) is dict and set(ref) == {'path', 'sha256'},
         'Malformed composite reference')
    path = local_path(root, ref['path'])
    need(path.is_file() and 0 < path.stat().st_size <= 16_777_216 and
         file_digest(path) == ref['sha256'],
         'Composite reference missing or changed')
    def unique(pairs):
        value = {}
        for key, item in pairs:
            need(key not in value, 'Duplicate composite evidence field')
            value[key] = item
        return value
    value = json.loads(path.read_bytes(), object_pairs_hook=unique)
    verify_seal(value)
    return value


def verify(root, source, manifest, grant):
    """Check both leaf grants, their completion receipts, and the exact union."""
    verify_seal(manifest)
    verify_seal(grant)
    need(grant.get('schema') == CLEARANCE and
         grant.get('policy') == POLICY, 'Wrong composite clearance policy')
    decision = read_ref(root, grant.get('composition_decision_ref'))
    need(decision.get('schema') == POLICY and
         decision.get('decision') == 'COMPOSE_EXACT_FILES' and
         decision.get('reviewer') and decision.get('decided_at') and
         grant.get('composition_decision_identity') == decision['identity'],
         'Missing externally reviewed composition decision')
    need(source.get('approval_composition') == dict(
         policy=POLICY, decision_identity=decision['identity']),
         'Source does not retain composite authority binding')
    common = ('source_id', 'locator', 'revision', 'split', 'repository_group',
              'domain_group', 'split_group', 'category', 'subcategory',
              'license_id', 'target_cap_tokens', 'category_caps_tokens')
    need(all(decision.get(k) == source.get(k) for k in common) and
         decision.get('attribution') == grant.get('attribution') and
         decision.get('obligations') == grant.get('obligations'),
         'Composite source policy or terms changed')
    original_registry = read_ref(root, decision.get('original_registry_ref'))
    original_spec = read_ref(root, decision.get('original_spec_ref'))
    active_registry = read_ref(root, decision.get('active_registry_ref'))
    active_spec = read_ref(root, decision.get('active_spec_ref'))
    catalog = read_ref(root, decision.get('catalog_ref'))
    reference = decision.get('references_ref')
    need(type(reference) is dict and set(reference) == {'path', 'sha256'} and
         file_digest(local_path(root, reference['path'])) == reference['sha256'],
         'Protected references changed after composition')
    review = local_path(root, decision.get('review_path'))
    cache = local_path(root, decision.get('cache_path'))
    reviewed = verify_bundle(review, root)
    need(reviewed['complete'] and reviewed['training_allowed'] is False and
         reviewed['counts']['EVIDENCED'] == 0,
         'Composite original review evidence changed')
    request_path = review/'request.json'
    request = json.loads(request_path.read_bytes())
    verify_seal(request)
    for ref in (decision['original_registry_ref'],decision['original_spec_ref'],
                decision['catalog_ref'],reference):
        need(request['inputs'].get(ref['path']) == ref['sha256'],
             'Composite review is not bound to original policy')
    status = cache/'status.json'
    need(request['inputs'].get(status.relative_to(Path(root).resolve()).as_posix()) ==
         file_digest(status), 'Composite cache differs from reviewed inventory')
    need(original_spec.get('registry_identity') == original_registry['identity'] and
         active_spec.get('registry_identity') == active_registry['identity'] and
         active_spec.get('group_caps') == original_spec.get('group_caps') and
         active_spec.get('categories') == original_spec.get('categories') and
         active_spec.get('three_d') == original_spec.get('three_d'),
         'Original allocation or active policy changed')
    need(catalog.get('schema') == original_registry.get('schema') and
         len([s for s in catalog.get('sources', [])
              if s['source_id'] == source['source_id'] and
              s['locator'] == source['locator'] and
              s['revision'] == source['revision']]) == 1,
         'Composite discovery source differs from original review')
    original = [s for s in original_registry['sources']
                if s['source_id'] == source['source_id']]
    active = [s for s in active_registry['sources']
              if s['source_id'] == source['source_id']]
    need(len(original) == len(active) == 1 and
         original[0]['status'] == 'REVIEW' and
         not original[0].get('file_manifest') and
         not original[0].get('clearance_record') and
         active[0]['status'] == 'ACCEPT' and
         all(source.get(k) == active[0].get(k) for k in source
             if k not in ('allow_paths','file_manifest','clearance_record',
                          'approval_composition')) and
         all(original[0].get(k) == active[0].get(k) for k in
             ('source_id','locator','revision','license_id','repository_group',
              'domain_group','split_group','category','subcategory','deny_paths')) and
         len(original_registry['sources']) == len(active_registry['sources']) and
         all(s == next((a for a in active_registry['sources']
                        if a['source_id'] == s['source_id']), None)
             for s in original_registry['sources'] if s['source_id'] != source['source_id']),
         'Composite baseline or unrelated source changed')
    components = decision.get('components')
    need(type(components) is list and len(components) == 2 and
         components == sorted(components, key=lambda c: c['decision_identity']) and
         grant.get('components') == components and
         len({c['decision_identity'] for c in components}) == 2,
         'Missing or ambiguous independent components')
    expected = []
    evidence = {}
    active_components = 0
    common_notice = None
    for component in components:
        need(type(component) is dict and set(component) == {
            'decision_ref', 'decision_identity', 'manifest_ref',
            'manifest_identity', 'clearance_ref', 'clearance_identity',
            'complete_ref', 'complete_identity', 'plan_ref'},
            'Malformed component lineage')
        leaf_decision = read_ref(root, component['decision_ref'])
        leaf_manifest = read_ref(root, component['manifest_ref'])
        leaf_clearance = read_ref(root, component['clearance_ref'])
        complete = read_ref(root, component['complete_ref'])
        need(leaf_decision['identity'] == component['decision_identity'] and
             leaf_manifest['identity'] == component['manifest_identity'] and
             leaf_clearance['identity'] == component['clearance_identity'] and
             complete['identity'] == component['complete_identity'] and
             leaf_decision.get('schema') == FIRST and
             leaf_decision.get('decision') == 'APPROVE_EXACT_FILES' and
             leaf_decision.get('reviewer') and
             leaf_clearance.get('schema') != CLEARANCE and
             leaf_clearance.get('governance_decision_identity') == leaf_decision['identity'] and
             leaf_clearance.get('file_manifest_identity') == leaf_manifest['identity'] and
             leaf_clearance.get('decision') == 'ACCEPT' and
             complete.get('policy') == FIRST and
             complete.get('decision_identity') == leaf_decision['identity'] and
             complete.get('training_authorized') is False and
             complete.get('production_tokens') == 0,
             'Component is not a completed independent human grant')
        need(all(leaf_decision.get(k) == decision.get(k) for k in
                 ('source_id', 'locator', 'revision', 'license_id')) and
             leaf_decision['owner'] == decision['category'] and
             leaf_decision['subcategory'] == decision['subcategory'] and
             leaf_decision['split'] == decision['split'] and
             leaf_decision['attribution'] == decision['attribution'] and
             leaf_decision['obligations'] == decision['obligations'] and
             leaf_decision['cap_tokens'] >= decision['target_cap_tokens'] and
             leaf_clearance.get('attribution') == decision['attribution'] and
             leaf_clearance.get('obligations') == decision['obligations'] and
             leaf_manifest.get('source_id') == decision['source_id'] and
             leaf_manifest.get('revision') == decision['revision'],
             'Incompatible component source, terms, or cap')
        leaf_dir = local_path(root, component['complete_ref']['path']).parent
        need(local_path(root, component['complete_ref']['path']) ==
             leaf_dir/'complete.json', 'Component completion receipt alias')
        if (local_path(root, decision['active_registry_ref']['path']) ==
                leaf_dir/'registry.json'):
            active_components += 1
            need(active[0]['file_manifest'] == component['manifest_ref']['path'] and
                 active[0]['clearance_record'] == component['clearance_ref']['path'],
                 'Active component grant differs from registered source')
        artifacts = complete.get('artifacts')
        need(type(artifacts) is dict and artifacts and
             complete.get('files') == len(leaf_decision['files']),
             'Incomplete component receipt')
        for name, sha in artifacts.items():
            artifact = (leaf_dir / name).resolve()
            need(artifact.is_relative_to(leaf_dir) and artifact.is_file() and
                 file_digest(artifact) == sha, 'Component artifact changed')
        for key, name in (('manifest_ref', 'files.json'),
                          ('clearance_ref', 'clearance.json'),
                          ('plan_ref', 'plan.jsonl')):
            path = local_path(root, component[key]['path'])
            need(path == leaf_dir / name and
                 artifacts.get(name) == component[key]['sha256'],
                 'Component path differs from completion receipt')
        need(file_digest(local_path(root, component['plan_ref']['path'])) ==
             component['plan_ref']['sha256'], 'Component plan changed')
        leaf_files = list(iter_manifest_files(root, leaf_manifest))
        need(len(leaf_files) == len(leaf_decision['files']) and
             [(f['path'], f['sha256']) for f in leaf_files] ==
             [(f['path'], f['sha256']) for f in leaf_decision['files']],
             'Component manifest differs from exact-file decision')
        need(all(f['bytes'] == approved['bytes'] and
                 f['content_type'] == approved['content_type'] and
                 f.get('ownership_decision', {}).get('rationale_ref') ==
                 leaf_decision['identity'] and
                 approved['rights_ref'] in leaf_clearance['evidence']
                 for f, approved in zip(leaf_files, leaf_decision['files'])) and
             all(ref in leaf_clearance['evidence'] for ref in
                 leaf_decision['source_scope_evidence']),
             'Component rights or ownership evidence omitted')
        plan_path = local_path(root, component['plan_ref']['path'])
        with plan_path.open('rb') as stream:
            plan_rows = [json.loads(line) for line in stream if line.strip()]
        need(len(plan_rows) == len(leaf_files) and
             len({row['path'].casefold() for row in plan_rows}) == len(plan_rows) and
             {(row['path'], row['sha256']) for row in plan_rows} ==
             {(f['path'], f['sha256']) for f in leaf_files} and
             all(row.get('source_clearance_identity') == leaf_clearance['identity']
                 and row.get('attribution') == leaf_clearance['attribution']
                 and row.get('license_evidence') == leaf_clearance['evidence']
                 for row in plan_rows),
             'Component plan is not the exact independently cleared set')
        file_by_path = {f['path']: f for f in leaf_files}
        for row in plan_rows:
            verify_seal(row)
            f = file_by_path[row['path']]
            need(all(row.get(k) == f.get(k) for k in
                     ('bytes','owner','subcategory','content_type','license_id',
                      'ownership_decision')),
                 'Component plan changed file or ownership evidence')
        for f in leaf_files:
            need(f.get('owner') == decision['category'] and
                 f.get('subcategory') == decision['subcategory'] and
                 f.get('license_id') == decision['license_id'],
                 'Component file rights or ownership conflict')
            expected.append(f)
        for ref in leaf_clearance.get('evidence', []):
            path = local_path(root, ref['path'])
            need(file_digest(path) == ref['sha256'], 'Leaf rights evidence changed')
            old = evidence.setdefault(ref['path'], ref['sha256'])
            need(old == ref['sha256'], 'Conflicting component evidence')
        notice = sorted((r['path'], r['sha256'])
                        for r in leaf_clearance['evidence']
                        if Path(r['path']).name.casefold() in
                        {'notice', 'notice.txt', 'notice.md'})
        if common_notice is None:
            common_notice = notice
        else:
            need(notice == common_notice,
                 'Independent batches have incompatible NOTICE applicability')
    paths = [f['path'].casefold() for f in expected]
    need(active_components == 1 and len(paths) == len(set(paths)) and
         len(expected) == len(decision.get('files', [])),
         'Overlapping or missing composition files')
    union = sorted(expected, key=lambda f: f['path'])
    need(decision['files'] == [{'path': f['path'], 'sha256': f['sha256']}
                               for f in union] and
         source.get('allow_paths') == [f['path'] for f in union] and
         list(iter_manifest_files(root, manifest)) == union and
         grant.get('file_manifest_identity') == manifest['identity'],
         'Aggregate manifest differs from approved exact union')
    refs = grant.get('evidence')
    for ref in (grant['composition_decision_ref'],
                *(component[key] for component in components for key in
                  ('decision_ref', 'manifest_ref', 'clearance_ref',
                   'complete_ref', 'plan_ref'))):
        old = evidence.setdefault(ref['path'], ref['sha256'])
        need(old == ref['sha256'], 'Conflicting component reference')
    expected_refs = [dict(path=p, sha256=sha) for p, sha in sorted(evidence.items())]
    need(refs == expected_refs,
         'Aggregate evidence is not the exact independent evidence union')
    return decision
