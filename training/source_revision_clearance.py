"""Prepare provisional exact-file R2 clearance using the existing review validators.

This stage cannot activate a revision or produce a production plan. Every selected
file needs a fresh, externally pinned R2 decision and R2-bound rights evidence.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from training.category_ownership import decision as ownership_decision
from training.clearance_evidence import verify_bundle
from training.language_validation import validate_label
from training.production_acquisition import read_plan
from training.production_corpus import atomic_json, canonical, file_digest
from training.production_spec import (local_path, need, path_allowed, seal,
                                      source_manifest, validate)
from training.source_admission import cached_sources, holdout_groups
from training.source_approval import payload
from training.source_first_clearance import (
    BOUNDED_RIGHTS, SHA, _read, _ref, validate_candidate_batch,
    validate_decision_scope, validate_rights_record, validate_source_license,
)
from training.source_prepare import ROOT, git, screening
from training.source_revision_inventory import build as compare_revisions
from training.notice_attribution import source_clearance

POLICY = 'reviewed-revision-file-clearance-v1'
EXTRA_FIELDS = frozenset({'old_revision', 'comparison_sha256',
                          'revision_license_evidence', 'excluded_paths',
                          'removed_paths'})


def _pin(path, expected):
    need(type(expected) is str and SHA.fullmatch(expected) and
         path.is_file() and file_digest(path) == expected,
         'Required external byte pin missing or stale: '+str(path))


def _paths(values, label):
    need(type(values) is list and values == sorted(set(values)) and
         all(type(p) is str and bool(p) for p in values),
         'Missing, duplicated or unordered '+label)
    return set(values)


def validate_disposition(decision, comparison):
    """Account for every complete R2 inventory path and every removed R1 path."""
    need(decision['old_revision'] == comparison['old_revision'] and
         decision['revision'] == comparison['new_revision'] and
         decision['comparison_sha256'] == 'sha256:'+hashlib.sha256(
             canonical(comparison)).hexdigest(),
         'R2 decision does not bind exact R1/R2 comparison')
    classified = {row['path']: row for row in comparison['files']}
    need(len(classified) == len(comparison['files']),
         'Duplicate comparison file')
    available = {p for p, row in classified.items() if row['new'] is not None}
    removed = {p for p, row in classified.items() if row['new'] is None}
    approved = {f['path'] for f in decision['files']}
    excluded = _paths(decision['excluded_paths'], 'R2 exclusions')
    need(_paths(decision['removed_paths'], 'R1 removals') == removed and
         approved.isdisjoint(excluded) and approved | excluded == available and
         all(classified[p]['new']['screening'] != 'EXCLUDED' for p in approved),
         'Incomplete or conflicting R1/R2 file disposition')
    return classified


def build(decision_path, output, *, expected_decision_sha256,
          old_cache, new_cache, registry_path, spec_path, manifest_path,
          references_path, comparison_dir, expected_comparison_sha256,
          expected_comparison_receipt_sha256, expected_registry_sha256,
          expected_spec_sha256, expected_manifest_sha256,
          expected_old_clearance_sha256,
          expected_references_sha256, expected_old_inventory_sha256,
          expected_new_inventory_sha256, expected_old_receipt_sha256,
          expected_new_receipt_sha256, review, catalog,
          expected_review_receipt_sha256, root=ROOT):
    """Publish only provisional R2 manifest, clearance and ownership evidence."""
    root = Path(root).resolve()
    def resolve(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else local_path(root, str(path))
    (decision_path, output, old_cache, new_cache, registry_path, spec_path,
     manifest_path, references_path, comparison_dir, review, catalog) = map(resolve,
        (decision_path, output, old_cache, new_cache, registry_path, spec_path,
         manifest_path, references_path, comparison_dir, review, catalog))
    # The active registry pins a path, not the bytes of its R1 clearance.
    # Require an independent byte pin for that historical decision as well.
    inputs = [decision_path, registry_path, spec_path, manifest_path,
              references_path, comparison_dir/'comparison.json',
              comparison_dir/'complete.json', review/'request.json',
              review/'complete.json', review/'evidence.jsonl', catalog,
              new_cache.parent/'status.json', old_cache/'complete.json',
              new_cache/'complete.json', old_cache/'request.json',
              new_cache/'request.json']
    need(output.is_relative_to(root/'artifacts') and
         all(p.is_relative_to(root) and not p.is_relative_to(output)
             for p in inputs) and
         all(not output.is_relative_to(p) and not p.is_relative_to(output)
             for p in (old_cache, new_cache, comparison_dir, review)),
         'R2 provisional output overlaps inputs or escapes artifacts')
    for path, expected in (
        (decision_path, expected_decision_sha256),
        (comparison_dir/'comparison.json', expected_comparison_sha256),
        (comparison_dir/'complete.json', expected_comparison_receipt_sha256),
        (review/'complete.json', expected_review_receipt_sha256),
        (registry_path, expected_registry_sha256),
        (spec_path, expected_spec_sha256),
        (manifest_path, expected_manifest_sha256),
        (references_path, expected_references_sha256),
        (old_cache/'complete.json', expected_old_receipt_sha256),
        (new_cache/'complete.json', expected_new_receipt_sha256),
        (old_cache/'inventory.jsonl', expected_old_inventory_sha256),
        (new_cache/'inventory.jsonl', expected_new_inventory_sha256)):
        _pin(path, expected)
    registry = _read(registry_path, maximum=16_777_216, canonical_required=False)
    decision = _read(decision_path, maximum=1_048_576)
    matched_source = [s for s in registry['sources']
                      if s['source_id'] == decision.get('source_id')]
    need(len(matched_source) == 1 and
         type(matched_source[0].get('clearance_record')) is str,
         'Missing uniquely registered historical clearance')
    old_clearance_path = local_path(root, matched_source[0]['clearance_record'])
    need(not old_clearance_path.is_relative_to(output),
         'R1 clearance cannot be output-local')
    _pin(old_clearance_path, expected_old_clearance_sha256)
    inputs.append(old_clearance_path)
    need(decision.get('schema') == POLICY and
         type(decision.get('comparison_sha256')) is str and
         decision['comparison_sha256'] == expected_comparison_sha256,
         'Missing revision-specific, comparison-bound decision')
    comparison = _read(comparison_dir/'comparison.json', maximum=16_777_216)
    receipt = _read(comparison_dir/'complete.json')
    need(receipt.get('comparison_identity') == comparison['identity'] and
         receipt.get('comparison_sha256') == expected_comparison_sha256 and
         receipt.get('review_only') is True and
         receipt.get('training_authorized') is False and
         receipt.get('production_tokens') == 0,
         'Comparison receipt does not bind review-only report')
    # Re-run the accepted 038B comparator against all trusted input pins. Since
    # both comparison files must already exist, its replay cannot publish new data.
    recomputed = compare_revisions(old_cache, new_cache, registry_path, spec_path,
        manifest_path, references_path, comparison_dir,
        source_id=decision['source_id'], locator=decision['locator'],
        old_revision=decision['old_revision'], new_revision=decision['revision'],
        expected_registry_sha256=expected_registry_sha256,
        expected_spec_sha256=expected_spec_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_references_sha256=expected_references_sha256,
        expected_old_inventory_sha256=expected_old_inventory_sha256,
        expected_new_inventory_sha256=expected_new_inventory_sha256,
        expected_old_receipt_sha256=expected_old_receipt_sha256,
        expected_new_receipt_sha256=expected_new_receipt_sha256, root=root)
    need(canonical(recomputed) == canonical(comparison) and
         file_digest(comparison_dir/'complete.json') == expected_comparison_receipt_sha256,
         'Comparison replay or completion changed')
    spec = _read(spec_path, maximum=16_777_216, canonical_required=False)
    validate(spec, registry, root)
    matched = [s for s in registry['sources'] if s['source_id'] == decision['source_id']]
    need(len(matched) == 1, 'Missing or ambiguous cleared R1 source')
    old_source = matched[0]
    need(old_source['locator'] == decision['locator'] and
         old_source['revision'] == decision['old_revision'] and
         old_source['status'] == 'ACCEPT' and
         old_source.get('file_manifest') and old_source.get('clearance_record') and
         old_source['split'] == decision['split'] and
         decision['cap_tokens'] <= old_source['target_cap_tokens'] and
         decision['cap_tokens'] <= old_source['category_caps_tokens'][old_source['category']] and
         all(decision['cap_tokens'] <= spec['group_caps'][kind][old_source[kind]]
             for kind in ('repository_group', 'domain_group')),
         'R2 identity, split or established cap differs from cleared R1')
    old_clearance = source_clearance(old_source, root)  # Also checks composites.
    old_manifest = source_manifest(root, old_source)
    need(old_source['split_group'] not in holdout_groups(references_path, registry),
         'Protected R1/R2 source group conflict')
    license_ref = decision.get('revision_license_evidence')
    need(type(license_ref) is dict and
         {'path', 'sha256', 'upstream_path', 'text_sha256'} <= set(license_ref) and
         license_ref != old_source['license_evidence'],
         'Fresh R2 license evidence required')
    source = dict(old_source, revision=decision['revision'],
                  license_evidence=license_ref,
                  name=old_source.get('name') or
                       old_source['locator'].removeprefix('https://github.com/'))
    catalog_value = _read(catalog, maximum=16_777_216, canonical_required=False)
    cache_source = dict(old_source, revision=decision['revision'])
    need(catalog_value['schema'] == registry['schema'] and
         [s for s in catalog_value['sources'] if s['source_id'] == source['source_id']]
         == [cache_source],
         'R2 review catalog does not match compared cache source')
    caches = [entry for entry in cached_sources(new_cache.parent, catalog_value,
              spec, root) if entry[1]['source_id'] == source['source_id']]
    need(len(caches) == 1 and caches[0][0] == new_cache,
         'Missing or ambiguous R2 review cache')
    directory, _, inventory, done = caches[0]
    inv_rows = list(read_plan(inventory))
    rows = {r['path']: r for r in inv_rows}
    need(len(rows) == len(inv_rows) == sum(done['counts'].values()),
         'Incomplete or duplicate R2 inventory')
    approved = validate_decision_scope(decision, source, rows,
        policy=POLICY, extra_fields=EXTRA_FIELDS)
    need(decision['attribution'] == old_clearance['attribution'] and
         set(old_clearance.get('obligations', [])) <= set(decision['obligations']),
         'Existing attribution or NOTICE obligations cannot be dropped')
    classified = validate_disposition(decision, comparison)
    # First clearance and composition materialize their grant as an exact
    # allowlist. Recover the pre-grant routing policy from the receipt-bound
    # historical cache request, so its independent limits survive a revision.
    old_grant_paths = {f['path'] for f in old_manifest['files']}
    old_allow = old_source['allow_paths']
    original_allow = _read(old_cache/'request.json').get('source', {}).get('allow_paths')
    need(type(original_allow) is list, 'Missing original source path policy')
    if original_allow:
        need(all(path_allowed(f['path'], original_allow,
                 old_source['deny_paths']) for f in approved),
             'R2 grant exceeds original source routing policy')
    if set(old_allow) != old_grant_paths:
        need(all(path_allowed(f['path'], old_allow, old_source['deny_paths'])
                 for f in approved),
             'R2 grant exceeds established source scope')
    bundle = verify_bundle(review, root)
    need(bundle['complete'] and bundle['training_allowed'] is False and
         bundle['counts']['EVIDENCED'] == 0 and
         file_digest(review/'complete.json') == expected_review_receipt_sha256,
         'Missing or changed complete review-only bundle')
    request = _read(review/'request.json', canonical_required=False)
    for p in (catalog, references_path, registry_path, spec_path,
              new_cache.parent/'status.json', inventory):
        need(request['inputs'].get(p.relative_to(root).as_posix()) == file_digest(p),
             'R2 review does not bind compared cache/source/policy')
    review_list = [r for r in read_plan(review/'evidence.jsonl')
                   if r['source_id'] == source['source_id']]
    reviewed = {r['path']: r for r in review_list}
    need(len(review_list) == len(reviewed) == len(rows),
         'Incomplete or duplicate R2 review decisions')
    evidence = []
    for ref in decision['source_scope_evidence']:
        _ref(root, ref, output)
        need(ref not in evidence, 'Duplicate source scope evidence')
        evidence.append(ref)
    notice_paths = [p for p in rows if Path(p).name.casefold() in
                    {'notice', 'notice.txt', 'notice.md'}]
    need(not notice_paths or license_ref.get('canonical_evidence') is not None,
         'R2 NOTICE requires pinned canonical license/notice evidence')
    validate_source_license(root, output, source, rows, directory, evidence)
    prepared = []
    bounded_candidates = []
    for f in approved:
        validate_label(f['language'], f['content_type'])
        row = rows[f['path']]
        need(classified[f['path']]['new']['sha256'] == row['sha256'] and
             classified[f['path']]['new']['git_object'] == row['git_object'] and
             reviewed[f['path']]['status'] == 'REVIEW' and
             all(reviewed[f['path']][key] == row[key] for key in
                 ('source_id', 'revision', 'path', 'sha256', 'bytes')) and
             reviewed[f['path']]['cache_inventory_sha256'] == done['inventory_sha256'],
             'R2 file is not exact reviewed material')
        raw = payload(directory, row)
        screened, _ = screening(f['path'], row['git_mode'], row['bytes'], raw,
                                cache_source, spec)
        need(screened != 'EXCLUDED', 'R2 file is excluded by screening')
        rights = _read(_ref(root, f['rights_ref'], output), maximum=16384)
        bounded = {}
        if rights.get('schema') == BOUNDED_RIGHTS:
            bounded = dict(
                scope_decision=_read(_ref(root, rights['source_scope_ref'], output)),
                dossier=_read(_ref(root, rights['candidate_ref'], output)),
                history=_read(_ref(root, rights['file_history_ref'], output)),
                inventory_identity=done['inventory_sha256'])
            actual = git(directory/'git', 'log', '--max-count=257', '--format=%H',
                         source['revision'], '--', f['path']).stdout.decode('ascii').splitlines()
            shallow = git(directory/'git', 'rev-parse',
                          '--is-shallow-repository').stdout.strip() == b'true'
            if shallow:
                boundary = set((directory/'git/shallow').read_text().splitlines())
                actual = [oid for oid in actual if oid not in boundary]
            need(bounded['history']['commits'] and
                 all(oid in actual for oid in bounded['history']['commits']),
                 'R2 exact-path history differs from Git cache')
            bounded_candidates.append(bounded['dossier'])
        for support in validate_rights_record(
                rights, decision, f, raw=raw, git_object=row['git_object'],
                source_license_ref=license_ref, **bounded):
            _ref(root, support, output)
        evidence.append(f['rights_ref'])
        own = seal(dict(source_id=source['source_id'], revision=source['revision'],
                        path=f['path'], sha256=f['sha256'], owner=decision['owner'],
                        subcategory=decision['subcategory'], decision_type='manual_review',
                        rationale_ref=decision['identity'], reviewer=decision['reviewer']))
        own_path = output/'ownership'/(hashlib.sha256(f['path'].encode()).hexdigest()+'.json')
        prepared.append((f, row, own_path, own))
    if bounded_candidates:
        need(len(bounded_candidates) == len(approved) and
             'candidate_batch_ref' in decision,
             'Bounded R2 files require one exact candidate batch')
        batch = _read(_ref(root, decision['candidate_batch_ref'], output))
        validate_candidate_batch(batch, decision, source, done['inventory_sha256'],
                                 bounded_candidates)
    else:
        need('candidate_batch_ref' not in decision,
             'Unneeded candidate batch cannot confer R2 authority')
    evidence.extend(dict(path=p.relative_to(root).as_posix(), sha256=file_digest(p))
                    for p in (decision_path, comparison_dir/'comparison.json',
                              comparison_dir/'complete.json', inventory,
                              review/'complete.json'))
    need(len({ref['path'] for ref in evidence}) == len(evidence),
         'Duplicate or conflicting R2 evidence')
    expected_files = {output/'files.json', output/'clearance.json',
                      output/'complete.json', *(item[2] for item in prepared)}
    if output.exists():
        need(all(p in expected_files for p in output.rglob('*') if p.is_file()),
             'Unrecognized provisional output artifact')
    output.mkdir(parents=True, exist_ok=True)
    (output/'ownership').mkdir(exist_ok=True)
    def publish(path, value):
        encoded = canonical(value)
        if path.exists():
            need(path.read_bytes() == encoded, 'Conflicting provisional R2 replay')
        else:
            atomic_json(path, value)
    for _, _, path, own in prepared:
        publish(path, own)
    files = []
    for f, _, path, _ in prepared:
        files.append(dict(path=f['path'], sha256=f['sha256'], bytes=f['bytes'],
            owner=decision['owner'], subcategory=decision['subcategory'],
            content_type=f['content_type'], license_id=decision['license_id'],
            tags=[], ownership_decision=ownership_decision(
                source_id=source['source_id'], revision=source['revision'],
                path=f['path'], sha256=f['sha256'], owner=decision['owner'],
                subcategory=decision['subcategory'], decision_type='manual_review',
                rationale_ref=decision['identity'], evidence_ref=dict(
                    path=path.relative_to(root).as_posix(), sha256=file_digest(path)))))
    manifest = seal(dict(source_id=source['source_id'], revision=source['revision'],
                         files=files))
    publish(output/'files.json', manifest)
    clearance = seal(dict(schema=POLICY, source_id=source['source_id'],
        locator=source['locator'], old_revision=decision['old_revision'],
        revision=source['revision'], file_manifest_identity=manifest['identity'],
        decision='PROVISIONAL', evidence=evidence,
        attribution=decision['attribution'], obligations=decision['obligations'],
        governance_decision_identity=decision['identity'],
        comparison_identity=comparison['identity'], training_authorized=False))
    publish(output/'clearance.json', clearance)
    artifacts = [output/'files.json', output/'clearance.json',
                 *(item[2] for item in prepared)]
    complete = seal(dict(policy=POLICY, source_id=source['source_id'],
        old_revision=decision['old_revision'], revision=source['revision'],
        decision_identity=decision['identity'], comparison_identity=comparison['identity'],
        manifest_identity=manifest['identity'], clearance_identity=clearance['identity'],
        approved_paths=[f['path'] for f in approved],
        excluded_paths=decision['excluded_paths'], removed_paths=decision['removed_paths'],
        inputs={p.relative_to(root).as_posix():file_digest(p) for p in
                sorted(set(inputs) | {local_path(root, ref['path'])
                                     for ref in evidence})},
        artifacts={p.relative_to(output).as_posix():file_digest(p) for p in artifacts},
        review_only=True, training_authorized=False, production_tokens=0))
    publish(output/'complete.json', complete)
    return complete


def verify_provisional(output, *, expected_complete_sha256, root=ROOT):
    """Read-only byte/identity audit of a published provisional R2 bundle.

    Full input-lineage revalidation is performed by replaying ``build`` with
    the original trusted pins; this audit never fills in missing artifacts.
    """
    root = Path(root).resolve()
    output = Path(output).resolve()
    need(output.is_relative_to(root/'artifacts'),
         'Provisional R2 bundle escapes artifacts')
    _pin(output/'complete.json', expected_complete_sha256)
    complete = _read(output/'complete.json')
    need(complete.get('policy') == POLICY and
         complete.get('review_only') is True and
         complete.get('training_authorized') is False and
         complete.get('production_tokens') == 0,
         'Not a provisional R2 completion')
    for name, sha in complete['inputs'].items():
        _pin(local_path(root, name), sha)
    expected = {'complete.json'}
    for name, sha in complete['artifacts'].items():
        path = (output/name).resolve()
        need(path.is_relative_to(output), 'Provisional artifact escapes output')
        _pin(path, sha)
        expected.add(name)
    observed = {p.relative_to(output).as_posix() for p in output.rglob('*')
                if p.is_file()}
    need(observed == expected, 'Missing or unrecognized provisional artifact')
    manifest = _read(output/'files.json')
    clearance = _read(output/'clearance.json')
    need(manifest['identity'] == complete['manifest_identity'] and
         clearance['identity'] == complete['clearance_identity'] and
         clearance['schema'] == POLICY and clearance['decision'] == 'PROVISIONAL' and
         clearance['training_authorized'] is False and
         clearance['file_manifest_identity'] == manifest['identity'] and
         clearance['governance_decision_identity'] == complete['decision_identity'] and
         clearance['comparison_identity'] == complete['comparison_identity'] and
         clearance['old_revision'] == complete['old_revision'] and
         manifest['source_id'] == clearance['source_id'] == complete['source_id'] and
         manifest['revision'] == clearance['revision'] == complete['revision'] and
         [f['path'] for f in manifest['files']] == complete['approved_paths'],
         'Provisional manifest/clearance identity mismatch')
    need(all(complete['inputs'].get(ref['path']) == ref['sha256']
             for ref in clearance['evidence']),
         'Source rights/obligation evidence omitted from completion')
    for f in manifest['files']:
        ref = f['ownership_decision']['evidence_ref']
        path = local_path(root, ref['path'])
        need(path.is_relative_to(output/'ownership') and
             ref['sha256'] == complete['artifacts'].get(
                 path.relative_to(output).as_posix()),
             'Missing or altered R2 ownership evidence')
        _pin(path, ref['sha256'])
        own = _read(path)
        need(all(own[key] == f[key] for key in ('path', 'sha256', 'owner', 'subcategory')) and
             own['source_id'] == complete['source_id'] and
             own['revision'] == complete['revision'] and
             own['rationale_ref'] == complete['decision_identity'],
             'Ownership evidence differs from exact R2 decision')
    return complete
