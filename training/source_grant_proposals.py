"""Offline Batch 01 file-grant review dossiers; never issues a grant.

Existing package/file profiles may identify evidence worth reviewing. Neither a
root license nor a proposal supplies source clearance, ownership approval,
conditional approval, or training authority.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from training.clearance_evidence import verify_bundle
from training.production_acquisition import read_plan
from training.production_corpus import atomic_json, atomic_replace, canonical, file_digest
from training.production_spec import need, seal, verify_seal, local_path, validate
from training.source_admission import cached_sources
from training.source_approval import package_scope, mit_file_scope, file_reason, mit_file_reason
from training.source_prepare import ROOT

POLICY = 'source-grant-review-proposals-v1'


def _payload(directory, row):
    path = local_path(directory, row['local_blob'])
    need(path.is_relative_to(directory/'files') and path.is_file()
         and path.stat().st_size == row['bytes'] and file_digest(path) == row['sha256'],
         'Cached proposal payload corrupt')
    if row.get('git_object'):
        digest = hashlib.sha1(f"blob {row['bytes']}\0".encode())
        with path.open('rb') as stream:
            while block := stream.read(65536):
                digest.update(block)
        need(digest.hexdigest() == row['git_object'], 'Cached proposal Git object changed')
    need(row['bytes'] <= 1_048_576, 'Unbounded review payload')
    return path.read_bytes()


def _source_evidence(source, root):
    reference = source.get('license_evidence')
    if not isinstance(reference, dict):
        return None
    path = local_path(root, reference['path'])
    need(file_digest(path) == reference['sha256'], 'Source license metadata changed')
    value = json.loads(path.read_bytes()); verify_seal(value)
    need(value['url'] == 'https://api.github.com/repos/'+source['name']+'/license?ref='+source['revision']
         and value['body']['path'] == reference['upstream_path'],
         'Source license metadata revision/path mismatch')
    return reference


def _file_scope(source, rows, directory, root):
    if source['status'] != 'ACCEPT':
        return None, 'conditional or unapproved source requires source-specific governance'
    if _source_evidence(source, root) is None:
        return None, 'source license metadata not verified'
    scope, reason = package_scope(source, rows, directory, root)
    if scope is None and source['license_id'] == 'MIT':
        scope, reason = mit_file_scope(source, rows, directory, root)
    return scope, reason


def compile_proposals(cache, catalog, evidence, references, output, *, root=ROOT):
    root = Path(root).resolve()
    cache, catalog, evidence, references, output = map(lambda p: Path(p).resolve(),
                                                       (cache, catalog, evidence, references, output))
    need(output.is_relative_to(root/'artifacts') and not output.is_relative_to(cache)
         and not output.is_relative_to(evidence), 'Unsafe proposal output')
    base_reg_path = root/'training/specifications/source_registry_v1.json'
    base_spec_path = root/'training/specifications/production_corpus_v1.json'
    base_reg = json.loads(base_reg_path.read_bytes())
    spec = json.loads(base_spec_path.read_bytes())
    validate(spec, base_reg, root)
    registry = json.loads(catalog.read_bytes()); verify_seal(registry)
    bundle = verify_bundle(evidence, root)
    need(bundle['complete'] and bundle['training_allowed'] is False
         and bundle['counts']['EVIDENCED'] == 0, 'Only complete review-only evidence is supported')
    request = json.loads((evidence/'request.json').read_bytes())
    caches = list(cached_sources(cache, registry, spec, root))
    need(caches, 'No completed cached sources')
    expected_paths = [catalog, references, base_reg_path, base_spec_path,
                      cache/'status.json']
    for directory, _, inventory, _ in caches:
        expected_paths.extend((inventory, directory/'request.json', directory/'complete.json'))
    expected = {p.relative_to(root).as_posix(): file_digest(p) for p in expected_paths}
    need(request['inputs'] == expected,
         'Review-only evidence does not bind exactly this cache/catalog/policy/reference set')
    evidence_rows = {}
    for row in read_plan(evidence/'evidence.jsonl'):
        verify_seal(row)
        key = (row['source_id'], row['path'])
        need(key not in evidence_rows and row['request_identity'] == request['identity'],
             'Duplicate/stale file evidence')
        evidence_rows[key] = row
    input_paths = [*expected_paths, evidence/'request.json',
                   evidence/'complete.json', evidence/'evidence.jsonl']
    inputs = {p.relative_to(root).as_posix(): file_digest(p) for p in input_paths}
    proposal_request = seal(dict(policy=POLICY, inputs=inputs,
                                 clearance_identity=bundle['identity']))
    counts = Counter(); per_source = {}; output_rows = []; candidates = 0
    for directory, source, inventory, done in caches:
        rows = list(read_plan(inventory))
        need(len({r['path'] for r in rows}) == len(rows)
             and len(rows) == sum(done['counts'].values()), 'Incomplete/duplicate cache inventory')
        by_path = {r['path']:r for r in rows}
        scope, scope_reason = _file_scope(source, by_path, directory, root)
        source_counts = Counter()
        for row in rows:
            key = (source['source_id'], row['path'])
            need(key in evidence_rows, 'Missing file clearance evidence')
            observed = evidence_rows.pop(key)
            need(all(observed[k] == row[k] for k in ('source_id','revision','path','sha256','bytes'))
                 and observed['cache_inventory_sha256'] == done['inventory_sha256']
                 and observed['status'] in ('REVIEW','EXCLUDED'),
                 'File evidence/cache binding mismatch')
            candidate = False; reason = observed['reason']
            if observed['status'] == 'REVIEW':
                data = _payload(directory, row)
                if scope:
                    checker = mit_file_reason if scope.get('profile') == 'mit-attributed-file-v1' else file_reason
                    file_issue = checker(row, data, source, scope, by_path)
                    candidate = file_issue is None
                    reason = 'existing file-scope evidence candidate; external ownership/source approval still required' if candidate else file_issue
                else:
                    reason = scope_reason or 'no existing file-scope evidence profile'
            if candidate: candidates += 1
            source_counts[observed['status']] += 1
            counts[observed['status']] += 1
            output_rows.append(seal(dict(policy=POLICY, request_identity=proposal_request['identity'],
                source_id=row['source_id'], locator=row['locator'], revision=row['revision'],
                path=row['path'], sha256=row['sha256'], bytes=row['bytes'],
                cache_inventory_sha256=done['inventory_sha256'],
                clearance_evidence_identity=observed['identity'],
                clearance_status=observed['status'], reason=reason,
                source_status=source['status'], license_id=source['license_id'],
                license_evidence=source.get('license_evidence'),
                source_license_scope=source.get('license_scope'),
                allow_paths=source.get('allow_paths', []), deny_paths=source.get('deny_paths', []),
                file_scope_profile=scope.get('profile','package') if candidate else None,
                file_scope_evidence=scope['evidence'] if candidate else [],
                rights_scope_candidate=scope if candidate else None,
                owner_claim=source['category'], subcategory_claim=source['subcategory'],
                ownership_approved=False, source_clearance_approved=False,
                conditional_approval_identity=None, notice_obligations_approved=False,
                protected_reference_status=observed['status'],
                grantable=False, training_allowed=False)))
        per_source[source['source_id']] = dict(source_counts)
    need(not evidence_rows and sum(counts.values()) == bundle['reviewed_files'],
         'Unexpected/missing clearance evidence rows')
    payload = b''.join(canonical(row)+b'\n' for row in output_rows)
    proposal_sha = 'sha256:'+hashlib.sha256(payload).hexdigest()
    receipt = seal(dict(policy=POLICY, request_identity=proposal_request['identity'],
                        total=sum(counts.values()), reviewed=counts['REVIEW'],
                        excluded=counts['EXCLUDED'], file_scope_candidates=candidates,
                        legitimately_grantable=0, per_source=per_source,
                        proposals_sha256=proposal_sha, training_allowed=False))
    if output.exists():
        need((output/'request.json').is_file() and (output/'proposals.jsonl').is_file()
             and (output/'complete.json').is_file()
             and json.loads((output/'request.json').read_bytes()) == proposal_request
             and json.loads((output/'complete.json').read_bytes()) == receipt
             and file_digest(output/'proposals.jsonl') == proposal_sha,
             'Proposal replay differs; use a new output')
        return receipt
    output.mkdir(parents=True)
    atomic_json(output/'request.json', proposal_request)
    temp = output/'proposals.part'
    temp.write_bytes(payload)
    atomic_replace(temp, output/'proposals.jsonl')
    atomic_json(output/'complete.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--catalog', required=True, type=Path)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--references', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = compile_proposals(**vars(args))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
