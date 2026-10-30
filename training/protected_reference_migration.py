"""Compile retained Phase 1 and GSM8K holdouts into protected-reference v1.

This is an offline exclusion inventory, never clearance or training authority.
Historical inputs are read and pinned; a new output directory is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from training.experiments.acquisition_readiness_gate import retained_heldout_inventory
from training.production_corpus import atomic_json, atomic_replace, canonical, file_digest
from training.production_spec import need, seal, verify_seal
from training.protected_references import make_reference, read_inventory, validate_records
from training.source_prepare import ROOT

POLICY = 'retained-protected-reference-migration-v1'
PHASE1_COUNT = 59
GSM8K_COUNT = 32
PHASE1 = Path('artifacts/acquisition_readiness_gate_v1')
GSM8K = Path('artifacts/gsm8k_admission_v1')


def _unique(pairs):
    value = {}
    for key, item in pairs:
        need(key not in value, 'Duplicate JSON field in retained evidence')
        value[key] = item
    return value


def _rows(path):
    rows = []
    with path.open('rb') as stream:
        while line := stream.readline(1_048_577):
            need(len(line) <= 1_048_576 and bool(line.strip()), 'Malformed retained row')
            rows.append(json.loads(line, object_pairs_hook=_unique))
    return rows


def _digest(text):
    return 'sha256:' + hashlib.sha256(text.encode('utf-8')).hexdigest()


def compile_records(phase1, old_phase1, gsm8k, combined, decisions, *,
                    gsm8k_sha256, decisions_sha256, combined_sha256):
    """Require complete one-to-one legacy coverage before deriving components."""
    need(len(phase1) == len(old_phase1) == PHASE1_COUNT, 'Phase 1 item count changed')
    need(len(gsm8k) == len(decisions) == GSM8K_COUNT, 'GSM8K item count changed')
    need(len(combined) == PHASE1_COUNT + GSM8K_COUNT, 'Combined holdout count changed')
    old_ids = [r['reference_id'] for r in old_phase1]
    gsm_ids = [r['reference_id'] for r in gsm8k]
    combined_ids = [r['reference_id'] for r in combined]
    need(len(set(combined_ids)) == len(combined_ids)
         and set(combined_ids) == set(old_ids + gsm_ids)
         and len(set(old_ids + gsm_ids)) == len(combined_ids),
         'Missing/extra/duplicate retained item')
    combined_by_id = dict(zip(combined_ids, combined))
    for row in old_phase1 + gsm8k:
        need(combined_by_id[row['reference_id']] == row,
             'Combined inventory differs from original retained item')

    phase1_by_id = {r['reference_id']: r for r in phase1}
    need(len(phase1_by_id) == PHASE1_COUNT and set(phase1_by_id) == set(old_ids),
         'Phase 1 derivation omitted or duplicated an item')
    for old in old_phase1:
        need(type(old) is dict and set(old) == {'reference_id', 'text', 'content_type', 'language', 'provenance'},
             'Malformed Phase 1 retained record')
        current = phase1_by_id[old['reference_id']]
        need(all(current[key] == old[key] for key in ('reference_id', 'text', 'content_type', 'provenance'))
             and current['language'] == (old['language'] or 'und')
             and current['role'] == 'reference'
             and current['protected_split'] == ('validation' if old['reference_id'].startswith('fit_validation:') else 'evaluation'),
             'Phase 1 retained semantics changed')

    by_document = {}
    for decision in decisions:
        verify_seal(decision)
        doc = decision['document_identity']
        need(doc not in by_document, 'Duplicate GSM8K decision identity')
        by_document[doc] = decision
    records = list(phase1)
    for old in gsm8k:
        need(type(old) is dict and set(old) == {'reference_id', 'text', 'content_type', 'language', 'provenance'},
             'Malformed GSM8K retained record')
        provenance = old['provenance']
        need(type(provenance) is dict and old['content_type'] == 'prose' and old['language'] == 'en',
             'Ambiguous GSM8K route or provenance')
        doc = provenance['document_identity']
        need(old['reference_id'] == 'gsm8k:' + doc and doc in by_document,
             'GSM8K item/decision mismatch')
        decision = by_document[doc]
        need(decision['admission_status'] == 'HELD_OUT'
             and decision['training_eligibility'] == 'EXCLUDE'
             and decision['production_grant'] is False
             and decision['rights_status'] == 'CLEAR'
             and decision['source_id'] == provenance['source_id']
             and decision['revision'] == provenance['revision']
             and decision['reason'] == provenance['reason']
             and decision['sha256'] == _digest(old['text']),
             'GSM8K holdout decision/content mismatch')
        need(provenance['dataset_id'] == 'openai/gsm8k'
             and type(provenance['aliases']) is list,
             'Ambiguous GSM8K source or aliases')
        components = json.loads(old['text'], object_pairs_hook=_unique)
        need(type(components) is dict and set(components) == {'question', 'answer'}
             and all(type(components[part]) is str and components[part].strip()
                     for part in ('question', 'answer')),
             'GSM8K question/answer cannot be represented losslessly')
        for role in ('reference', 'question', 'answer'):
            records.append(make_reference(
                work_id=provenance['dataset_id'], work_version=provenance['revision'],
                item_id=doc, reference_id=old['reference_id'] if role == 'reference'
                else old['reference_id']+':'+role,
                role=role, protected_split='evaluation', aliases=provenance['aliases'],
                text=old['text'] if role == 'reference' else components[role],
                content_type='prose', language=old['language'],
                provenance=dict(source_id=provenance['source_id'], revision=provenance['revision'],
                                original_record=old, original_record_sha256=_digest(old['text']),
                                original_inventory_sha256=gsm8k_sha256,
                                combined_inventory_sha256=combined_sha256,
                                decision=decision, decision_identity=decision['identity'],
                                decisions_sha256=decisions_sha256, component_key=role,
                                upstream_split=decision['upstream_split'])))
    need(len(records) == PHASE1_COUNT + 3*GSM8K_COUNT,
         'Protected component count changed')
    return list(validate_records(records))


def compile_inventory(root=ROOT, output=None):
    root = Path(root).resolve()
    need(output is not None, 'New output directory required')
    output = Path(output).resolve()
    need(output.is_relative_to(root) and not output.is_relative_to(root/PHASE1)
         and not output.is_relative_to(root/GSM8K), 'Output overlaps retained input')
    p1_manifest = root/PHASE1/'heldout_references_manifest.json'
    p1_legacy = root/PHASE1/'heldout_references.jsonl'
    gsm_file = root/GSM8K/'gsm8k_holdout_references.jsonl'
    combined_file = root/GSM8K/'combined_holdout_references.jsonl'
    decisions_file = root/GSM8K/'decisions.jsonl'
    gsm_result_file = root/GSM8K/'result.json'
    manifest = json.loads(p1_manifest.read_bytes(), object_pairs_hook=_unique)
    result = json.loads(gsm_result_file.read_bytes(), object_pairs_hook=_unique)
    verify_seal(manifest); verify_seal(result)
    need(manifest['references'] == PHASE1_COUNT and manifest['production_clearance_granted'] is False
         and manifest['training_tokens_created'] == 0
         and file_digest(p1_legacy) == manifest['inventory_sha256'],
         'Phase 1 retained manifest/inventory mismatch')
    need(result['combined_reference_count'] == PHASE1_COUNT + GSM8K_COUNT
         and result['historical_references_preserved'] == PHASE1_COUNT
         and result['held_out'] == result['excluded_from_training'] == GSM8K_COUNT
         and result['production_grants_created'] == result['training_eligible_documents'] == 0
         and file_digest(combined_file) == result['combined_references_sha256'],
         'Combined GSM8K retained receipt/inventory mismatch')
    for source in manifest['inputs']:
        need(file_digest(root/source['path']) == source['sha256'],
             'Retained source input hash mismatch')
        _rows(root/source['path'])  # Reject ambiguous duplicate JSON fields.
    phase1, producer_inputs = retained_heldout_inventory(root)
    need(producer_inputs == manifest['inputs'], 'Retained producer inputs changed')
    paths = [p1_manifest, p1_legacy, gsm_file, combined_file, decisions_file, gsm_result_file]
    input_hashes = {p.relative_to(root).as_posix(): file_digest(p) for p in paths}
    records = compile_records(
        phase1, _rows(p1_legacy), _rows(gsm_file), _rows(combined_file), _rows(decisions_file),
        gsm8k_sha256=input_hashes[gsm_file.relative_to(root).as_posix()],
        decisions_sha256=input_hashes[decisions_file.relative_to(root).as_posix()],
        combined_sha256=input_hashes[combined_file.relative_to(root).as_posix()])
    input_hashes.update({x['path']: x['sha256'] for x in manifest['inputs']})
    payload = b''.join(canonical(record)+b'\n' for record in records)
    inventory_sha = 'sha256:' + hashlib.sha256(payload).hexdigest()
    receipt = seal(dict(policy=POLICY, inputs=input_hashes,
                        phase1_items=PHASE1_COUNT, gsm8k_items=GSM8K_COUNT,
                        source_items=PHASE1_COUNT+GSM8K_COUNT,
                        protected_records=len(records),
                        protected_split_gsm8k='evaluation',
                        inventory_sha256=inventory_sha, training_allowed=False,
                        production_grants_created=0))
    if output.exists():
        need((output/'manifest.json').is_file() and (output/'references.jsonl').is_file(),
             'Existing output is incomplete/unowned')
        need(json.loads((output/'manifest.json').read_bytes()) == receipt
             and file_digest(output/'references.jsonl') == inventory_sha,
             'Existing protected inventory differs; use a new output')
        need(len(list(read_inventory(output/'references.jsonl'))) == len(records),
             'Existing protected inventory invalid')
        return receipt
    output.mkdir(parents=True)
    temp = output/'references.part'
    temp.write_bytes(payload)
    atomic_replace(temp, output/'references.jsonl')
    need(len(list(read_inventory(output/'references.jsonl'))) == len(records),
         'Produced protected inventory invalid')
    atomic_json(output/'manifest.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compile_inventory(output=args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
