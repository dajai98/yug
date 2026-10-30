"""Synthetic retained holdouts; tests never require ignored local artifacts."""
import hashlib
import json
from pathlib import Path

import pytest

from training.experiments.acquisition_readiness_gate import retained_heldout_inventory
from training.production_corpus import canonical, file_digest
from training.production_spec import SpecificationError, seal
from training.protected_reference_migration import compile_inventory, compile_records
from training.protected_references import read_inventory, normalized_text


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b''.join(canonical(row)+b'\n' for row in rows))


@pytest.fixture
def retained(tmp_path):
    fit = []
    for i in range(35):
        text = f'print("retained fit {i}")\n'
        fit.append(dict(document_id=f'fit-{i:02d}', text=text,
                        content_hash=hashlib.sha256(text.encode()).hexdigest(),
                        partition='validation', provenance={'path':f'src/file_{i}.py',
                                                            'source':'fixture-fit'},
                        primary_domain='code', language='python'))
    transfer = []
    for i in range(24):
        text = f'# Retained transfer {i}\n'
        transfer.append(dict(id=f'doc-{i:02d}', text=text,
                             text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                             path=f'docs/doc_{i}.md', repo='fixture/transfer', commit='a'*40))
    write_rows(tmp_path/'artifacts/tokenizer_fit_sample_gate_v1/validation.jsonl', fit)
    write_rows(tmp_path/'artifacts/tokenizer_transfer_gate_v1/documents.jsonl', transfer)
    current, sources = retained_heldout_inventory(tmp_path)
    old = [{key: row[key] for key in ('reference_id','text','content_type','language','provenance')}
           for row in current]
    for row in old:
        if row['language'] == 'und':
            row['language'] = ''
    p1 = tmp_path/'artifacts/acquisition_readiness_gate_v1'
    write_rows(p1/'heldout_references.jsonl', old)
    (p1/'heldout_references_manifest.json').write_bytes(canonical(seal(dict(
        inputs=sources, references=59,
        inventory_sha256=file_digest(p1/'heldout_references.jsonl'),
        production_clearance_granted=False, training_tokens_created=0))))
    gsm = []
    decisions = []
    for i in range(32):
        doc = 'sha256:' + f'{i+1:064x}'
        text = canonical(dict(question=f'Question {i}?', answer=f'Answer {i}.')).decode()
        provenance = dict(dataset_id='openai/gsm8k', document_identity=doc,
                          source_id='fixture-gsm8k', revision='b'*40,
                          reason='benchmark excluded', aliases=['gsm8k'])
        gsm.append(dict(reference_id='gsm8k:'+doc, text=text,
                        content_type='prose', language='en', provenance=provenance))
        decisions.append(seal(dict(document_identity=doc, source_id='fixture-gsm8k',
                                   revision='b'*40, reason='benchmark excluded',
                                   sha256='sha256:'+hashlib.sha256(text.encode()).hexdigest(),
                                   admission_status='HELD_OUT', training_eligibility='EXCLUDE',
                                   production_grant=False, rights_status='CLEAR',
                                   upstream_split='train')))
    g = tmp_path/'artifacts/gsm8k_admission_v1'
    write_rows(g/'gsm8k_holdout_references.jsonl', gsm)
    write_rows(g/'decisions.jsonl', decisions)
    write_rows(g/'combined_holdout_references.jsonl', sorted(old+gsm,key=lambda row:row['reference_id']))
    (g/'result.json').write_bytes(canonical(seal(dict(
        combined_reference_count=91, historical_references_preserved=59,
        held_out=32, excluded_from_training=32,
        production_grants_created=0, training_eligible_documents=0,
        combined_references_sha256=file_digest(g/'combined_holdout_references.jsonl')))))
    return tmp_path


def legacy_inputs(root):
    p1=root/'artifacts/acquisition_readiness_gate_v1'
    g=root/'artifacts/gsm8k_admission_v1'
    current,_=retained_heldout_inventory(root)
    rows=lambda p:[json.loads(line) for line in p.read_bytes().splitlines()]
    return dict(phase1=current, old_phase1=rows(p1/'heldout_references.jsonl'),
                gsm8k=rows(g/'gsm8k_holdout_references.jsonl'),
                combined=rows(g/'combined_holdout_references.jsonl'),
                decisions=rows(g/'decisions.jsonl'),
                gsm8k_sha256=file_digest(g/'gsm8k_holdout_references.jsonl'),
                decisions_sha256=file_digest(g/'decisions.jsonl'),
                combined_sha256=file_digest(g/'combined_holdout_references.jsonl'))


def test_complete_deterministic_lossless_and_non_authorizing(retained):
    before={p.relative_to(retained).as_posix():file_digest(p)
            for p in (retained/'artifacts').rglob('*') if p.is_file()}
    output=retained/'new_inventory'
    first=compile_inventory(retained,output)
    records=list(read_inventory(output/'references.jsonl'))
    assert (first['phase1_items'],first['gsm8k_items'],first['source_items'],first['protected_records']) == (59,32,91,155)
    assert not first['training_allowed'] and first['production_grants_created']==0
    assert len(records)==155 and {r['reference_id'] for r in records[:59]} == {r['reference_id'] for r in legacy_inputs(retained)['old_phase1']}
    for i in range(32):
        composite,question,answer=records[59+3*i:62+3*i]
        assert composite['role']=='reference' and composite['text']==composite['provenance']['original_record']['text']
        assert composite['reference_id']==composite['provenance']['original_record']['reference_id']
        assert question['role']=='question' and answer['role']=='answer'
        assert composite['protected_split']==question['protected_split']==answer['protected_split']=='evaluation'
        assert (composite['work_id'],composite['work_version'],composite['item_id']) == (question['work_id'],question['work_version'],question['item_id']) == (answer['work_id'],answer['work_version'],answer['item_id'])
        original=question['provenance']['original_record']
        assert original==answer['provenance']['original_record']
        assert json.loads(original['text'])=={'question':question['text'],'answer':answer['text']}
        assert question['provenance']['decision_identity']==answer['provenance']['decision_identity']
        assert question['raw_text_sha256']=='sha256:'+hashlib.sha256(question['text'].encode()).hexdigest()
        assert question['normalized_text_sha256']=='sha256:'+hashlib.sha256(
            normalized_text(question['text'], question['content_type']).encode()).hexdigest()
    assert compile_inventory(retained,output)==first
    assert before=={p.relative_to(retained).as_posix():file_digest(p)
                    for p in (retained/'artifacts').rglob('*') if p.is_file()}


@pytest.mark.parametrize('damage', ['missing_phase1','missing_gsm','duplicate_id','malformed','changed_decision','ambiguous_question'])
def test_incomplete_or_ambiguous_legacy_fails_closed(retained, damage):
    values=legacy_inputs(retained)
    if damage=='missing_phase1': values['combined'].pop(0)
    elif damage=='missing_gsm': values['gsm8k'].pop()
    elif damage=='duplicate_id': values['gsm8k'][1]['reference_id']=values['gsm8k'][0]['reference_id']
    elif damage=='malformed': values['old_phase1'][0].pop('provenance')
    elif damage=='changed_decision':
        values['decisions'][0]=seal(dict(values['decisions'][0],production_grant=True))
    else:
        values['gsm8k'][0]['text']='{"question":"Q","question":"duplicate","answer":"A"}'
        values['combined']=[dict(r, text=values['gsm8k'][0]['text']) if r['reference_id']==values['gsm8k'][0]['reference_id'] else r for r in values['combined']]
        values['decisions'][0]=seal(dict(values['decisions'][0],sha256='sha256:'+hashlib.sha256(values['gsm8k'][0]['text'].encode()).hexdigest()))
    with pytest.raises((SpecificationError,KeyError,ValueError)):
        compile_records(**values)


def test_changed_source_or_existing_output_fails_closed(retained):
    output=retained/'new_inventory'
    compile_inventory(retained,output)
    (output/'references.jsonl').write_bytes(b'corrupt')
    with pytest.raises(SpecificationError):compile_inventory(retained,output)
    (retained/'artifacts/tokenizer_fit_sample_gate_v1/validation.jsonl').write_bytes(b'changed')
    with pytest.raises(SpecificationError,match='source input hash'):
        compile_inventory(retained,retained/'other_output')
