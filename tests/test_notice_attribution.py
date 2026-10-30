"""Diagnostic obligations only; no fixture grants production rights."""
import json

import pytest

from training.notice_attribution import (FILENAME, build_notice, reference,
                                         verify_file)
from training.production_acquisition import ProductionBuild, validate_clearance
from training.production_corpus import canonical, file_digest
from training.production_spec import SpecificationError, seal
from tests.test_production_acquisition import Tokenizer, fixture_plan
from tests.test_corpus_gate import setup, bundle, builder


@pytest.mark.parametrize('mutation', ['attribution','evidence','source','revision','reseal_same_terms',
                                      'transplant','item_attribution','item_evidence',
                                      'item_license','missing_binding'])
def test_item_binds_exact_source_obligations(tmp_path, mutation):
    spec,registry,plan,records=fixture_plan(tmp_path)
    source=registry['sources'][0]
    record=records[0]
    assert validate_clearance(record,source,spec,tmp_path)
    grant_path=tmp_path/'clear0.json'
    grant=json.loads(grant_path.read_bytes())
    if mutation=='attribution':grant['attribution']='Unrelated attribution'
    elif mutation=='evidence':
        other=tmp_path/'other.txt';other.write_text('Other evidence')
        grant['evidence']=[dict(path=other.name,sha256=file_digest(other))]
    elif mutation=='source':grant['source_id']='different-source'
    elif mutation=='revision':grant['revision']='f'*40
    elif mutation=='reseal_same_terms':grant['unapproved_extra']='changed source clearance bytes'
    elif mutation=='transplant':grant=json.loads((tmp_path/'clear1.json').read_bytes())
    elif mutation=='item_attribution':record=seal(dict(record,attribution='Unrelated attribution'))
    elif mutation=='item_evidence':
        other=tmp_path/'other.txt';other.write_text('Other evidence')
        record=seal(dict(record,license_evidence=[dict(path=other.name,sha256=file_digest(other))]))
    elif mutation=='item_license':record=seal(dict(record,license_id='Apache-2.0'))
    elif mutation=='missing_binding':
        record=dict(record);record.pop('source_clearance_identity');record=seal(record)
    if mutation in {'attribution','evidence','source','revision','reseal_same_terms','transplant'}:
        grant_path.write_bytes(canonical(seal(grant)))
    with pytest.raises(SpecificationError):
        validate_clearance(record,source,spec,tmp_path)


def test_resealed_source_grant_cannot_verify_or_export(tmp_path):
    spec,registry,plan,records=fixture_plan(tmp_path)
    with ProductionBuild(tmp_path/'build',spec,registry,repository_root=tmp_path,
                         diagnostic=True) as build:
        assert build.run(plan,Tokenizer())['state']=='COMPLETE'
    grant_path=tmp_path/'clear0.json'
    grant=seal(dict(json.loads(grant_path.read_bytes()),attribution='Wrong notice'))
    grant_path.write_bytes(canonical(grant))
    with ProductionBuild(tmp_path/'build',spec,registry,repository_root=tmp_path,
                         diagnostic=True,expected_plan_sha256=file_digest(plan)) as build:
        with pytest.raises(SpecificationError,match='source-clearance identity'):
            build.verify()
        with pytest.raises(SpecificationError):
            build.export_shards(tmp_path/'build/export')
    assert not (tmp_path/'build/export/prepared.json').exists()


def test_notice_is_deterministic_and_only_accepted_sources(tmp_path):
    spec,registry,plan,records=fixture_plan(tmp_path)
    notice_file=tmp_path/'NOTICE';notice_file.write_text('Fixture NOTICE text')
    grant_path=tmp_path/'clear0.json';grant=json.loads(grant_path.read_bytes())
    grant['evidence'].append(dict(path='NOTICE',sha256=file_digest(notice_file)))
    grant=seal(grant);grant_path.write_bytes(canonical(grant))
    records[0]=seal(dict(records[0],license_evidence=grant['evidence'],
                         source_clearance_identity=grant['identity']))
    plan.write_bytes(b''.join(canonical(r)+b'\n' for r in records))
    pairs=list(enumerate(records))
    one=build_notice(pairs,{s['source_id']:s for s in registry['sources']},tmp_path,
                     plan_identity=file_digest(plan),admission_sha256='sha256:'+'a'*64)
    two=build_notice(reversed(pairs),{s['source_id']:s for s in registry['sources']},tmp_path,
                     plan_identity=file_digest(plan),admission_sha256='sha256:'+'a'*64)
    assert canonical(one)==canonical(two) and one['identity']==two['identity']
    assert one['accepted_items']==2 and len(one['entries'])==2
    assert sum(e['accepted_items'] for e in one['entries'])==2
    assert sum(bool(e['notice_references']) for e in one['entries'])==1
    assert all(e['separate_notice_applicability'] in ('RECORDED','UNDETERMINED')
               for e in one['entries'])
    assert reference(one)['sha256']==reference(two)['sha256']
    removed=json.loads(canonical(one))
    next(e for e in removed['entries'] if e['notice_references'])['notice_references']=[]
    removed=seal(removed)
    (tmp_path/FILENAME).write_bytes(canonical(removed))
    with pytest.raises(SpecificationError,match='differ from accepted'):
        verify_file(tmp_path,reference(removed),expected=one)
    with ProductionBuild(tmp_path/'build',spec,registry,repository_root=tmp_path,
                         diagnostic=True) as build:
        build.run(plan,Tokenizer());build.export_shards(tmp_path/'build/export')
        artifact=json.loads((tmp_path/'build/export'/FILENAME).read_bytes())
        assert artifact['accepted_items']==2
        assert any(e['notice_references'] for e in artifact['entries'])


@pytest.mark.parametrize('mutation', ['delete','attribution','license_hash','missing_entry',
                                      'extra_entry','swapped_source','substituted_valid'])
def test_notice_tamper_blocks_export_and_authorized_replay(tmp_path, mutation):
    data=setup(tmp_path);receipts=bundle(tmp_path,data)
    with builder(tmp_path,data) as build:
        build.run(data[3][0],Tokenizer(),intended_use='commercial_training',
                  decision_receipts=receipts)
        out=tmp_path/'build/export'
        build.export_shards(out,intended_use='commercial_training',decision_receipts=receipts)
        original=json.loads((out/FILENAME).read_bytes())
        assert list(build.replay_for_use(out,split='train',intended_use='commercial_training',
                                         decision_receipts=receipts))
        if mutation=='delete':(out/FILENAME).unlink()
        else:
            value=json.loads(canonical(original))
            if mutation=='attribution':value['entries'][0]['attribution']='Wrong attribution'
            elif mutation=='license_hash':value['entries'][0]['license_evidence'][0]['sha256']='sha256:'+'0'*64
            elif mutation=='missing_entry':value['entries'].pop()
            elif mutation=='extra_entry':
                value['entries'].append(dict(value['entries'][0],source_id='unrelated'))
            elif mutation=='swapped_source':value['entries'][0]['source_id']='other-source'
            else:value['accepted_items']+=1
            (out/FILENAME).write_bytes(canonical(seal(value)))
        with pytest.raises(Exception):
            build.replay_for_use(out,split='train',intended_use='commercial_training',
                                 decision_receipts=receipts)
        with pytest.raises(Exception):
            build.export_shards(out,intended_use='commercial_training',decision_receipts=receipts)


def test_resealed_notice_still_needs_exact_accepted_content(tmp_path):
    data=setup(tmp_path);receipts=bundle(tmp_path,data)
    with builder(tmp_path,data) as build:
        build.run(data[3][0],Tokenizer(),intended_use='commercial_training',
                  decision_receipts=receipts)
        out=tmp_path/'build/export';build.export_shards(out,intended_use='commercial_training',
                                                       decision_receipts=receipts)
        notice=json.loads((out/FILENAME).read_bytes())
        notice['entries'][0]['attribution']='Resealed but wrong'
        notice=seal(notice);(out/FILENAME).write_bytes(canonical(notice))
        with pytest.raises(SpecificationError,match='differ from accepted'):
            verify_file(out,reference(notice),expected=build._expected_notice(notice['admission_sha256']))
