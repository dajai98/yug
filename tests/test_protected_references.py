"""Synthetic protected work/item/role records; no real benchmark material."""
import json
import sqlite3

import pytest

from training.protected_references import make_reference, validate_records, read_inventory, SCHEMA
from training.production_acquisition import ProductionBuild
from training.production_corpus import canonical, file_digest
from training.production_dedup import DedupIndex
from training.production_spec import SpecificationError
from tests.test_production_acquisition import BASE, Tokenizer, fixture_plan, prior_inventory, approval_kwargs
from tests.test_corpus_gate import bundle


def row(role='question', text=BASE, *, item='q001', reference_id=None):
    return make_reference(work_id='synthetic-benchmark', work_version='v1',
        item_id=item, reference_id=reference_id or item+'-'+role,
        role=role, protected_split='test', aliases=['synthetic-benchmark-alias'],
        text=text, content_type='prose', language='en',
        provenance={'publisher':'diagnostic fixture', 'revision':'v1'})


def inventory(path, rows):
    path.write_bytes(b''.join(canonical(r)+b'\n' for r in rows))
    return path


def test_valid_roles_relationship_aliases_and_determinism(tmp_path):
    records=[row('question'), row('answer','Distinct answer evidence.'),
             row('solution','Distinct solution evidence.'),
             row('reference','Unrelated generic holdout.',item='other')]
    assert [r['role'] for r in validate_records(records)]==['question','answer','solution','reference']
    assert len({(r['work_id'],r['work_version'],r['item_id']) for r in records[:3]})==1
    path=inventory(tmp_path/'references.jsonl',records)
    assert list(read_inventory(path))==list(read_inventory(path))==records
    assert records[0]['schema']==SCHEMA and records[0]['aliases']


@pytest.mark.parametrize('field,value',[
    ('schema',None),('work_id',None),('work_version',''),('item_id',None),
    ('reference_id',''),('role',None),('role','rationale'),
    ('protected_split',None),('protected_split','train'),
    ('aliases','not-a-list'),('aliases',['bad alias']),
    ('provenance',{}),('provenance',{'source':''}),('text',None),
    ('content_type','binary'),('language',''),('raw_text_sha256','sha256:'+'0'*64),
    ('normalized_text_sha256','sha256:'+'0'*64),
])
def test_missing_malformed_or_mismatched_field_rejected(field,value):
    damaged=row();damaged[field]=value
    with pytest.raises(SpecificationError):list(validate_records([damaged]))


def test_raw_and_normalized_hashes_are_independently_bound():
    r=make_reference(work_id='w',work_version='v1',item_id='i',reference_id='r',
        role='question',protected_split='test',aliases=[],provenance={'source':'fixture'},
        text='first   second',content_type='prose',language='en')
    assert r['raw_text_sha256']!=r['normalized_text_sha256']
    assert list(validate_records([r]))==[r]
    r['text']='first second'
    with pytest.raises(SpecificationError,match='raw hash'):list(validate_records([r]))


def test_duplicate_id_role_split_and_old_format_rejected():
    first=row()
    cases=([first,dict(first)],
           [first,row('answer',reference_id=first['reference_id'])],
           [first,row('question',reference_id='other-id')],
           [first,dict(row('answer'),protected_split='evaluation')])
    for records in cases:
        with pytest.raises(SpecificationError):list(validate_records(records))
    with pytest.raises(SpecificationError,match='schema'):
        list(validate_records([{'reference_id':'old','text':BASE,'provenance':{'source':'old'}}]))


def test_duplicate_json_field_rejected(tmp_path):
    r=row();payload=canonical(r).decode()
    payload=payload.replace('"role":"question"','"role":"answer","role":"question"')
    path=tmp_path/'references.jsonl';path.write_text(payload+'\n')
    with pytest.raises(SpecificationError,match='Duplicate.*field'):list(read_inventory(path))


def test_repeated_reference_registration_does_not_change_index(tmp_path):
    spec,registry,plan,_=fixture_plan(tmp_path,texts=['Distinct new training document about ocean tides.'])
    refs=inventory(tmp_path/'references.jsonl',[
        row('question'),row('answer','Distinct protected answer.'),
        row('solution','Distinct protected solution.')])
    with ProductionBuild(tmp_path/'build',spec,registry,repository_root=tmp_path,
                         reference_plan=refs,prior_production=prior_inventory(tmp_path),
                         expected_plan_sha256=file_digest(plan),**approval_kwargs(tmp_path,file_digest(plan))) as build:
        build.register_references()
        before=list(build.db.execute('SELECT item,exact,split,provenance FROM dd ORDER BY item'))
        assert build.get('references_registered')==3
        build.register_references()
        assert list(build.db.execute('SELECT item,exact,split,provenance FROM dd ORDER BY item'))==before


def test_missing_indexed_answer_fails_closed_on_resume(tmp_path):
    answer=' '.join('answer'+str(i) for i in range(60))[:277]
    spec,registry,plan,_=fixture_plan(tmp_path,texts=[answer])
    refs=inventory(tmp_path/'references.jsonl',[row('answer',answer)])
    prior=prior_inventory(tmp_path)
    with ProductionBuild(tmp_path/'build',spec,registry,repository_root=tmp_path,
                         reference_plan=refs,prior_production=prior,
                         expected_plan_sha256=file_digest(plan),**approval_kwargs(tmp_path,file_digest(plan))) as build:
        build.register_references()
        indexed=build.db.execute("SELECT id FROM dd WHERE item LIKE 'reference:%'").fetchone()[0]
        with build.db:
            build.db.execute('DELETE FROM anchors WHERE doc=?',(indexed,))
            build.db.execute('DELETE FROM dd WHERE id=?',(indexed,))
        assert build.get('references_registered')==1
        build.dedup.verify()  # The generic index check does not prove registry completeness.
    with ProductionBuild(tmp_path/'build',spec,registry,repository_root=tmp_path,
                         reference_plan=refs,prior_production=prior,
                         expected_plan_sha256=file_digest(plan),**approval_kwargs(tmp_path,file_digest(plan))) as build:
        with pytest.raises(SpecificationError,match='Protected reference index'):
            build.run(plan,Tokenizer())
        assert build.db.execute('SELECT count(*) FROM totals').fetchone()[0]==0
        assert build.db.execute("SELECT count(*) FROM items WHERE stage='ACCEPTED'").fetchone()[0]==0


def test_same_process_inventory_change_blocks_export_and_replay(tmp_path):
    answer=' '.join('answer'+str(i) for i in range(60))[:277]
    question=row('question',' '.join('question'+str(i) for i in range(60)))
    spec,registry,plan,_=fixture_plan(tmp_path,texts=[answer])
    refs=inventory(tmp_path/'references.jsonl',[question])
    receipts=bundle(tmp_path,(spec,registry,refs,[plan]))
    with ProductionBuild(tmp_path/'build',spec,registry,repository_root=tmp_path,
                         reference_plan=refs,prior_production=prior_inventory(tmp_path),
                         expected_plan_sha256=file_digest(plan),**approval_kwargs(tmp_path,file_digest(plan))) as build:
        result=build.run(plan,Tokenizer(),intended_use='commercial_training',decision_receipts=receipts)
        assert result['accepted_clean_train_tokens']==277
        out=tmp_path/'build/export'
        assert build.export_shards(out,intended_use='commercial_training',
                                   decision_receipts=receipts)['total_token_count']==277
        inventory(refs,[question,row('answer',answer)])
        with pytest.raises(SpecificationError,match='Held-out reference inventory changed'):
            build.export_shards(out,intended_use='commercial_training',decision_receipts=receipts)
        with pytest.raises(SpecificationError):
            build.export_shards(tmp_path/'build/new-export',intended_use='commercial_training',
                                decision_receipts=receipts)
        with pytest.raises(SpecificationError,match='Held-out reference inventory changed'):
            build.replay_for_use(out,split='train',intended_use='commercial_training',
                                 decision_receipts=receipts)


def test_old_unstructured_inventory_cannot_start_admission(tmp_path):
    spec,registry,plan,_=fixture_plan(tmp_path,texts=['Distinct new training document about ocean tides.'])
    refs=inventory(tmp_path/'references.jsonl',[
        {'reference_id':'old','text':BASE,'content_type':'prose',
         'language':'en','provenance':{'source':'old fixture'}}])
    with ProductionBuild(tmp_path/'build',spec,registry,repository_root=tmp_path,
                         reference_plan=refs,prior_production=prior_inventory(tmp_path),
                         expected_plan_sha256=file_digest(plan),**approval_kwargs(tmp_path,file_digest(plan))) as build:
        with pytest.raises(SpecificationError,match='schema'):
            build.run(plan,Tokenizer())
        assert build.db.execute('SELECT count(*) FROM totals').fetchone()[0]==0
        assert build.db.execute('SELECT count(*) FROM dd').fetchone()[0]==0


def production(tmp_path,candidate,protected,*,authorized=False):
    spec,registry,plan,_=fixture_plan(tmp_path,texts=[candidate])
    refs=inventory(tmp_path/'references.jsonl',protected)
    data=(spec,registry,refs,[plan])
    receipts=bundle(tmp_path,data) if authorized else None
    with ProductionBuild(tmp_path/'build',spec,registry,repository_root=tmp_path,
                         reference_plan=refs,prior_production=prior_inventory(tmp_path),
                         expected_plan_sha256=file_digest(plan),**approval_kwargs(tmp_path,file_digest(plan))) as build:
        result=build.run(plan,Tokenizer(),intended_use='commercial_training' if authorized else 'production_preparation',
                         decision_receipts=receipts)
        return result,build,receipts


def test_answer_absent_reproduces_277_then_explicit_answer_blocks(tmp_path):
    # The byte-level test double emits exactly one token per ASCII byte.
    answer=' '.join('answer'+str(i) for i in range(60))[:277]
    assert len(answer.encode('utf-8'))==277
    question=row('question',' '.join('question'+str(i) for i in range(60)))
    answer_row=row('answer',answer)
    solution=row('solution',' '.join('solution'+str(i) for i in range(60)))
    missing=tmp_path/'missing';missing.mkdir()
    first,_,_=production(missing,answer,[question])
    assert first['stages']=={'ACCEPTED':1} and first['accepted_clean_train_tokens']==277
    complete=tmp_path/'complete';complete.mkdir()
    spec,registry,plan,_=fixture_plan(complete,texts=[answer])
    refs=inventory(complete/'references.jsonl',[question,answer_row,solution])
    receipts=bundle(complete,(spec,registry,refs,[plan]))
    with ProductionBuild(complete/'build',spec,registry,repository_root=complete,
                         reference_plan=refs,prior_production=prior_inventory(complete),
                         expected_plan_sha256=file_digest(plan),**approval_kwargs(complete,file_digest(plan))) as build:
        result=build.run(plan,Tokenizer(),intended_use='commercial_training',decision_receipts=receipts)
        assert result['stages']=={'DUPLICATE':1}
        assert result['accepted_clean_train_tokens']==0
        assert build.db.execute('SELECT count(*) FROM totals').fetchone()[0]==0
        assert build.db.execute('SELECT count(*) FROM items WHERE stage="ACCEPTED"').fetchone()[0]==0
        assert build.db.execute('SELECT reason FROM items').fetchone()[0]
        out=complete/'build/export'
        manifest=build.export_shards(out,intended_use='commercial_training',decision_receipts=receipts)
        assert manifest['total_token_count']==0
        assert list(build.replay_for_use(out,split='train',intended_use='commercial_training',
                                         decision_receipts=receipts))==[]


@pytest.mark.parametrize('variant',[
    lambda s:s,
    lambda s:s.replace('word30','edited30'),
    lambda s:s+' Additional surrounding context.',
    lambda s:'Preface context. '+s,
    lambda s:' '.join(s.split()[45:]+s.split()[:45]),
    lambda s:' '.join(s.split()[:50]),
    lambda s:s.replace('word30','word30,'),
])
def test_question_changed_source_path_and_supported_near_copies(tmp_path,variant):
    candidate=variant(BASE)
    result,_,_=production(tmp_path,candidate,[row('question')])
    assert result['stages']=={'DUPLICATE':1} and result['accepted_clean_train_tokens']==0


@pytest.mark.parametrize('excerpt',[False,True])
def test_solution_independently_protected(tmp_path,excerpt):
    solution=' '.join('solution'+str(i) for i in range(80))
    candidate=' '.join(solution.split()[:50]) if excerpt else solution
    result,_,_=production(tmp_path,candidate,[row('solution',solution)])
    assert result['stages']=={'DUPLICATE':1} and result['accepted_clean_train_tokens']==0


def test_existing_dedup_diagnostic_residuals_remain_visible(tmp_path):
    # A common short four-word fragment can be overmatched; this is NOT fixed here.
    db=sqlite3.connect(tmp_path/'dedup.sqlite');index=DedupIndex(db)
    try:
        index.add('a','public static void main','prose','en','validation',{})
        index.add('b','Public static void main.','prose','en','train',{})
        assert index.assignment('a')['cluster_id']==index.assignment('b')['cluster_id']
    finally:db.close()
    # Renamed Python identifiers can evade the unchanged lexical shingles.
    db=sqlite3.connect(tmp_path/'renamed.sqlite');index=DedupIndex(db)
    original=('def compute_total(prices, tax_rate):\n'
              '    subtotal = sum(prices)\n'
              '    discounted = subtotal - rebate\n'
              '    return discounted * (1 + tax_rate)\n')
    renamed=('def calculate_amount(values, levy):\n'
             '    amount = sum(values)\n'
             '    net = amount - credit\n'
             '    return net * (1 + levy)\n')
    try:
        index.add('original',original,'code','python','validation',{})
        index.add('renamed',renamed,'code','python','train',{})
        assert index.assignment('original')['cluster_id']!=index.assignment('renamed')['cluster_id']
    finally:db.close()
