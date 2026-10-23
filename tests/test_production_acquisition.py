"""Near-dedup and production-coordinator tests; fixtures never train a model."""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import pytest
from training.production_corpus import canonical, file_digest, identity
from training.production_spec import seal, SpecificationError, TOKENIZER
from training.production_dedup import DedupIndex, features, POLICY
from training.production_acquisition import ProductionBuild, item_key
from training.experiments.production_corpus_spec import build

ROOT=Path(__file__).resolve().parents[1]
BASE=' '.join('word'+str(i) for i in range(90))

def prior_inventory(root, builds=(), name='prior.json'):
    path=root/name
    path.write_bytes(canonical(seal(dict(schema='yug-prior-production-v1',complete=True,
        genesis=not builds,builds=[dict(path=p.relative_to(root).as_posix(),sha256=file_digest(p)) for p in builds]))))
    return path

@pytest.fixture
def index(tmp_path):
    db=sqlite3.connect(tmp_path/'dedup.sqlite')
    idx=DedupIndex(db)
    yield idx
    db.close()

@pytest.mark.parametrize('second,kind,lang',[
    (BASE,'prose',''),
    (BASE.replace('word40','edited40'),'prose',''),
    ('\n  '.join(BASE.upper().split()),'prose',''),
    (' '.join(BASE.split()[45:]+BASE.split()[:45]),'prose',''),
])
def test_exact_edited_format_reordered(index,second,kind,lang):
    with index.db:
        index.add('a',BASE,kind,lang,'train',{'category':'code'})
        index.add('b',second,kind,lang,'train',{'category':'docs'})
    assert index.assignment('a')==index.assignment('b')
    assert index.db.execute('SELECT count(*) FROM edges').fetchone()[0]==1

def test_unrelated_not_clustered(index):
    with index.db:
        index.add('a',BASE,'prose','','train',{})
        index.add('b','Entirely separate natural content about ocean tides and the lunar gravitational cycle.','prose','','train',{})
    assert index.assignment('a')['cluster_id']!=index.assignment('b')['cluster_id']

@pytest.mark.parametrize('lang,a,b',[
 ('python','def add(a, b):\n    # Add two inputs\n    result = a + b\n    return result\n','def add(a,b):\n\n result=a+b # changed comment\n return result\n'),
 ('c_cpp','int add(int a, int b) { /* implementation */ int r = a + b; return r; }','int add ( int a,int b ){\n// revised comment\nint r=a+b;return r;}')])
def test_code_comments_and_layout(index,lang,a,b):
    with index.db:
        index.add('a',a,'code',lang,'train',{})
        index.add('b',b,'code',lang,'train',{})
    assert index.assignment('a')==index.assignment('b')

def test_code_strings_preserved():
    a=features('const char *url = "https://example.com/a";','code','c_cpp')
    b=features('const char *url = "https://different.example/b";','code','c_cpp')
    assert a!=b

def test_order_independent_labels_and_validation_precedence(tmp_path):
    outputs=[]
    for order in [('a','b','c'),('c','b','a')]:
        db=sqlite3.connect(tmp_path/(''.join(order)+'.db'));idx=DedupIndex(db)
        with db:
            for k in order:idx.add(k,BASE.replace('word40',k+'40'),'prose','','validation' if k=='b' else 'train',{'id':k})
        outputs.append([idx.assignment(k) for k in ('a','b','c')]);db.close()
    assert outputs[0]==outputs[1]
    assert {x['canonical_item'] for x in outputs[0]}=={'b'}
    assert {x['split'] for x in outputs[0]}=={'validation'}

def test_dedup_resume_and_corrupt_index(tmp_path):
    path=tmp_path/'dd.db'
    db=sqlite3.connect(path);idx=DedupIndex(db)
    with db:idx.add('a',BASE,'prose','','train',{'origin':'first'})
    before=idx.assignment('a');db.commit();db.close()
    db=sqlite3.connect(path);idx=DedupIndex(db)
    with db:
        idx.add('a',BASE,'prose','','train',{'origin':'first'})
        idx.add('b',BASE.replace('word45','changed45'),'prose','','train',{})
    assert idx.assignment('b')==before
    idx.verify()
    db.execute('DELETE FROM anchors WHERE doc=(SELECT id FROM dd WHERE item=?)',('a',));db.commit()
    with pytest.raises(SpecificationError,match='anchor'):idx.verify()
    db.close()

def test_transaction_rollback(index):
    with pytest.raises(RuntimeError):
        with index.db:
            index.add('a',BASE,'prose','','train',{})
            raise RuntimeError('crash')
    assert index.db.execute('SELECT count(*) FROM dd').fetchone()[0]==0

class Tokenizer:
    """Test double; real frozen-tokenizer integration has a separate test."""
    identity=TOKENIZER;vocab_size=4096
    def __init__(self):self.calls=0
    def encode(self,text):self.calls+=1;return list(text.encode('utf-8'))

def fixture_plan(tmp_path,texts=None,splits=None):
    spec,registry=build(ROOT)
    spec=copy.deepcopy(spec);registry=copy.deepcopy(registry)
    # Two explicit test source groups; production registry never promoted.
    sources=[copy.deepcopy(next(s for s in registry['sources'] if s['source_id']==sid)) for sid in ('gh-psf-requests','gh-expressjs-express')]
    texts=texts or [BASE, 'Ocean tides are driven by the gravitational interaction of the earth moon and sun. Coastal shape modifies the tidal range.']
    splits=splits or ['train']*len(texts)
    license_file=tmp_path/'license.txt';license_file.write_text('Diagnostic fixture provenance. Not production clearance.')
    evidence=[dict(path='license.txt',sha256=file_digest(license_file))]
    spec['evidence']=evidence;records=[]
    for i,(source,text,split) in enumerate(zip(sources,texts,splits)):
        source.update(locator='https://example.invalid/'+str(i),revision=str(i+1)*40,origin_verified=True,split=split,
            file_manifest=f'files{i}.json',clearance_record=f'clear{i}.json',allow_paths=['src/**'])
        payload=text.encode();payload_file=tmp_path/f'input{i}.txt';payload_file.write_bytes(payload)
        owner='code' if i==0 else 'software_engineering'
        f=dict(path=f'src/doc{i}.txt',owner=owner,content_type='prose',sha256=file_digest(payload_file),bytes=len(payload),license_id='MIT')
        m=seal(dict(source_id=source['source_id'],revision=source['revision'],files=[f]));(tmp_path/f'files{i}.json').write_text(json.dumps(m))
        c=seal(dict(source_id=source['source_id'],revision=source['revision'],file_manifest_identity=m['identity'],decision='ACCEPT',evidence=evidence,attribution='Fixture'))
        (tmp_path/f'clear{i}.json').write_text(json.dumps(c))
        record=seal(dict(source_id=source['source_id'],revision=source['revision'],**f,
            clearance_status='ACCEPT',subcategory='python' if i==0 else 'testing',split=split,split_group=source['split_group'],
            split_unit_kind='repository',allow_rule='src/**',deny_rules_checked=source['deny_paths'],
            language='en',license_evidence=evidence,attribution='Fixture',quality_review='ACCEPT',input_path=payload_file.name))
        records.append(record)
    registry['sources']=sources[:len(texts)];registry=seal(registry);spec['registry_identity']=registry['identity'];spec=seal(spec)
    plan=tmp_path/'plan.jsonl';plan.write_bytes(b''.join(canonical(r)+b'\n' for r in records))
    return spec,registry,plan,records

def test_production_resume_no_refetch_or_reencode(tmp_path):
    spec,registry,plan,records=fixture_plan(tmp_path)
    tok=Tokenizer();calls=[]
    def fetch(r):calls.append(r['identity']);yield (tmp_path/r['input_path']).read_bytes()
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True) as b:
        assert b.run(plan,tok,fetch=fetch,max_items=1)['state']=='PARTIAL'
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True) as b:
        result=b.run(plan,tok,fetch=fetch)
        assert result['state']=='COMPLETE' and result['plan_complete']
        assert len(calls)==2 and tok.calls==2
        assert b.run(plan,tok,fetch=fetch)['accepted_clean_train_tokens']==result['accepted_clean_train_tokens']
        assert len(calls)==2 and tok.calls==2
        manifest=b.export_shards()
        assert manifest['total_token_count']==result['accepted_clean_train_tokens']
        assert tok.calls==2
        assert not result['actual_corpus_complete']

def test_cross_category_duplicate_only_one_accounted(tmp_path):
    spec,registry,plan,records=fixture_plan(tmp_path,[BASE,BASE.replace('word40','edited40')])
    tok=Tokenizer()
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True) as b:
        result=b.run(plan,tok)
        assert result['stages']=={'ACCEPTED':1,'DUPLICATE':1}
        assert tok.calls==1
        assert len([n for n in result['category_tokens'].values() if n])==1
        b.export_shards()
        audit=[json.loads(line) for line in (tmp_path/'out/admission.jsonl').read_text().splitlines()]
        assert len(audit)==2 and all(r['clearance']['license_evidence'] for r in audit)

def test_validation_duplicate_never_leaks_to_train(tmp_path):
    spec,registry,plan,records=fixture_plan(tmp_path,[BASE,BASE],['train','validation'])
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True) as b:
        result=b.run(plan,Tokenizer())
        assert result['accepted_clean_train_tokens']==0
        assert sum(result['category_validation_tokens'].values())>0
        shards=b.export_shards()
        assert shards['splits']['train']['total_token_count']==0
        assert shards['splits']['validation']['total_token_count']>0

@pytest.mark.parametrize('field,value',[('clearance_status','REVIEW'),('license_id','uncleared'),('sha256','sha256:'+'f'*64),('revision','a'*40),('split','validation'),('owner','graphics'),('allow_rule','**'),('deny_rules_checked',[])])
def test_clearance_cannot_be_bypassed(tmp_path,field,value):
    spec,registry,plan,records=fixture_plan(tmp_path)
    records[0][field]=value;records[0]=seal(records[0]);plan.write_bytes(b''.join(canonical(r)+b'\n' for r in records))
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True) as b:
        with pytest.raises(SpecificationError):b.run(plan,Tokenizer())
        assert b.get('state')=='FAILED'
        assert b.db.execute("SELECT count(*) FROM items WHERE stage='ACCEPTED'").fetchone()[0]==0

def test_changed_clearance_or_plan_refused(tmp_path):
    spec,registry,plan,records=fixture_plan(tmp_path)
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True) as b:
        b.run(plan,Tokenizer(),max_items=1)
        records[0]['attribution']='different';records[0]=seal(records[0]);plan.write_bytes(b''.join(canonical(r)+b'\n' for r in records))
        with pytest.raises(SpecificationError,match='Plan changed'):b.run(plan,Tokenizer())

@pytest.mark.parametrize('kind',['raw','normalized','tokens'])
def test_committed_corruption_fails_closed(tmp_path,kind):
    spec,registry,plan,records=fixture_plan(tmp_path)
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True) as b:
        b.run(plan,Tokenizer())
        item=item_key(records[0]);path=b.path(kind,item,'.bin' if kind=='tokens' else '')
        path.write_bytes(b'corrupt')
        with pytest.raises(SpecificationError,match='corrupt'):b.run(plan,Tokenizer())
        assert b.get('state')=='FAILED'

def test_partial_download_recovery_and_wrong_payload(tmp_path):
    spec,registry,plan,records=fixture_plan(tmp_path,texts=[BASE])
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True) as b:
        def interrupted(r):
            yield b'partial'
            raise OSError('connection lost')
        with pytest.raises(OSError):b.run(plan,Tokenizer(),fetch=interrupted)
        assert b.get('state')=='FAILED'
        assert b.run(plan,Tokenizer())['state']=='COMPLETE'

def test_real_frozen_tokenizer_cached_shard_equivalence(tmp_path):
    from yug.tokenizer.bpe import ByteLevelBpeTokenizer
    tokenizer=ByteLevelBpeTokenizer.load(ROOT/'training/resources/phase1/bpe_fitter_optimization_gate_v1/comparison/run-4096-0/tokenizer.json')
    spec,registry,plan,records=fixture_plan(tmp_path,texts=[BASE])
    from training.token_shards import TokenShardStream
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True) as b:
        result=b.run(plan,tokenizer);m=b.export_shards()
        replay=TokenShardStream(tmp_path/'out/shards',split='train',tokenizer_identity=TOKENIZER,prepared_corpus_identity=m['prepared_corpus_identity'])
        assert list(replay)==tokenizer.encode(BASE)
        assert result['accepted_clean_train_tokens']==len(tokenizer.encode(BASE))

def test_heldout_inventory_exclusion_and_mandatory_pin(tmp_path):
    spec,registry,plan,records=fixture_plan(tmp_path,texts=[BASE])
    with pytest.raises(SpecificationError,match='requires pinned'):
        ProductionBuild(tmp_path/'no-reference',spec,registry,repository_root=tmp_path)
    refs=tmp_path/'heldout.jsonl'
    refs.write_bytes(canonical(dict(reference_id='heldout-test',text=BASE.replace('word30','variant30'),content_type='prose',language='en',provenance={'source':'diagnostic heldout'}))+b'\n')
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,reference_plan=refs,prior_production=prior_inventory(tmp_path)) as b:
        result=b.run(plan,Tokenizer())
        assert result['accepted_clean_train_tokens']==0 and result['stages']['DUPLICATE']==1
        assert result['heldout_references']==1
        assert b.export_shards()['total_token_count']==0

def test_accounting_corruption_detected(tmp_path):
    spec,registry,plan,records=fixture_plan(tmp_path)
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True) as b:
        b.run(plan,Tokenizer())
        b.db.execute("UPDATE totals SET tokens=0 WHERE kind='category'");b.db.commit()
        with pytest.raises(SpecificationError,match='accounting totals'):b.run(plan,Tokenizer())

def test_quota_rejection_and_generated_filter(tmp_path):
    spec,registry,plan,records=fixture_plan(tmp_path,[BASE,'// Code generated -- DO NOT EDIT\n'+BASE])
    registry['sources'][0]['target_cap_tokens']=1;registry=seal(registry);spec['registry_identity']=registry['identity'];spec=seal(spec)
    tok=Tokenizer()
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True) as b:
        result=b.run(plan,tok)
        assert result['stages']=={'FILTERED':1,'QUOTA_REJECTED':1}
        assert result['accepted_clean_train_tokens']==0
        assert tok.calls==1

def test_no_promotion_of_conditional_source(tmp_path):
    spec,registry,plan,records=fixture_plan(tmp_path)
    registry['sources'][0]['status']='CONDITIONAL';registry=seal(registry);spec['registry_identity']=registry['identity'];spec=seal(spec)
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True) as b:
        with pytest.raises(SpecificationError):b.run(plan,Tokenizer())

def test_source_group_conflict_rejected_before_build(tmp_path):
    spec,registry,plan,records=fixture_plan(tmp_path,splits=['train','validation'])
    registry['sources'][1]['split_group']=registry['sources'][0]['split_group']
    registry=seal(registry);spec['registry_identity']=registry['identity'];spec=seal(spec)
    with pytest.raises(SpecificationError,match='crosses'):
        ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True)

def test_wrong_acquisition_hash_never_accounted(tmp_path):
    spec,registry,plan,records=fixture_plan(tmp_path,texts=[BASE])
    def wrong(r):yield b'x'*r['bytes']
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True) as b:
        with pytest.raises(SpecificationError,match='hash/size'):b.run(plan,Tokenizer(),fetch=wrong)
        assert b.progress('FAILED')['accepted_clean_train_tokens']==0


def test_exact_duplicate_anchor_storage_does_not_multiply(index):
    with index.db:
        for i in range(50):index.add(str(i),BASE,'prose','','train',{'i':i})
    assert index.db.execute('SELECT count(*) FROM anchors').fetchone()[0]<=128
    assert index.db.execute('SELECT count(*) FROM edges').fetchone()[0]==49
    index.verify()


def test_streamed_manifest_inventory(tmp_path):
    spec,registry,plan,records=fixture_plan(tmp_path,texts=[BASE])
    m=json.loads((tmp_path/'files0.json').read_text());files=m.pop('files');m.pop('identity')
    inventory=tmp_path/'inventory.jsonl';inventory.write_bytes(b''.join(canonical(f)+b'\n' for f in files))
    m=seal(dict(m,files_jsonl='inventory.jsonl',files_sha256=file_digest(inventory),file_count=1))
    (tmp_path/'files0.json').write_text(json.dumps(m))
    c=json.loads((tmp_path/'clear0.json').read_text());c['file_manifest_identity']=m['identity'];(tmp_path/'clear0.json').write_text(json.dumps(seal(c)))
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True) as b:
        assert b.run(plan,Tokenizer())['state']=='COMPLETE'

def test_candidate_overflow_fails_closed(tmp_path,monkeypatch):
    monkeypatch.setitem(POLICY,'candidate_limit',1)
    db=sqlite3.connect(tmp_path/'overflow.db');idx=DedupIndex(db)
    with db:
        idx.add('a',BASE,'prose','','train',{})
        idx.add('b',BASE.replace('word40','edited40'),'prose','','train',{})
    with pytest.raises(SpecificationError,match='overflow'):
        with db:idx.add('c',BASE.replace('word50','changed50'),'prose','','train',{})
    assert db.execute('SELECT count(*) FROM dd').fetchone()[0]==2
    db.close()

def test_production_request_duplicate_rejected(tmp_path):
    spec,registry,plan,records=fixture_plan(tmp_path,texts=[BASE])
    plan.write_bytes(canonical(records[0])+b'\n'+canonical(records[0])+b'\n')
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True) as b:
        with pytest.raises(SpecificationError,match='Duplicate item'):b.run(plan,Tokenizer())

def test_split_build_and_uninterrupted_shards_identical(tmp_path):
    spec,registry,plan,records=fixture_plan(tmp_path)
    outputs=[]
    for name,interrupt in [('full',False),('resume',True)]:
        with ProductionBuild(tmp_path/name,spec,registry,repository_root=tmp_path,diagnostic=True) as b:
            tok=Tokenizer()
            if interrupt:
                while b.run(plan,tok,max_items=1)['state']!='COMPLETE':pass
            else:b.run(plan,tok)
            outputs.append(b.export_shards())
    assert outputs[0]==outputs[1]
    for split in ('train','validation'):
        for entry in outputs[0]['splits'][split]['shards']:
            assert (tmp_path/'full/shards'/entry['path']).read_bytes()==(tmp_path/'resume/shards'/entry['path']).read_bytes()

def test_structure_preserving_normalization_route(tmp_path):
    text='def multiply(a, b):\r\n\t# Keep input indentation\r\n\treturn a * b  \r\n'
    spec,registry,plan,records=fixture_plan(tmp_path,texts=[text])
    r=records[0];r['content_type']='code';r['language']='python';r=seal(r)
    m=json.loads((tmp_path/'files0.json').read_text());m['files'][0]['content_type']='code';m=seal(m)
    (tmp_path/'files0.json').write_text(json.dumps(m))
    c=json.loads((tmp_path/'clear0.json').read_text());c['file_manifest_identity']=m['identity'];(tmp_path/'clear0.json').write_text(json.dumps(seal(c)))
    plan.write_bytes(canonical(r)+b'\n')
    from training.experiments.representative_mini_corpus import normalize_structured_text
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True) as b:
        b.run(plan,Tokenizer())
        normalized=b.path('normalized',item_key(r)).read_text()
        assert normalized==normalize_structured_text(text)
        assert '\treturn' in normalized

def test_acquisition_gate_no_network_and_cap_table(monkeypatch):
    from training.experiments.acquisition_readiness_gate import evaluate
    import urllib.request
    def forbidden(*a,**k):raise AssertionError('Network forbidden')
    monkeypatch.setattr(urllib.request,'urlopen',forbidden)
    result=evaluate(ROOT)
    assert result['gate']=='NOT_READY'
    assert sum(r['target_tokens'] for r in result['category_capacity'])==2_000_000_000
    assert all(r['current_approved_clean_capacity']==0 for r in result['category_capacity'])
    assert sum(t['target_tokens'] for t in result['three_d_capacity'])==100_000_000
    assert result['dry_run']['planned_download_bytes']==0

def test_capacity_respects_shared_caps():
    from training.experiments.acquisition_readiness_gate import capacity
    sources=[dict(source_id=str(i),status='REVIEW',training_allowed_after_clearance=True,allowed_categories=['code'],split='unassigned',repository_group='shared',domain_group='domain',target_cap_tokens=100,category_caps_tokens={'code':100}) for i in range(3)]
    spec={'group_caps':{'repository_group':{'shared':150},'domain_group':{'domain':120}}}
    assert capacity(spec,{'sources':sources},'code',{'REVIEW'})==120
    assert capacity(spec,{'sources':sources},'code',{'ACCEPT'})==0


@pytest.mark.parametrize('limit',[0,-1,True,1.5])
def test_invalid_resume_chunk_limit(tmp_path,limit):
    spec,registry,plan,records=fixture_plan(tmp_path,texts=[BASE])
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True) as b:
        with pytest.raises(SpecificationError,match='positive integer'):b.run(plan,Tokenizer(),max_items=limit)
        assert b.db.execute('SELECT count(*) FROM items').fetchone()[0]==0


def test_clearance_sidecar_mutation_rejected_on_resume(tmp_path):
    spec,registry,plan,records=fixture_plan(tmp_path,texts=[BASE])
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True) as b:
        b.run(plan,Tokenizer())
        p=b.path('clearances',item_key(records[0]),'.json')
        r=json.loads(p.read_bytes());r['attribution']='modified';p.write_bytes(canonical(r))
        with pytest.raises(SpecificationError,match='Clearance changed'):b.run(plan,Tokenizer())



def test_heldout_inventory_is_reproducible_and_checks_payloads(tmp_path):
    """Exercise both historical formats using diagnostic text, never corpus copies."""
    from training.experiments.acquisition_readiness_gate import retained_heldout_inventory
    from training.production_acquisition import digest
    from training.production_corpus import PipelineError
    text = 'Diagnostic heldout fixture, not production data.'
    fit = dict(text=text, content_hash=digest(text.encode())[7:], partition='validation',
               provenance={'path':'example.py'}, primary_domain='code',
               document_id='diagnostic-fit', language='python')
    transfer = dict(text=text, text_sha256=digest(text.encode())[7:],
                    path='example.md', id='diagnostic-transfer')
    paths = [tmp_path/'artifacts/tokenizer_fit_sample_gate_v1/validation.jsonl',
             tmp_path/'artifacts/tokenizer_transfer_gate_v1/documents.jsonl']
    with pytest.raises(PipelineError): retained_heldout_inventory(tmp_path)
    for path, row in zip(paths, [fit, transfer]):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical(row)+b'\n')
    a,inputs=retained_heldout_inventory(tmp_path)
    b,other=retained_heldout_inventory(tmp_path)
    assert canonical(a)==canonical(b) and inputs==other
    assert {r['reference_id'] for r in a}=={'fit_validation:diagnostic-fit','transfer_evaluation:diagnostic-transfer'}
    assert all(r['provenance']['artifact_sha256'].startswith('sha256:') for r in a)
    assert a[0]['language']=='python' and a[1]['content_type']=='markdown'
    transfer['text']='Changed payload'
    paths[1].write_bytes(canonical(transfer)+b'\n')
    with pytest.raises(ValueError,match='hash mismatch'):retained_heldout_inventory(tmp_path)


def test_coordinator_dry_run_never_acquires_or_encodes(tmp_path,monkeypatch):
    spec,registry,plan,records=fixture_plan(tmp_path)
    def forbidden(*a,**k):raise AssertionError('Dry run must not acquire or encode')
    monkeypatch.setattr(ProductionBuild,'_acquire',forbidden)
    monkeypatch.setattr(ProductionBuild,'_cached_tokens',forbidden)
    import socket
    monkeypatch.setattr(socket.socket,'connect',forbidden)
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True) as b:
        first=b.dry_run(plan);second=b.dry_run(plan)
        assert first['scheduled_download_files']==2
        assert first['scheduled_download_bytes']==sum(r['bytes'] for r in records)
        assert first['downloaded_bytes']==first['encoded_tokens']==0
        assert second['stages']=={'PENDING':2}
        assert not list((tmp_path/'out/raw').iterdir())
        assert not list((tmp_path/'out/tokens').iterdir())


def test_coordinator_dry_run_rejects_invalid_clearance(tmp_path):
    spec,registry,plan,records=fixture_plan(tmp_path,texts=[BASE])
    records[0]['sha256']='sha256:'+'0'*64
    plan.write_bytes(canonical(seal(records[0]))+b'\n')
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True) as b:
        with pytest.raises(SpecificationError,match='sha256 differs'):b.dry_run(plan)
        assert b.db.execute('SELECT count(*) FROM items').fetchone()[0]==0


def test_coordinator_dry_run_does_not_relabel_acquired_build(tmp_path):
    spec,registry,plan,records=fixture_plan(tmp_path,texts=[BASE])
    with ProductionBuild(tmp_path/'out',spec,registry,repository_root=tmp_path,diagnostic=True) as b:
        b.run(plan,Tokenizer())
        with pytest.raises(SpecificationError,match='unacquired'):b.dry_run(plan)
        assert b.get('state')=='COMPLETE'
