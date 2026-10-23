import copy
import json
import pytest
from training import dataset_clearance as d
from training.production_spec import seal, SpecificationError
from training.production_corpus import canonical

@pytest.fixture
def item():
    row={'question':'What is 2+2?','answer':'4'}
    source=dict(source_id='hf-test',dataset_id='openai/gsm8k',locator='https://huggingface.co/datasets/openai/gsm8k',
                revision='a'*40,license_id='MIT',license_evidence={'path':'evidence.json','sha256':'sha256:'+'b'*64})
    payload='sha256:'+'c'*64;path=d.PILOT[source['dataset_id']]
    record=dict(source_id=source['source_id'],locator=source['locator'],revision=source['revision'],
        path=path+'/row-00000000.json',sha256=d.identity_bytes(canonical(row)),bytes=len(canonical(row)),
        provenance=dict(dataset_id=source['dataset_id'],revision=source['revision'],payload_path=path,
            payload_sha256=payload,row_index=0,upstream_split='train',license_evidence=source['license_evidence'],serialization='canonical-json-row-v1'))
    evidence=seal(dict(**{k:record[k] for k in ('source_id','revision','path','sha256','bytes')},
        status='REVIEW',reason='No exact document rights grant',scope=None))
    return dict(record=record,source=source,raw_row=row,evidence=evidence,payload_sha=payload,request_identity='request')

def test_license_metadata_alone_never_clears(item):
    r=d.decision(**item)
    assert r['status']=='REVIEW' and not r['training_allowed'] and not r['production_admitted']
    assert r==d.decision(**copy.deepcopy(item))
    assert r['license_evidence']==item['source']['license_evidence']

@pytest.mark.parametrize('field,value',[('dataset_id','other/data'),('revision','b'*40),('row_index',-1),
    ('payload_sha256','wrong'),('upstream_split','test'),('license_evidence',{}),('payload_path','other.parquet')])
def test_provenance_corruption(item,field,value):
    item['record']['provenance'][field]=value
    with pytest.raises(SpecificationError):d.decision(**item)

def test_row_reconstruction(item):
    item['raw_row']['answer']='5'
    with pytest.raises(SpecificationError):d.decision(**item)

def test_exclusion_preserved(item):
    item['evidence']=seal(dict(item['evidence'],status='EXCLUDED',reason='held out'))
    assert d.decision(**item)['status']=='EXCLUDE'

def test_clear_needs_exact_scope(item):
    item['evidence']=seal(dict(item['evidence'],status='EVIDENCED'))
    with pytest.raises(SpecificationError):d.decision(**item)
    item['evidence']=seal(dict(item['evidence'],scope={'clearance_identity':'exact-grant'},license_id='MIT',license_evidence=[{'path':'grant'}]))
    r=d.decision(**item)
    assert r['status']=='CLEAR' and not r['training_allowed']

def test_clearance_wrong_document(item):
    item['evidence']=seal(dict(item['evidence'],sha256='other'))
    with pytest.raises(SpecificationError):d.decision(**item)
