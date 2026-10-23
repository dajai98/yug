import hashlib
import io
import json
import pytest
from training import dataset_transport as t
from training.production_corpus import canonical, atomic_json, file_digest
from training.production_spec import seal, SpecificationError
from training.source_admission import cached_sources
from training.source_prepare import ROOT

@pytest.fixture
def setup(tmp_path):
    did='openai/gsm8k'; rev='a'*40
    e=tmp_path/'license.json'
    atomic_json(e,seal(dict(url=f'https://huggingface.co/api/datasets/{did}/revision/{rev}',body=dict(id=did,sha=rev,private=False,gated=False,cardData={'license':'mit'}))))
    s=dict(source_id='hf-test',dataset_id=did,locator='https://huggingface.co/datasets/'+did,revision=rev,
        license_id='MIT',status='REVIEW',allow_paths=[],training_allowed_after_clearance=False,category='reasoning',subcategory='math_reasoning',
        license_evidence=dict(path='license.json',sha256=file_digest(e)))
    cat=tmp_path/'catalog.json';atomic_json(cat,seal(dict(sources=[s])))
    data=b'fake parquet'; sha=hashlib.sha256(data).hexdigest()
    def metadata(s,path):return dict(url=f"https://huggingface.co/api/datasets/{did}/tree/{rev}/main",body=[dict(type='file',path=path,size=len(data),lfs=dict(oid=sha,size=len(data)))])
    def download(url,path,size,hash):path.write_bytes(data)
    args=dict(catalog=cat,spec=ROOT/'training/specifications/production_corpus_v1.json',output=tmp_path/'cache',root=tmp_path,
              fetch=True,limit=2,budget_mib=16,metadata_fetch=metadata,payload_fetch=download,rows=lambda p,n:iter([{'question':'one','answer':'two'}]))
    return args,s

def test_cache_compatibility_and_offline_resume(setup):
    a,s=setup;first=t.build(**a)
    assert first['documents']==1 and first['production_admitted_documents']==0
    def forbidden(*args):raise AssertionError('Unexpected processing/network')
    assert t.build(**dict(a,fetch=False,metadata_fetch=forbidden,payload_fetch=forbidden,rows=forbidden),resume=True)==first
    reg=json.loads(a['catalog'].read_bytes());spec=json.loads(a['spec'].read_bytes())
    assert len(list(cached_sources(a['output'],reg,spec,a['root'])))==1
    record=json.loads((a['output']/'hf-test/inventory.jsonl').read_bytes())
    assert record['screening']=='UNRESOLVED' and record['provenance']['upstream_split']=='train'

@pytest.mark.parametrize('target',['payload.parquet','inventory.jsonl','files/00000000.json'])
def test_corruption(setup,target):
    a,_=setup;t.build(**a);(a['output']/'hf-test'/target).write_bytes(b'corrupt')
    with pytest.raises((ValueError,KeyError)):t.build(**a,resume=True)

def test_changed_pin_and_budget(setup):
    a,_=setup;t.build(**a)
    with pytest.raises(SpecificationError):t.build(**dict(a,limit=3),resume=True)
    reg=json.loads(a['catalog'].read_bytes());reg['sources'][0]['revision']='b'*40;atomic_json(a['catalog'],seal(reg))
    with pytest.raises(SpecificationError):t.build(**a,resume=True)

def test_no_fetch_and_size_bound(setup):
    a,_=setup
    with pytest.raises(SpecificationError):t.build(**dict(a,fetch=False))
    with pytest.raises(SpecificationError):t.build(**dict(a,budget_mib=1),resume=True)

def test_interrupted_extraction_resumes(setup):
    a,_=setup
    def interrupted(p,n):
        yield {'question':'one','answer':'two'}
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):t.build(**dict(a,rows=interrupted))
    assert t.build(**a,resume=True)['documents']==1

class Response(io.BytesIO):
    def __init__(self,data,status=200,headers=None):super().__init__(data);self.status=status;self.headers=headers or {}

URL='https://huggingface.co/datasets/openai/gsm8k/resolve/'+'a'*40+'/main/train.parquet'

@pytest.mark.parametrize('range_supported',[True,False])
def test_download_resume(tmp_path,range_supported):
    path=tmp_path/'payload.parquet';path.with_suffix('.part').write_bytes(b'ab')
    def opener(req,timeout):
        assert req.headers['Range']=='bytes=2-'
        return Response(b'cd',206,{'Content-Range':'bytes 2-3/4'}) if range_supported else Response(b'abcd')
    t.download(URL,path,4,hashlib.sha256(b'abcd').hexdigest(),opener=opener)
    assert path.read_bytes()==b'abcd'

def test_hash_and_range_rejection(tmp_path):
    p=tmp_path/'x.parquet'
    with pytest.raises(ValueError):t.download(URL,p,4,'0'*64,opener=lambda *a,**k:Response(b'abcd'))
    assert not p.exists() and not p.with_suffix('.part').exists()
    with pytest.raises(ValueError):t.download(URL,p,4,'0'*64,opener=lambda *a,**k:Response(b'abcd',206,{'Content-Range':'wrong'}))

def test_redirect_rejected():
    with pytest.raises(SpecificationError):t.Redirect().redirect_request(None,None,302,'',{},'http://localhost/data')

def test_official_cdn_redirect():
    import urllib.request
    req=urllib.request.Request(URL)
    result=t.Redirect().redirect_request(req,None,302,'',{},'https://us.aws.cdn.hf.co/blob')
    assert result.full_url=='https://us.aws.cdn.hf.co/blob'

def test_parquet_limit(tmp_path):
    pa=pytest.importorskip('pyarrow'); pq=pytest.importorskip('pyarrow.parquet')
    p=tmp_path/'x.parquet';pq.write_table(pa.table({'question':['a','b','c'],'answer':['1','2','3']}),p)
    assert list(t.parquet_rows(p,2))==[{'question':'a','answer':'1'},{'question':'b','answer':'2'}]


def test_explicit_selection_resume_and_binding(setup):
    a,s=setup
    evidence=a['root']/'license.json'
    value=json.loads(evidence.read_bytes())
    value['body']['siblings']=[{'rfilename':'main/train.parquet'}]
    atomic_json(evidence,seal(value))
    reg=json.loads(a['catalog'].read_bytes())
    reg['sources'][0]['license_evidence']['sha256']=file_digest(evidence)
    atomic_json(a['catalog'],seal(reg))
    selection={s['dataset_id']:dict(revision=s['revision'],path='main/train.parquet')}
    first=t.build(**a,selection=selection)
    assert t.build(**dict(a,fetch=False),selection=selection,resume=True)==first
    record=json.loads((a['output']/'hf-test/inventory.jsonl').read_bytes())
    assert record['provenance']['upstream_split']=='unspecified'
    for path in ('../train.parquet','/train.parquet','main/missing.parquet'):
        with pytest.raises(SpecificationError):
            t.build(**a,selection={s['dataset_id']:dict(revision=s['revision'],path=path)},resume=True)
    with pytest.raises(SpecificationError):
        t.build(**a,selection={s['dataset_id']:dict(revision='b'*40,path='main/train.parquet')},resume=True)


def test_bounded_jsonl(tmp_path):
    p=tmp_path/'payload.parquet'
    p.write_bytes(b'{"text":"one"}\n{"text":"two"}\ninvalid')
    assert list(t.payload_rows(p,2,'data.jsonl'))==[{'text':'one'},{'text':'two'}]
    p.write_bytes(b'x'*(t.MAX_DOC+1))
    with pytest.raises(SpecificationError):list(t.payload_rows(p,1,'data.jsonl'))
    p.write_bytes(b'[]\n')
    with pytest.raises(SpecificationError):list(t.payload_rows(p,1,'data.jsonl'))
