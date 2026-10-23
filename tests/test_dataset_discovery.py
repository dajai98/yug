import json
import pytest
from training import dataset_discovery as d
from training.production_spec import seal, SpecificationError
from training.production_corpus import canonical
from training.source_prepare import ROOT, classify
from training.source_expansion import build as expand

@pytest.fixture
def args(tmp_path):
    reg=tmp_path/'registry.json'
    reg.write_bytes(canonical(seal(dict(schema='yug-production-source-registry-v1',sources=[]))))
    return dict(registry=reg,spec=ROOT/'training/specifications/production_corpus_v1.json',
        seeds=[dict(dataset_id='org/data',category='code',subcategory='python')],output=tmp_path/'out',root=tmp_path)

def provider(license='mit',**changes):
    calls=[]
    def fetch(url):
        assert '/api/datasets/org/data/revision/' in url
        calls.append(url)
        return dict(dict(id='org/data',sha='a'*40,cardData={'license':license},private=False,gated=False),**changes)
    return fetch,calls

def test_pinning_replay_and_no_admission_bypass(args):
    fetch,calls=provider(); first=d.build(**args,fetcher=fetch)
    assert len(calls)==2 and calls[1].endswith('a'*40)
    assert first['counts']==dict(ACCEPTED=1,REVIEW=0,REJECTED=0)
    assert d.build(**args,resume=True)==first
    catalog=json.loads((args['output']/'catalog.json').read_bytes())
    s=catalog['sources'][0]
    assert s['status']=='REVIEW' and s['target_cap_tokens']==0 and s['allow_paths']==[]
    assert s['file_manifest'] is None and not s['training_allowed_after_clearance']
    assert classify(s)!='CACHE_READY'
    result=expand(args['output']/'catalog.json',args['spec'],args['root']/'expanded')
    assert result['candidate_sources']==0  # existing transport/clearance gates remain closed

@pytest.mark.parametrize('license,status',[(None,'REVIEW'),('other','REVIEW'),(['mit','apache-2.0'],'REVIEW'),
    ('gpl-3.0','REJECTED'),('cc-by-nc-4.0','REJECTED'),('MIT OR Apache-2.0','REJECTED'),(['mit'],'ACCEPTED')])
def test_licenses(args,license,status):
    fetch,_=provider(license); assert d.build(**args,fetcher=fetch)['counts'][status]==1

@pytest.mark.parametrize('changes',[{'private':True},{'gated':'auto'},{'tags':['license:apache-2.0']},{'gated':None}])
def test_access_and_conflicts_review(args,changes):
    fetch,_=provider(**changes); assert d.build(**args,fetcher=fetch)['counts']['REVIEW']==1


@pytest.mark.parametrize('license',['cc-by-sa-3.0','cc-by-sa-4.0','mpl-2.0','Apache-2.0 WITH LLVM-exception'])
def test_conditional_is_review_not_auto_approval(args,license):
    fetch,_=provider(license)
    result=d.build(**args,fetcher=fetch)
    assert result['counts']['REVIEW']==1 and result['counts']['ACCEPTED']==0
    source=json.loads((args['output']/'catalog.json').read_bytes())['sources'][0]
    assert source['conditional_license']['tier']=='CONDITIONAL'
    assert source['status']=='REVIEW' and not source['training_allowed_after_clearance']
    assert source['target_cap_tokens']==0 and source['allow_paths']==[]
    assert d.build(**args,resume=True)==result

@pytest.mark.parametrize('changes',[{'id':'other/data'},{'sha':'main'}])
def test_identity_mismatch(args,changes):
    fetch,_=provider(**changes)
    with pytest.raises(SpecificationError): d.build(**args,fetcher=fetch)

def test_pinned_response_must_match(args):
    fetch,_=provider()
    def bad(url):
        data=fetch(url)
        if url.endswith('a'*40): data['sha']='b'*40
        return data
    with pytest.raises(SpecificationError): d.build(**args,fetcher=bad)

def test_interrupt_reuses_first_pin(args):
    fetch,calls=provider()
    def stop(url):
        if url.endswith('a'*40): raise KeyboardInterrupt()
        return fetch(url)
    with pytest.raises(KeyboardInterrupt): d.build(**args,fetcher=stop)
    d.build(**args,resume=True,fetcher=fetch)
    assert sum(u.endswith('main') for u in calls)==1

def test_corruption_and_changed_request(args):
    fetch,_=provider(); d.build(**args,fetcher=fetch)
    with pytest.raises(SpecificationError):
        d.build(**dict(args,seeds=[dict(dataset_id='org/other',category='code',subcategory='python')]),resume=True)
    (args['output']/'source-00/metadata.json').write_text('{}')
    with pytest.raises(SpecificationError): d.build(**args,resume=True)

@pytest.mark.parametrize('url',['https://huggingface.co/datasets/org/data/resolve/main/data.parquet',
    'https://datasets-server.huggingface.co/rows', 'https://example.com/api/datasets/org/data/revision/main'])
def test_transport_rejects_payload_urls(url):
    with pytest.raises(SpecificationError): d.fetch_json(url)

def test_no_implicit_network_or_oversized_response(args):
    with pytest.raises(SpecificationError): d.build(**args)
    fetch,_=provider(extra='x'*d.MAX_BYTES)
    with pytest.raises(SpecificationError): d.build(**args,resume=True,fetcher=fetch)

def test_registry_null_locator(args):
    args['registry'].write_bytes(canonical(seal(dict(schema='yug-production-source-registry-v1',sources=[{'locator':None}]))))
    fetch,_=provider()
    assert d.build(**args,fetcher=fetch)['counts']['ACCEPTED']==1
