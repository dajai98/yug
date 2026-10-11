"""Bounded metadata discovery with synthetic responses; never fetch corpus files."""
import base64
import hashlib
import json
import pytest
from training import source_discovery as discovery
from training.production_corpus import canonical
from training.production_spec import seal, SpecificationError
from training.source_expansion import build as expand
from training.source_prepare import ROOT, classify


@pytest.fixture
def inputs(tmp_path):
    registry = tmp_path/'registry.json'
    registry.write_bytes(canonical(seal(dict(schema='yug-production-source-registry-v1', sources=[]))))
    return dict(registry=registry, spec=ROOT/'training/specifications/production_corpus_v1.json',
                output=tmp_path/'out', root=tmp_path,
                seeds=[dict(locator='https://github.com/example/library', branch='main', category='code', subcategory='python')])


def provider(license_id='MIT'):
    calls = []; revision = 'a'*40; data = b'Fixture license evidence\n'
    def fetch(url):
        calls.append(url)
        if url.endswith('/git/ref/heads/main'):
            return {'ref': 'refs/heads/main', 'object': {'sha': revision, 'type': 'commit'}}
        assert url.endswith('/license?ref='+revision)
        return dict(license={'spdx_id': license_id}, encoding='base64', type='file',
                    content=base64.b64encode(data).decode(), size=len(data), path='LICENSE',
                    sha=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest(),
                    html_url='https://github.com/example/library/blob/'+revision+'/LICENSE')
    return fetch, calls


def test_discovery_expansion_compatibility_and_offline_resume(inputs):
    fetch, calls = provider()
    result = discovery.build(**inputs, fetcher=fetch)
    assert result['counts'] == dict(ACCEPTED=1, REJECTED=0, REVIEW=0)
    assert len(calls) == 2
    assert result == discovery.build(**inputs, resume=True)
    registry = json.loads((inputs['output']/'catalog.json').read_bytes())
    source = registry['sources'][0]
    assert classify(source) == 'CACHE_READY'
    assert source['target_cap_tokens'] == 0 and source['file_manifest'] is None
    assert source['clearance_record'] is None and not source['origin_verified']
    assert source['allow_paths'] == [] and source['split'] == 'unassigned'
    assert expand(inputs['output']/'catalog.json', inputs['spec'], inputs['root']/'expanded')['candidate_sources'] == 1


@pytest.mark.parametrize('license_id,status', [('GPL-3.0','REJECTED'), ('MIT OR GPL-3.0','REJECTED'),
                                             ('NOASSERTION','REVIEW'), (None,'REVIEW')])
def test_license_allowlist(inputs, license_id, status):
    fetch, _ = provider(license_id)
    result = discovery.build(**inputs, fetcher=fetch)
    assert result['counts'][status] == 1
    assert json.loads((inputs['output']/'catalog.json').read_bytes())['sources'] == []


@pytest.mark.parametrize('field,value', [('sha','0'*40), ('size',0), ('encoding','plain'),
                                      ('path','vendor/LICENSE'), ('html_url','https://github.com/example/library/blob/main/LICENSE')])
def test_invalid_license_binding_fails_closed(inputs, field, value):
    fetch, _ = provider()
    def damaged(url):
        result = fetch(url)
        if '/license?' in url: result[field] = value
        return result
    with pytest.raises(ValueError): discovery.build(**inputs, fetcher=damaged)
    assert not (inputs['output']/'catalog.json').exists()


def test_interrupt_resume_keeps_first_pin(inputs):
    fetch, calls = provider()
    def interrupted(url):
        if '/license?' in url: raise KeyboardInterrupt()
        return fetch(url)
    with pytest.raises(KeyboardInterrupt): discovery.build(**inputs, fetcher=interrupted)
    assert discovery.build(**inputs, fetcher=fetch, resume=True)['counts']['ACCEPTED'] == 1
    assert sum(url.endswith('/git/ref/heads/main') for url in calls) == 1


def test_corrupt_response_rejected(inputs):
    fetch, _ = provider(); discovery.build(**inputs, fetcher=fetch)
    (inputs['output']/'source-00/license.json').write_bytes(b'corrupt')
    with pytest.raises(ValueError): discovery.build(**inputs, resume=True, fetcher=fetch)


def test_no_implicit_network(inputs):
    with pytest.raises(SpecificationError, match='explicit --fetch'):
        discovery.build(**inputs)


def test_existing_source_not_rediscovered(inputs):
    catalog = seal(dict(schema='yug-production-source-registry-v1', sources=[dict(
        source_id='existing', locator=inputs['seeds'][0]['locator'])]))
    inputs['registry'].write_bytes(canonical(catalog))
    assert discovery.build(**inputs)['counts']['REJECTED'] == 1


def test_bounds_and_changed_resume(inputs):
    with pytest.raises(SpecificationError, match='1..10'):
        discovery.build(**dict(inputs, seeds=inputs['seeds']*11))
    fetch, _ = provider(); discovery.build(**inputs, fetcher=fetch)
    inputs['seeds'][0]['locator'] = 'https://github.com/example/other'
    with pytest.raises(SpecificationError, match='inputs changed'):
        discovery.build(**inputs, resume=True)


def test_fetcher_restricts_url_before_http():
    with pytest.raises(SpecificationError, match='Non-metadata'):
        discovery.fetch_json('https://github.com/example/library/archive/main.zip')
    with pytest.raises(SpecificationError, match='Non-metadata'):
        discovery.fetch_json('https://api.github.com/repos/example/library/commits/HEAD')


def test_ref_response_cannot_retain_patch_fields(inputs):
    fetch, _ = provider()
    def invalid(url):
        result = fetch(url)
        if '/git/ref/' in url: result['files'] = [{'patch': 'not metadata'}]
        return result
    with pytest.raises(SpecificationError, match='non-ref metadata'):
        discovery.build(**inputs, fetcher=invalid)
    assert not (inputs['output']/'source-00/commit.json').exists()
