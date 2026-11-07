"""Bounded metadata discovery with synthetic responses; never fetch corpus files."""
import base64
import hashlib
import json
import urllib.error
from pathlib import Path

import pytest

from training import source_discovery as discovery
from training.production_corpus import canonical
from training.production_spec import SpecificationError, seal
from training.source_expansion import build as expand
from training.source_prepare import ROOT, classify

MIT_TEXT = b"""MIT License
Copyright (c) 2026 Example

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE."""

BSD_TEXT = b"""BSD 2-Clause License
Copyright (c) 2026 Example
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""


@pytest.fixture
def inputs(tmp_path):
    registry = tmp_path/'registry.json'
    registry.write_bytes(canonical(seal(dict(schema='yug-production-source-registry-v1', sources=[]))))
    return dict(registry=registry, spec=ROOT/'training/specifications/production_corpus_v1.json',
                output=tmp_path/'out', root=tmp_path,
                seeds=[dict(locator='https://github.com/example/library', branch='main', category='code', subcategory='python')])


def provider(license_id='MIT', data=b'Fixture license evidence\n'):
    calls = []
    revision = 'a'*40
    blob_sha = hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
    tree_sha = 'b'*40
    def fetch(url):
        calls.append(url)
        if url.endswith('/git/ref/heads/main'):
            return {'ref': 'refs/heads/main', 'object': {'sha': revision, 'type': 'commit'}}
        if url.endswith('/git/commits/'+revision):
            return {'sha': revision, 'tree': {'sha': tree_sha}}
        if url.endswith('/git/trees/'+tree_sha):
            return {'sha': tree_sha, 'truncated': False, 'tree': [
                {'path': 'LICENSE', 'type': 'blob', 'mode': '100644', 'sha': blob_sha}]}
        if url.endswith('/git/blobs/'+blob_sha):
            return dict(sha=blob_sha, encoding='base64', size=len(data),
                        content=base64.b64encode(data).decode())
        assert url.endswith('/license?ref='+revision)
        return dict(license={'spdx_id': license_id}, encoding='base64', type='file',
                    content=base64.b64encode(data).decode(), size=len(data), path='LICENSE',
                    sha=blob_sha,
                    html_url='https://github.com/example/library/blob/'+revision+'/LICENSE')
    return fetch, calls


def canonical_provider(primary, companions=None, *, include_primary=True):
    fetch, calls = provider('NOASSERTION', primary)
    files = ({'LICENSE': primary} if include_primary else {}) | (companions or {})
    hashes = {name: hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
              for name, data in files.items()}

    def supplied(url):
        if url.endswith('/git/trees/'+'b'*40):
            return {'sha': 'b'*40, 'truncated': False, 'tree': [
                {'path': name, 'type': 'blob', 'mode': '100644', 'sha': hashes[name]}
                for name in files]}
        for name, sha in hashes.items():
            if url.endswith('/git/blobs/'+sha):
                data = files[name]
                return dict(sha=sha, encoding='base64', size=len(data),
                            content=base64.b64encode(data).decode())
        return fetch(url)
    return supplied, calls


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


@pytest.mark.parametrize('license_id',['CC-BY-SA-3.0','CC-BY-SA-4.0','MPL-2.0','Apache-2.0 WITH LLVM-exception'])
def test_conditional_metadata_never_grants(inputs,license_id):
    fetch,_=provider(license_id)
    result=discovery.build(**inputs,fetcher=fetch)
    assert result['counts']['REVIEW']==1 and result['counts']['ACCEPTED']==0
    source=json.loads((inputs['output']/'catalog.json').read_bytes())['sources'][0]
    assert source['status']=='CONDITIONAL' and source['conditional_license']['approval_required']
    assert not source['training_allowed_after_clearance'] and not source['allow_paths']
    assert classify(source)=='CACHE_READY'
    assert discovery.build(**inputs,resume=True)==result


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


@pytest.mark.parametrize('corrupt',[False,True])
def test_noassertion_uses_verified_full_text(inputs,corrupt):
    data=(Path(__file__).parent/'fixtures/license_evidence/Apache-2.0.txt').read_bytes()
    fetch,_=provider('NOASSERTION', data)
    def retained(url):
        result=fetch(url)
        if '/license?' in url:
            result.update(content=base64.b64encode(data).decode(),size=len(data),
                          sha='0'*40 if corrupt else hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest())
        return result
    if corrupt:
        with pytest.raises(ValueError): discovery.build(**inputs,fetcher=retained)
    else:
        result=discovery.build(**inputs,fetcher=retained)
        assert result['counts']['ACCEPTED']==1
        assert discovery.build(**inputs,resume=True)==result
        source=json.loads((inputs['output']/'catalog.json').read_bytes())['sources'][0]
        assert source['license_id']=='Apache-2.0' and source['allow_paths']==[]


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


@pytest.mark.parametrize('terms,expected', [
    (MIT_TEXT, 'MIT'),
    (BSD_TEXT, 'BSD-2-Clause'),
    ((Path(__file__).parent/'fixtures/license_evidence/Apache-2.0.txt').read_bytes(),
     'Apache-2.0'),
])
def test_noassertion_canonical_root_terms_resolve_for_review_only(inputs, terms, expected):
    fetch, _ = canonical_provider(terms)
    result = discovery.build(**inputs, fetcher=fetch)
    assert result['counts']['ACCEPTED'] == 1
    source = json.loads((inputs['output']/'catalog.json').read_bytes())['sources'][0]
    assert source['license_id'] == expected
    assert source['allow_paths'] == [] and source['clearance_record'] is None
    assert source['training_allowed_after_clearance'] is False
    ref = source['license_evidence']['canonical_evidence']
    record = json.loads((inputs['root']/ref['path']).read_bytes())
    assert record['identity'] == ref['identity'] and record['authority'] == 'NONE'
    assert record['revision'] == 'a'*40
    assert record['files'][0]['path'] == 'LICENSE'
    assert record['files'][0]['text'].encode() == terms
    assert record['files'][0]['sha256'] == 'sha256:'+hashlib.sha256(terms).hexdigest()
    assert discovery.build(**inputs, resume=True) == result


def test_missing_canonical_root_file_stays_review(inputs):
    fetch, _ = canonical_provider(MIT_TEXT, include_primary=False)
    result = discovery.build(**inputs, fetcher=fetch)
    assert result['counts']['REVIEW'] == 1
    decision = json.loads((inputs['output']/'decisions.json').read_bytes())['sources'][0]
    ref = decision['canonical_evidence']
    assert json.loads((inputs['root']/ref['path']).read_bytes())['files'] == []


def test_conflicting_canonical_licenses_stay_review(inputs):
    apache = (Path(__file__).parent/'fixtures/license_evidence/Apache-2.0.txt').read_bytes()
    fetch, _ = canonical_provider(MIT_TEXT, {'COPYING': apache})
    result = discovery.build(**inputs, fetcher=fetch)
    assert result['counts']['REVIEW'] == 1
    decision = json.loads((inputs['output']/'decisions.json').read_bytes())['sources'][0]
    record = json.loads((inputs['root']/decision['canonical_evidence']['path']).read_bytes())
    assert len(record['files']) == 2 and record['resolved_license'] is None


@pytest.mark.parametrize('terms', [MIT_TEXT+b'\nAdditional restriction', b'Custom terms.'])
def test_custom_or_modified_canonical_terms_stay_review(inputs, terms):
    fetch, _ = canonical_provider(terms)
    assert discovery.build(**inputs, fetcher=fetch)['counts']['REVIEW'] == 1


def test_readme_only_claim_does_not_resolve(inputs):
    fetch, _ = canonical_provider(MIT_TEXT, {'README.md': MIT_TEXT}, include_primary=False)

    def readme(url):
        result = fetch(url)
        if '/license?' in url:
            result.update(path='README.md',
                          html_url='https://github.com/example/library/blob/'+'a'*40+'/README.md')
        return result
    assert discovery.build(**inputs, fetcher=readme)['counts']['REVIEW'] == 1
    decision = json.loads((inputs['output']/'decisions.json').read_bytes())['sources'][0]
    record = json.loads((inputs['root']/decision['canonical_evidence']['path']).read_bytes())
    assert record['files'] == []


def test_canonical_receipt_tamper_and_revision_mismatch_fail(inputs):
    fetch, _ = canonical_provider(MIT_TEXT)

    def wrong_commit(url):
        result = fetch(url)
        if '/git/commits/' in url:
            result['sha'] = 'c'*40
        return result
    with pytest.raises(SpecificationError, match='commit/tree mismatch'):
        discovery.build(**inputs, fetcher=wrong_commit)
    inputs['output'] = inputs['root']/'good'
    discovery.build(**inputs, fetcher=fetch)
    evidence = inputs['output']/'source-00/canonical_evidence.json'
    evidence.write_bytes(b'{}')
    with pytest.raises(ValueError):
        discovery.build(**inputs, resume=True)


def test_notice_and_copyright_companions_are_retained(inputs):
    fetch, _ = canonical_provider(MIT_TEXT, {'NOTICE': b'Notice obligations\n',
                                              'COPYRIGHT': b'Copyright context\n'})
    result = discovery.build(**inputs, fetcher=fetch)
    assert result['counts']['REVIEW'] == 1
    decision = json.loads((inputs['output']/'decisions.json').read_bytes())['sources'][0]
    assert 'companion requires obligation review' in decision['reason']
    ref = decision['canonical_evidence']
    record = json.loads((inputs['root']/ref['path']).read_bytes())
    assert {f['path'] for f in record['files']} == {'LICENSE', 'NOTICE', 'COPYRIGHT'}
    assert {f['path']: f['text'] for f in record['files']}['NOTICE'] == 'Notice obligations\n'
    assert all(f['git_blob'] and f['sha256'] for f in record['files'])
    assert json.loads((inputs['output']/'catalog.json').read_bytes())['sources'] == []


def test_unavailable_canonical_blob_stays_review(inputs):
    fetch, _ = canonical_provider(MIT_TEXT)

    def unavailable(url):
        if '/git/blobs/' in url:
            raise urllib.error.URLError('offline')
        return fetch(url)
    result = discovery.build(**inputs, fetcher=unavailable)
    assert result['counts']['REVIEW'] == 1
    assert not json.loads((inputs['output']/'catalog.json').read_bytes())['sources']
