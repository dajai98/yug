"""Registered REVIEW continuation with disposable Git and synthetic human decisions."""
import base64
import copy
import hashlib
import json
import shutil

import pytest

from tests.test_source_prepare import command, fixture as git_fixture
from training import corpus_workflow, source_discovery, source_first_clearance
from training.license_evidence import MIT_TERMS
from training.production_acquisition import read_plan
from training.production_corpus import canonical, file_digest
from training.production_spec import SpecificationError, seal, validate
from training.protected_references import make_reference
from training.source_admission import build as admit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))
    return path


@pytest.fixture
def registered(tmp_path, monkeypatch):
    root = tmp_path
    _, spec, fetch_git, calls = git_fixture.__wrapped__(root)
    upstream = root / 'upstream'
    license_bytes = ('MIT License\nCopyright (c) 2026 Fixture\n\n' + MIT_TERMS + '\n').encode()
    guide = b'# Fixture guide\n\nAn original guide to a small example project.\n'
    (upstream / 'LICENSE').write_bytes(license_bytes)
    (upstream / 'docs').mkdir()
    (upstream / 'docs/guide.md').write_bytes(guide)
    command('add', '.', cwd=upstream)
    command('-c', 'core.hooksPath='+str(root/'no-hooks'), 'commit', '-m', 'review files', cwd=upstream)
    revision = command('rev-parse', 'HEAD', cwd=upstream)
    sid = 'gh-example-project'
    license_blob = hashlib.sha1(f'blob {len(license_bytes)}\0'.encode()+license_bytes).hexdigest()
    license_path = write(root/'evidence/license.json', seal(dict(
        url=f'https://api.github.com/repos/example/project/license?ref={revision}',
        body=dict(path='LICENSE', size=len(license_bytes), sha=license_blob,
                  content=base64.b64encode(license_bytes).decode(),
                  license=dict(spdx_id='MIT')))))
    supporting = write(root/'evidence/support.json', seal(dict(
        statement='Synthetic fixture publication and provenance review; no real authority')))
    base = json.loads((corpus_workflow.ROOT/'training/specifications/source_registry_v1.json').read_bytes())
    source = copy.deepcopy(next(s for s in base['sources'] if s['source_id']=='candidate-angle'))
    source.update(source_id=sid, name='example/project', locator='https://github.com/example/project',
        revision=revision, status='REVIEW', category='software_docs', subcategory='curated_subset',
        license_id='MIT', license_evidence=dict(path='evidence/license.json',
            sha256=file_digest(license_path), upstream_path='LICENSE',
            text_sha256='sha256:'+hashlib.sha256(license_bytes).hexdigest()),
        license_text_sha256='sha256:'+hashlib.sha256(license_bytes).hexdigest(),
        allowed_categories=['software_docs'], target_cap_tokens=10000,
        category_caps_tokens={'software_docs':10000}, repository_group=sid,
        domain_group=sid, split_group=sid, split='unassigned', allow_paths=[],
        deny_paths=spec['filters']['deny_paths'], clearance_record=None,
        file_manifest=None, origin_verified=False, training_allowed_after_clearance=False)
    registry = seal(dict(schema='yug-production-source-registry-v1', sources=[source]))
    spec['group_caps']['repository_group'][sid] = 10000
    spec['group_caps']['domain_group'][sid] = 10000
    spec = seal(dict(spec, registry_identity=registry['identity']))
    for entry in spec['evidence']:
        target = root/entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(corpus_workflow.ROOT/entry['path'], target)
    validate(spec, registry, root)
    registry_path = write(root/'training/specifications/source_registry_v1.json', registry)
    spec_path = write(root/'training/specifications/production_corpus_v1.json', spec)
    reference = make_reference(work_id='fixture', work_version='1', item_id='1',
        reference_id='heldout', role='reference', protected_split='test',
        text='Unrelated protected astronomical observations.', content_type='prose',
        language='en', provenance={'source_id':'another-project'})
    references = root/'references.jsonl'
    references.write_bytes(canonical(reference)+b'\n')
    seed = dict(source_id=sid, locator=source['locator'], revision=revision,
                category=source['category'], subcategory=source['subcategory'])
    batch = write(root/'batch.json', dict(seeds=[seed],
        registry=registry_path.relative_to(root).as_posix(),
        spec=spec_path.relative_to(root).as_posix(),
        references=references.relative_to(root).as_posix()))
    monkeypatch.setattr(corpus_workflow.source_prepare, 'fetch', fetch_git)
    monkeypatch.setattr(corpus_workflow.source_discovery, 'fetch_json',
                        lambda _: pytest.fail('registered source cannot fetch metadata'))
    return dict(root=root, source=source, registry=registry, spec=spec, seed=seed,
        registry_path=registry_path, spec_path=spec_path, references=references,
        license_path=license_path, supporting=supporting, batch=batch,
        output=root/'artifacts/workflow', calls=calls, guide=guide)


def prepare(registered):
    r = registered
    report = corpus_workflow.prepare(r['batch'], r['output'], root=r['root'], fetch=True)
    assert report['phase']=='HUMAN_REVIEW' and report['training_allowed'] is False
    assert report['sources'][0]['state']=='HUMAN_REVIEW'
    assert report['sources'][0]['file_count'] > 0
    assert not (r['output']/'admission').exists()
    assert not (r['output']/'production').exists()
    assert corpus_workflow.verify(r['output'], root=r['root']) == report
    return report


def decision(registered):
    r = registered
    source = r['source']
    cache = r['output']/'cache'/source['source_id']
    rows = {row['path']:row for row in read_plan(cache/'inventory.jsonl')}
    row = rows['docs/guide.md']
    license_ref = dict(path='evidence/license.json', sha256=file_digest(r['license_path']))
    support_ref = dict(path='evidence/support.json', sha256=file_digest(r['supporting']))
    rights_path = r['root']/'decisions/guide-rights.json'
    write(rights_path, seal(dict(schema=source_first_clearance.RIGHTS,
        source_id=source['source_id'], locator=source['locator'], revision=source['revision'],
        path=row['path'], sha256=row['sha256'], owner=source['category'],
        subcategory=source['subcategory'], license_id='MIT', first_party=True,
        third_party=False, generated=False, reviewer='synthetic fixture reviewer',
        basis='Synthetic exact-file human review for test only', evidence=[support_ref])))
    grant_file = dict(path=row['path'],sha256=row['sha256'],bytes=row['bytes'],
        content_type='markdown',language='en', rights_ref=dict(
            path=rights_path.relative_to(r['root']).as_posix(),sha256=file_digest(rights_path)))
    d = seal(dict(schema=source_first_clearance.POLICY,
        decision='APPROVE_EXACT_FILES', reviewer='synthetic fixture reviewer',
        decided_at='2026-11-19', source_id=source['source_id'], locator=source['locator'],
        revision=source['revision'], owner=source['category'],
        subcategory=source['subcategory'], license_id='MIT', split='train',
        cap_tokens=1000, attribution='Fixture copyright and MIT license',
        obligations=['preserve MIT attribution'],
        source_scope_evidence=sorted([license_ref,support_ref],key=lambda x:x['path']),
        files=[grant_file]))
    decision_path = write(r['root']/'decisions/decision.json', d)
    return decision_path


def grant(registered, decision_path, *, output=None, pin=None):
    r = registered
    return source_first_clearance.build(decision_path, output or r['root']/'artifacts/grant',
        expected_decision_sha256=pin if pin is not None else file_digest(decision_path),
        cache=r['output']/'cache', catalog=r['output']/'discovery/catalog.json',
        review=r['output']/'clearance', references=r['references'], root=r['root'],
        base_registry=r['registry_path'], base_spec=r['spec_path'])


def test_registered_review_to_clearance_to_admission(registered):
    r = registered
    prepare(r)
    assert len(r['calls']) == 1
    assert corpus_workflow.prepare(r['batch'],r['output'],root=r['root'],resume=True) == \
        corpus_workflow.verify(r['output'],root=r['root'])
    assert len(r['calls']) == 1
    upstream = [r['registry_path'],r['spec_path'],r['license_path'],r['supporting'],
                r['output']/'clearance/evidence.jsonl']
    before = {p:file_digest(p) for p in upstream}
    d = decision(r)
    completed = grant(r,d)
    assert completed['files']==1 and completed['production_tokens']==0
    assert completed['training_authorized'] is False
    output = r['root']/'artifacts/grant'
    approved = json.loads((output/'registry.json').read_bytes())
    assert len(approved['sources'])==1
    assert approved['sources'][0]['status']=='ACCEPT'
    assert approved['sources'][0]['source_id']==r['source']['source_id']
    assert approved['sources'][0]['split_group']==r['source']['split_group']
    assert approved['sources'][0]['target_cap_tokens']<=1000
    assert json.loads((output/'spec.json').read_bytes())['group_caps']==r['spec']['group_caps']
    assert completed==grant(r,d)
    admission = admit(r['output']/'cache',r['registry_path'],r['spec_path'],
        output/'registry.json',output/'spec.json',output/'plan.jsonl',r['references'],
        r['root']/'artifacts/admission',root=r['root'],
        cache_registry=r['output']/'discovery/catalog.json')
    assert admission['counts']['CLEARED']==1
    assert admission['newly_authorized_files']==0 and admission['new_production_tokens']==0
    assert admission['cleared_cache_matches']==1
    assert before=={p:file_digest(p) for p in upstream}


@pytest.mark.parametrize('change', ['wrong_revision','wrong_locator','accepted','cleared',
                                   'alias','duplicate','wrong_id'])
def test_registered_seed_rejects_wrong_or_ambiguous_identity(registered, change):
    r = registered
    registry = copy.deepcopy(r['registry'])
    seed = dict(r['seed'])
    source = registry['sources'][0]
    if change=='wrong_revision':seed['revision']='0'*40
    elif change=='wrong_locator':seed['locator']='https://github.com/example/other'
    elif change=='accepted':source['status']='ACCEPT'
    elif change=='cleared':source['file_manifest']='artifacts/old-files.json'
    elif change=='alias':registry['sources'].append(dict(source,source_id='gh-example-alias'))
    elif change=='duplicate':registry['sources'].append(copy.deepcopy(source))
    else:seed['source_id']='gh-example-other'
    path = write(r['root']/'alternate-registry.json',seal(registry))
    with pytest.raises(SpecificationError):
        source_discovery.build(path,r['spec_path'],r['root']/'artifacts/failed-discovery',
                               seeds=[seed],root=r['root'])


@pytest.mark.parametrize('change', ['missing_pin','wrong_pin','forged_rights','stale_review',
                                   'wrong_split','altered_split_group','oversized_cap',
                                   'missing_evidence','conflicting_approvals'])
def test_registered_clearance_fails_closed(registered, change):
    r = registered
    prepare(r)
    d = decision(r)
    pin = file_digest(d)
    if change=='missing_pin':pin=None
    elif change=='wrong_pin':pin='sha256:'+'0'*64
    elif change=='forged_rights':
        rights = r['root']/'decisions/guide-rights.json'
        value = json.loads(rights.read_bytes()); value['first_party']=False
        write(rights,seal(value))
    elif change=='stale_review':
        r['references'].write_bytes(r['references'].read_bytes()+b' ')
    elif change=='altered_split_group':
        catalog = r['output']/'discovery/catalog.json'
        value = json.loads(catalog.read_bytes())
        value['sources'][0]['split_group'] = 'other-group'
        write(catalog,seal(value))
    elif change in ('wrong_split','oversized_cap'):
        value = json.loads(d.read_bytes())
        if change=='wrong_split':value['split']='unassigned'
        else:value['cap_tokens']=10001
        write(d,seal(value));pin=file_digest(d)
    elif change=='conflicting_approvals':
        value = json.loads(d.read_bytes())
        value['files'].append(dict(value['files'][0]))
        write(d,seal(value));pin=file_digest(d)
    else:r['supporting'].write_bytes(b'changed independent evidence')
    if change=='missing_pin':
        with pytest.raises(SpecificationError):
            source_first_clearance.build(d,r['root']/'artifacts/failed-grant',
                expected_decision_sha256=None,cache=r['output']/'cache',
                catalog=r['output']/'discovery/catalog.json',
                review=r['output']/'clearance',references=r['references'],root=r['root'],
                base_registry=r['registry_path'],base_spec=r['spec_path'])
    else:
        with pytest.raises((SpecificationError, ValueError)):
            grant(r,d,output=r['root']/'artifacts/failed-grant',pin=pin)
    assert not (r['root']/'artifacts/failed-grant/complete.json').exists()


def test_registered_grant_recovers_from_interrupted_publication(registered, monkeypatch):
    r = registered
    prepare(r)
    d = decision(r)
    original = source_first_clearance.atomic_json
    def interrupted(path,value):
        original(path,value)
        if path.name=='files.json':raise KeyboardInterrupt()
    monkeypatch.setattr(source_first_clearance,'atomic_json',interrupted)
    with pytest.raises(KeyboardInterrupt):grant(r,d)
    monkeypatch.setattr(source_first_clearance,'atomic_json',original)
    assert grant(r,d)['files']==1


def test_registered_existing_path_scope_cannot_expand(registered):
    r = registered
    r['source']['allow_paths'] = ['docs/other.md']
    r['registry'] = seal(dict(r['registry'], sources=[r['source']]))
    r['spec'] = seal(dict(r['spec'], registry_identity=r['registry']['identity']))
    write(r['registry_path'], r['registry'])
    write(r['spec_path'], r['spec'])
    prepare(r)
    d = decision(r)
    with pytest.raises(SpecificationError, match='established source/group cap'):
        grant(r, d, output=r['root']/'artifacts/failed-grant')
