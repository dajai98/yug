"""Synthetic R2 decisions exercise real Git, review, rights and comparison gates."""
import base64
import copy
import hashlib
import json

import pytest

from tests.test_registered_source_continuation import (
    registered, prepare, decision as first_decision, grant, write,
)
from tests.test_source_prepare import command
from training import (clearance_evidence, source_prepare,
                      source_revision_clearance as revision_clearance,
                      source_revision_inventory)
from training.notice_attribution import source_clearance
from training.production_acquisition import read_plan
from training.production_corpus import file_digest
from training.production_spec import SpecificationError, seal


def _rights(case, row, revision, name):
    source = case['source']
    ref = dict(path='evidence/support.json', sha256=file_digest(case['supporting']))
    rights = seal(dict(schema='reviewed-first-file-rights-v1',
        source_id=source['source_id'], locator=source['locator'], revision=revision,
        path=row['path'], sha256=row['sha256'], owner=source['category'],
        subcategory=source['subcategory'], license_id='MIT', first_party=True,
        third_party=False, generated=False, reviewer='synthetic fixture reviewer',
        basis='Synthetic exact R2 file review; not real authority', evidence=[ref]))
    path = write(case['root']/f'decisions/{name}-rights.json', rights)
    return dict(path=row['path'], sha256=row['sha256'], bytes=row['bytes'],
        content_type=('markdown' if row['path'].endswith('.md') else
                      'code' if row['path'].endswith('.py') else 'prose'),
        language='python' if row['path'].endswith('.py') else 'en', rights_ref=dict(
            path=path.relative_to(case['root']).as_posix(), sha256=file_digest(path)))


@pytest.fixture
def r2_case(registered):
    r = registered
    prepare(r)
    root = r['root']
    sid = r['source']['source_id']
    r1 = r['source']['revision']
    old_cache = r['output']/'cache'/sid
    old_rows = {row['path']:row for row in read_plan(old_cache/'inventory.jsonl')}
    first_path = first_decision(r)
    first = json.loads(first_path.read_bytes())
    first['files'] += [_rights(r, old_rows[p], r1, 'r1-'+p.replace('/', '-'))
                       for p in ('README.md', 'src/main.py')]
    first['files'].sort(key=lambda f:f['path'])
    first_path = write(first_path, seal(first))
    grant(r, first_path)
    active = root/'artifacts/grant'
    active_registry = json.loads((active/'registry.json').read_bytes())
    active_source = next(s for s in active_registry['sources'] if s['source_id']==sid)
    spec = json.loads((active/'spec.json').read_bytes())
    historical = {p:file_digest(p) for p in (
        first_path, active/'files.json', active/'clearance.json',
        active/'registry.json', active/'spec.json', old_cache/'complete.json')}
    upstream = root/'upstream'
    (upstream/'README.md').write_bytes(b'# Changed fixture readme\n')
    (upstream/'src/main.py').unlink()
    (upstream/'docs/added.md').write_bytes(b'# Added fixture guide\n')
    command('add', '-A', cwd=upstream)
    command('-c', 'core.hooksPath='+str(root/'no-hooks'),
            'commit', '-m', 'r2 fixture', cwd=upstream)
    r2 = command('rev-parse', 'HEAD', cwd=upstream)
    next_source = dict(active_source, revision=r2)
    catalog = write(root/'artifacts/r2-catalog.json', seal(dict(
        schema=active_registry['schema'], sources=[next_source])))
    catalog_value = json.loads(catalog.read_bytes())
    def fetcher(cache, source):
        command('--git-dir', cache, 'fetch', '--depth=2', upstream, source['revision'])
    cache_root = root/'artifacts/r2-cache'
    status = source_prepare.run(catalog_value, spec, cache_root, fetcher=fetcher)
    assert status[0]['status']=='CACHED'
    new_cache = cache_root/sid
    review = root/'artifacts/r2-review'
    result = clearance_evidence.build(cache_root, review, root=root,
        review_only=True, cache_registry=catalog,
        root_registry=active/'registry.json', root_spec=active/'spec.json',
        references=r['references'])
    assert result['complete'] and result['counts']['EVIDENCED']==0
    comparison_dir = root/'artifacts/r2-comparison'
    comparison_args = dict(source_id=sid, locator=active_source['locator'],
        old_revision=r1, new_revision=r2,
        expected_registry_sha256=file_digest(active/'registry.json'),
        expected_spec_sha256=file_digest(active/'spec.json'),
        expected_manifest_sha256=file_digest(active/'files.json'),
        expected_references_sha256=file_digest(r['references']),
        expected_old_inventory_sha256=file_digest(old_cache/'inventory.jsonl'),
        expected_new_inventory_sha256=file_digest(new_cache/'inventory.jsonl'),
        expected_old_receipt_sha256=file_digest(old_cache/'complete.json'),
        expected_new_receipt_sha256=file_digest(new_cache/'complete.json'), root=root)
    comparison = source_revision_inventory.build(old_cache, new_cache,
        active/'registry.json', active/'spec.json', active/'files.json',
        r['references'], comparison_dir, **comparison_args)
    license_bytes = (upstream/'LICENSE').read_bytes()
    old_license = json.loads(r['license_path'].read_bytes())
    fresh_license = seal(dict(old_license,
        url=f'https://api.github.com/repos/example/project/license?ref={r2}'))
    license_path = write(root/'evidence/license-r2.json', fresh_license)
    license_ref = dict(path='evidence/license-r2.json',
        sha256=file_digest(license_path), upstream_path='LICENSE',
        text_sha256='sha256:'+hashlib.sha256(license_bytes).hexdigest())
    new_rows = {row['path']:row for row in read_plan(new_cache/'inventory.jsonl')}
    selected = ('LICENSE', 'README.md', 'docs/added.md', 'docs/guide.md')
    files = [_rights(r, new_rows[p], r2, 'r2-'+p.replace('/', '-'))
             for p in selected]
    decision_value = seal(dict(schema=revision_clearance.POLICY,
        decision='APPROVE_EXACT_FILES', reviewer='synthetic fixture reviewer',
        decided_at='2026-11-20', source_id=sid, locator=active_source['locator'],
        old_revision=r1, revision=r2, owner=active_source['category'],
        subcategory=active_source['subcategory'], license_id='MIT', split='train',
        cap_tokens=active_source['target_cap_tokens'],
        attribution='Fixture copyright and MIT license',
        obligations=['preserve MIT attribution'],
        source_scope_evidence=sorted([
            dict(path=license_ref['path'], sha256=license_ref['sha256']),
            dict(path='evidence/support.json', sha256=file_digest(r['supporting']))],
            key=lambda ref:ref['path']),
        revision_license_evidence=license_ref,
        comparison_sha256=file_digest(comparison_dir/'comparison.json'),
        removed_paths=sorted(row['path'] for row in comparison['files']
                             if row['new'] is None),
        excluded_paths=sorted(set(new_rows)-set(selected)), files=files))
    decision_path = write(root/'decisions/r2-decision.json', decision_value)
    kwargs = dict(expected_decision_sha256=file_digest(decision_path),
        old_cache=old_cache, new_cache=new_cache,
        registry_path=active/'registry.json', spec_path=active/'spec.json',
        manifest_path=active/'files.json', references_path=r['references'],
        comparison_dir=comparison_dir,
        expected_comparison_sha256=file_digest(comparison_dir/'comparison.json'),
        expected_comparison_receipt_sha256=file_digest(comparison_dir/'complete.json'),
        expected_old_clearance_sha256=file_digest(active/'clearance.json'),
        review=review, catalog=catalog,
        expected_review_receipt_sha256=file_digest(review/'complete.json'),
        root=root, **{k:v for k,v in comparison_args.items()
                      if k.startswith('expected_')})
    return dict(r=r, kwargs=kwargs, decision=decision_path,
        decision_value=decision_value, comparison=comparison,
        output=root/'artifacts/r2-provisional', historical=historical,
        old_source=active_source, active=active)


def test_r2_real_comparison_review_rights_and_provisional_output(r2_case):
    c = r2_case
    receipt = revision_clearance.build(c['decision'], c['output'], **c['kwargs'])
    assert receipt['approved_paths'] == ['LICENSE', 'README.md',
                                         'docs/added.md', 'docs/guide.md']
    assert receipt['removed_paths'] == ['src/main.py']
    assert receipt['review_only'] and not receipt['training_authorized']
    assert receipt['production_tokens']==0
    assert json.loads((c['output']/'clearance.json').read_bytes())['decision']=='PROVISIONAL'
    assert not any((c['output']/name).exists() for name in
                   ('registry.json', 'spec.json', 'plan.jsonl'))
    assert c['historical']=={p:file_digest(p) for p in c['historical']}
    trial = dict(c['old_source'], revision=c['decision_value']['revision'],
        file_manifest=(c['output']/'files.json').relative_to(c['r']['root']).as_posix(),
        clearance_record=(c['output']/'clearance.json').relative_to(c['r']['root']).as_posix())
    with pytest.raises(SpecificationError):
        source_clearance(trial, c['r']['root'])


def test_r2_replay_and_conflicting_output(r2_case):
    c=r2_case
    first=revision_clearance.build(c['decision'],c['output'],**c['kwargs'])
    assert revision_clearance.build(c['decision'],c['output'],**c['kwargs'])==first
    (c['output']/'files.json').write_bytes(b'corrupt')
    with pytest.raises(SpecificationError):
        revision_clearance.build(c['decision'],c['output'],**c['kwargs'])


def test_explicit_exclusion_of_unchanged_r1_file(r2_case):
    c=r2_case
    d=copy.deepcopy(c['decision_value'])
    d['files']=[f for f in d['files'] if f['path']!='docs/guide.md']
    d['excluded_paths'].append('docs/guide.md')
    d['excluded_paths'].sort()
    write(c['decision'],seal(d))
    kw=dict(c['kwargs'],expected_decision_sha256=file_digest(c['decision']))
    result=revision_clearance.build(c['decision'],c['output'],**kw)
    assert 'docs/guide.md' not in result['approved_paths']
    assert 'docs/guide.md' in result['excluded_paths']
    assert c['historical']=={p:file_digest(p) for p in c['historical']}


def test_interrupted_publication_recovers_exactly(r2_case, monkeypatch):
    c=r2_case
    original=revision_clearance.atomic_json
    calls=[]
    def interrupted(path, value):
        calls.append(path)
        if len(calls)==2:
            raise RuntimeError('synthetic interruption')
        return original(path,value)
    monkeypatch.setattr(revision_clearance,'atomic_json',interrupted)
    with pytest.raises(RuntimeError,match='synthetic interruption'):
        revision_clearance.build(c['decision'],c['output'],**c['kwargs'])
    monkeypatch.setattr(revision_clearance,'atomic_json',original)
    result=revision_clearance.build(c['decision'],c['output'],**c['kwargs'])
    before={p:file_digest(p) for p in c['output'].rglob('*') if p.is_file()}
    assert revision_clearance.build(c['decision'],c['output'],**c['kwargs'])==result
    assert before=={p:file_digest(p) for p in before}


def test_read_only_audit_requires_all_ownership_evidence(r2_case):
    c=r2_case
    revision_clearance.build(c['decision'],c['output'],**c['kwargs'])
    pin=file_digest(c['output']/'complete.json')
    assert revision_clearance.verify_provisional(c['output'],
        expected_complete_sha256=pin,root=c['r']['root'])['production_tokens']==0
    next((c['output']/'ownership').glob('*.json')).unlink()
    with pytest.raises(SpecificationError):
        revision_clearance.verify_provisional(c['output'],
            expected_complete_sha256=pin,root=c['r']['root'])


@pytest.mark.parametrize('damage', [
    'missing_pin', 'stale_pin', 'r1_decision', 'missing_unchanged',
    'missing_added', 'duplicate', 'wrong_revision', 'wrong_source',
    'rights_stale', 'rights_forged', 'missing_rights', 'missing_owner',
    'license_old', 'notice_dropped', 'protected', 'allow_conflict',
    'screened_out', 'comparison_pin', 'inventory_pin', 'review_pin',
    'old_clearance_pin', 'blob_modified', 'wrong_hash', 'cap_increase',
    'split_change', 'missing_removed', 'review_corrupt',
    'comparison_forged',
])
def test_r2_fail_closed(r2_case, damage):
    c=r2_case
    d=copy.deepcopy(c['decision_value'])
    kw=dict(c['kwargs'])
    if damage=='missing_pin': kw['expected_decision_sha256']=None
    elif damage=='stale_pin': kw['expected_decision_sha256']='sha256:'+'0'*64
    elif damage=='r1_decision': d['schema']='reviewed-first-source-clearance-v1'
    elif damage=='missing_unchanged':
        d['files']=[f for f in d['files'] if f['path']!='docs/guide.md']
    elif damage=='missing_added':
        d['files']=[f for f in d['files'] if f['path']!='docs/added.md']
    elif damage=='duplicate': d['files'].append(d['files'][0])
    elif damage=='wrong_revision': d['revision']='0'*40
    elif damage=='wrong_source': d['source_id']='gh-other'
    elif damage=='rights_stale':
        f=d['files'][0]
        f['rights_ref']['sha256']='sha256:'+'0'*64
    elif damage=='rights_forged':
        f=d['files'][0]
        rights_path=c['r']['root']/f['rights_ref']['path']
        rights=json.loads(rights_path.read_bytes())
        rights['revision']='0'*40
        write(rights_path, seal(rights))
        f['rights_ref']['sha256']=file_digest(rights_path)
    elif damage=='missing_rights':
        d['files'][0]['rights_ref']['path']='evidence/absent.json'
    elif damage=='missing_owner':
        f=d['files'][0]
        rights_path=c['r']['root']/f['rights_ref']['path']
        rights=json.loads(rights_path.read_bytes())
        rights['owner']='different'
        write(rights_path, seal(rights))
        f['rights_ref']['sha256']=file_digest(rights_path)
    elif damage=='license_old':
        d['revision_license_evidence']=c['old_source']['license_evidence']
    elif damage=='notice_dropped': d['obligations']=[]
    elif damage=='protected':
        kw['expected_references_sha256']='sha256:'+'0'*64
    elif damage=='allow_conflict':
        d['excluded_paths'].append(d['files'][0]['path'])
        d['excluded_paths'].sort()
    elif damage=='screened_out':
        rows={r['path']:r for r in read_plan(kw['new_cache']/'inventory.jsonl')}
        d['files'].append(_rights(c['r'],rows['vendor/copied.py'],d['revision'],'forbidden'))
        d['files'].sort(key=lambda f:f['path'])
        d['excluded_paths'].remove('vendor/copied.py')
    elif damage=='comparison_pin': kw['expected_comparison_sha256']='sha256:'+'0'*64
    elif damage=='inventory_pin': kw['expected_new_inventory_sha256']='sha256:'+'0'*64
    elif damage=='review_pin': kw['expected_review_receipt_sha256']='sha256:'+'0'*64
    elif damage=='old_clearance_pin':
        kw['expected_old_clearance_sha256']='sha256:'+'0'*64
    elif damage=='blob_modified':
        row=next(r for r in read_plan(kw['new_cache']/'inventory.jsonl')
                 if r['path']=='README.md')
        (kw['new_cache']/row['local_blob']).write_bytes(b'changed cached blob')
    elif damage=='wrong_hash': d['files'][0]['sha256']='sha256:'+'0'*64
    elif damage=='cap_increase':
        d['cap_tokens']=c['old_source']['target_cap_tokens']+1
    elif damage=='split_change': d['split']='validation'
    elif damage=='missing_removed': d['removed_paths']=[]
    elif damage=='review_corrupt':
        (kw['review']/'evidence.jsonl').write_bytes(b'corrupt')
    elif damage=='comparison_forged':
        directory=kw['comparison_dir']
        path=directory/'comparison.json'
        report=json.loads(path.read_bytes())
        next(f for f in report['files'] if f['path']=='README.md')[
            'classification']='UNCHANGED'
        report=seal(report)
        write(path,report)
        complete_path=directory/'complete.json'
        complete=json.loads(complete_path.read_bytes())
        complete['comparison_identity']=report['identity']
        complete['comparison_sha256']=file_digest(path)
        write(complete_path,seal(complete))
        d['comparison_sha256']=file_digest(path)
        kw['expected_comparison_sha256']=file_digest(path)
        kw['expected_comparison_receipt_sha256']=file_digest(complete_path)
    if d != c['decision_value']:
        c['decision']=write(c['decision'], seal(d))
        kw['expected_decision_sha256']=file_digest(c['decision'])
    with pytest.raises((SpecificationError, ValueError, KeyError)):
        revision_clearance.build(c['decision'], c['output'], **kw)
    assert not c['output'].exists()
