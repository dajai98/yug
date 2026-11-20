"""Synthetic fixture authority only; no real source or training approval."""
import copy
import hashlib
import json

import pytest

from tests.test_registered_source_continuation import (
    registered, prepare, decision as first_decision, grant, write)
from tests.test_source_prepare import command
from training import source_first_clearance
from training.corpus_gate import (BUNDLE, SCHEMA as USE_SCHEMA,
                                  POLICY as USE_POLICY, binding, verify_receipt)
from training.approval_governance import Governance, event
from training.category_ownership import decision as ownership_decision
from training.composite_lineage import verify
from training.production_acquisition import ProductionBuild, read_plan
from training.production_corpus import canonical, file_digest
from training.production_spec import SpecificationError, seal, validate
from training.production_spec import TOKENIZER
from training.source_admission import build as admit
from training.source_approval_composition import build as compose
from tests.test_production_acquisition import approval_kwargs, prior_inventory


def two_batches(r):
    root = r['root']
    upstream = root/'upstream'
    guide = ('# First guide\n\n' + ' '.join(f'word{i}' for i in range(90)) + '\n').encode()
    (upstream/'docs/guide.md').write_bytes(guide)
    other = ('# Second guide\n\n' + ' '.join(f'topic{i}' for i in range(90)) + '\n').encode()
    (upstream/'docs/other.md').write_bytes(other)
    command('add', '.', cwd=upstream)
    command('-c', 'core.hooksPath='+str(root/'no-hooks'), 'commit',
            '-m', 'second exact file', cwd=upstream)
    revision = command('rev-parse', 'HEAD', cwd=upstream)
    r['source']['revision'] = revision
    r['seed']['revision'] = revision
    license_record = json.loads(r['license_path'].read_bytes())
    license_record['url'] = (
        f'https://api.github.com/repos/example/project/license?ref={revision}')
    write(r['license_path'], seal(license_record))
    r['source']['license_evidence']['sha256'] = file_digest(r['license_path'])
    r['registry'] = seal(dict(r['registry'], sources=[
        r['source'] if s['source_id']==r['source']['source_id'] else s
        for s in r['registry']['sources']]))
    r['spec'] = seal(dict(r['spec'], registry_identity=r['registry']['identity']))
    write(r['registry_path'], r['registry'])
    write(r['spec_path'], r['spec'])
    write(r['batch'], dict(seeds=[r['seed']],
        registry=r['registry_path'].relative_to(root).as_posix(),
        spec=r['spec_path'].relative_to(root).as_posix(),
        references=r['references'].relative_to(root).as_posix()))
    prepare(r)
    a = first_decision(r)
    a_value = json.loads(a.read_bytes())
    rows = {row['path']: row for row in read_plan(
        r['output']/'cache'/r['source']['source_id']/'inventory.jsonl')}
    row = rows['docs/other.md']
    support = dict(path='evidence/support.json',sha256=file_digest(r['supporting']))
    rights = write(root/'decisions/other-rights.json', seal(dict(
        schema=source_first_clearance.RIGHTS,
        source_id=r['source']['source_id'], locator=r['source']['locator'],
        revision=revision, path=row['path'], sha256=row['sha256'],
        owner=r['source']['category'], subcategory=r['source']['subcategory'],
        license_id='MIT', first_party=True, third_party=False,
        generated=False, reviewer='synthetic fixture reviewer',
        basis='Independent synthetic exact-file review for test only',
        evidence=[support])))
    b_value = copy.deepcopy(a_value)
    b_value['files'] = [dict(path=row['path'], sha256=row['sha256'],
        bytes=row['bytes'], content_type='markdown', language='en',
        rights_ref=dict(path=rights.relative_to(root).as_posix(),
                        sha256=file_digest(rights)))]
    b = write(root/'decisions/decision-b.json', seal(b_value))
    grant(r, a, output=root/'artifacts/grant-a')
    grant(r, b, output=root/'artifacts/grant-b')
    return a, b


def composition_decision(r, a, b, *, b_folder='grant-b'):
    root = r['root']
    def ref(path):
        return dict(path=path.relative_to(root).as_posix(), sha256=file_digest(path))
    components = []
    for folder_name, decision_path in (('grant-a',a),(b_folder,b)):
        folder = root/'artifacts'/folder_name
        components.append(dict(decision_ref=ref(decision_path),
            decision_identity=json.loads(decision_path.read_bytes())['identity'],
            manifest_ref=ref(folder/'files.json'),
            manifest_identity=json.loads((folder/'files.json').read_bytes())['identity'],
            clearance_ref=ref(folder/'clearance.json'),
            clearance_identity=json.loads((folder/'clearance.json').read_bytes())['identity'],
            complete_ref=ref(folder/'complete.json'),
            complete_identity=json.loads((folder/'complete.json').read_bytes())['identity'],
            plan_ref=ref(folder/'plan.jsonl')))
    components.sort(key=lambda c:c['decision_identity'])
    active = json.loads((root/'artifacts/grant-a/registry.json').read_bytes())
    source = active['sources'][0]
    files = sorted([dict(path=f['path'], sha256=f['sha256'])
        for component in components
        for f in json.loads((root/component['manifest_ref']['path']).read_bytes())['files']],
        key=lambda f:f['path'])
    d = seal(dict(schema='reviewed-exact-file-composition-v1',
        decision='COMPOSE_EXACT_FILES', reviewer='synthetic composition reviewer',
        decided_at='2026-11-19', source_id=source['source_id'],
        locator=source['locator'], revision=source['revision'], split=source['split'],
        repository_group=source['repository_group'], domain_group=source['domain_group'],
        split_group=source['split_group'], category=source['category'],
        subcategory=source['subcategory'], license_id=source['license_id'],
        target_cap_tokens=source['target_cap_tokens'],
        category_caps_tokens=source['category_caps_tokens'],
        attribution='Fixture copyright and MIT license',
        obligations=['preserve MIT attribution'], files=files,
        components=components, original_registry_ref=ref(r['registry_path']),
        original_spec_ref=ref(r['spec_path']),
        active_registry_ref=ref(root/'artifacts/grant-a/registry.json'),
        active_spec_ref=ref(root/'artifacts/grant-a/spec.json'),
        catalog_ref=ref(r['output']/'discovery/catalog.json'),
        references_ref=ref(r['references']),
        cache_path=(r['output']/'cache').relative_to(root).as_posix(),
        review_path=(r['output']/'clearance').relative_to(root).as_posix()))
    return write(root/'decisions/composition.json', d)


def pins(path):
    return [c['decision_ref']['sha256'] for c in
            json.loads(path.read_bytes())['components']]


def unrelated_cleared_source(r):
    """Add a separate synthetic, validator-backed prior clearance for source C."""
    root = r['root']
    upstream = root/'upstream-c'
    upstream.mkdir()
    command('init', str(upstream))
    command('config', 'user.name', 'Fixture C', cwd=upstream)
    command('config', 'user.email', 'fixture-c@example.invalid', cwd=upstream)
    data = ('# Prior source\n\n' + ' '.join(f'legacy{i}' for i in range(90)) + '\n').encode()
    (upstream/'docs').mkdir()
    (upstream/'docs/legacy.md').write_bytes(data)
    command('add', '.', cwd=upstream)
    command('-c', 'core.hooksPath='+str(root/'no-hooks'), 'commit',
            '-m', 'fixture C', cwd=upstream)
    revision = command('rev-parse', 'HEAD', cwd=upstream)
    sid = 'gh-unrelated-cleared'
    path = 'docs/legacy.md'
    sha = 'sha256:'+hashlib.sha256(data).hexdigest()
    license_value = json.loads(r['license_path'].read_bytes())
    license_value['url'] = f'https://api.github.com/repos/unrelated/cleared/license?ref={revision}'
    license_path = write(root/'evidence/c-license.json', seal(license_value))
    license_ref = dict(path=license_path.relative_to(root).as_posix(),
                       sha256=file_digest(license_path))
    support = write(root/'evidence/c-support.json', seal(dict(
        statement='Independent synthetic fixture C rights review only')))
    support_ref = dict(path=support.relative_to(root).as_posix(),
                       sha256=file_digest(support))
    rights = write(root/'decisions/c-rights.json', seal(dict(
        source_id=sid, revision=revision, path=path, sha256=sha,
        decision='APPROVE_EXACT_FILE', reviewer='synthetic fixture C reviewer',
        evidence=[license_ref, support_ref])))
    rights_ref = dict(path=rights.relative_to(root).as_posix(),
                      sha256=file_digest(rights))
    own = write(root/'artifacts/c-clearance/ownership.json', seal(dict(
        source_id=sid, revision=revision, path=path, sha256=sha,
        owner=r['source']['category'], subcategory=r['source']['subcategory'],
        decision_type='manual_review', rationale_ref=json.loads(rights.read_bytes())['identity'],
        reviewer='synthetic fixture C reviewer')))
    own_ref = dict(path=own.relative_to(root).as_posix(),sha256=file_digest(own))
    file = dict(path=path, sha256=sha, bytes=len(data), owner=r['source']['category'],
        subcategory=r['source']['subcategory'], content_type='markdown',
        license_id='MIT', tags=[], ownership_decision=ownership_decision(
            source_id=sid, revision=revision, path=path, sha256=sha,
            owner=r['source']['category'], subcategory=r['source']['subcategory'],
            decision_type='manual_review', rationale_ref=json.loads(rights.read_bytes())['identity'],
            evidence_ref=own_ref))
    manifest_path = write(root/'artifacts/c-clearance/files.json', seal(dict(
        source_id=sid, revision=revision, files=[file])))
    evidence = [license_ref, support_ref, rights_ref, own_ref]
    clearance_path = write(root/'artifacts/c-clearance/clearance.json', seal(dict(
        source_id=sid, revision=revision,
        file_manifest_identity=json.loads(manifest_path.read_bytes())['identity'],
        decision='ACCEPT', evidence=evidence,
        attribution='Fixture C copyright and MIT license',
        obligations=['preserve MIT attribution'],
        governance_decision_identity=json.loads(rights.read_bytes())['identity'])))
    c = copy.deepcopy(r['source'])
    c.update(source_id=sid, name='unrelated/cleared',
        locator='https://github.com/unrelated/cleared', revision=revision,
        status='ACCEPT', status_authority='synthetic fixture C review',
        license_evidence=dict(c['license_evidence'], path=license_ref['path'],
                              sha256=license_ref['sha256']),
        repository_group=sid, domain_group=sid, split_group=sid,
        split='train', target_cap_tokens=1000,
        category_caps_tokens={c['category']:1000}, allow_paths=[path],
        file_manifest=manifest_path.relative_to(root).as_posix(),
        clearance_record=clearance_path.relative_to(root).as_posix(),
        origin_verified=True, training_allowed_after_clearance=True)
    for kind in ('repository_group','domain_group'):
        r['spec']['group_caps'][kind][sid] = 1000
    r['registry'] = seal(dict(r['registry'],sources=[*r['registry']['sources'],c]))
    r['spec'] = seal(dict(r['spec'],registry_identity=r['registry']['identity']))
    write(r['registry_path'],r['registry'])
    write(r['spec_path'],r['spec'])
    blob = root/'artifacts/c-cached/files/docs/legacy.md'
    blob.parent.mkdir(parents=True,exist_ok=True)
    blob.write_bytes(data)
    record = seal(dict(source_id=sid, revision=revision, **file,
        clearance_status='ACCEPT', split='train', split_group=sid,
        split_unit_kind='repository', allow_rule=path,
        deny_rules_checked=c['deny_paths'], language='en',
        license_evidence=evidence, attribution='Fixture C copyright and MIT license',
        source_clearance_identity=json.loads(clearance_path.read_bytes())['identity'],
        quality_review='ACCEPT', input_path=blob.relative_to(root).as_posix()))
    from training.production_acquisition import validate_clearance
    validate(r['spec'],r['registry'],root)
    validate_clearance(record,c,r['spec'],root,file)
    return c, record


def pinned_prior_plan(r, record):
    root = r['root']
    active = list(read_plan(root/'artifacts/grant-a/plan.jsonl'))
    rows = sorted([*active, record], key=lambda row: (
        row['source_id'], row['revision'], row['path'], row['sha256']))
    from training.production_acquisition import item_key
    rows.sort(key=item_key)
    path = root/'artifacts/prior-approved-plan.jsonl'
    path.write_bytes(b''.join(canonical(row)+b'\n' for row in rows))
    return path


def fixture_receipt(record, source, spec, root):
    path=root/'synthetic-intended-use.txt'
    if not path.exists():
        path.write_text('Synthetic fixture approval, rights and provenance only.')
    return seal(dict(schema=USE_SCHEMA, policy=USE_POLICY,
        binding=binding(record,source,spec), decision='ACCEPT',
        authority=dict(kind='human',id='fixture reviewer'),
        decided_at='2026-11-19', human_approval_required=True,
        human_approved=True, intended_uses=['commercial_training'],
        reason='Fixture decision', obligations='Retain fixture attribution',
        restrictions='Fixture only; no real-world authority',
        evidence=[dict(role=role,path=path.name,sha256=file_digest(path))
                  for role in ('approval','rights_scope','provenance')]))


def test_two_independent_grants_compose_and_admit(registered):
    r = registered
    a,b = two_batches(r)
    d = composition_decision(r,a,b)
    leaf_paths = [r['root']/'artifacts'/f'grant-{name}'/file
        for name in ('a','b') for file in
        ('files.json','clearance.json','registry.json','spec.json','plan.jsonl','complete.json')]
    before = {p:file_digest(p) for p in leaf_paths}
    output = r['root']/'artifacts/composite'
    result = compose(d,output,expected_decision_sha256=file_digest(d),
                     expected_component_decision_sha256=pins(d),root=r['root'])
    assert result['files']==2 and result['training_authorized'] is False
    assert result['production_tokens']==0
    assert result == compose(d,output,expected_decision_sha256=file_digest(d),
                             expected_component_decision_sha256=pins(d),root=r['root'])
    assert before == {p:file_digest(p) for p in leaf_paths}
    registry = json.loads((output/'registry.json').read_bytes())
    spec = json.loads((output/'spec.json').read_bytes())
    assert len(registry['sources'])==1
    assert registry['sources'][0]['target_cap_tokens'] == json.loads(
        (r['root']/'artifacts/grant-a/registry.json').read_bytes())['sources'][0]['target_cap_tokens']
    assert spec['group_caps'] == r['spec']['group_caps']
    validate(spec,registry,r['root'])
    grant_value=json.loads((output/'clearance.json').read_bytes())
    manifest=json.loads((output/'files.json').read_bytes())
    verify(r['root'],registry['sources'][0],manifest,grant_value)
    admission=admit(r['output']/'cache',r['registry_path'],r['spec_path'],
        output/'registry.json',output/'spec.json',output/'plan.jsonl',
        r['references'],r['root']/'artifacts/admission',root=r['root'],
        cache_registry=r['output']/'discovery/catalog.json')
    assert admission['counts']['CLEARED']==2
    assert admission['new_production_tokens']==0


@pytest.mark.parametrize('change', ['missing_pin','wrong_pin','missing_leaf_pin',
                                   'wrong_leaf_pin','overlap','hash',
                                   'cap','attribution','missing_leaf','altered_leaf'])
def test_composition_fails_closed(registered,change):
    r=registered
    a,b=two_batches(r)
    d=composition_decision(r,a,b)
    pin=file_digest(d)
    leaf_pins=pins(d)
    if change=='missing_pin':pin=None
    elif change=='wrong_pin':pin='sha256:'+'0'*64
    elif change=='missing_leaf_pin':leaf_pins=None
    elif change=='wrong_leaf_pin':leaf_pins[1]='sha256:'+'0'*64
    elif change in ('overlap','hash','cap','attribution'):
        value=json.loads(d.read_bytes())
        if change=='overlap':value['files'][1]=dict(value['files'][0])
        elif change=='hash':value['files'][1]['sha256']='sha256:'+'0'*64
        elif change=='cap':value['target_cap_tokens']+=1
        else:value['attribution']='altered terms'
        write(d,seal(value));pin=file_digest(d)
    elif change=='missing_leaf':
        (r['root']/'artifacts/grant-b/complete.json').unlink()
    else:
        (r['root']/'artifacts/grant-b/files.json').write_bytes(b'forged')
    with pytest.raises((SpecificationError,ValueError,KeyError)):
        compose(d,r['root']/'artifacts/failed',
                expected_decision_sha256=pin,
                expected_component_decision_sha256=leaf_pins,root=r['root'])
    assert not (r['root']/'artifacts/failed/complete.json').exists()


@pytest.mark.parametrize('stage',['files','plan'])
def test_interrupted_composition_replays_identically(registered,monkeypatch,stage):
    r=registered
    a,b=two_batches(r)
    d=composition_decision(r,a,b)
    from training import source_approval_composition as module
    original=module.atomic_json
    original_replace=module.atomic_replace
    def interrupt(path,value):
        original(path,value)
        if stage=='files' and path.name=='files.json':raise KeyboardInterrupt()
    def interrupt_replace(source,target):
        if stage=='plan' and target.name=='plan.jsonl':raise KeyboardInterrupt()
        return original_replace(source,target)
    monkeypatch.setattr(module,'atomic_json',interrupt)
    monkeypatch.setattr(module,'atomic_replace',interrupt_replace)
    output=r['root']/'artifacts/composite'
    with pytest.raises(KeyboardInterrupt):
        compose(d,output,expected_decision_sha256=file_digest(d),
                expected_component_decision_sha256=pins(d),root=r['root'])
    monkeypatch.setattr(module,'atomic_json',original)
    monkeypatch.setattr(module,'atomic_replace',original_replace)
    assert compose(d,output,expected_decision_sha256=file_digest(d),
                   expected_component_decision_sha256=pins(d),root=r['root'])['files']==2


@pytest.mark.parametrize('target', ['decision','manifest','clearance','complete',
                                    'rights','references','review','cache'])
def test_retained_lineage_mutation_invalidates_composite(registered,target):
    r=registered
    a,b=two_batches(r)
    d=composition_decision(r,a,b)
    output=r['root']/'artifacts/composite'
    compose(d,output,expected_decision_sha256=file_digest(d),
            expected_component_decision_sha256=pins(d),root=r['root'])
    paths=dict(decision=b,manifest=r['root']/'artifacts/grant-b/files.json',
        clearance=r['root']/'artifacts/grant-b/clearance.json',
        complete=r['root']/'artifacts/grant-b/complete.json',
        rights=r['root']/'decisions/other-rights.json',references=r['references'],
        review=r['output']/'clearance/complete.json',
        cache=r['output']/'cache/status.json')
    path=paths[target]
    path.write_bytes(path.read_bytes()+b' ')
    registry=json.loads((output/'registry.json').read_bytes())
    spec=json.loads((output/'spec.json').read_bytes())
    with pytest.raises((SpecificationError,ValueError)):
        validate(spec,registry,r['root'])


def test_unrelated_registered_source_is_preserved(registered):
    r=registered
    unrelated=copy.deepcopy(r['source'])
    unrelated.update(source_id='gh-unrelated-project',
        name='unrelated/project',locator='https://github.com/unrelated/project',
        repository_group='gh-unrelated-project',
        domain_group='gh-unrelated-project',split_group='gh-unrelated-project')
    r['registry']=seal(dict(r['registry'],sources=[r['source'],unrelated]))
    for kind in ('repository_group','domain_group'):
        r['spec']['group_caps'][kind]['gh-unrelated-project']=10000
    r['spec']=seal(dict(r['spec'],registry_identity=r['registry']['identity']))
    write(r['registry_path'],r['registry'])
    write(r['spec_path'],r['spec'])
    validate(r['spec'],r['registry'],r['root'])
    a,b=two_batches(r)
    d=composition_decision(r,a,b)
    output=r['root']/'artifacts/composite'
    compose(d,output,expected_decision_sha256=file_digest(d),
            expected_component_decision_sha256=pins(d),root=r['root'])
    after=json.loads((output/'registry.json').read_bytes())
    assert next(s for s in after['sources']
                if s['source_id']=='gh-unrelated-project')==unrelated
    assert json.loads((output/'spec.json').read_bytes())['group_caps']==r['spec']['group_caps']


def test_cleared_unrelated_source_survives_composition_and_admission(registered):
    r=registered
    c, c_record=unrelated_cleared_source(r)
    a,b=two_batches(r)
    prior=pinned_prior_plan(r,c_record)
    d=composition_decision(r,a,b)
    output=r['root']/'artifacts/composite'
    args=dict(expected_decision_sha256=file_digest(d),
              expected_component_decision_sha256=pins(d),root=r['root'],
              prior_approved_plan=prior,expected_prior_plan_sha256=file_digest(prior))
    result=compose(d,output,**args)
    assert result['files']==3 and result['carried_forward_files']==1
    assert result['training_authorized'] is False and result['production_tokens']==0
    assert result==compose(d,output,**args)
    registry=json.loads((output/'registry.json').read_bytes())
    assert next(s for s in registry['sources'] if s['source_id']==c['source_id'])==c
    rows=list(read_plan(output/'plan.jsonl'))
    assert len(rows)==3 and len({row['identity'] for row in rows})==3
    assert next(row for row in rows if row['source_id']==c['source_id'])==c_record
    assert {row['path'] for row in rows if row['source_id']==r['source']['source_id']}=={
        'docs/guide.md','docs/other.md'}
    admission=admit(r['output']/'cache',r['registry_path'],r['spec_path'],
        output/'registry.json',output/'spec.json',output/'plan.jsonl',
        r['references'],r['root']/'artifacts/admission',root=r['root'],
        cache_registry=r['output']/'discovery/catalog.json')
    assert admission['production_plan_files']==3
    assert {row['identity'] for row in read_plan(r['root']/'artifacts/admission/plan.jsonl')
            if row['source_id']==c['source_id']}=={c_record['identity']}
    assert admission['new_production_tokens']==0


@pytest.mark.parametrize('mutation',['missing_plan','missing_pin','wrong_pin',
                                     'changed_bytes','forged_clearance','duplicate',
                                     'omitted_c','wrong_revision','wrong_split',
                                     'changed_c_grant','changed_c_rights',
                                     'changed_c_payload'])
def test_prior_approved_plan_fails_closed(registered,mutation):
    r=registered
    _,c_record=unrelated_cleared_source(r)
    a,b=two_batches(r)
    prior=pinned_prior_plan(r,c_record)
    d=composition_decision(r,a,b)
    pin=file_digest(prior)
    if mutation=='missing_plan': prior=None;pin=None
    elif mutation=='missing_pin': pin=None
    elif mutation=='wrong_pin': pin='sha256:'+'0'*64
    elif mutation=='changed_bytes': prior.write_bytes(prior.read_bytes()+b' ')
    elif mutation in ('changed_c_grant','changed_c_rights','changed_c_payload'):
        if mutation=='changed_c_grant':
            path=r['root']/'artifacts/c-clearance/clearance.json'
        elif mutation=='changed_c_rights':
            path=r['root']/'decisions/c-rights.json'
        else:
            path=r['root']/c_record['input_path']
        path.write_bytes(path.read_bytes()+b' ')
    else:
        rows=list(read_plan(prior))
        c_row=next(row for row in rows if row['source_id']==c_record['source_id'])
        if mutation=='forged_clearance':
            c_row['source_clearance_identity']='sha256:'+'0'*64
            c_row=seal(c_row)
            rows=[c_row if row['source_id']==c_record['source_id'] else row
                  for row in rows]
        elif mutation=='duplicate': rows.append(c_row)
        elif mutation=='omitted_c': rows=[row for row in rows
                                           if row['source_id']!=c_record['source_id']]
        elif mutation=='wrong_revision':
            c_row['revision']='0'*40;c_row=seal(c_row)
            rows=[c_row if row['source_id']==c_record['source_id'] else row
                  for row in rows]
        else:
            c_row['split']='validation';c_row=seal(c_row)
            rows=[c_row if row['source_id']==c_record['source_id'] else row
                  for row in rows]
        from training.production_acquisition import item_key
        prior.write_bytes(b''.join(canonical(row)+b'\n' for row in
                                   sorted(rows,key=item_key)))
        pin=file_digest(prior)
    with pytest.raises((SpecificationError,ValueError,KeyError)):
        compose(d,r['root']/'artifacts/failed',
                expected_decision_sha256=file_digest(d),
                expected_component_decision_sha256=pins(d),root=r['root'],
                prior_approved_plan=prior,expected_prior_plan_sha256=pin)
    assert not (r['root']/'artifacts/failed/complete.json').exists()


@pytest.mark.parametrize('stage',['files','plan'])
def test_multisource_interruption_recovers_without_rewriting_prior(registered,monkeypatch,stage):
    r=registered
    _,c_record=unrelated_cleared_source(r)
    a,b=two_batches(r)
    prior=pinned_prior_plan(r,c_record)
    prior_bytes=prior.read_bytes()
    d=composition_decision(r,a,b)
    from training import source_approval_composition as module
    original_json=module.atomic_json
    original_replace=module.atomic_replace
    def interrupt_json(path,value):
        original_json(path,value)
        if stage=='files' and path.name=='files.json':raise KeyboardInterrupt()
    def interrupt_replace(source,target):
        if stage=='plan' and target.name=='plan.jsonl':raise KeyboardInterrupt()
        return original_replace(source,target)
    monkeypatch.setattr(module,'atomic_json',interrupt_json)
    monkeypatch.setattr(module,'atomic_replace',interrupt_replace)
    output=r['root']/'artifacts/composite'
    args=dict(expected_decision_sha256=file_digest(d),
              expected_component_decision_sha256=pins(d),root=r['root'],
              prior_approved_plan=prior,expected_prior_plan_sha256=file_digest(prior))
    with pytest.raises(KeyboardInterrupt):compose(d,output,**args)
    monkeypatch.setattr(module,'atomic_json',original_json)
    monkeypatch.setattr(module,'atomic_replace',original_replace)
    result=compose(d,output,**args)
    assert result['files']==3 and result['carried_forward_files']==1
    assert prior.read_bytes()==prior_bytes
    assert result==compose(d,output,**args)


def test_multisource_production_dedup_and_new_use_authority(registered):
    r=registered
    c,c_record=unrelated_cleared_source(r)
    a,b=two_batches(r)
    prior=pinned_prior_plan(r,c_record)
    root=r['root']
    c_plan=root/'artifacts/c-only-plan.jsonl'
    c_plan.write_bytes(canonical(c_record)+b'\n')
    class WordTokenizer:
        identity=TOKENIZER
        vocab_size=4096
        def encode(self,text):
            return [int(hashlib.sha256(word.encode()).hexdigest()[:4],16)%4096
                    for word in text.split()]
    prior_build=root/'artifacts/c-production'
    prior_kwargs=approval_kwargs(root,file_digest(c_plan))
    with ProductionBuild(prior_build,r['spec'],r['registry'],
            repository_root=root,reference_plan=r['references'],
            prior_production=prior_inventory(root),
            expected_plan_sha256=file_digest(c_plan),**prior_kwargs) as first:
        assert first.run(c_plan,WordTokenizer())['state']=='COMPLETE'
        assert first.db.execute("SELECT count(*) FROM items WHERE stage='ACCEPTED'").fetchone()[0]==1
    prior_inventory_path=prior_inventory(root,builds=(prior_build/'state.sqlite',),
                                         name='c-prior.json')
    d=composition_decision(r,a,b)
    output=root/'artifacts/composite'
    compose(d,output,expected_decision_sha256=file_digest(d),
            expected_component_decision_sha256=pins(d),root=root,
            prior_approved_plan=prior,expected_prior_plan_sha256=file_digest(prior))
    plan=output/'plan.jsonl'
    registry=json.loads((output/'registry.json').read_bytes())
    spec=json.loads((output/'spec.json').read_bytes())
    with pytest.raises(SpecificationError):
        ProductionBuild(root/'artifacts/stale-prospective-approval',spec,registry,
            repository_root=root,reference_plan=r['references'],
            prior_production=prior_inventory_path,
            expected_plan_sha256=file_digest(plan),**prior_kwargs)
    old_use=fixture_receipt(c_record,c,r['spec'],root)
    with pytest.raises(SpecificationError,match='binding'):
        verify_receipt(old_use,c_record,c,spec,root,'commercial_training')
    kwargs=approval_kwargs(root,file_digest(plan))
    log=kwargs['governance_log']
    previous=Governance(log,kwargs['expected_governance_sha256']).head
    components=json.loads((output/'clearance.json').read_bytes())['components']
    for component in components:
        row=event(approval_identity=component['decision_identity'],
            approval_type='source_component',
            scope=dict(source_id=r['source']['source_id'],revision=r['source']['revision'],
                       component_decision_identity=component['decision_identity']),
            previous_status='NONE',new_status='ACTIVE',
            transition_at='2026-11-19T00:00:00Z',reason_ref='synthetic fixture authority',
            previous_event_identity=previous)
        with log.open('ab') as stream:stream.write(canonical(row)+b'\n')
        previous=row['identity']
    kwargs['expected_governance_sha256']=file_digest(log)
    with ProductionBuild(root/'artifacts/combined-production',spec,registry,
            repository_root=root,reference_plan=r['references'],
            prior_production=prior_inventory_path,
            expected_plan_sha256=file_digest(plan),**kwargs) as combined:
        assert combined.run(plan,WordTokenizer())['state']=='COMPLETE'
        stages=dict(combined.db.execute('SELECT stage,count(*) FROM items GROUP BY stage'))
        assert stages.get('ACCEPTED')==2 and stages.get('DUPLICATE')==1
        assert combined.db.execute("SELECT sum(tokens) FROM totals WHERE kind='source'").fetchone()[0]>0
        assert combined.export_shards(root/'artifacts/combined-production/export')
        current=Governance(log,combined.expected_governance_sha256)
        component=components[0]
        revoked=event(approval_identity=component['decision_identity'],
            approval_type='source_component',
            scope=dict(source_id=r['source']['source_id'],revision=r['source']['revision'],
                       component_decision_identity=component['decision_identity']),
            previous_status='ACTIVE',new_status='REVOKED',
            transition_at='2026-11-19T00:00:01Z',
            reason_ref='synthetic post-commit revocation',
            previous_event_identity=current.head)
        with log.open('ab') as stream:stream.write(canonical(revoked)+b'\n')
        combined.expected_governance_sha256=file_digest(log)
        with pytest.raises(SpecificationError,match='revoked'):
            combined.export_shards(root/'artifacts/combined-production/after-revocation')


def test_incompatible_notice_scope_rejected(registered):
    r=registered
    a,b=two_batches(r)
    notice=r['root']/'notices/NOTICE.txt'
    notice.parent.mkdir(parents=True,exist_ok=True)
    notice.write_text('Synthetic second-batch-only NOTICE obligation.')
    value=json.loads(b.read_bytes())
    value['source_scope_evidence'].append(dict(
        path=notice.relative_to(r['root']).as_posix(),sha256=file_digest(notice)))
    value['source_scope_evidence'].sort(key=lambda x:x['path'])
    alternate=write(r['root']/'decisions/decision-b-notice.json',seal(value))
    grant(r,alternate,output=r['root']/'artifacts/grant-b-notice')
    decision_path=composition_decision(r,a,alternate,b_folder='grant-b-notice')
    with pytest.raises(SpecificationError,match='NOTICE'):
        compose(decision_path,r['root']/'artifacts/notices-conflict',
                expected_decision_sha256=file_digest(decision_path),
                expected_component_decision_sha256=pins(decision_path),root=r['root'])


@pytest.mark.parametrize('mutation',['extra_file','changed_hash','extra_evidence',
                                     'missing_component','circular_component',
                                     'missing_marker','broadened_allow'])
def test_shared_lineage_verifier_rejects_altered_union(registered,mutation):
    r=registered
    a,b=two_batches(r)
    d=composition_decision(r,a,b)
    output=r['root']/'artifacts/composite'
    compose(d,output,expected_decision_sha256=file_digest(d),
            expected_component_decision_sha256=pins(d),root=r['root'])
    registry=json.loads((output/'registry.json').read_bytes())
    source=registry['sources'][0]
    manifest=json.loads((output/'files.json').read_bytes())
    clearance=json.loads((output/'clearance.json').read_bytes())
    if mutation=='extra_file':
        manifest['files'].append(copy.deepcopy(manifest['files'][0]))
        manifest=seal(manifest)
        clearance=seal(dict(clearance,file_manifest_identity=manifest['identity']))
    elif mutation=='changed_hash':
        manifest['files'][0]['sha256']='sha256:'+'0'*64
        manifest=seal(manifest)
        clearance=seal(dict(clearance,file_manifest_identity=manifest['identity']))
    elif mutation=='extra_evidence':
        extra=r['root']/'extra-evidence.txt'
        extra.write_text('No independent approval')
        clearance['evidence'].append(dict(path=extra.relative_to(r['root']).as_posix(),
                                          sha256=file_digest(extra)))
        clearance=seal(clearance)
    elif mutation=='missing_component':
        clearance['components'].pop()
        clearance=seal(clearance)
    elif mutation=='missing_marker':
        source.pop('approval_composition')
    elif mutation=='broadened_allow':
        source['allow_paths'].append('docs/unapproved.md')
    else:
        clearance['components'][0]['clearance_ref']=dict(
            path=(output/'clearance.json').relative_to(r['root']).as_posix(),
            sha256=file_digest(output/'clearance.json'))
        clearance=seal(clearance)
    with pytest.raises((SpecificationError,ValueError)):
        verify(r['root'],source,manifest,clearance)


def test_both_components_must_be_current_for_production(registered):
    r=registered
    a,b=two_batches(r)
    d=composition_decision(r,a,b)
    output=r['root']/'artifacts/composite'
    compose(d,output,expected_decision_sha256=file_digest(d),
            expected_component_decision_sha256=pins(d),root=r['root'])
    plan=output/'plan.jsonl'
    kwargs=approval_kwargs(r['root'],file_digest(plan))
    log=kwargs['governance_log']
    history=Governance(log,kwargs['expected_governance_sha256'])
    components=json.loads((output/'clearance.json').read_bytes())['components']
    events=[]
    previous=history.head
    for c in components:
        row=event(approval_identity=c['decision_identity'],
            approval_type='source_component',
            scope=dict(source_id=r['source']['source_id'],revision=r['source']['revision'],
                       component_decision_identity=c['decision_identity']),
            previous_status='NONE',new_status='ACTIVE',
            transition_at='2026-11-19T00:00:00Z',reason_ref='synthetic leaf approval',
            previous_event_identity=previous)
        events.append(row);previous=row['identity']
    with log.open('ab') as stream:
        for row in events:stream.write(canonical(row)+b'\n')
    kwargs['expected_governance_sha256']=file_digest(log)
    spec=json.loads((output/'spec.json').read_bytes())
    registry=json.loads((output/'registry.json').read_bytes())
    with ProductionBuild(r['root']/'artifacts/production',spec,registry,
            repository_root=r['root'],reference_plan=r['references'],
            prior_production=prior_inventory(r['root']),
            expected_plan_sha256=file_digest(plan),**kwargs) as build:
        build.require_current_approvals(plan)
        history=Governance(log,kwargs['expected_governance_sha256'])
        c=components[1]
        scope=dict(source_id=r['source']['source_id'],revision=r['source']['revision'],
                   component_decision_identity=c['decision_identity'])
        revoked=event(approval_identity=c['decision_identity'],
            approval_type='source_component',scope=scope,
            previous_status='ACTIVE',new_status='REVOKED',
            transition_at='2026-11-19T00:00:01Z',reason_ref='synthetic revocation',
            previous_event_identity=history.head)
        with log.open('ab') as stream:stream.write(canonical(revoked)+b'\n')
        with pytest.raises(SpecificationError,match='stale or changed'):
            build.require_current_approvals(plan)
        build.expected_governance_sha256=file_digest(log)
        with pytest.raises(SpecificationError,match='revoked'):
            build.require_current_approvals(plan)


def test_synthetic_production_accounting_notice_and_export(registered):
    r=registered
    a,b=two_batches(r)
    d=composition_decision(r,a,b)
    output=r['root']/'artifacts/composite'
    compose(d,output,expected_decision_sha256=file_digest(d),
            expected_component_decision_sha256=pins(d),root=r['root'])
    plan=output/'plan.jsonl'
    kwargs=approval_kwargs(r['root'],file_digest(plan))
    log=kwargs['governance_log']
    history=Governance(log,kwargs['expected_governance_sha256'])
    previous=history.head
    for c in json.loads((output/'clearance.json').read_bytes())['components']:
        row=event(approval_identity=c['decision_identity'],
            approval_type='source_component',
            scope=dict(source_id=r['source']['source_id'],revision=r['source']['revision'],
                       component_decision_identity=c['decision_identity']),
            previous_status='NONE',new_status='ACTIVE',
            transition_at='2026-11-19T00:00:00Z',reason_ref='synthetic test authority',
            previous_event_identity=previous)
        with log.open('ab') as stream:stream.write(canonical(row)+b'\n')
        previous=row['identity']
    kwargs['expected_governance_sha256']=file_digest(log)
    spec=json.loads((output/'spec.json').read_bytes())
    registry=json.loads((output/'registry.json').read_bytes())
    class WordTokenizer:
        identity=TOKENIZER
        vocab_size=4096
        def encode(self,text):
            return [int(hashlib.sha256(word.encode()).hexdigest()[:4],16)%4096
                    for word in text.split()]
    with ProductionBuild(r['root']/'artifacts/production',spec,registry,
            repository_root=r['root'],reference_plan=r['references'],
            prior_production=prior_inventory(r['root']),
            expected_plan_sha256=file_digest(plan),**kwargs) as build:
        report=build.run(plan,WordTokenizer())
        assert report['state']=='COMPLETE'
        assert build.db.execute("SELECT count(*) FROM items WHERE stage='ACCEPTED'").fetchone()[0]==2
        assert build.db.execute("SELECT sum(tokens) FROM totals WHERE kind='source'").fetchone()[0]>0
        exported=build.export_shards(r['root']/'artifacts/production/export')
        assert exported
        assert (r['root']/'artifacts/production/export/notice_attribution.json').is_file()
        with pytest.raises(SpecificationError):
            build.replay_for_use(r['root']/'artifacts/production/export',
                split='train',intended_use='commercial_training',decision_receipts=None)
        decisions=[fixture_receipt(row,registry['sources'][0],spec,r['root'])
                   for row in read_plan(plan)]
        receipt_file=write(r['root']/'synthetic-use-receipts.json',
                           seal(dict(schema=BUNDLE,receipts=decisions)))
        history=Governance(log,build.expected_governance_sha256)
        previous=history.head
        for item in decisions:
            row=event(approval_identity=item['identity'],
                approval_type='decision_receipt',
                scope=dict(clearance_identity=item['binding']['clearance_identity']),
                previous_status='NONE',new_status='ACTIVE',
                transition_at='2026-11-19T00:00:01Z',
                reason_ref='synthetic intended-use approval',
                previous_event_identity=previous)
            with log.open('ab') as stream:stream.write(canonical(row)+b'\n')
            previous=row['identity']
        build.expected_governance_sha256=file_digest(log)
        destination=r['root']/'artifacts/production/authorized'
        build.export_shards(destination,intended_use='commercial_training',
                            decision_receipts=receipt_file)
        assert list(build.replay_for_use(destination,split='train',
            intended_use='commercial_training',decision_receipts=receipt_file))
        notice=destination/'notice_attribution.json'
        retained_notice=notice.read_bytes()
        notice.write_bytes(retained_notice+b' ')
        with pytest.raises(SpecificationError):
            list(build.replay_for_use(destination,split='train',
                intended_use='commercial_training',decision_receipts=receipt_file))
        notice.write_bytes(retained_notice)
    prior=prior_inventory(r['root'],
        builds=(r['root']/'artifacts/production/state.sqlite',),name='prior-2.json')
    second_decision=json.loads(d.read_bytes())
    second_decision['reviewer']='second synthetic composition reviewer'
    second_decision=write(r['root']/'decisions/composition-2.json',
                          seal(second_decision))
    output2=r['root']/'artifacts/composite-2'
    compose(second_decision,output2,
            expected_decision_sha256=file_digest(second_decision),
            expected_component_decision_sha256=pins(second_decision),root=r['root'])
    plan2=output2/'plan.jsonl'
    kwargs2=approval_kwargs(r['root'],file_digest(plan2))
    log2=kwargs2['governance_log']
    previous=Governance(log2,kwargs2['expected_governance_sha256']).head
    for c in json.loads((output2/'clearance.json').read_bytes())['components']:
        row=event(approval_identity=c['decision_identity'],
            approval_type='source_component',
            scope=dict(source_id=r['source']['source_id'],revision=r['source']['revision'],
                       component_decision_identity=c['decision_identity']),
            previous_status='NONE',new_status='ACTIVE',
            transition_at='2026-11-19T00:00:00Z',reason_ref='synthetic test authority',
            previous_event_identity=previous)
        with log2.open('ab') as stream:stream.write(canonical(row)+b'\n')
        previous=row['identity']
    kwargs2['expected_governance_sha256']=file_digest(log2)
    spec2=json.loads((output2/'spec.json').read_bytes())
    registry2=json.loads((output2/'registry.json').read_bytes())
    with ProductionBuild(r['root']/'artifacts/production-2',spec2,registry2,
            repository_root=r['root'],reference_plan=r['references'],
            prior_production=prior,expected_plan_sha256=file_digest(plan2),
            **kwargs2) as second:
        report=second.run(plan2,WordTokenizer())
        assert report['state']=='COMPLETE'
        assert second.db.execute("SELECT count(*) FROM items WHERE stage='ACCEPTED'").fetchone()[0]==0
        assert second.db.execute("SELECT count(*) FROM items WHERE stage='DUPLICATE'").fetchone()[0]==2
    current=approval_kwargs(r['root'],file_digest(plan))
    log=current['governance_log']
    history=Governance(log,current['expected_governance_sha256'])
    component=json.loads((output/'clearance.json').read_bytes())['components'][0]
    revoked=event(approval_identity=component['decision_identity'],
        approval_type='source_component',
        scope=dict(source_id=r['source']['source_id'],revision=r['source']['revision'],
                   component_decision_identity=component['decision_identity']),
        previous_status='ACTIVE',new_status='REVOKED',
        transition_at='2026-11-19T00:00:02Z',
        reason_ref='synthetic post-commit revocation',
        previous_event_identity=history.head)
    with log.open('ab') as stream:stream.write(canonical(revoked)+b'\n')
    current['expected_governance_sha256']=file_digest(log)
    with ProductionBuild(r['root']/'artifacts/production',spec,registry,
            repository_root=r['root'],reference_plan=r['references'],
            prior_production=r['root']/'prior.json',
            expected_plan_sha256=file_digest(plan),**current) as historical:
        historical.verify()
        with pytest.raises(SpecificationError,match='revoked'):
            list(historical.replay_for_use(r['root']/'artifacts/production/authorized',
                split='train',intended_use='commercial_training',
                decision_receipts=receipt_file))
