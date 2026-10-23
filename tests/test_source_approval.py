"""Explicit file/license/package proof; never blanket root-license approval."""
import base64
import hashlib
import json
from pathlib import Path
import pytest
from tests.test_source_admission import inputs
from training import source_approval as approval
from training.clearance_evidence import build as evidence_build
from training.production_corpus import canonical, file_digest
from training.production_spec import seal, SpecificationError
from training.production_acquisition import read_plan
from training.source_admission import build as admit


@pytest.fixture
def setup(inputs):
    if approval.tomllib is None: pytest.skip('This package evidence profile requires stdlib tomllib (Python 3.11+)')
    root=inputs['root']; directory=next(inputs['cache'].iterdir())
    registry=json.loads(inputs['root_registry'].read_bytes()); spec=json.loads(inputs['root_spec'].read_bytes())
    source=dict(registry['sources'][0],source_id='new-package',name='example/package',locator='https://github.com/example/package',
                repository_group='new-package',domain_group='new-package',split_group='new-package',split='unassigned',
                allow_paths=[],file_manifest=None,clearance_record=None,origin_verified=False,target_cap_tokens=0,
                category='code',subcategory='python',allowed_categories=['code'],category_caps_tokens={'code':0},license_id='MIT')
    license=('The MIT License\nCopyright (c) Test Author\n'+approval.MIT_TERMS+'\n').encode()
    metadata=b'''[project]
name = "package"
license = "MIT"
license-files = ["LICENSE"]
authors = [{name = "Test Author"}]
[project.urls]
Repository = "https://github.com/example/package"
[tool.hatch.build.targets.wheel]
packages = ["src/package"]
'''
    body=dict(license={'spdx_id':'MIT'},path='LICENSE',content=base64.b64encode(license).decode())
    ep=root/'license-evidence.json'; ep.write_bytes(canonical(seal(dict(url='https://api.github.com/repos/example/package/license?ref='+source['revision'],body=body))))
    source['license_evidence']=dict(path=ep.relative_to(root).as_posix(),sha256=file_digest(ep),upstream_path='LICENSE',text_sha256='sha256:'+hashlib.sha256(license).hexdigest())
    files={'LICENSE':license,'pyproject.toml':metadata,'src/package/main.py':b'# SPDX-License-Identifier: MIT\ndef add(a, b):\n    return a + b\n',
           'src/package/unknown.py':b'def multiply(a, b):\n    return a * b\n','vendor/copied.py':b'copied'}
    def rebuild(files):
        rows=[]
        for i,(path,data) in enumerate(sorted(files.items())):
            blob=directory/'files'/str(i); blob.write_bytes(data)
            rows.append(dict(source_id=source['source_id'],revision=source['revision'],locator=source['locator'],path=path,bytes=len(data),
                             sha256=file_digest(blob),git_object=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest(),
                             git_mode='100644',local_blob=blob.relative_to(directory).as_posix(),screening='UNRESOLVED',category='code',subcategory='python'))
        inv=directory/'inventory.jsonl'; inv.write_bytes(b''.join(canonical(r)+b'\n' for r in rows))
        request=seal(dict(source=source,filters=spec['filters'])); (directory/'request.json').write_bytes(canonical(request))
        (directory/'complete.json').write_bytes(canonical(seal(dict(request_identity=request['identity'],revision=source['revision'],inventory_sha256=file_digest(inv),counts={'UNRESOLVED':len(rows)}))))
        catalog=root/'catalog.json'; catalog.write_bytes(canonical(seal(dict(registry,sources=[source]))))
        return catalog
    catalog=rebuild(files)
    args=dict(cache=inputs['cache'],catalog=catalog,evidence=root/'evidence',registry=inputs['approved_registry'],spec=inputs['approved_spec'],
              plan=inputs['approved_plan'],references=inputs['references'],output=root/'approval',root=root,validation_groups=0)
    def classify():
        evidence_build(**dict(inputs,root_registry=catalog,output=args['evidence']))
    return args,files,rebuild,classify


def test_exact_grants_caps_splits_and_admission(setup):
    args,_,_,classify=setup; classify()
    result=approval.build(**args)
    assert result['counts']=={'APPROVED':1,'EXCLUDED':1,'REVIEW':3}
    assert result['new_plan_files']==1 and result['accepted_tokens_added']==0
    assert result['assignments']=={'new-package':'train'}
    assert 0 < result['source_caps']['new-package'] <= 100000
    records=list(read_plan(args['output']/'plan.jsonl')); granted=[r for r in records if r['source_id']=='new-package']
    assert len(granted)==1 and granted[0]['allow_rule']=='src/package/main.py'
    admitted=admit(cache=args['cache'],root_registry=args['catalog'],root_spec=args['spec'],
                   approved_registry=args['output']/'registry.json',approved_spec=args['output']/'spec.json',approved_plan=args['output']/'plan.jsonl',
                   references=args['references'],output=args['root']/'admission',root=args['root'])
    assert admitted['counts']['CLEARED']==1 and admitted['production_plan_files']==3


@pytest.mark.parametrize('case',['no_spdx','conflicting','copied','nested_license','package_url','package_license','whole_repo','syntax'])
def test_ambiguity_stays_review(setup,case):
    args,files,rebuild,classify=setup
    if case=='no_spdx': files['src/package/main.py']=b'def add(a,b):\n    return a+b\n'
    elif case=='conflicting': files['src/package/main.py']+=b'# SPDX-License-Identifier: GPL-3.0\n'
    elif case=='copied': files['src/package/main.py']+=b'# Adapted from another project\n'
    elif case=='nested_license': files['src/package/LICENSE']=b'Other terms'
    elif case=='package_url': files['pyproject.toml']=files['pyproject.toml'].replace(b'example/package',b'example/other')
    elif case=='package_license': files['pyproject.toml']=files['pyproject.toml'].replace(b'"MIT"',b'"BSD-3-Clause"')
    elif case=='whole_repo': files['pyproject.toml']=files['pyproject.toml'].replace(b'"src/package"',b'"."')
    else: files['src/package/main.py']+=b'not python syntax !!!'
    rebuild(files); classify(); result=approval.build(**args)
    assert result['counts']['APPROVED']==0 and result['new_plan_files']==0


def test_resume_deterministic_cache_read_only(setup):
    args,_,_,classify=setup; classify()
    before={p:file_digest(p) for p in args['cache'].rglob('*') if p.is_file()}
    first=approval.build(**args)
    (args['output']/'diagnostic.log').write_text('Incidental test output is not grant evidence.')
    assert first==approval.build(**args,resume=True)
    assert before=={p:file_digest(p) for p in args['cache'].rglob('*') if p.is_file()}


def test_explicit_validation_assignment(setup):
    args,_,_,classify=setup; classify()
    result=approval.build(**dict(args,validation_groups=1,cap=25))
    assert result['assignments']=={'new-package':'validation'} and result['source_caps']=={'new-package':25}
    assert all(r['split']=='validation' for r in read_plan(args['output']/'plan.jsonl') if r['source_id']=='new-package')


@pytest.mark.parametrize('damage',['blob','license','output'])
def test_corruption_fails_closed(setup,damage):
    args,_,_,classify=setup; classify()
    if damage=='output': approval.build(**args); path=args['output']/'plan.jsonl'
    elif damage=='license': path=args['root']/'license-evidence.json'
    else: path=next((next(args['cache'].iterdir())/'files').iterdir())
    path.write_bytes(b'corrupt')
    with pytest.raises((ValueError,SpecificationError)): approval.build(**args,resume=damage=='output')


def test_interrupted_publication_resumes(setup,monkeypatch):
    args,_,_,classify=setup; classify(); original=approval.atomic_json
    def interrupted(path,value):
        original(path,value)
        if Path(path).name.endswith('-files.json'): raise KeyboardInterrupt()
    monkeypatch.setattr(approval,'atomic_json',interrupted)
    with pytest.raises(KeyboardInterrupt): approval.build(**args)
    monkeypatch.setattr(approval,'atomic_json',original)
    assert approval.build(**args,resume=True)['new_plan_files']==1


def test_holdout_source_never_materialized(setup):
    args,_,_,classify=setup
    args['references'].write_bytes(canonical(dict(reference_id='held',text='Retained benchmark text about algorithms.',
                                                  content_type='prose',language='en',provenance={'source_id':'new-package'}))+b'\n')
    classify(); result=approval.build(**args)
    assert result['counts']=={'APPROVED':0,'EXCLUDED':5,'REVIEW':0}


def test_policy_resume_cannot_change_split_or_cap(setup):
    args,_,_,classify=setup; classify(); approval.build(**args)
    with pytest.raises(SpecificationError,match='Resume request changed'):
        approval.build(**dict(args,cap=500),resume=True)


def test_missing_toml_parser_stays_review(setup,monkeypatch):
    args,_,_,classify=setup; classify()
    monkeypatch.setattr(approval,'tomllib',None)
    result=approval.build(**args)
    assert result['counts']=={'APPROVED':0,'EXCLUDED':1,'REVIEW':4}


@pytest.mark.parametrize('value',[b'{text = "MIT"}',b'{file = "LICENSE"}'])
def test_legacy_package_forms_still_require_file_evidence(setup,value):
    args,files,rebuild,classify=setup
    files['pyproject.toml']=files['pyproject.toml'].replace(b'license = "MIT"',b'license = '+value).replace(b'license-files = ["LICENSE"]\n',b'')
    rebuild(files); classify()
    assert approval.build(**args)['counts']=={'APPROVED':1,'EXCLUDED':1,'REVIEW':3}


@pytest.mark.parametrize('case', ['valid','package_present','no_attribution','wrong_owner','conflict','copied','nested',
                                'generated','outside','project_conflict','canonical','binary','vendor','body_copyright'])
def test_mit_file_profile(setup, case):
    args,files,rebuild,classify=setup
    if case != 'package_present':
        files['pyproject.toml']=files['pyproject.toml'].split(b'[tool.hatch.build.targets.wheel]')[0].replace(b'license-files = ["LICENSE"]\n',b'')
    content=b'# SPDX-FileCopyrightText: 2023 Test Author <test@example.com>\n#\n# SPDX-License-Identifier: MIT\n\ndef add(a,b):\n    return a+b\n'
    if case=='no_attribution': content=content.replace(b'Test Author',b'')
    if case=='wrong_owner': content=content.replace(b'Test Author',b'Other Owner')
    if case=='conflict': content+=b'# SPDX-License-Identifier: BSD-3-Clause\n'
    if case=='copied': content+=b'# adapted from another project\n'
    if case=='nested': files['src/package/LICENSE']=b'Other terms'
    if case=='generated': content+=b'# automatically generated; do not edit\n'
    if case=='project_conflict': files['pyproject.toml']=files['pyproject.toml'].replace(b'"MIT"',b'"BSD-3-Clause"')
    if case=='canonical': files['pyproject.toml']=files['pyproject.toml'].replace(b'example/package',b'example/other')
    if case=='binary': content+=b'\0'
    if case=='body_copyright': content+=b'# Copyright (c) Another Person\n'
    del files['src/package/main.py']
    path={'outside':'other/main.py','vendor':'src/package/vendor/main.py'}.get(case,'src/package/main.py')
    files[path]=content
    rebuild(files);classify();result=approval.build(**args)
    assert result['new_plan_files']==(1 if case in ('valid','package_present') else 0)
    if case=='valid':
        assert result==approval.build(**args,resume=True)
        admitted=admit(cache=args['cache'],root_registry=args['catalog'],root_spec=args['spec'],
                       approved_registry=args['output']/'registry.json',approved_spec=args['output']/'spec.json',
                       approved_plan=args['output']/'plan.jsonl',references=args['references'],output=args['root']/'admission',root=args['root'])
        assert admitted['counts']['CLEARED']==1


def test_mit_profile_does_not_expand_other_licenses():
    assert approval.mit_file_scope({'license_id':'Apache-2.0'}, {}, None, None)[0] is None


def test_mit_profile_rejects_extra_root_terms(setup):
    args,files,rebuild,classify=setup
    files['LICENSE']+=b'Additional restrictions\n'
    rebuild(files);classify()
    with pytest.raises(SpecificationError): approval.build(**args)


@pytest.mark.parametrize('preamble', [
    'MIT License\nCopyright (c) 2020 Test Author\n',
    'Copyright (c) 2020 Test Author\nMIT License\n',
    'Copyright (c) 2020 Test Author\n',
    'Copyright (c) 2020 Test Author\nThe MIT License\nCopyright (c) 2021 Second Author\n',
])
def test_mit_preamble_presentation(preamble):
    result=approval.mit_attribution(preamble+approval.MIT_TERMS)
    assert result and 'Copyright (c) 2020 Test Author' in result
    assert approval.mit_attribution((preamble+approval.MIT_TERMS).replace('\n','\r\n'))==result


@pytest.mark.parametrize('text', [
    'MIT License\n'+approval.MIT_TERMS,
    'Copyright (c) \n'+approval.MIT_TERMS,
    'MIT License\nMIT License\nCopyright (c) Someone\n'+approval.MIT_TERMS,
    'Copyright (c) Someone\nNoncommercial only\n'+approval.MIT_TERMS,
    'Copyright (c) Someone\n'+approval.MIT_TERMS+'\nAdditional restrictions',
    'Copyright (c) Someone\n'+approval.MIT_TERMS.replace('sell copies','use copies'),
    'Copyright (c) Someone\n'+approval.MIT_TERMS.split('THE SOFTWARE')[0],
])
def test_mit_preamble_does_not_relax_terms(text):
    assert approval.mit_attribution(text) is None
