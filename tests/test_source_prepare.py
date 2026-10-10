"""Offline source preparation tests using local, temporary Git repositories."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import pytest
from training import source_prepare as sp
from training.production_spec import seal,SpecificationError


def command(*args,cwd=None):
    return subprocess.run(['git',*args],cwd=cwd,stdin=subprocess.DEVNULL,capture_output=True,check=True).stdout.decode().strip()


@pytest.fixture
def fixture(tmp_path):
    repo=tmp_path/'upstream';repo.mkdir();command('init',str(repo))
    command('config','user.name','Fixture',cwd=repo);command('config','user.email','fixture@example.invalid',cwd=repo)
    for path,data in {'src/main.py':b'print("natural code")\n','vendor/copied.py':b'copied source','src/gen.py':b'# DO NOT EDIT\n','image.png':b'\x89PNG\0stuff','src/binary.txt':b'abc\0def','README.md':b'Example text\n'}.items():
        p=repo/path;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(data)
    command('add','.',cwd=repo);command('-c','core.hooksPath='+str(tmp_path/'no-hooks'),'commit','-m','fixture',cwd=repo)
    rev=command('rev-parse','HEAD',cwd=repo)
    source=dict(source_id='fixture',locator='https://github.com/example/fixture',revision=rev,status='ACCEPT',license_id='MIT',license_evidence='recorded historical evidence',category='code',subcategory='python',allow_paths=['src/**'],deny_paths=['**/vendor/**'])
    spec=seal(json.loads((sp.ROOT/'training/specifications/production_corpus_v1.json').read_bytes()))
    calls=[]
    def fetcher(cache,s):
        calls.append(s['source_id']);command('--git-dir',str(cache),'fetch','--depth=1',str(repo),s['revision'])
    return source,spec,fetcher,calls


@pytest.mark.parametrize('revision',['main','v1.0','abc123','A'*40,'',None])
def test_immutable_revision_enforced(fixture,revision):
    source,*_=fixture;source['revision']=revision
    assert sp.classify(source)=='BLOCKED_NO_REVISION'


def test_resume_inventory_hash_and_screening(tmp_path,fixture,monkeypatch):
    source,spec,fetcher,calls=fixture
    first=sp.prepare(source,spec,tmp_path/'out',fetcher=fetcher)
    rows=[json.loads(x) for x in (tmp_path/'out/fixture/inventory.jsonl').read_bytes().splitlines()]
    bypath={r['path']:r for r in rows}
    assert bypath['src/main.py']['sha256']=='sha256:'+hashlib.sha256(b'print("natural code")\n').hexdigest()
    assert bypath['src/main.py']['screening']=='UNRESOLVED'
    assert all(not r['training_allowed'] for r in rows)
    for path in ('vendor/copied.py','src/gen.py','image.png','src/binary.txt'):assert bypath[path]['screening']=='EXCLUDED'
    monkeypatch.setattr(sp,'inventory',lambda *a:pytest.fail('must reuse completed inventory'))
    again=sp.prepare(source,spec,tmp_path/'out',fetcher=fetcher,resume=True)
    sp.verify_seal(first);sp.verify_seal(again)
    assert len(calls)==1 and again['reused'] and again['inventory_sha256']==first['inventory_sha256']


def test_deterministic_fresh_inventory(tmp_path,fixture):
    source,spec,fetcher,calls=fixture
    a=sp.prepare(source,spec,tmp_path/'a',fetcher=fetcher)
    b=sp.prepare(source,spec,tmp_path/'b',fetcher=fetcher)
    assert a['inventory_sha256']==b['inventory_sha256']


@pytest.mark.parametrize('damage',['revision','blob','inventory','git_object','config'])
def test_corruption_and_wrong_revision_fail_closed(tmp_path,fixture,damage):
    source,spec,fetcher,calls=fixture;out=tmp_path/'out'
    sp.prepare(source,spec,out,fetcher=fetcher);base=out/'fixture'
    if damage=='revision':(base/'git/HEAD').write_text('0'*40+'\n')
    elif damage=='blob':next((base/'files').glob('*.blob')).write_bytes(b'broken')
    elif damage=='inventory':(base/'inventory.jsonl').write_bytes(b'broken')
    elif damage=='config':(base/'git/config').write_text('broken')
    else:
        objects=[p for p in (base/'git/objects').rglob('*') if p.is_file() and p.suffix!='.idx']
        objects[0].chmod(0o600)
        objects[0].write_bytes(b'broken')
    with pytest.raises((SpecificationError,subprocess.SubprocessError)):sp.prepare(source,spec,out,fetcher=fetcher,resume=True)
    assert len(calls)==1


def test_interrupted_inventory_resume_no_refetch(tmp_path,fixture,monkeypatch):
    source,spec,fetcher,calls=fixture;original=sp.inventory
    def interrupted(*args):
        original(*args);raise KeyboardInterrupt()
    monkeypatch.setattr(sp,'inventory',interrupted)
    with pytest.raises(KeyboardInterrupt):sp.prepare(source,spec,tmp_path/'out',fetcher=fetcher)
    monkeypatch.setattr(sp,'inventory',original)
    sp.prepare(source,spec,tmp_path/'out',fetcher=fetcher,resume=True)
    assert len(calls)==1


def test_dry_run_and_failed_source_isolation(tmp_path,fixture):
    source,spec,fetcher,calls=fixture
    other=dict(source,source_id='other');reg=seal(dict(sources=[source,other]))
    sp.run(reg,spec,tmp_path/'dry',dry_run=True,fetcher=lambda *a:pytest.fail('dry run fetch'))
    assert not list((tmp_path/'dry').rglob('git'))
    attempts=[]
    def intermittent(cache,s):
        attempts.append(s['source_id'])
        if s['source_id']=='fixture':raise OSError('network unavailable')
        fetcher(cache,s)
    rows=sp.run(reg,spec,tmp_path/'out',retries=0,fetcher=intermittent)
    assert [r['status'] for r in rows]==['FAILED','CACHED']
    assert attempts==['fixture','other']
    # Missing commit from interrupted fetch can be fetched on explicit resume.
    rows=sp.run(reg,spec,tmp_path/'out',resume=True,retries=0,fetcher=fetcher)
    assert all(r['status']=='CACHED' for r in rows)


def test_source_limit_and_blocked_licenses(tmp_path,fixture):
    source,spec,fetcher,calls=fixture
    unknown=dict(source,source_id='unknown',license_id='unknown')
    assert sp.classify(unknown)=='BLOCKED_LICENSE'
    rows=sp.run(seal(dict(sources=[source,dict(source,source_id='other'),unknown])),spec,tmp_path/'out',limit=1,fetcher=fetcher)
    assert len(calls)==1 and rows[-1]['classification']=='BLOCKED_LICENSE'


def test_interrupted_fetch_reuses_downloaded_commit(tmp_path,fixture):
    source,spec,fetcher,calls=fixture
    def interrupted(repo,s):
        fetcher(repo,s)
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):sp.prepare(source,spec,tmp_path/'out',fetcher=interrupted)
    sp.prepare(source,spec,tmp_path/'out',resume=True,fetcher=fetcher)
    assert len(calls)==1


def test_fetch_retries_are_bounded(tmp_path,fixture,monkeypatch):
    source,spec,fetcher,calls=fixture;attempts=[]
    def fail(*args):attempts.append(1);raise OSError('offline')
    monkeypatch.setattr(sp.time,'sleep',lambda _:None)
    with pytest.raises(OSError):sp.prepare(source,spec,tmp_path/'out',retries=2,fetcher=fail)
    assert len(attempts)==3
