"""Synthetic local-Git tests; fixture clearances are not production authority."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from training.production_corpus import atomic_json, canonical, file_digest
from training.production_spec import SpecificationError, seal, verify_seal
from training.protected_references import make_reference
from training import source_prepare as sp
from training import source_revision_inventory as revision


def command(*args, cwd=None):
    return subprocess.run(['git', *map(str, args)], cwd=cwd,
                          capture_output=True, check=True).stdout.decode().strip()


@pytest.fixture
def case(tmp_path):
    root = sp.ROOT
    repo = tmp_path / 'upstream'
    repo.mkdir()
    command('init', repo)
    command('config', 'user.name', 'Fixture', cwd=repo)
    command('config', 'user.email', 'fixture@example.invalid', cwd=repo)
    first = {'a.py': b'print(1)\n', 'b.py': b'print(2)\n',
             'c.py': b'print(3)\n', 'oldname.py': b'print(4)\n',
             'unapproved.py': b'print(5)\n', 'vendor/copied.py': b'copied\n'}
    def write_files(files):
        for path, data in files.items():
            target = repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    write_files(first)
    command('add', '.', cwd=repo)
    command('-c', 'core.hooksPath=' + str(tmp_path / 'no-hooks'),
            'commit', '-m', 'r1', cwd=repo)
    r1 = command('rev-parse', 'HEAD', cwd=repo)
    (repo / 'c.py').unlink()
    (repo / 'oldname.py').rename(repo / 'newname.py')
    write_files({'b.py': b'print(20)\n', 'added.py': b'print(6)\n'})
    command('add', '-A', cwd=repo)
    command('-c', 'core.hooksPath=' + str(tmp_path / 'no-hooks'),
            'commit', '-m', 'r2', cwd=repo)
    r2 = command('rev-parse', 'HEAD', cwd=repo)
    source = dict(source_id='gh-example-fixture',
        locator='https://github.com/example/fixture', revision=r1,
        status='ACCEPT', category='code', subcategory='python',
        allowed_categories=['code'], target_cap_tokens=10000,
        category_caps_tokens={'code': 10000}, repository_group='gh-example-fixture',
        domain_group='gh-example-fixture', split_group='gh-example-fixture',
        split='train', allow_paths=['**'], deny_paths=['vendor/**'],
        license_id='MIT', license_evidence='synthetic test fixture only',
        origin_verified=True, training_allowed_after_clearance=True,
        file_manifest=(tmp_path / 'manifest.json').relative_to(root).as_posix(),
        clearance_record=(tmp_path / 'clearance.json').relative_to(root).as_posix())
    registry = json.loads((root / 'training/specifications/source_registry_v1.json').read_bytes())
    spec = json.loads((root / 'training/specifications/production_corpus_v1.json').read_bytes())
    registry['sources'].append(source)
    registry = seal(registry)
    spec['group_caps']['repository_group'][source['repository_group']] = 10000
    spec['group_caps']['domain_group'][source['domain_group']] = 10000
    spec = seal(dict(spec, registry_identity=registry['identity']))
    reg_path, spec_path = tmp_path / 'registry.json', tmp_path / 'spec.json'
    atomic_json(reg_path, registry)
    atomic_json(spec_path, spec)
    def fetcher(cache, candidate):
        command('--git-dir', cache, 'fetch', '--depth=2', repo, candidate['revision'])
    # R1 was cached before its later synthetic clearance and manifest existed.
    old_review_source = dict(source, file_manifest=None, clearance_record=None,
                             allow_paths=[], training_allowed_after_clearance=False)
    sp.prepare(old_review_source, spec, tmp_path / 'r1', fetcher=fetcher)
    next_source = dict(source, revision=r2)
    sp.prepare(next_source, spec, tmp_path / 'r2', fetcher=fetcher)
    old_cache = tmp_path / 'r1' / source['source_id']
    new_cache = tmp_path / 'r2' / source['source_id']
    rows = {r['path']: r for r in map(json.loads,
            (old_cache / 'inventory.jsonl').read_bytes().splitlines())}
    files = [dict(path=p, sha256=rows[p]['sha256'], bytes=rows[p]['bytes'],
                  owner='code', content_type='code') for p in ('a.py', 'b.py', 'c.py')]
    manifest = seal(dict(source_id=source['source_id'], revision=r1, files=files))
    atomic_json(tmp_path / 'manifest.json', manifest)
    marker = tmp_path / 'fixture-evidence.txt'
    marker.write_text('synthetic test evidence, not human approval', encoding='utf-8')
    clearance = seal(dict(source_id=source['source_id'], revision=r1,
                          file_manifest_identity=manifest['identity'],
                          decision='ACCEPT', attribution='synthetic test fixture',
                          evidence=[dict(path=marker.relative_to(root).as_posix(),
                                         sha256=file_digest(marker))]))
    atomic_json(tmp_path / 'clearance.json', clearance)
    refs = tmp_path / 'references.jsonl'
    protected = make_reference(work_id='unrelated', work_version='1',
        item_id='item', reference_id='unrelated-ref', role='reference',
        protected_split='test', text='Unrelated synthetic protected text.',
        content_type='prose', language='en',
        provenance={'source_id': 'unrelated-source'})
    refs.write_bytes(canonical(protected) + b'\n')
    args = dict(source_id=source['source_id'], locator=source['locator'],
        old_revision=r1, new_revision=r2, expected_registry_sha256=file_digest(reg_path),
        expected_spec_sha256=file_digest(spec_path),
        expected_manifest_sha256=file_digest(tmp_path / 'manifest.json'),
        expected_references_sha256=file_digest(refs),
        expected_old_inventory_sha256=file_digest(old_cache / 'inventory.jsonl'),
        expected_new_inventory_sha256=file_digest(new_cache / 'inventory.jsonl'),
        expected_old_receipt_sha256=file_digest(old_cache / 'complete.json'),
        expected_new_receipt_sha256=file_digest(new_cache / 'complete.json'), root=root)
    paths = (old_cache, new_cache, reg_path, spec_path,
             tmp_path / 'manifest.json', refs, tmp_path / 'comparison')
    return paths, args, tmp_path, repo


def test_actual_git_comparison_and_old_approval_scope(case):
    paths, args, _, _ = case
    report = revision.build(*paths, **args)
    verify_seal(report)
    kinds = {r['path']: r['classification'] for r in report['files']}
    assert kinds == {
        'a.py': 'UNCHANGED', 'b.py': 'CHANGED', 'c.py': 'REMOVED',
        'oldname.py': 'REMOVED', 'newname.py': 'ADDED',
        'unapproved.py': 'UNCHANGED', 'vendor/copied.py': 'UNCHANGED',
        'added.py': 'ADDED'}
    assert report['approval_scope']['previously_approved_unchanged'] == ['a.py']
    assert report['approval_scope']['previously_approved_changed'] == ['b.py']
    assert report['approval_scope']['previously_approved_removed'] == ['c.py']
    assert report['approval_scope']['newly_added'] == ['added.py', 'newname.py']
    assert report['approval_scope']['previously_unapproved_in_r1'] == [
        'oldname.py', 'unapproved.py', 'vendor/copied.py']
    assert report['approval_scope']['ineligible_in_r2'] == ['vendor/copied.py']
    assert report['review_only'] and not report['training_authorized']
    assert report['production_tokens'] == 0
    assert revision.build(*paths, **args) == report


@pytest.mark.parametrize('damage', [
    'missing_old', 'missing_new', 'bad_old_hash', 'bad_new_hash',
    'wrong_locator', 'wrong_revision', 'wrong_source', 'bad_old_policy',
    'new_policy_change', 'stale_manifest', 'changed_refs',
    'protected_conflict', 'old_scope_conflict',
    'bad_blob', 'bad_receipt', 'resealed_receipt', 'partial_inventory', 'duplicate_inventory',
    'casefold_inventory', 'unsafe_path', 'unsupported_mode',
    'duplicate_json_field', 'conflicting_output', 'conflicting_completion'])
def test_invalid_inputs_fail_closed(case, damage):
    paths, args, tmp, _ = case
    paths = list(paths)
    old, new = paths[:2]
    if damage == 'missing_old': (old / 'complete.json').unlink()
    elif damage == 'missing_new': (new / 'complete.json').unlink()
    elif damage == 'bad_old_hash': args['expected_old_inventory_sha256'] = 'sha256:' + '0'*64
    elif damage == 'bad_new_hash': args['expected_new_inventory_sha256'] = 'sha256:' + '0'*64
    elif damage == 'wrong_locator': args['locator'] = 'https://github.com/other/fixture'
    elif damage == 'wrong_revision': args['new_revision'] = 'a'*40
    elif damage == 'wrong_source': args['source_id'] = 'gh-other-fixture'
    elif damage in ('bad_old_policy', 'new_policy_change'):
        cache = old if damage == 'bad_old_policy' else new
        request = json.loads((cache / 'request.json').read_bytes())
        if damage == 'bad_old_policy': request['source']['category'] = 'nepali'
        else: request['source']['allow_paths'] = []
        request = seal(request)
        atomic_json(cache / 'request.json', request)
        receipt = json.loads((cache / 'complete.json').read_bytes())
        receipt['request_identity'] = request['identity']
        atomic_json(cache / 'complete.json', seal(receipt))
        args['expected_old_receipt_sha256' if cache == old else
             'expected_new_receipt_sha256'] = file_digest(cache / 'complete.json')
    elif damage == 'stale_manifest': (tmp / 'manifest.json').write_bytes(b'forged')
    elif damage == 'changed_refs': (tmp / 'references.jsonl').write_bytes(b'changed')
    elif damage == 'protected_conflict':
        protected = make_reference(work_id='fixture', work_version='1',
            item_id='item', reference_id='fixture-ref', role='reference',
            protected_split='test', text='Synthetic protected fixture.',
            content_type='prose', language='en',
            provenance={'source_id': args['source_id']})
        paths[5].write_bytes(canonical(protected) + b'\n')
        args['expected_references_sha256'] = file_digest(paths[5])
    elif damage == 'old_scope_conflict':
        manifest = json.loads(paths[4].read_bytes())
        manifest['files'][0]['sha256'] = 'sha256:' + '0'*64
        atomic_json(paths[4], seal(manifest))
        args['expected_manifest_sha256'] = file_digest(paths[4])
    elif damage == 'bad_blob': next((new / 'files').glob('*.blob')).write_bytes(b'broken')
    elif damage == 'bad_receipt': (new / 'complete.json').write_bytes(b'{}')
    elif damage == 'resealed_receipt':
        receipt = json.loads((new / 'complete.json').read_bytes())
        receipt['extra'] = 'not in pinned receipt'
        atomic_json(new / 'complete.json', seal(receipt))
    elif damage == 'duplicate_json_field':
        request = new / 'request.json'
        content = request.read_bytes()
        request.write_bytes(content[:-1] + b',"source":{}}')
    elif damage in ('partial_inventory', 'duplicate_inventory', 'casefold_inventory',
                    'unsafe_path', 'unsupported_mode'):
        inv = new / 'inventory.jsonl'
        rows = list(map(json.loads, inv.read_bytes().splitlines()))
        if damage == 'partial_inventory': rows.pop()
        elif damage == 'duplicate_inventory': rows.append(copy.deepcopy(rows[-1]))
        elif damage == 'casefold_inventory': rows[0]['path'] = rows[0]['path'].upper()
        elif damage == 'unsafe_path': rows[0]['path'] = '../escape.py'
        else: rows[0]['git_mode'] = '120000'
        inv.write_bytes(b''.join(canonical(r) + b'\n' for r in rows))
        done = json.loads((new / 'complete.json').read_bytes())
        done['inventory_sha256'] = file_digest(inv)
        done['counts'] = {'UNRESOLVED': len(rows)} if damage in ('partial_inventory', 'duplicate_inventory') else done['counts']
        atomic_json(new / 'complete.json', seal(done))
        args['expected_new_inventory_sha256'] = file_digest(inv)
        args['expected_new_receipt_sha256'] = file_digest(new / 'complete.json')
    else:
        revision.build(*paths, **args)
        (paths[-1] / ('comparison.json' if damage == 'conflicting_output'
                      else 'complete.json')).write_bytes(b'forged')
    with pytest.raises((SpecificationError, ValueError, OSError, subprocess.SubprocessError)):
        revision.build(*paths, **args)


def test_interrupted_publication_recovers_without_changing_report(case, monkeypatch):
    paths, args, _, _ = case
    original = revision.atomic_json
    def interrupted(path, value):
        if path.name == 'complete.json':
            raise KeyboardInterrupt()
        return original(path, value)
    monkeypatch.setattr(revision, 'atomic_json', interrupted)
    with pytest.raises(KeyboardInterrupt):
        revision.build(*paths, **args)
    before = (paths[-1] / 'comparison.json').read_bytes()
    monkeypatch.setattr(revision, 'atomic_json', original)
    revision.build(*paths, **args)
    assert (paths[-1] / 'comparison.json').read_bytes() == before
    assert (paths[-1] / 'complete.json').is_file()


def test_actual_git_symlink_mode_is_rejected(case):
    paths, args, tmp, repo = case
    oid = command('hash-object', '-w', 'a.py', cwd=repo)
    command('update-index', '--add', '--cacheinfo',
            f'120000,{oid},link.py', cwd=repo)
    command('-c', 'core.hooksPath=' + str(tmp / 'no-hooks'),
            'commit', '-m', 'unsafe-mode', cwd=repo)
    r3 = command('rev-parse', 'HEAD', cwd=repo)
    spec = json.loads(paths[3].read_bytes())
    registry = json.loads(paths[2].read_bytes())
    source = next(s for s in registry['sources'] if s['source_id'] == args['source_id'])
    source = dict(source, revision=r3)
    def fetcher(cache, candidate):
        command('--git-dir', cache, 'fetch', '--depth=2', repo, candidate['revision'])
    sp.prepare(source, spec, tmp / 'r3', fetcher=fetcher)
    paths = list(paths)
    paths[1] = tmp / 'r3' / args['source_id']
    args['new_revision'] = r3
    args['expected_new_inventory_sha256'] = file_digest(paths[1] / 'inventory.jsonl')
    args['expected_new_receipt_sha256'] = file_digest(paths[1] / 'complete.json')
    with pytest.raises(SpecificationError, match='Unsupported Git entry'):
        revision.build(*paths, **args)


def test_complete_empty_r2_classifies_every_old_file_as_removed(case):
    paths, args, tmp, repo = case
    command('rm', '--cached', '-r', '.', cwd=repo)
    command('-c', 'core.hooksPath=' + str(tmp / 'no-hooks'),
            'commit', '-m', 'empty-r2', cwd=repo)
    r3 = command('rev-parse', 'HEAD', cwd=repo)
    spec = json.loads(paths[3].read_bytes())
    registry = json.loads(paths[2].read_bytes())
    source = next(s for s in registry['sources'] if s['source_id'] == args['source_id'])
    source = dict(source, revision=r3)
    def fetcher(cache, candidate):
        command('--git-dir', cache, 'fetch', '--depth=2', repo, candidate['revision'])
    sp.prepare(source, spec, tmp / 'r3', fetcher=fetcher)
    paths = list(paths)
    paths[1] = tmp / 'r3' / args['source_id']
    args['new_revision'] = r3
    args['expected_new_inventory_sha256'] = file_digest(paths[1] / 'inventory.jsonl')
    args['expected_new_receipt_sha256'] = file_digest(paths[1] / 'complete.json')
    report = revision.build(*paths, **args)
    assert report['counts'] == {'REMOVED': 6}
    assert report['approval_scope']['previously_approved_removed'] == [
        'a.py', 'b.py', 'c.py']


def test_ambiguous_registered_locator_alias_fails(case):
    paths, args, _, _ = case
    registry = json.loads(paths[2].read_bytes())
    original = next(s for s in registry['sources'] if s['source_id'] == args['source_id'])
    alias = dict(original, source_id='gh-example-alias', status='REVIEW',
                 clearance_record=None, file_manifest=None,
                 training_allowed_after_clearance=False)
    registry['sources'].append(alias)
    registry = seal(registry)
    atomic_json(paths[2], registry)
    spec = json.loads(paths[3].read_bytes())
    atomic_json(paths[3], seal(dict(spec, registry_identity=registry['identity'])))
    args['expected_registry_sha256'] = file_digest(paths[2])
    args['expected_spec_sha256'] = file_digest(paths[3])
    with pytest.raises(SpecificationError, match='Ambiguous'):
        revision.build(*paths, **args)
