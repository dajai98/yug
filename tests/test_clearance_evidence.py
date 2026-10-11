"""Offline evidence tests: no authority inferred from permissive root licenses."""
import json
import sqlite3
from pathlib import Path

import pytest

from tests.test_source_admission import inputs
from training import clearance_evidence as evidence
from training.production_acquisition import read_plan
from training.production_corpus import canonical, file_digest
from training.production_spec import seal, SpecificationError
from training.source_admission import build as admit
from training.production_build import compile_inventory


def rewrite_rows(inputs, change):
    directory = next(inputs['cache'].iterdir())
    inventory = directory/'inventory.jsonl'
    rows = list(read_plan(inventory)); change(rows, directory)
    inventory.write_bytes(b''.join(canonical(r)+b'\n' for r in rows))
    done = json.loads((directory/'complete.json').read_bytes())
    done['inventory_sha256'] = file_digest(inventory)
    (directory/'complete.json').write_bytes(canonical(seal(done)))


def test_owned_scope_and_admission_compatibility(inputs):
    result = evidence.build(**inputs)
    assert result['counts'] == {'EVIDENCED': 1, 'EXCLUDED': 1, 'REVIEW': 1}
    rows = list(read_plan(inputs['output']/'evidence.jsonl'))
    assert rows[0]['scope']['clearance_identity']
    assert rows[0]['license_id'] and rows[0]['license_evidence']
    assert rows[1]['scope'] is None  # Even an SPDX MIT label is not a grant.
    assert all(r['training_allowed'] is False for r in rows)
    out = inputs['root']/'admission'
    result = admit(**dict(inputs, output=out), evidence=inputs['output'])
    assert result['counts'] == {'CLEARED': 1, 'EXCLUDED': 1, 'REVIEW': 1}
    spec = json.loads((out/'spec.json').read_bytes())
    registry = json.loads((out/'registry.json').read_bytes())
    _, receipt = compile_inventory(spec, registry, inputs['root'], out/'runner', out/'plan.jsonl')
    assert receipt['scheduled'] == 2


def test_deterministic_replay_and_no_cache_mutation(inputs):
    before = {p: file_digest(p) for p in inputs['cache'].rglob('*') if p.is_file()}
    first = evidence.build(**inputs)
    assert first == evidence.build(**inputs, resume=True)
    assert first == evidence.build(**dict(inputs, output=inputs['root']/'other'))
    assert before == {p: file_digest(p) for p in inputs['cache'].rglob('*') if p.is_file()}


@pytest.mark.parametrize('damage', ['revision', 'source', 'hash', 'inventory', 'receipt', 'git_object', 'missing'])
def test_corruption_fails_closed(inputs, damage):
    directory = next(inputs['cache'].iterdir())
    if damage in ('revision', 'source', 'git_object', 'missing'):
        def change(rows, directory):
            if damage == 'revision': rows[0]['revision'] = 'a'*40
            elif damage == 'source': rows[0]['source_id'] = 'wrong'
            elif damage == 'git_object': rows[0]['git_object'] = '0'*40
            else: rows[0]['local_blob'] = None
        rewrite_rows(inputs, change)
    else:
        path = {'hash': directory/'files/0', 'inventory': directory/'inventory.jsonl',
                'receipt': directory/'complete.json'}[damage]
        path.write_bytes(b'corrupt')
    with pytest.raises((ValueError, SpecificationError)):
        evidence.build(**inputs)
    assert not (inputs['output']/'complete.json').exists()


@pytest.mark.parametrize('path,payload', [
    ('vendor/copied.py', b'copy'), ('third_party/copied.py', b'copy'),
    ('src/image.png', b'PNG'), ('src/file.py', b'\x00binary'),
    ('src/generated.py', b'# Code generated automatically\n'),
])
def test_exclusions(inputs, path, payload):
    def change(rows, directory):
        blob = directory/rows[1]['local_blob']; blob.write_bytes(payload)
        rows[1].update(path=path, sha256=file_digest(blob), bytes=len(payload))
    # Avoid collision with the fixture vendor path.
    if path == 'vendor/copied.py': path = 'vendor/other.py'
    rewrite_rows(inputs, change)
    assert evidence.build(**inputs)['counts']['EXCLUDED'] == 2


def test_ambiguous_license_remains_review(inputs):
    def change(rows, directory):
        blob = directory/rows[1]['local_blob']
        blob.write_bytes(b'# License: MIT or other terms; third-party ownership unknown\n')
        rows[1].update(sha256=file_digest(blob), bytes=blob.stat().st_size)
    rewrite_rows(inputs, change)
    evidence.build(**inputs)
    row = list(read_plan(inputs['output']/'evidence.jsonl'))[1]
    assert row['status'] == 'REVIEW' and row['license_id'] is None


def test_limit_resume_and_partial_admission_rejection(inputs):
    assert not evidence.build(**inputs, limit=1)['complete']
    with pytest.raises(SpecificationError, match='Partial evidence'):
        admit(**dict(inputs, output=inputs['root']/'admission'), evidence=inputs['output'])
    result = evidence.build(**inputs, resume=True)
    assert result['complete'] and result['reviewed_files'] == 3
    with pytest.raises(SpecificationError, match='Limit cannot shrink'):
        evidence.build(**inputs, resume=True, limit=1)


@pytest.mark.parametrize('stage', ['decision', 'publication'])
def test_interruption_resumes(inputs, monkeypatch, stage):
    original = evidence._decision if stage == 'decision' else evidence.atomic_json
    if stage == 'decision':
        calls = 0
        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2: raise KeyboardInterrupt()
            return original(*args, **kwargs)
        monkeypatch.setattr(evidence, '_decision', interrupted)
    else:
        evidence.build(**inputs, limit=1)
        def interrupted(path, value):
            if Path(path).name == 'complete.json': raise KeyboardInterrupt()
            return original(path, value)
        monkeypatch.setattr(evidence, 'atomic_json', interrupted)
    with pytest.raises(KeyboardInterrupt):
        evidence.build(**inputs, resume=stage == 'publication')
    monkeypatch.setattr(evidence, '_decision' if stage == 'decision' else 'atomic_json', original)
    assert evidence.build(**inputs, resume=True)['complete']


def test_dry_run_no_writes_or_network(inputs, monkeypatch):
    import socket
    import subprocess
    def forbidden(*args, **kwargs): raise AssertionError('Network/process forbidden')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(subprocess, 'run', forbidden)
    before = {p: file_digest(p) for p in inputs['root'].rglob('*') if p.is_file()}
    result = evidence.build(**inputs, dry_run=True)
    assert result['complete'] and not inputs['output'].exists()
    assert before == {p: file_digest(p) for p in inputs['root'].rglob('*') if p.is_file()}


@pytest.mark.parametrize('damage', ['evidence', 'state', 'license'])
def test_resume_integrity(inputs, damage):
    evidence.build(**inputs)
    if damage == 'evidence': (inputs['output']/'evidence.jsonl').write_bytes(b'corrupt')
    elif damage == 'state':
        with sqlite3.connect(inputs['output']/'state.sqlite') as db:
            db.execute("UPDATE decisions SET payload='{}' WHERE position=0")
    else:
        row = next(read_plan(inputs['output']/'evidence.jsonl'))
        (inputs['root']/row['license_evidence'][0]['path']).write_bytes(b'changed license')
    with pytest.raises((ValueError, SpecificationError)):
        evidence.build(**inputs, resume=True)


def test_evidence_cannot_promote_ungranted_file(inputs):
    evidence.build(**inputs)
    path = inputs['output']/'evidence.jsonl'
    rows = list(read_plan(path)); rows[1] = seal(dict(rows[1], status='EVIDENCED'))
    path.write_bytes(b''.join(canonical(r)+b'\n' for r in rows))
    receipt = inputs['output']/'complete.json'
    value = json.loads(receipt.read_bytes()); value['evidence_sha256'] = file_digest(path)
    receipt.write_bytes(canonical(seal(value)))
    with pytest.raises(SpecificationError, match='classification mismatch'):
        admit(**dict(inputs, output=inputs['root']/'admission'), evidence=inputs['output'])


def test_holdout_source_exclusion(inputs):
    records = list(read_plan(inputs['approved_plan']))
    inputs['approved_plan'].write_bytes(canonical(records[1])+b'\n')
    inputs['references'].write_bytes(canonical(dict(provenance={'source_id': records[0]['source_id']}))+b'\n')
    assert evidence.build(**inputs)['counts'] == {'EVIDENCED': 0, 'EXCLUDED': 3, 'REVIEW': 0}


def test_corrupt_checkpoint_positions(inputs):
    evidence.build(**inputs)
    with sqlite3.connect(inputs['output']/'state.sqlite') as db:
        db.execute('UPDATE decisions SET position=100 WHERE position=1')
    with pytest.raises(SpecificationError, match='checkpoint positions'):
        evidence.build(**inputs, resume=True)


def test_cli_rejects_output_in_cache_without_mutation(inputs, monkeypatch):
    import sys
    before = {p: file_digest(p) for p in inputs['cache'].rglob('*') if p.is_file()}
    monkeypatch.setattr(evidence, 'ROOT', inputs['root'])
    monkeypatch.setattr(sys, 'argv', ['clearance_evidence', '--cache', str(inputs['cache']),
                                    '--output', str(inputs['cache']/'bad')])
    assert evidence.main() == 2
    assert before == {p: file_digest(p) for p in inputs['cache'].rglob('*') if p.is_file()}


def test_cli_flags(inputs, monkeypatch, capsys):
    import sys
    monkeypatch.setattr(evidence, 'ROOT', inputs['root'])
    original = evidence.build
    def configured(**kwargs):
        return original(**dict(inputs, **kwargs))
    monkeypatch.setattr(evidence, 'build', configured)
    base = ['clearance_evidence', '--cache', str(inputs['cache']), '--output', str(inputs['output'])]
    monkeypatch.setattr(sys, 'argv', base+['--dry-run', '--limit', '1'])
    assert evidence.main() == 0 and not inputs['output'].exists()
    assert json.loads(capsys.readouterr().out)['reviewed_files'] == 1
    monkeypatch.setattr(sys, 'argv', base+['--limit', '1'])
    assert evidence.main() == 0
    monkeypatch.setattr(sys, 'argv', base+['--resume'])
    assert evidence.main() == 0
