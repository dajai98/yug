"""Source expansion never guesses licensing or bypasses admission."""
import json
import socket
from pathlib import Path
import pytest

from training import source_expansion as module
from training.production_corpus import canonical, file_digest
from training.production_spec import seal, SpecificationError
from training.source_prepare import ROOT, run


@pytest.fixture
def inputs(tmp_path):
    registry = json.loads((ROOT/'training/specifications/source_registry_v1.json').read_bytes())
    source = next(s for s in registry['sources'] if s['source_id'] == 'gh-psf-requests')
    registry = seal(dict(registry, sources=[source]))
    r = tmp_path/'registry.json'; r.write_bytes(canonical(registry))
    return dict(registry=r, spec=ROOT/'training/specifications/production_corpus_v1.json', output=tmp_path/'out')


def change(inputs, **updates):
    r = json.loads(inputs['registry'].read_bytes()); r['sources'][0].update(updates)
    inputs['registry'].write_bytes(canonical(seal(r)))


def test_replay_and_preparation_compatibility(inputs, monkeypatch):
    def forbidden(*a, **k): raise AssertionError('No network')
    monkeypatch.setattr(socket, 'socket', forbidden)
    before = file_digest(inputs['registry'])
    first = module.build(**inputs)
    assert first['candidate_sources'] == 1 and first['rejected_sources'] == 0
    assert first == module.build(**inputs, resume=True)
    assert first == module.build(**dict(inputs, output=inputs['output'].with_name('other')))
    registry = json.loads((inputs['output']/'candidates.json').read_bytes())
    assert registry['sources'] == json.loads(inputs['registry'].read_bytes())['sources']
    spec = json.loads(inputs['spec'].read_bytes())
    assert run(registry, spec, inputs['output']/'prepare', dry_run=True)[0]['status'] == 'CACHE_READY'
    assert before == file_digest(inputs['registry'])


@pytest.mark.parametrize('updates', [
    {'license_id': None}, {'license_id': 'UNKNOWN'}, {'license_id': 'MIT OR unknown'},
    {'license_id': 'GPL-3.0'}, {'license_evidence': None}, {'revision': 'main'},
    {'locator': 'https://example.com/source'}, {'status': 'REVIEW'},
    {'status': 'REJECT'}, {'category': 'invented'}, {'split_group': None},
])
def test_uncertainty_rejected(inputs, updates):
    change(inputs, **updates)
    result = module.build(**inputs)
    assert result['candidate_sources'] == 0 and result['rejected_sources'] == 1


def test_duplicate_project_not_expanded(inputs):
    r = json.loads(inputs['registry'].read_bytes())
    r['sources'].append(dict(r['sources'][0], source_id='duplicate'))
    inputs['registry'].write_bytes(canonical(seal(r)))
    result = module.build(**inputs)
    assert result['candidate_sources'] == result['rejected_sources'] == 1


@pytest.mark.parametrize('target', ['inputs', 'output'])
def test_corruption_fails_closed(inputs, target):
    module.build(**inputs)
    path = inputs['registry'] if target == 'inputs' else inputs['output']/'candidates.json'
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError): module.build(**inputs, resume=True)


def test_interruption_resume(inputs, monkeypatch):
    original = module.atomic_json
    def interrupt(path, value):
        original(path, value)
        if Path(path).name == 'candidates.json': raise KeyboardInterrupt()
    monkeypatch.setattr(module, 'atomic_json', interrupt)
    with pytest.raises(KeyboardInterrupt): module.build(**inputs)
    monkeypatch.setattr(module, 'atomic_json', original)
    assert module.build(**inputs, resume=True)['candidate_sources'] == 1


def test_changed_input_resume_rejected(inputs):
    module.build(**inputs)
    change(inputs, revision='a'*40)
    with pytest.raises(SpecificationError, match='inputs changed'): module.build(**inputs, resume=True)
