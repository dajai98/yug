"""Integrity checks for the isolated, encode-only transfer evidence driver."""
import importlib.util
import json
from pathlib import Path
import pytest

@pytest.fixture
def probe():
    path = Path(__file__).resolve().parents[1] / 'artifacts/tokenizer_transfer_gate_v1/probe.py'
    spec = importlib.util.spec_from_file_location('transfer_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_transfer_detects_normalized_exact_overlap(probe):
    result = probe.overlap([{'id': 'new', 'text': 'Alpha  BETA\nGamma'}], [{'id': 'old', 'text': 'alpha beta gamma'}])
    assert result[0]['exact'] == ['old']

def test_transfer_detects_contained_excerpt(probe):
    original = ' '.join('word' + str(i) for i in range(80))
    excerpt = ' '.join(original.split()[10:50])
    result = probe.overlap([{'id': 'new', 'text': excerpt}], [{'id': 'old', 'text': original}])
    assert result[0]['exact'] == []
    assert result[0]['flagged_pairs'][0]['containment'] == 1

def test_transfer_reports_shared_boilerplate_without_whole_doc_match(probe):
    line = 'Copyright Example Authors. Permission granted to reuse this source.'
    result = probe.overlap([{'id': 'new', 'text': line + '\nnew implementation'}], [{'id': 'old', 'text': line + '\nold implementation'}])
    assert result[0]['exact'] == []
    assert result[0]['matching_lines'] == [line]

def test_transfer_rejects_modified_payload_before_encoding(probe, tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'R', tmp_path)
    (tmp_path / 'manifest.json').write_text(json.dumps({'documents_sha256': 'incorrect'}))
    (tmp_path / 'documents.jsonl').write_text('{}\n')
    with pytest.raises(AssertionError):
        probe.evaluate('must-not-exist.json')
    assert not (tmp_path / 'must-not-exist.json').exists()
