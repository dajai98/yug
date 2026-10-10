import hashlib
import json
from dataclasses import replace

import pytest

from training.corpus_prep import AcceptedDocument
from training.experiments import tokenizer_fit_sample_gate as gate


def document(number, family='code'):
    text = f'Original fixture document {number}: ' + 'abcdef ' * 20
    return AcceptedDocument(str(number), text, 'source', 'fixture', 'MIT', 'en', family,
                            hashlib.sha256(text.encode()).hexdigest())


def record(doc, split):
    return dict(gate.asdict(doc), partition=split, provenance={'url': 'fixture-only'})


def test_selection_is_order_independent_whole_documents_and_bounded():
    docs = [document(f'{family}-{n}', family) for family in gate.FAMILIES for n in range(20)]
    selected = gate.select_train(docs, cap=6000)
    assert selected == gate.select_train(list(reversed(docs)), cap=6000)
    for family in gate.FAMILIES:
        subset = [d for d in selected if d.primary_domain == family]
        assert subset
        assert sum(len(d.text.encode()) for d in subset) <= 6000 * gate.WEIGHTS[family] // 61
        assert all(d in docs for d in subset)


@pytest.mark.parametrize('kind', ['id', 'hash', 'partition', 'text', 'provenance', 'duplicate'])
def test_contamination_and_corruption_rejected(kind):
    train = [record(document(1), 'train')]
    val = [record(document(2), 'validation')]
    if kind == 'id':
        val[0]['document_id'] = train[0]['document_id']
    elif kind == 'hash':
        val[0] = dict(train[0], partition='validation', document_id='different-id')
    elif kind == 'partition':
        val[0]['partition'] = 'train'
    elif kind == 'text':
        train[0]['text'] += 'corruption'
    elif kind == 'provenance':
        train[0]['provenance'] = {}
    else:
        train *= 2
    with pytest.raises(ValueError):
        gate.validate_rows(train, val)


def test_manifest_binds_order_and_checksums(tmp_path):
    train = [record(document(1), 'train'), record(document(2), 'train')]
    val = [record(document(3), 'validation')]
    gate.write_rows(tmp_path / 'train.jsonl', train)
    gate.write_rows(tmp_path / 'validation.jsonl', val)
    manifest = dict(files={name: gate.sha(tmp_path / name) for name in ('train.jsonl','validation.jsonl')},
                    train_identity=gate.identity(train), validation_identity=gate.identity(val))
    manifest['identity'] = gate.identity(manifest)
    gate.write_json(tmp_path / 'manifest.json', manifest)
    assert gate.load_sample(tmp_path)[1:] == (train, val)
    assert gate.identity(train) != gate.identity(list(reversed(train)))
    gate.write_rows(tmp_path / 'train.jsonl', list(reversed(train)))
    with pytest.raises(ValueError, match='checksum'):
        gate.load_sample(tmp_path)


def test_metrics_use_real_encoded_counts_and_roundtrip():
    from yug.tokenizer.bpe import ByteLevelBpeTokenizer
    rows = [record(document(family, family), 'train') for family in gate.FAMILIES]
    tok = ByteLevelBpeTokenizer.train([r['text'] for r in rows], 320)
    measured = gate.measure(tok, rows)
    assert measured == gate.measure(tok, rows)
    assert measured['totals']['tokens'] == sum(len(tok.encode(r['text'])) for r in rows)
    assert measured['totals']['utf8_bytes'] == sum(len(r['text'].encode()) for r in rows)
    assert measured['groups']['code']['identifier_count'] > 0


def test_failed_worker_is_a_failed_command(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(gate, 'load_sample', lambda root: ({'identity': 'fixture'}, [], []))
    calls = []
    def failed(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=42)
    monkeypatch.setattr(gate.subprocess, 'run', failed)
    with pytest.raises(RuntimeError, match='Worker failed resource gate'):
        gate.run_comparison(tmp_path)
    assert len(calls) == 1
    assert json.loads((tmp_path / 'execution.json').read_text())[0]['returncode'] == 42
