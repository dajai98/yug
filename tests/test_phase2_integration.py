"""Final bounded Phase 2 corpus integration checks; synthetic authority only."""
import json
import sqlite3
from contextlib import closing

import pytest

from tests.test_corpus_gate import builder, bundle
from tests.test_production_acquisition import BASE, Tokenizer, approval_kwargs, prior_inventory
from tests.test_production_cross_batch import setup
from training.production_acquisition import ProductionBuild
from training.production_corpus import file_digest
from training.production_spec import SpecificationError
from yug.tokenizer.bpe import ByteLevelBpeTokenizer


def _complete(root, *, interrupted):
    root.mkdir()
    texts = [BASE, ' '.join('planet' + str(i) for i in range(88))]
    spec, registry, references, plans = setup(root, texts=texts,
                                               splits=['train', 'validation'])
    plan = root / 'combined-plan.jsonl'
    plan.write_bytes(b''.join(path.read_bytes() for path in plans))
    decisions = bundle(root, (spec, registry, references, plans),
                       ['commercial_training', 'evaluation'])
    pin = file_digest(plan)
    tokenizer = ByteLevelBpeTokenizer.load(
        'training/resources/phase1/bpe_fitter_optimization_gate_v1/comparison/run-4096-0/tokenizer.json')
    with ProductionBuild(root / 'build', spec, registry, repository_root=root,
                         reference_plan=references, prior_production=prior_inventory(root),
                         expected_plan_sha256=pin, **approval_kwargs(root, pin)) as build:
        if interrupted:
            assert build.run(plan, tokenizer, max_items=1,
                intended_use='commercial_training', decision_receipts=decisions)['state'] == 'PARTIAL'
        progress = build.run(plan, tokenizer, intended_use='commercial_training',
                             decision_receipts=decisions)
        assert progress['stages'] == {'ACCEPTED': 2}
        output = root / 'build' / 'export'
        shards = build.export_shards(output, intended_use='commercial_training',
                                     decision_receipts=decisions)
        assert len(list(build.replay_for_use(output, split='train',
            intended_use='commercial_training', decision_receipts=decisions))) == 303
        assert len(list(build.replay_for_use(output, split='validation',
            intended_use='commercial_training', decision_receipts=decisions))) == 471
        with closing(sqlite3.connect(root / 'corpus-ledger.sqlite')) as db:
            assert db.execute('SELECT count(*) FROM commits').fetchone()[0] == 1
            assert db.execute('SELECT count(*) FROM item_evidence').fetchone()[0] == 2
        files = ['prepared.json', 'admission.jsonl', 'notice_attribution.json',
                 'shards/manifest.json', 'shards/train-000000.bin',
                 'shards/validation-000000.bin']
        return (shards['manifest_identity'],
                {name: (output / name).read_bytes() for name in files})


def test_real_tokenizer_production_resume_and_authorized_replay_match(tmp_path):
    uninterrupted = _complete(tmp_path / 'full', interrupted=False)
    resumed = _complete(tmp_path / 'resumed', interrupted=True)
    assert uninterrupted == resumed


def test_final_integrity_failure_releases_uncommitted_admission(tmp_path):
    data = setup(tmp_path)
    decisions = bundle(tmp_path, data)
    with builder(tmp_path, data) as build:
        original_verify = build.verify
        calls = 0
        raw = None
        saved = None

        def corrupt_at_final_check():
            nonlocal calls, raw, saved
            calls += 1
            if calls == 2:
                raw = next((tmp_path / 'build' / 'raw').iterdir())
                saved = raw.read_bytes()
                raw.write_bytes(b'corrupt')
            return original_verify()

        build.verify = corrupt_at_final_check
        with pytest.raises(SpecificationError, match='raw payload corrupt'):
            build.run(data[3][0], Tokenizer(), intended_use='commercial_training',
                      decision_receipts=decisions)
        build.verify = original_verify
        assert build.get('state') == 'FAILED'
        assert build.db.execute("SELECT count(*) FROM items WHERE stage='ACCEPTED'").fetchone()[0] == 0
        assert build.db.execute('SELECT coalesce(sum(tokens),0) FROM totals').fetchone()[0] == 0
        with closing(sqlite3.connect(tmp_path / 'corpus-ledger.sqlite')) as db:
            assert db.execute("SELECT count(*) FROM builds WHERE state='ACTIVE'").fetchone()[0] == 0
            assert db.execute('SELECT count(*) FROM usage').fetchone()[0] == 0
            assert db.execute('SELECT count(*) FROM commits').fetchone()[0] == 0
        with pytest.raises(SpecificationError):
            build.export_shards(tmp_path / 'build' / 'failed-export',
                                intended_use='commercial_training', decision_receipts=decisions)
        raw.write_bytes(saved)
        class NoEncode(Tokenizer):
            def encode(self, _):
                raise AssertionError('retry re-tokenized cached content')
        def no_fetch(_):
            raise AssertionError('retry refetched cached content')
        assert build.run(data[3][0], NoEncode(), fetch=no_fetch,
            intended_use='commercial_training', decision_receipts=decisions)['accepted_clean_train_tokens'] == 619
        with closing(sqlite3.connect(tmp_path / 'corpus-ledger.sqlite')) as db:
            assert db.execute('SELECT count(*) FROM commits').fetchone()[0] == 1
            assert db.execute("SELECT tokens FROM totals WHERE kind='source'").fetchone()[0] == 619
