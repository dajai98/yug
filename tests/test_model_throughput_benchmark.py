"""Fast tests for training/experiments/model_throughput_benchmark.py.

None of these run the full 30+100-step 5M-model benchmark; the real-size
configs are only instantiated (parameter checks). Timing runs use a tiny model.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from training.experiments import model_throughput_benchmark as mtb  # noqa: E402
from yug.tokenizer.bpe import ByteLevelBpeTokenizer  # noqa: E402

TINY = dict(
    vocab_sizes=(300, 400), hidden_size=16, num_layers=1, num_heads=2, ffn_size=32,
    sequence_length=8, context_length=8, batch_size=2, warmup_steps=1, measured_steps=3,
    eval_batches=2, threads=1, seed=7,
)


@pytest.fixture(scope="module")
def tiny_result():
    return mtb.run_benchmark(**TINY)


def _primary_config(v):
    return mtb.build_model_config(v, mtb.placeholder_tokenizer_identity(v),
                                  context_length=mtb.PRIMARY_CONTEXT_LENGTH, **mtb.PRIMARY_ARCH)


def test_both_primary_configs_valid():
    for v in mtb.PRIMARY_VOCAB_SIZES:
        cfg = _primary_config(v)
        assert cfg.hidden_size // cfg.num_heads == 32
        assert cfg.normalization == "rmsnorm" and cfg.positional_encoding == "rope"
        assert cfg.activation == "silu" and cfg.bias is False and cfg.tie_embeddings is True
    assert mtb.PRIMARY_SEQUENCE_LENGTH <= mtb.PRIMARY_CONTEXT_LENGTH


def test_actual_parameter_counts_and_body_identical():
    bodies = set()
    for v in mtb.PRIMARY_VOCAB_SIZES:
        model = mtb.YugTransformer(_primary_config(v))
        actual = model.count_parameters()
        assert actual == mtb.PRIMARY_EXPECTED_PARAMETERS[v]
        assert actual == mtb.expected_parameter_count(v, 256, 6, 768)
        vocab = v * 256
        bodies.add(actual - vocab)
    assert bodies == {mtb.PRIMARY_EXPECTED_BODY_PARAMETERS}


def test_vocab_and_body_shares():
    p4, p8 = (mtb.PRIMARY_EXPECTED_PARAMETERS[v] for v in (4096, 8192))
    assert 4096 * 256 == 1_048_576 and 8192 * 256 == 2_097_152
    assert round(100 * 1_048_576 / p4, 2) == 21.04
    assert round(100 * 2_097_152 / p8, 2) == 34.76
    assert p8 - p4 == 1_048_576


def test_result_schema(tiny_result):
    r = tiny_result
    for key in ("experiment_name", "environment", "threading", "corpus", "configs", "comparison"):
        assert key in r
    for key in ("python_version", "torch_version", "platform", "torch_num_threads_default"):
        assert key in r["environment"]
    for c in r["configs"].values():
        for key in ("parameters", "batch", "training", "evaluation", "memory", "checkpoint", "tokenizer_identity"):
            assert key in c
        t = c["training"]
        for key in ("total_measured_seconds", "mean_step_seconds", "median_step_seconds", "min_step_seconds",
                    "max_step_seconds", "stdev_step_seconds", "training_tokens_per_second", "bytes_per_second",
                    "initial_loss_sanity_only", "final_loss_sanity_only"):
            assert key in t
        assert t["bytes_per_second"] == "UNKNOWN"
        assert c["batch"]["tokens_per_step"] == 2 * 8
        p = c["parameters"]
        assert p["vocab_dependent"] + p["body"] == p["actual"]
        assert p["vocab_share"] + p["body_share"] == pytest.approx(1.0)
    assert "mean_step_time_ratio" in r["comparison"]
    assert "tokens_per_second_ratio" in r["comparison"]


def test_flags_and_no_quality_verdict(tiny_result):
    assert tiny_result["not_a_milestone"] is True
    assert tiny_result["nb13_assigned"] is False
    assert tiny_result["model_quality_verdict"] == "NOT_GENERATED"

    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from keys(v)

    assert not any(re.search(r"winner|better|best|verdict_", str(k)) for k in keys(tiny_result))


def test_warmup_excluded_from_measured_steps():
    class FakeTrainer:
        calls = 0

        def train_step(self, x, y):
            self.calls += 1
            return float(self.calls)

    times = iter([0.0, 1.0, 1.0, 3.0, 3.0, 6.0])
    trainer = FakeTrainer()
    out = mtb.time_training_steps(trainer, [(None, None)], warmup_steps=4, measured_steps=3,
                                  timer=lambda: next(times))
    assert trainer.calls == 7
    assert out["step_seconds"] == [1.0, 2.0, 3.0]
    assert out["initial_loss"] == 1.0 and out["final_loss"] == 7.0
    with pytest.raises(StopIteration):
        next(times)  # exactly 2 timer calls per measured step, none for warm-up


def test_real_train_step_updates_weights():
    cfg = mtb.build_model_config(300, mtb.placeholder_tokenizer_identity(300), hidden_size=16, num_layers=1,
                                 num_heads=2, ffn_size=32, context_length=8)
    mtb.seed_everything(0)
    model = mtb.YugTransformer(cfg)
    before = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    tc = mtb.TrainingConfig(seed=0, learning_rate=1e-2, batch_size=2, sequence_length=8, max_steps=2,
                            eval_interval=100, checkpoint_interval=100)
    trainer = mtb.Trainer(model, opt, tc, cfg, cfg.tokenizer_identity)
    batches = mtb.build_batches(mtb.synthetic_token_ids(300, 2 * 9 * 2, 0), 8, 2, shuffle_seed=None)
    mtb.time_training_steps(trainer, batches, warmup_steps=1, measured_steps=1)
    assert trainer.step == 2
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_deterministic_small_run():
    a = mtb.run_benchmark(**TINY)
    b = mtb.run_benchmark(**TINY)
    for v in ("300", "400"):
        ta, tb = a["configs"][v]["training"], b["configs"][v]["training"]
        assert ta["initial_loss_sanity_only"] == pytest.approx(tb["initial_loss_sanity_only"], rel=1e-6)
        assert ta["final_loss_sanity_only"] == pytest.approx(tb["final_loss_sanity_only"], rel=1e-6)


def test_checkpoint_roundtrip(tiny_result):
    for c in tiny_result["configs"].values():
        assert c["checkpoint"]["save_success"] is True
        assert c["checkpoint"]["reload_success"] is True, c["checkpoint"]["error"]


def test_peak_memory_is_int_or_none():
    peak = mtb.peak_process_memory_bytes()
    assert peak is None or (isinstance(peak, int) and peak > 0)


def _tiny_tokenizer(tmp_path):
    tok = ByteLevelBpeTokenizer.train(["hello world "] * 300, vocab_size=270)
    path = tmp_path / "tok.json"
    tok.save(path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world " * 300, encoding="utf-8")
    return tok, path, corpus


def test_supplied_tokenizer_success_and_blocks(tmp_path):
    tok, path, corpus = _tiny_tokenizer(tmp_path)
    v = tok.vocab_size
    kwargs = {**TINY, "vocab_sizes": (v,), "tokenizer_paths": {v: path}, "corpus_dir": corpus}
    ok = mtb.run_benchmark(**kwargs, expected_identities={v: tok.identity})
    cfg = ok["configs"][str(v)]
    assert cfg["data"]["tokenizer_identity_verified"] is True
    assert cfg["tokenizer_identity"] == tok.identity

    with pytest.raises(mtb.BenchmarkBlockedError):  # identity mismatch
        mtb.run_benchmark(**kwargs, expected_identities={v: "sha256:" + "0" * 64})
    with pytest.raises(mtb.BenchmarkBlockedError):  # vocab mismatch
        mtb.run_benchmark(**{**kwargs, "vocab_sizes": (v + 10,), "tokenizer_paths": {v + 10: path}})
    with pytest.raises(mtb.BenchmarkBlockedError):  # real mode needs both tokenizer and corpus
        mtb.run_benchmark(**{**TINY, "vocab_sizes": (v,), "tokenizer_paths": {v: path}})


def test_parameter_mismatch_is_blocking(monkeypatch):
    monkeypatch.setattr(mtb, "expected_parameter_count", lambda *a, **k: 1)
    with pytest.raises(mtb.BenchmarkBlockedError):
        mtb.run_benchmark(**TINY)


def test_production_yug_does_not_import_training():
    yug_dir = Path(__file__).resolve().parents[1] / "yug"
    if not yug_dir.is_dir():
        pytest.skip("yug/ not found relative to tests/")
    pattern = re.compile(r"^\s*(from|import)\s+training(\.|\s|$)", re.MULTILINE)
    offenders = [str(p) for p in yug_dir.rglob("*.py") if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []