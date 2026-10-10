"""Execute the three real consumer functions with optimizer steps disabled.

Checkpoint save/load and successful evaluation are real. No corpus acquisition
or experiment training occurs in these contract tests.
"""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import training.evaluate as ev
from tests.test_real_corpus_experiment import _fixture_documents
from tests.test_training_evaluate import _tiny_tokenizer, _write_checkpoint


@pytest.mark.parametrize("consumer", ["closure", "real", "scaling"])
@pytest.mark.parametrize("failure", [None, "fingerprint", "serialization"])
def test_consumer_preserves_full_report_or_failure(tmp_path, monkeypatch, consumer, failure):
    from training.experiments import nb08_experiment as base
    from training.experiments import nb08_experiment_v2 as v2
    from training.experiments import nb08_experiment_v4 as v4
    from training.experiments import nb12_checkpoint_closure as closure
    from training.experiments import real_corpus_experiment as real
    from training.experiments import scaling_scale3_budget_extension as scaling
    from training.loop import Trainer
    from yug.native import generation
    import tempfile

    tokenizer = _tiny_tokenizer()
    documents = _fixture_documents(12)
    monkeypatch.setattr(tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(v4, "build_locked_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(v4, "SCALE_CORPORA", {"scale_3": [d.text for d in documents]})
    monkeypatch.setattr(generation, "generate", lambda model, ids, **kwargs: ids)

    # Drive checkpoint orchestration, never backward() or optimizer.step().
    steps = []
    def advance_only(self, inputs, targets):
        self.step += 1
        steps.append(self.step)
        return 2.0
    monkeypatch.setattr(Trainer, "train_step", advance_only)
    monkeypatch.setattr(Trainer, "evaluate", lambda self, batches: 2.0)
    def forbidden(*args, **kwargs):
        pytest.fail("consumer contract test attempted optimizer training")
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)

    calls = []
    original_evaluate = ev.evaluate_checkpoint
    def fail(*args, **kwargs):
        raise ev.EvaluationError("injected contract failure")
    if failure == "fingerprint":
        monkeypatch.setattr(ev, "semantic_fingerprint", fail)
    def recording_evaluate(*args, **kwargs):
        call = {"provenance": kwargs["data_provenance"]}
        calls.append(call)
        report = original_evaluate(*args, **kwargs)
        call["report"] = report.to_dict()
        if failure == "serialization":
            return SimpleNamespace(loss=report.loss, perplexity=report.perplexity, to_dict=fail)
        return report
    monkeypatch.setattr(ev, "evaluate_checkpoint", recording_evaluate)

    if consumer == "closure":
        directory, _ = _write_checkpoint(tmp_path, tokenizer)
        experiment = dict(checkpoint_dir=str(directory), initial_train_loss=3.0,
                          final_train_loss_full_eval=2.0, final_val_loss=2.0,
                          best_val_loss=2.0, uniform_baseline_loss=4.0,
                          unigram_baseline_loss=4.0)
        monkeypatch.setattr(v4, "run_single_scale", lambda *args: experiment)
        from yug.brain.adapters.yug_native import YugNativeAdapter
        from yug.runtime.core import YugCore
        async def loaded(self):
            return None
        monkeypatch.setattr(YugNativeAdapter, "_ensure_loaded", loaded)
        monkeypatch.setattr(YugCore, "respond", lambda *args: SimpleNamespace(
            success=True, content="fixture response", error=None))
        result = closure.run()
        if failure:
            assert result["verdict"] == "FAIL"
            assert "injected contract failure" in result["verdict_reason"]
            assert "evaluation" not in result
        else:
            assert result["verdict"] == "PASS"
            artifact = result["evaluation"]
            assert artifact["report"] == calls[0]["report"]
            assert artifact["loss"] == artifact["report"]["loss"]
            assert artifact["perplexity"] == artifact["report"]["perplexity"]
    else:
        if consumer == "real":
            monkeypatch.setattr(real, "MAX_STEPS", 1)
            monkeypatch.setattr(real, "EVAL_CHECKPOINTS", [1])
            monkeypatch.setattr(real, "RETAINED_CHECKPOINTS", [1])
            monkeypatch.setattr(real, "train_tokenizer_on_train_partition", lambda prep: tokenizer)
            for name, value in (("HIDDEN_SIZE", 8), ("NUM_LAYERS", 1), ("NUM_HEADS", 2), ("FFN_SIZE", 16)):
                monkeypatch.setattr(real, name, value)
            result = real.run_real_corpus_experiment(documents)
        else:
            monkeypatch.setattr(scaling, "MAX_STEPS", 1)
            monkeypatch.setattr(scaling, "EXTENDED_CURVE_CHECKPOINTS", [1])
            monkeypatch.setattr(scaling, "RETAINED_CHECKPOINTS", [1])
            monkeypatch.setattr(scaling, "GENERATION_CHECKPOINTS", [])
            for name, value in (("V2_HIDDEN_SIZE", 8), ("V2_NUM_LAYERS", 1), ("V2_NUM_HEADS", 2), ("V2_FFN_SIZE", 16)):
                monkeypatch.setattr(v2, name, value)
            result = scaling.run()
        assert steps == [1]
        artifact = result["checkpoints_saved"][1]
        if failure:
            assert artifact["reload_eval_success"] is False
            assert artifact["evaluation_report"] is None
            assert "injected contract failure" in artifact["reload_eval_detail"]
            assert "failure" in result["stop_reason"]
        else:
            assert artifact["reload_eval_success"] is True
            assert artifact["evaluation_report"] == calls[0]["report"]
            assert artifact["reload_eval_loss"] == artifact["evaluation_report"]["loss"]
            assert artifact["reload_eval_perplexity"] == artifact["evaluation_report"]["perplexity"]
    assert len(calls) == 1
    assert calls[0]["provenance"] == {
        "corpus_identity": None, "split_name": "validation",
        "split_method": "training.split.split_sentences",
        "split_seed": real.SEED if consumer == "real" else base.SEED,
        "split_fraction": real.VAL_FRACTION if consumer == "real" else base.VAL_FRACTION,
    }
