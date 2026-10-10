"""Frozen Phase C contracts, using untrained checkpoints and independent oracles."""
import hashlib
import json
import math
import shutil
import struct
from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import training.evaluate as ev
from training.errors import TrainingError
from training.loop import Trainer
from training.loss import evaluation_nll_count
from tests.test_training_evaluate import _tiny_tokenizer, _write_checkpoint


def reference_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def semantic_fields(report):
    return {key: getattr(report, key) for key in (
        "evaluation_semantics", "model_weights_identity", "model_identity",
        "tokenizer_identity", "eval_data_identity", "data_provenance", "sequence_length",
    )}


@pytest.fixture
def checkpoint(tmp_path):
    tokenizer = _tiny_tokenizer()
    directory, config = _write_checkpoint(tmp_path, tokenizer)
    return directory, tokenizer, config


def evaluate_fixture(checkpoint, **kwargs):
    directory, tokenizer, _ = checkpoint
    options = dict(sequence_length=4, batch_size=2)
    options.update(kwargs)
    return ev.evaluate_checkpoint(directory, tokenizer, ["abcdefghijklmnopq"], **options)


def test_partition_invariance_and_independent_fingerprint_rebuilding(checkpoint):
    reports = [evaluate_fixture(checkpoint, batch_size=n) for n in (1, 2, 3, 10)]
    assert len({r.evaluation_fingerprint for r in reports}) == 1
    assert len({r.execution_fingerprint for r in reports}) == 4
    assert {(r.encoded_token_count, r.eval_sequence_count, r.eval_token_count)
            for r in reports} == {(17, 3, 12)}
    for report in reports:
        assert report.loss == pytest.approx(reports[0].loss, abs=1e-6)
        assert report.loss == report.total_nll / report.eval_token_count
        assert report.perplexity == math.exp(report.loss)
        assert report.evaluation_fingerprint == reference_hash({
            "schema": "yug-evaluation-identity-v1", **semantic_fields(report),
        })
        assert report.execution_fingerprint == reference_hash({
            "schema": "yug-evaluation-execution-v1",
            "evaluation_fingerprint": report.evaluation_fingerprint,
            "batch_size": report.batch_size, "execution_config": report.execution_config,
        })
        assert json.loads(report.to_json()) == report.to_dict()
        assert set(report.data_provenance.values()) == {None}


def test_effective_execution_setting_changes_only_execution(checkpoint):
    original = torch.get_num_threads()
    first = evaluate_fixture(checkpoint)
    try:
        torch.set_num_threads(1 if original != 1 else 2)
        second = evaluate_fixture(checkpoint)
        assert second.execution_config["torch_num_threads"] == torch.get_num_threads()
        assert first.evaluation_fingerprint == second.evaluation_fingerprint
        assert first.execution_fingerprint != second.execution_fingerprint
    finally:
        torch.set_num_threads(original)


def test_path_step_and_bookkeeping_seed_are_report_only(checkpoint, tmp_path):
    first = evaluate_fixture(checkpoint)
    directory, tokenizer, config = checkpoint
    copy = tmp_path / "relocated"
    shutil.copytree(directory, copy)
    metadata_path = copy / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["step"] += 10
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    second = evaluate_fixture((copy, tokenizer, config), seed=985)
    assert first.checkpoint_dir != second.checkpoint_dir
    assert first.checkpoint_step != second.checkpoint_step
    assert first.seed != second.seed
    assert first.evaluation_fingerprint == second.evaluation_fingerprint
    assert first.execution_fingerprint == second.execution_fingerprint
    assert second.data_provenance["split_seed"] is None


def test_changed_loaded_weight_and_model_config_are_semantic(checkpoint):
    first = evaluate_fixture(checkpoint)
    directory, _, config = checkpoint
    state = torch.load(directory / "state.pt", map_location="cpu")
    next(iter(state["model"].values())).view(-1)[0] += 0.5
    torch.save(state, directory / "state.pt")
    second = evaluate_fixture(checkpoint)
    assert second.model_identity == first.model_identity
    assert second.model_weights_identity != first.model_weights_identity
    assert second.evaluation_fingerprint != first.evaluation_fingerprint
    replace(config, dropout=0.2).save(directory / "model_config.json")
    third = evaluate_fixture(checkpoint)
    assert third.model_weights_identity == second.model_weights_identity
    assert third.model_identity != second.model_identity
    assert third.evaluation_fingerprint != second.evaluation_fingerprint


@pytest.mark.parametrize("key,value", [
    ("tokenizer_identity", "sha256:" + "1" * 64),
    ("sequence_length", 3),
    ("evaluation_semantics", "yug-causal-lm-eval-v2"),
    ("data_provenance", {"split_seed": 12}),
    ("data_provenance", {"split_name": "test"}),
    ("data_provenance", {"split_method": "declared-method"}),
    ("data_provenance", {"split_fraction": 0.3}),
    ("data_provenance", {"corpus_identity": "sha256:" + "2" * 64}),
])
def test_each_semantic_field_changes_both_identities(checkpoint, key, value):
    report = evaluate_fixture(checkpoint)
    fields = semantic_fields(report)
    fields[key] = value
    changed = ev.semantic_fingerprint(**fields)
    assert changed != report.evaluation_fingerprint
    assert ev.execution_fingerprint(changed, report.batch_size, report.execution_config) != report.execution_fingerprint


def test_exact_ordered_data_changes_semantics(checkpoint):
    report = evaluate_fixture(checkpoint)
    documents = [["ab", "cd"], ["cd", "ab"], ["abcd"], ["ab", "", "cd"],
                 ["ab ", "cd"], ["ab", "cD"], ["ab", "\u00e9"], ["ab", "e\u0301"]]
    identities = [ev.evaluation_data_identity(docs) for docs in documents]
    assert len(set(identities)) == len(documents)
    assert len({ev.semantic_fingerprint(**{**semantic_fields(report), "eval_data_identity": i})
                for i in identities}) == len(documents)


def test_metrics_and_counts_are_not_identity_inputs(checkpoint):
    report = evaluate_fixture(checkpoint)
    # Deliberately synthetic accounting fixture: identity builders cannot police
    # a caller's source data. They must not hash derived counts or metrics.
    alternate = replace(report, encoded_token_count=22, eval_sequence_count=4,
                        eval_token_count=16, model_parameter_count=report.model_parameter_count + 1,
                        total_nll=32.0, loss=2.0, perplexity=math.exp(2.0))
    assert alternate.evaluation_fingerprint == report.evaluation_fingerprint
    assert alternate.execution_fingerprint == report.execution_fingerprint
    for fields in ({"eval_token_count": 0}, {"eval_sequence_count": 999},
                   {"loss": report.loss + 1}, {"total_nll": float("nan")},
                   {"perplexity": float("inf")}, {"evaluation_fingerprint": "sha256:" + "0" * 64}):
        with pytest.raises(ev.EvaluationError):
            replace(report, **fields)
    report.execution_config["torch_num_threads"] += 1
    with pytest.raises(ev.EvaluationError, match="fingerprint mismatch"):
        report.to_dict()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_weight_value_identity_dtype_zero_layout_and_order(dtype, monkeypatch):
    values = torch.tensor([[0.0, 1.5], [-2.0, 4.0]], dtype=dtype)
    expected = ev.model_weights_identity({"z": values.double(), "a": torch.tensor([-0.0])})
    monkeypatch.setattr(ev, "_WEIGHT_CHUNK_VALUES", 1)
    assert expected == ev.model_weights_identity({"a": torch.tensor([0.0], dtype=dtype), "z": values})
    noncontiguous = values.T.contiguous().T
    assert not noncontiguous.is_contiguous()
    assert expected == ev.model_weights_identity({"z": noncontiguous, "a": torch.tensor([0.0])})
    assert ev.model_weights_identity({"x": values}) != ev.model_weights_identity({"x": values.flatten()})
    assert ev.model_weights_identity({"x": values}) != ev.model_weights_identity({"y": values})


def test_weight_rounding_changes_identity_and_binary64_oracle():
    precise = torch.tensor([0.1], dtype=torch.float64)
    assert ev.model_weights_identity({"x": precise}) != ev.model_weights_identity({"x": precise.float()})
    tensor_digest = "sha256:" + hashlib.sha256(struct.pack("<ddd", 0.0, 1.5, -2.0)).hexdigest()
    assert ev.model_weights_identity({"x": torch.tensor([-0.0, 1.5, -2.0])}) == reference_hash({
        "schema": "yug-eval-weight-values-v1",
        "tensors": [{"name": "x", "shape": [3], "values_sha256": tensor_digest}],
    })


@pytest.mark.parametrize("bad", [torch.tensor([1]), torch.tensor([True]),
    torch.tensor([1j]), torch.tensor([float("nan")]), torch.tensor([float("inf")]),
    torch.eye(2).to_sparse(), torch.empty(1, device="meta")])
def test_unsupported_or_nonfinite_weights_fail(bad):
    with pytest.raises(ev.EvaluationError):
        ev.model_weights_identity({"weight": bad})


def test_canonical_serialization_fixed_bytes_and_digest():
    payload = {"schema": "yug-eval-data-v1", "documents": ["a", "", "\u00e9"]}
    literal = b'{"documents":["a","","\\u00e9"],"schema":"yug-eval-data-v1"}'
    assert ev.canonical_json_bytes(payload) == literal
    assert ev.evaluation_data_identity(payload["documents"]) == (
        "sha256:ff24db84792c57b90a55f42035df22822ed960d161c42db70e0f43c3b5b97173"
    )
    assert ev.canonical_json_bytes({"b": None, "a": [2, 1]}) == b'{"a":[2,1],"b":null}'
    for bad in (float("nan"), float("inf"), object()):
        with pytest.raises(ev.EvaluationError):
            ev.canonical_json_bytes({"bad": bad})


def test_nll_ignored_targets_and_fp64_independent_oracle():
    logits = torch.tensor([[[1., 2., 3.], [float("nan"), 0., 0.]],
                           [[-2., 1., 4.], [2., 0., 1.]]])
    targets = torch.tensor([[2, -100], [0, 1]])
    valid_logits = logits[targets != -100].double()
    valid_targets = targets[targets != -100]
    oracle = (valid_logits.logsumexp(-1) - valid_logits.gather(1, valid_targets[:, None]).squeeze(1)).sum().item()
    total, count = evaluation_nll_count(logits, targets)
    assert count == 3
    assert total == pytest.approx(oracle, abs=1e-6)
    assert evaluation_nll_count(logits, torch.full_like(targets, -100)) == (0.0, 0)
    with pytest.raises((IndexError, RuntimeError)):
        evaluation_nll_count(logits, torch.full_like(targets, -1))
    with pytest.raises(TrainingError, match="Non-finite"):
        evaluation_nll_count(logits, torch.zeros_like(targets))


@pytest.mark.parametrize("training", [True, False])
def test_trainer_global_nll_with_ignored_batch_and_mode_restoration(training):
    class LogitsModel(torch.nn.Module):
        def forward(self, logits):
            assert not self.training
            assert not torch.is_grad_enabled()
            return logits
    model = LogitsModel().train(training)
    trainer = Trainer(model, None, SimpleNamespace(device="cpu"), None, "unused")
    logits = torch.tensor([[[1., 0.]], [[0., 2.]], [[3., 0.]]])
    targets = torch.tensor([[0], [1], [1]])
    batches = [(logits[:2], targets[:2]), (logits[:1], torch.tensor([[-100]])), (logits[2:], targets[2:])]
    expected = sum(math.log1p(math.exp(v)) for v in (-1, -2, 3)) / 3
    assert trainer.evaluate(batches) == pytest.approx(expected, abs=1e-7)
    assert model.training is training
    for invalid in ([], [(logits, torch.full_like(targets, -100))]):
        with pytest.raises(TrainingError, match="zero valid"):
            trainer.evaluate(invalid)
        assert model.training is training
    with pytest.raises((IndexError, RuntimeError)):
        trainer.evaluate([(logits, torch.full_like(targets, 99))])
    assert model.training is training


@pytest.mark.parametrize("failure", ["weights", "semantic", "execution", "settings", "autocast"])
def test_fingerprint_or_execution_failure_never_constructs_report(checkpoint, monkeypatch, failure):
    def fail(*args, **kwargs):
        raise ev.EvaluationError("injected identity failure")
    def no_report(*args, **kwargs):
        pytest.fail("partial report constructed after failure")
    monkeypatch.setattr(ev, "EvaluationReport", no_report)
    if failure == "settings":
        monkeypatch.setattr(torch, "get_num_threads", fail)
    elif failure == "autocast":
        monkeypatch.setattr(torch, "is_autocast_enabled", lambda *args: True)
    else:
        monkeypatch.setattr(ev, {"weights": "model_weights_identity", "semantic": "semantic_fingerprint",
                               "execution": "execution_fingerprint"}[failure], fail)
    with pytest.raises(ev.EvaluationError):
        evaluate_fixture(checkpoint)


def test_only_absent_execution_api_becomes_null(checkpoint, monkeypatch):
    monkeypatch.delattr(torch, "is_deterministic_algorithms_warn_only_enabled")
    assert evaluate_fixture(checkpoint).execution_config["deterministic_warn_only"] is None


def test_report_rejects_removed_null_provenance(checkpoint):
    report = evaluate_fixture(checkpoint)
    del report.data_provenance["split_seed"]
    with pytest.raises(ev.EvaluationError, match="explicit null"):
        report.to_dict()


def test_execution_change_during_evaluation_fails(checkpoint, monkeypatch):
    original = ev.capture_execution_config
    calls = []
    def changing_settings(model):
        settings = original(model)
        if calls:
            settings["torch_num_threads"] += 1
        calls.append(settings)
        return settings
    monkeypatch.setattr(ev, "capture_execution_config", changing_settings)
    with pytest.raises(ev.EvaluationError, match="settings changed"):
        evaluate_fixture(checkpoint)


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), float("-inf"), 1000.0])
def test_nonfinite_or_overflowing_perplexity_fails(loss):
    with pytest.raises(ev.EvaluationError):
        ev.perplexity_from_loss(loss)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("failure", ["ignored", "forward"])
def test_checkpoint_evaluator_restores_model_mode_on_failure(checkpoint, monkeypatch, training, failure):
    from yug.native.model import YugTransformer
    model = YugTransformer(checkpoint[2]).train(training)
    monkeypatch.setattr(ev, "YugTransformer", lambda config: model)
    if failure == "ignored":
        original = ev.iter_batches
        monkeypatch.setattr(ev, "iter_batches", lambda sequences, size: [
            (x, torch.full_like(y, -100)) for x, y in original(sequences, size)])
        error = ev.EvaluationError
    else:
        def forward(inputs):
            assert not model.training
            assert not torch.is_grad_enabled()
            raise RuntimeError("injected forward failure")
        monkeypatch.setattr(model, "forward", forward)
        error = RuntimeError
    with pytest.raises(error):
        evaluate_fixture(checkpoint)
    assert model.training is training
