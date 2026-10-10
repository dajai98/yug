"""Safety and accounting checks for diagnostic-only preprocessing tooling."""
import pytest

from training.experiments import preprocessing_scalability_probe as probe


def test_surrogate_is_bounded_nested_and_explicitly_not_training_corpus():
    small = probe.local_documents(2048)
    large = probe.local_documents(4096)
    assert small == large[:len(small)]
    assert sum(len(d.text) for d in large) == 4096
    assert all(len(d.text) <= 1024 for d in large)
    assert all(d.license == "DIAGNOSTIC_FIXTURE_ONLY" for d in large)
    for size in (-1, 0, probe.MAX_CHARS + 1):
        with pytest.raises(ValueError):
            probe.local_documents(size)


def test_dataset_diagnostic_rejects_unbounded_allocation():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="cap"):
        probe.worker("dataset", probe.MAX_TOKENS + 1, 1024, None)


def test_real_bounded_fit_and_encode_preserve_counts_and_tokenizer(tmp_path):
    pytest.importorskip("torch")
    fit = probe.worker("fit", 4096, 384, None)
    assert fit["prepared"]["train_documents"] + fit["prepared"]["validation_documents"] == fit["prepared"]["accepted"]
    assert fit["stages"]["fit"]["work_units"]["characters"] == fit["prepared"]["train_characters"]
    import json
    path = tmp_path / "diagnostic_tokenizer.json"
    path.write_text(json.dumps(fit["tokenizer"]), encoding="utf-8")
    first = probe.worker("encode", 4096, 384, path)
    second = probe.worker("encode", 4096, 384, path)
    assert first["tokenizer_identity"] == second["tokenizer_identity"] == fit["tokenizer"]["identity"]
    assert first["round_trips_ok"] and second["round_trips_ok"]
    for partition in ("train", "validation"):
        a = first[f"{partition}_counts"]
        b = second[f"{partition}_counts"]
        assert a["tokens"] == b["tokens"]
        assert a["sequences"] == b["sequences"] == a["tokens"] // 129
        assert a["correctness"] == b["correctness"]
        assert a["tensor_storage_bytes"] == a["sequences"] * 128 * 2 * 8


def test_batch_oracle_detects_shift_corruption():
    torch = pytest.importorskip("torch")
    from training.dataset import iter_batches, make_sequences
    ids = list(range(300))
    sequences = make_sequences(ids, 128)
    batches = iter_batches(sequences, 8)
    assert probe.batch_checks(ids, sequences, batches)["discarded_tail_tokens"] == 42
    batches[0][1][0, 0] = torch.tensor(99)
    with pytest.raises(AssertionError):
        probe.batch_checks(ids, sequences, batches)
