"""Value-level legacy oracle, demand bounds, repeatability and Trainer parity."""
import copy

import pytest
import torch

from training.dataset import (DocumentSequences, encode_corpus, iter_batches,
                              iter_encoded_tokens, iter_lazy_batches,
                              iter_sequences, make_sequences)
from training.errors import DatasetError, TrainingError
from yug.tokenizer.bpe import ByteLevelBpeTokenizer


@pytest.fixture
def tokenizer():
    return ByteLevelBpeTokenizer.train(["hello world\nhello code", "नेपाल"], 270)


@pytest.mark.parametrize("length,batch_size", [(1, 1), (4, 3), (8, 2), (19, 7)])
def test_values_against_independent_legacy_oracle(tokenizer, length, batch_size):
    texts = ["hello world!", "", "नेपाल code\n" * 9, "tail"]
    identity = tokenizer.identity
    ids = [i for text in texts for i in tokenizer.encode(text)]
    windows = [ids[i:i + length + 1]
               for i in range(0, len(ids) - length, length + 1)]
    assert encode_corpus(tokenizer, texts) == ids
    assert list(iter_encoded_tokens(tokenizer, iter(texts))) == ids
    assert make_sequences(ids, length) == windows
    assert list(iter_sequences(iter(ids), length)) == windows
    source = DocumentSequences(tokenizer, lambda: iter(texts), length)
    for _ in range(2):
        assert list(source) == windows
        actual = list(iter_lazy_batches(source, batch_size))
        eager = iter_batches(windows, batch_size)
        assert len(actual) == (len(windows) + batch_size - 1) // batch_size
        for index, ((x, y), (ex, ey)) in enumerate(zip(actual, eager)):
            group = windows[index * batch_size:(index + 1) * batch_size]
            assert x.tolist() == [s[:-1] for s in group]
            assert y.tolist() == [s[1:] for s in group]
            assert torch.equal(x, ex) and torch.equal(y, ey)
            assert x.dtype == y.dtype == torch.long
    assert tokenizer.identity == identity


@pytest.mark.parametrize("texts,message", [([], "at least one"), (["", ""], "zero tokens")])
def test_empty_inputs(tokenizer, texts, message):
    for call in (lambda: encode_corpus(tokenizer, texts),
                 lambda: list(iter_encoded_tokens(tokenizer, iter(texts))),
                 lambda: list(DocumentSequences(tokenizer, lambda: iter(texts), 4))):
        with pytest.raises(DatasetError, match=message):
            call()


@pytest.mark.parametrize("count", [0, 1, 4])
def test_too_short(count):
    for call in (lambda: make_sequences(list(range(count)), 4),
                 lambda: list(iter_sequences(iter(range(count)), 4))):
        with pytest.raises(DatasetError, match=f"has {count} tokens"):
            call()


def test_exact_window_tail_short_batch_and_document_crossing():
    tok = ByteLevelBpeTokenizer(merges=[], special_tokens={})
    texts = ["ab", "cdefg", "hijk"]
    source = DocumentSequences(tok, lambda: iter(texts), 2)
    assert list(source) == [[97, 98, 99], [100, 101, 102], [103, 104, 105]]
    batches = list(iter_lazy_batches(source, 2))
    assert batches[-1][0].tolist() == [[103, 104]]
    assert batches[-1][1].tolist() == [[104, 105]]
    assert list(iter_sequences(iter([1, 2, 3]), 2)) == [[1, 2, 3]]


def test_bounded_demand_and_no_read_ahead():
    consumed = []
    def tokens():
        for i in range(1000000):
            consumed.append(i)
            yield i
    stream = iter_lazy_batches(iter_sequences(tokens(), 4), 3)
    x, y = next(stream)
    assert consumed == list(range(15))
    assert x.tolist() == [[0, 1, 2, 3], [5, 6, 7, 8], [10, 11, 12, 13]]
    assert y.tolist() == [[1, 2, 3, 4], [6, 7, 8, 9], [11, 12, 13, 14]]
    stream.close()


@pytest.mark.parametrize("kind", ["sequence", "batch"])
@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_sizes_fail(kind, size):
    with pytest.raises(DatasetError):
        list(iter_sequences([1, 2], size) if kind == "sequence"
             else iter_lazy_batches([[1, 2]], size))


def test_trainer_matches_legacy_cycling_weights_losses_and_evaluation(tmp_path):
    from training.config import TrainingConfig
    from training.loop import Trainer
    from yug.native.config import ModelConfig
    from yug.native.model import YugTransformer
    tok = ByteLevelBpeTokenizer(merges=[], special_tokens={})
    texts = ["abcdefghijkl", "mnopqrstuvw"]
    config = ModelConfig.from_tokenizer(tok, hidden_size=8, num_layers=1,
                                       num_heads=2, ffn_size=16, context_length=4)
    tc = TrainingConfig(seed=12, learning_rate=0.001, batch_size=2, sequence_length=4,
                        max_steps=7, eval_interval=2, checkpoint_interval=100)
    torch.manual_seed(12)
    model = YugTransformer(config)
    twin = copy.deepcopy(model)
    old = Trainer(model, torch.optim.AdamW(model.parameters(), lr=.001), tc, config, tok.identity)
    new = Trainer(twin, torch.optim.AdamW(twin.parameters(), lr=.001), tc, config, tok.identity)
    eager = iter_batches(make_sequences(encode_corpus(tok, texts), 4), 2)
    expected = []
    for index in range(7):
        expected.append(old.train_step(*eager[index % len(eager)]))
        if old.step % 2 == 0:
            old.evaluate(eager)
    source = DocumentSequences(tok, lambda: iter(texts), 4)
    assert new.fit(source, source, str(tmp_path)) == expected
    assert new.step == old.step == 7
    for name, weight in model.state_dict().items():
        assert torch.equal(weight, twin.state_dict()[name])
    assert old.evaluate(eager) == new.evaluate(iter_lazy_batches(source, 2))
    for invalid in (iter([[1, 2, 3, 4, 5]]),):
        with pytest.raises(TrainingError, match="one-shot"):
            new.fit(invalid, source, str(tmp_path))


def test_phase_c_evaluation_receives_identical_batches(tmp_path, monkeypatch):
    from training import evaluate as ev
    from training.checkpoint import save_checkpoint
    from training.config import TrainingConfig
    from yug.native.config import ModelConfig
    from yug.native.model import YugTransformer
    tok = ByteLevelBpeTokenizer(merges=[], special_tokens={})
    config = ModelConfig.from_tokenizer(tok, hidden_size=8, num_layers=1,
                                       num_heads=2, ffn_size=16, context_length=4)
    model = YugTransformer(config)
    tc = TrainingConfig(seed=1, learning_rate=.001, batch_size=2, sequence_length=4,
                        max_steps=1, eval_interval=1, checkpoint_interval=1)
    save_checkpoint(tmp_path, model=model, optimizer=torch.optim.AdamW(model.parameters()),
                    model_config=config, training_config=tc, step=0, tokenizer_identity=tok.identity)
    texts = ["abcdefghi", "jklmnopqrstuvw"]
    before = ev.evaluate_checkpoint(str(tmp_path), tok, texts, sequence_length=4, batch_size=2, seed=1)
    monkeypatch.setattr(ev, "iter_batches", iter_lazy_batches)
    after = ev.evaluate_checkpoint(str(tmp_path), tok, texts, sequence_length=4, batch_size=2, seed=1)
    assert before.to_dict() == after.to_dict()
