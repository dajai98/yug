"""Production preparation/replay values, identity, integrity and recovery."""
import copy
from dataclasses import replace
import json

import pytest
import torch

from training.corpus_prep import (LicenseDecision, LicensePolicy, QualityConfig,
                                  RawDocument, prepare_corpus)
from training.dataset import DocumentSequences, encode_corpus, iter_batches, iter_lazy_batches, make_sequences
from training.errors import DatasetError
from training.production_corpus import (PipelineError, PreparedCorpus, canonical,
                                        identity, prepare_stream)
from training import token_shards as shards
from yug.tokenizer.bpe import ByteLevelBpeTokenizer

POLICY = LicensePolicy({"MIT": LicenseDecision.ALLOW, "NO": LicenseDecision.REJECT})
SOURCE = identity("local diagnostic corpus")


def documents():
    texts = ["hello useful code " + str(i) for i in range(8)]
    docs = [RawDocument(str(i), text, "local", "fixture", "MIT", "en", "code" if i % 2 else "docs")
            for i, text in enumerate(texts)]
    return docs + [replace(docs[0], document_id="dup"), replace(docs[1], license="NO"),
                   replace(docs[2], license="UNKNOWN"), replace(docs[3], text="  ")]


def prepare(tmp_path, tok, docs=None, **kwargs):
    return prepare_stream(iter(documents() if docs is None else docs), tmp_path,
                          license_policy=POLICY, source_identity=SOURCE,
                          sources={"local": {"revision": "fixture-v1"}},
                          tokenizer_identity=tok.identity, val_fraction=.25, seed=123,
                          quality_config=QualityConfig(min_length=1), **kwargs)


@pytest.fixture
def built(tmp_path):
    tok = ByteLevelBpeTokenizer.train([d.text for d in documents()[:8]], 270)
    prep = prepare(tmp_path / "prepared", tok)
    out = tmp_path / "tokens"
    shards.build_token_shards(prep, tok, out, shard_tokens=7)
    return tok, prep, out


def stream(built, split="train", **kwargs):
    tok, prep, out = built
    return shards.TokenShardStream(out, split=split,
                                   tokenizer_identity=kwargs.get("tokenizer_identity", tok.identity),
                                   prepared_corpus_identity=kwargs.get("prepared_corpus_identity", prep.identity))


def test_preparation_matches_legacy_and_deterministic_artifacts(tmp_path):
    tok = ByteLevelBpeTokenizer.train(["hello useful code"], 260)
    a = prepare(tmp_path / "a", tok)
    b = prepare(tmp_path / "b", tok)
    old = prepare_corpus(documents(), license_policy=POLICY, val_fraction=.25, seed=123,
                         quality_config=QualityConfig(min_length=1))
    assert list(a.documents("train")) == [d.text for d in old.train]
    assert list(a.documents("validation")) == [d.text for d in old.validation]
    assert a.manifest["counts"] == dict(input=12, accepted=8, rejected=3, review=1)
    assert a.manifest["rejection_reason_counts"] == old.manifest.rejection_reason_counts
    assert a.identity == b.identity
    for name in ("accepted.jsonl", "rejected.jsonl", "index.json", "manifest.json"):
        assert (a.directory / name).read_bytes() == (b.directory / name).read_bytes()
    for prepared, output in ((a, tmp_path / "ta"), (b, tmp_path / "tb")):
        shards.build_token_shards(prepared, tok, output, shard_tokens=7)
    for path in (tmp_path / "ta").glob("*.bin"):
        assert path.read_bytes() == (tmp_path / "tb" / path.name).read_bytes()
    assert (tmp_path / "ta/manifest.json").read_bytes() == (tmp_path / "tb/manifest.json").read_bytes()


def test_stratified_split_and_structured_normalization(tmp_path):
    from training.experiments.representative_mini_corpus import normalize_document, stratified_split, CATEGORIES
    tok = ByteLevelBpeTokenizer.train(["sample"], 260)
    docs = [replace(d, primary_domain="code") for d in documents()]
    old = prepare_corpus(docs, license_policy=POLICY, val_fraction=.25, seed=123,
                         quality_config=QualityConfig(min_length=1), text_normalizer=normalize_document)
    tr, va = stratified_split(old.accepted, val_fraction=.25, seed=123)
    new = prepare(tmp_path / "p", tok, docs, text_normalizer=normalize_document,
                  normalization_identity="representative-normalize-v1", category_order=CATEGORIES)
    assert list(new.documents("train")) == [d.text for d in tr]
    assert list(new.documents("validation")) == [d.text for d in va]


def test_identity_binds_inputs_policy_and_tokenizer(tmp_path):
    tok = ByteLevelBpeTokenizer.train(["hello"], 260)
    a = prepare(tmp_path / "a", tok)
    b = prepare(tmp_path / "b", tok, list(reversed(documents())))
    c = prepare(tmp_path / "c", ByteLevelBpeTokenizer(merges=[], special_tokens={}))
    d = prepare(tmp_path / "d", tok, text_normalizer=lambda doc: doc.text,
                normalization_identity="identity-normalizer-v1")
    assert len({a.identity, b.identity, c.identity, d.identity}) == 4
    with pytest.raises(PipelineError):
        PreparedCorpus(a.directory, expected_identity=b.identity)


@pytest.mark.parametrize("split", ["train", "validation"])
@pytest.mark.parametrize("length,batch", [(1, 1), (4, 3), (8, 2)])
def test_all_values_boundaries_and_repeatability(built, split, length, batch):
    tok, prep, _ = built
    texts = list(prep.documents(split))
    expected = encode_corpus(tok, texts)
    replay = stream(built, split)
    assert list(replay) == expected == list(replay)
    old = make_sequences(expected, length)
    assert list(replay.sequences(length)) == old
    assert list(DocumentSequences(tok, lambda: iter(texts), length)) == old
    actual = list(iter_lazy_batches(replay.sequences(length), batch))
    eager = iter_batches(old, batch)
    assert len(actual) == len(eager)
    for (a, b), (x, y) in zip(actual, eager):
        assert torch.equal(a, x) and torch.equal(b, y)
        assert a.dtype == b.dtype == torch.long


def rewrite_manifest(out, change):
    path = out / "manifest.json"
    m = json.loads(path.read_bytes())
    change(m)
    m.pop("manifest_identity")
    m["manifest_identity"] = identity(m)
    path.write_bytes(canonical(m))


@pytest.mark.parametrize("kind", ["bytes", "missing", "truncated", "oversized"])
def test_bad_shard_rejected_before_replay(built, kind):
    path = built[2] / "train-000000.bin"
    data = path.read_bytes()
    if kind == "missing":
        path.unlink()
    else:
        path.write_bytes(bytes([data[0] ^ 1]) + data[1:] if kind == "bytes" else
                         data[:-1] if kind == "truncated" else data + b"x")
    with pytest.raises(PipelineError):
        stream(built)


@pytest.mark.parametrize("kind", ["schema", "checksum", "order", "count", "total", "split", "path", "stream", "dtype"])
def test_manifest_corruption_rejected_even_with_recomputed_manifest_hash(built, kind):
    def change(m):
        first = m["splits"]["train"]["shards"][0]
        if kind == "schema": m["schema_version"] = "future"
        elif kind == "checksum": first["checksum"] = identity("wrong")
        elif kind == "order": first["index"] = 1
        elif kind == "count": first["token_count"] += 1
        elif kind == "total": m["total_token_count"] += 1
        elif kind == "split": first["split"] = "validation"
        elif kind == "path": first["path"] = "../outside.bin"
        elif kind == "stream": m["splits"]["train"]["ordered_token_stream_identity"] = identity("wrong")
        else: m["token_dtype"] = "uint16"
    rewrite_manifest(built[2], change)
    with pytest.raises(PipelineError):
        stream(built)


@pytest.mark.parametrize("field", ["tokenizer_identity", "prepared_corpus_identity"])
def test_wrong_identity(built, field):
    with pytest.raises(PipelineError):
        stream(built, **{field: identity("wrong")})


def test_split_separation_and_post_open_corruption(built):
    train, validation = stream(built), stream(built, "validation")
    assert list(train) != list(validation)
    path = built[2] / "validation-000000.bin"
    path.write_bytes(b"broken")
    assert list(train)
    with pytest.raises(PipelineError):
        list(validation)


@pytest.mark.parametrize("size", [0, 1])
def test_empty_and_single_document(tmp_path, size):
    tok = ByteLevelBpeTokenizer(merges=[], special_tokens={})
    if size == 0:
        with pytest.raises(PipelineError, match="at least one"):
            prepare(tmp_path / "p", tok, [])
        assert not (tmp_path / "p/manifest.json").exists()
        return
    prep = prepare(tmp_path / "p", tok, documents()[:1])
    shards.build_token_shards(prep, tok, tmp_path / "t", shard_tokens=3)
    built = tok, prep, tmp_path / "t"
    assert list(stream(built, "validation")) == []
    with pytest.raises(DatasetError, match="need at least"):
        list(stream(built, "validation").sequences(4))
    assert list(stream(built)) == tok.encode(documents()[0].text)


def test_resume_completed_shards_without_reencoding(tmp_path, monkeypatch):
    tok = ByteLevelBpeTokenizer(merges=[], special_tokens={})
    prep = prepare(tmp_path / "p", tok)
    calls = []
    encode = tok.encode
    monkeypatch.setattr(tok, "encode", lambda text: (calls.append(text), encode(text))[1])
    publish = shards._publish_bytes
    def interrupted(path, data):
        if path.name == "train-000001.bin":
            path.with_name(path.name + ".tmp").write_bytes(data[:1])
            raise RuntimeError("interrupted")
        publish(path, data)
    monkeypatch.setattr(shards, "_publish_bytes", interrupted)
    with pytest.raises(RuntimeError):
        shards.build_token_shards(prep, tok, tmp_path / "t", shard_tokens=7)
    assert (tmp_path / "t/train-000000.bin").exists()
    assert not (tmp_path / "t/manifest.json").exists()
    with pytest.raises(PipelineError):
        stream((tok, prep, tmp_path / "t"))
    with pytest.raises(PipelineError, match="Incompatible"):
        shards.build_token_shards(prep, tok, tmp_path / "t", shard_tokens=8)
    monkeypatch.setattr(shards, "_publish_bytes", publish)
    shards.build_token_shards(prep, tok, tmp_path / "t", shard_tokens=7)
    assert len(calls) == 8
    shards.build_token_shards(prep, tok, tmp_path / "t", shard_tokens=7)
    assert len(calls) == 8
    list(stream((tok, prep, tmp_path / "t")))
    assert len(calls) == 8


def test_interrupted_document_progress_reuses_only_committed_work(tmp_path, monkeypatch):
    tok = ByteLevelBpeTokenizer(merges=[], special_tokens={})
    prep = prepare(tmp_path / "p", tok)
    calls, write = [], shards.atomic_json
    encode = tok.encode
    monkeypatch.setattr(tok, "encode", lambda text: (calls.append(text), encode(text))[1])
    def interrupted(path, data):
        if path.name == "train.progress.json" and data["documents"] == 2:
            raise RuntimeError("power loss before journal publication")
        write(path, data)
    monkeypatch.setattr(shards, "atomic_json", interrupted)
    with pytest.raises(RuntimeError):
        shards.build_token_shards(prep, tok, tmp_path / "t", shard_tokens=7)
    monkeypatch.setattr(shards, "atomic_json", write)
    shards.build_token_shards(prep, tok, tmp_path / "t", shard_tokens=7)
    assert len(calls) == 9  # one uncommitted document retried, not the committed prefix
    assert list(stream((tok, prep, tmp_path / "t"))) == encode_corpus(tok, list(prep.documents("train")))


def test_tiny_training_and_exact_evaluation_report(tmp_path):
    from training.config import TrainingConfig
    from training.loop import Trainer
    from training.checkpoint import save_checkpoint
    from training.evaluate import evaluate_checkpoint
    from yug.native.config import ModelConfig
    from yug.native.model import YugTransformer
    tok = ByteLevelBpeTokenizer(merges=[], special_tokens={})
    docs = [replace(documents()[i], text=text) for i, text in enumerate(("abcdefghijkl", "mnopqrstuvwx"))]
    prep = prepare(tmp_path / "p", tok, docs)
    shards.build_token_shards(prep, tok, tmp_path / "t", shard_tokens=3)
    replay = stream((tok, prep, tmp_path / "t"))
    val_replay = stream((tok, prep, tmp_path / "t"), "validation")
    config = ModelConfig.from_tokenizer(tok, hidden_size=8, num_layers=1, num_heads=2, ffn_size=16, context_length=4)
    tc = TrainingConfig(seed=7, learning_rate=.001, batch_size=2, sequence_length=4,
                        max_steps=7, eval_interval=2, checkpoint_interval=100)
    torch.manual_seed(7)
    model = YugTransformer(config)
    twin = copy.deepcopy(model)
    a = Trainer(model, torch.optim.AdamW(model.parameters(), lr=.001), tc, config, tok.identity)
    b = Trainer(twin, torch.optim.AdamW(twin.parameters(), lr=.001), tc, config, tok.identity)
    losses_a = a.fit(DocumentSequences(tok, lambda: prep.documents("train"), 4),
                     DocumentSequences(tok, lambda: prep.documents("validation"), 4), str(tmp_path / "unused"))
    losses_b = b.fit(replay.sequences(4), val_replay.sequences(4), str(tmp_path / "unused"))
    assert losses_a == losses_b
    for name, weight in model.state_dict().items():
        assert torch.equal(weight, twin.state_dict()[name])
    assert a.evaluate(iter_lazy_batches(val_replay.sequences(4), 2)) == b.evaluate(iter_lazy_batches(val_replay.sequences(4), 2))
    reports = []
    for trainer in (a, b):
        save_checkpoint(tmp_path / "checkpoint", model=trainer.model, optimizer=trainer.optimizer,
                        model_config=config, training_config=tc, step=7, tokenizer_identity=tok.identity)
        reports.append(evaluate_checkpoint(str(tmp_path / "checkpoint"), tok,
                       list(prep.documents("validation")), sequence_length=4, batch_size=2, seed=7).to_dict())
    assert reports[0] == reports[1]


def test_interrupted_preparation_is_not_published(tmp_path):
    tok = ByteLevelBpeTokenizer(merges=[], special_tokens={})
    def interrupted():
        yield documents()[0]
        raise RuntimeError("source interrupted")
    with pytest.raises(RuntimeError):
        prepare(tmp_path / "p", tok, interrupted())
    assert not (tmp_path / "p/manifest.json").exists()
    with pytest.raises(PipelineError):
        PreparedCorpus(tmp_path / "p", expected_identity=SOURCE)
    with pytest.raises(PipelineError, match="already exists"):
        prepare(tmp_path / "p", tok)


def test_complete_prepared_payload_corruption_rejected(built):
    _, prep, _ = built
    with (prep.directory / "accepted.jsonl").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(PipelineError, match="checksum"):
        PreparedCorpus(prep.directory, expected_identity=prep.identity)


@pytest.mark.parametrize("kind", ["tokenizer", "corpus"])
def test_incompatible_completed_resume_rejected(built, tmp_path, kind):
    tok, prep, out = built
    if kind == "tokenizer":
        tok = ByteLevelBpeTokenizer(merges=[], special_tokens={})
    else:
        prep = prepare(tmp_path / "other", tok, list(reversed(documents())))
    with pytest.raises(PipelineError):
        shards.build_token_shards(prep, tok, out, shard_tokens=7)


def test_shard_build_lock_prevents_second_writer(tmp_path):
    tok = ByteLevelBpeTokenizer(merges=[], special_tokens={})
    prep = prepare(tmp_path / "p", tok)
    with shards._writer_lock(tmp_path / "tokens"):
        with pytest.raises(PipelineError, match="writer"):
            shards.build_token_shards(prep, tok, tmp_path / "tokens")


def test_zero_survivors_preserves_empty_partitions(tmp_path):
    tok = ByteLevelBpeTokenizer(merges=[], special_tokens={})
    prep = prepare(tmp_path / "p", tok, [replace(documents()[0], license="NO")])
    shards.build_token_shards(prep, tok, tmp_path / "t")
    assert list(stream((tok, prep, tmp_path / "t"))) == []
    assert list(stream((tok, prep, tmp_path / "t"), "validation")) == []


@pytest.mark.parametrize("permanent", [False, True])
def test_atomic_replace_retry_never_deletes_committed_target(tmp_path, monkeypatch, permanent):
    from training import production_corpus as pc
    path = tmp_path / "journal.json"
    pc.atomic_json(path, {"committed": 1})
    previous = path.read_bytes()
    real_replace, attempts = pc.os.replace, []
    def interrupted(source, target):
        attempts.append(1)
        assert path.read_bytes() == previous
        if permanent or len(attempts) < 3:
            raise PermissionError("sharing contention")
        return real_replace(source, target)
    monkeypatch.setattr(pc.os, "replace", interrupted)
    monkeypatch.setattr(pc.time, "sleep", lambda delay: None)
    if permanent or pc.os.name != "nt":
        with pytest.raises(PermissionError):
            pc.atomic_json(path, {"committed": 2})
        assert path.read_bytes() == previous
    else:
        pc.atomic_json(path, {"committed": 2})
        assert json.loads(path.read_bytes()) == {"committed": 2}
        assert len(attempts) == 3
