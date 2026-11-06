"""Historical prospective-use links never re-admit or re-account an item."""
import json
import sqlite3
from types import SimpleNamespace

import pytest

from training import historical_authority_link as links
from training.cumulative_accounting import GENESIS
from training.production_acquisition import ProductionBuild
from training.production_corpus import canonical, file_digest, identity
from training.production_spec import seal


def fixture(tmp_path, monkeypatch):
    root = tmp_path
    raw = "sha256:" + "a" * 64
    record = seal(dict(source_id="gh-python-attrs-attrs", revision="6" * 40,
                       path="src/attr/example.py", sha256=raw, bytes=7,
                       owner="code", split="train"))
    item = identity({k: record[k] for k in
                     ("source_id", "revision", "path", "sha256")})
    evidence = dict(item=item, record=record, source_id=record["source_id"],
                    owner="code", split="train", normalized_sha="sha256:" + "b" * 64,
                    token_sha="sha256:" + "e" * 64,
                    historical_state_sha256="sha256:" + "f" * 64, tokens=5)
    bid = "sha256:" + "c" * 64
    entry = dict(build_id=bid, parent=GENESIS,
                 historical_import_identity="sha256:" + "d" * 64,
                 evidence_identity=identity([evidence]))
    head = identity(entry)
    ledger = root / "ledger.sqlite"
    with sqlite3.connect(ledger) as db:
        db.executescript("""CREATE TABLE meta(key TEXT,value TEXT);
            CREATE TABLE builds(build_id TEXT,state TEXT);
            CREATE TABLE item_evidence(build_id TEXT,item TEXT,evidence TEXT);
            CREATE TABLE commits(ordinal INTEGER,build_id TEXT,head TEXT,entry TEXT);
            CREATE TABLE totals(kind TEXT,key TEXT,tokens INTEGER);""")
        db.execute("INSERT INTO meta VALUES ('head',?)", (head,))
        db.execute("INSERT INTO builds VALUES (?,'COMMITTED')", (bid,))
        db.execute("INSERT INTO item_evidence VALUES (?,?,?)",
                   (bid, item, canonical(evidence).decode()))
        db.execute("INSERT INTO commits VALUES (1,?,?,?)",
                   (bid, head, canonical(entry).decode()))
        db.execute("INSERT INTO totals VALUES ('category','code',5)")

    source = dict(source_id=record["source_id"], revision=record["revision"],
                  locator="https://github.com/python-attrs/attrs", split="train",
                  max_files=1, max_raw_bytes=7, max_accepted_tokens=5,
                  attribution="MIT attribution", notice_obligations=["Retain MIT notice"],
                  obligations=["Preserve notices"])
    decision = seal(dict(schema="yug-source-training-use-decision-v2",
                         decision="APPROVE_TRAINING_USE", intended_use="commercial_training",
                         owner="code", decided_at="2026-10-31T13:41:40Z",
                         files=[dict(path=record["path"], sha256=raw, bytes=7)],
                         governance_decision_identity="PENDING", **source))
    approval = seal(dict(schema="yug-human-prospective-use-approval-record-v1",
                         authority_scope="prospective_intended_use_decision_only",
                         decision="APPROVE_EXACT_SCOPE", intended_use="commercial_training",
                         decided_at=decision["decided_at"],
                         file_set_identity=identity(decision["files"]), **source))
    decision = seal(dict(decision, governance_decision_identity=approval["identity"]))
    manifest = seal(dict(mode="development"))
    result = dict(authority="NONE", files=1)
    monkeypatch.setattr(links, "verify_bundle", lambda *_args, **_kwargs: result)
    paths = [root / x for x in ("decision.json", "approval.json",
             "manifest.json", "verification.json")]
    for path, value in zip(paths[:3], (decision, approval, manifest), strict=True):
        path.write_bytes(canonical(value))
    verification = seal(dict(authority="NONE", decision_identity=decision["identity"],
        human_approval_record_identity=approval["identity"],
        development_bundle_identity=manifest["identity"],
        decision_sha256=file_digest(paths[0]),
        human_approval_record_sha256=file_digest(paths[1]),
        development_bundle_sha256=file_digest(paths[2]), file_count=1,
        accepted_token_cap=5, raw_byte_cap=7, result=result))
    paths[3].write_bytes(canonical(verification))
    ids = [v["identity"] for v in (decision, approval, manifest, verification)]
    return root, ledger, head, item, paths, ids


def test_link_readback_and_accounting_unchanged(tmp_path, monkeypatch):
    root, ledger, head, item, paths, ids = fixture(tmp_path, monkeypatch)
    before = file_digest(ledger)
    store = root / "links.jsonl"
    result = links.create_store(root, store, ledger, head, paths, ids,
                                created_at="2026-11-05T00:00:00Z")
    assert len(result["links"]) == 1
    assert result["head"] == result["links"][item]["identity"]
    assert result["links"][item]["historical_ingestion_at"] == "UNKNOWN"
    assert result["links"][item]["historical_accounting_state"] == "COMMITTED"
    assert result["links"][item]["authority_decided_at"] < result["links"][item]["link_created_at"]
    assert result["accounting_delta_documents"] == result["accounting_delta_tokens"] == 0
    assert file_digest(ledger) == before
    with sqlite3.connect(ledger.as_uri() + "?mode=ro", uri=True) as db:
        assert db.execute("SELECT count(*) FROM item_evidence").fetchone()[0] == 1
        assert db.execute("SELECT tokens FROM totals").fetchone()[0] == 5
    assert links.require_linked_historical_training_use(
        root, item, store, result["store_sha256"], ledger, head, paths, ids)
    candidate_path = root / "retention.json"
    candidate_id, candidate_sha = links.create_retention_candidate(
        root, candidate_path, store, result["store_sha256"], result["head"],
        ledger, before, head, paths, ids, created_at="2026-11-05T01:00:00Z")
    assert links.verify_retention_candidate(
        root, candidate_path, candidate_id, candidate_sha, store,
        result["store_sha256"], result["head"], ledger, before, head,
        paths, ids)["authority"] == "NONE"
    fake = SimpleNamespace(diagnostic=False, corpus_ledger_path=ledger,
                           expected_corpus_head=head, root=root)
    gated = ProductionBuild.require_historical_item_training_use(
        fake, item, link_store=store,
        expected_link_store_sha256=result["store_sha256"],
        material_paths=paths, expected_identities=ids,
        retention_candidate=candidate_path,
        expected_retention_identity=candidate_id,
        expected_retention_sha256=candidate_sha,
        expected_link_head=result["head"], expected_ledger_sha256=before)
    assert gated["authority"] == "NONE" and gated["link"]["item_identity"] == item
    with pytest.raises(ValueError, match="already exists"):
        links.create_store(root, store, ledger, head, paths, ids)


@pytest.mark.parametrize("field,value", [
    ("item_identity", "sha256:" + "0" * 64),
    ("revision", "7" * 40),
    ("path", "other.py"),
    ("raw_sha256", "sha256:" + "0" * 64),
    ("normalized_sha256", "sha256:" + "0" * 64),
    ("token_count", 6),
    ("accounting_delta_tokens", 1),
    ("link_created_at", "2026-10-01T00:00:00Z"),
])
def test_tampered_link_fails(tmp_path, monkeypatch, field, value):
    root, ledger, head, _, paths, ids = fixture(tmp_path, monkeypatch)
    store = root / "links.jsonl"
    links.create_store(root, store, ledger, head, paths, ids)
    event = json.loads(store.read_bytes())
    store.write_bytes(canonical(seal(dict(event, **{field: value}))) + b"\n")
    with pytest.raises(ValueError):
        links.verify_store(root, store, file_digest(store), ledger, head, paths, ids)


def test_wrong_decision_or_unlinked_item_fails(tmp_path, monkeypatch):
    root, ledger, head, item, paths, ids = fixture(tmp_path, monkeypatch)
    store = root / "links.jsonl"
    result = links.create_store(root, store, ledger, head, paths, ids)
    wrong = ["sha256:" + "0" * 64, *ids[1:]]
    with pytest.raises(ValueError, match="identity mismatch"):
        links.verify_store(root, store, result["store_sha256"], ledger, head, paths, wrong)
    with pytest.raises(ValueError, match="no prospective-use link"):
        links.require_linked_historical_training_use(
            root, "sha256:" + "0" * 64, store, result["store_sha256"],
            ledger, head, paths, ids)


def test_scope_mismatch_or_untrusted_store_fails(tmp_path, monkeypatch):
    root, ledger, head, _, paths, ids = fixture(tmp_path, monkeypatch)
    store = root / "links.jsonl"
    decision = json.loads(paths[0].read_bytes())
    decision["files"][0]["sha256"] = "sha256:" + "f" * 64
    paths[0].write_bytes(canonical(seal(decision)))
    with pytest.raises(ValueError):
        links.create_store(root, store, ledger, head, paths, ids)
    assert not store.exists()


def test_build_gate_rejects_unlinked_and_tampered(tmp_path, monkeypatch):
    root, ledger, head, item, paths, ids = fixture(tmp_path, monkeypatch)
    store = root / "links.jsonl"
    result = links.create_store(root, store, ledger, head, paths, ids)
    candidate_path = root / "retention.json"
    candidate_id, candidate_sha = links.create_retention_candidate(
        root, candidate_path, store, result["store_sha256"], result["head"],
        ledger, file_digest(ledger), head, paths, ids)
    fake = SimpleNamespace(diagnostic=False, corpus_ledger_path=ledger,
                           expected_corpus_head=head, root=root)
    arguments = dict(link_store=store,
                     expected_link_store_sha256=result["store_sha256"],
                     material_paths=paths, expected_identities=ids,
                     retention_candidate=candidate_path,
                     expected_retention_identity=candidate_id,
                     expected_retention_sha256=candidate_sha,
                     expected_link_head=result["head"],
                     expected_ledger_sha256=file_digest(ledger))
    for missing in ("expected_retention_identity", "expected_retention_sha256",
                    "expected_link_head"):
        incomplete = dict(arguments, **{missing: None})
        with pytest.raises(ValueError, match="pins required"):
            ProductionBuild.require_historical_item_training_use(
                fake, item, **incomplete)
    with pytest.raises(ValueError, match="store pin mismatch"):
        ProductionBuild.require_historical_item_training_use(
            fake, item, **dict(arguments,
                               expected_link_store_sha256="sha256:" + "0" * 64))
    with pytest.raises(ValueError, match="chained head mismatch"):
        ProductionBuild.require_historical_item_training_use(
            fake, item, **dict(arguments,
                               expected_link_head="sha256:" + "0" * 64))
    with pytest.raises(ValueError, match="no prospective-use link"):
        ProductionBuild.require_historical_item_training_use(
            fake, "sha256:" + "0" * 64, **arguments)
    event = json.loads(store.read_bytes())
    store.write_bytes(canonical(seal(dict(event, token_count=6))) + b"\n")
    with pytest.raises(ValueError, match="store pin mismatch"):
        ProductionBuild.require_historical_item_training_use(fake, item, **arguments)


def test_retention_candidate_tamper_fails(tmp_path, monkeypatch):
    root, ledger, head, _, paths, ids = fixture(tmp_path, monkeypatch)
    store = root / "links.jsonl"
    result = links.create_store(root, store, ledger, head, paths, ids)
    candidate_path = root / "retention.json"
    ledger_sha = file_digest(ledger)
    candidate_id, candidate_sha = links.create_retention_candidate(
        root, candidate_path, store, result["store_sha256"], result["head"],
        ledger, ledger_sha, head, paths, ids)
    candidate = json.loads(candidate_path.read_bytes())
    candidate_path.write_bytes(canonical(seal(dict(candidate, grants_training=True))))
    with pytest.raises(ValueError, match="byte pin mismatch"):
        links.verify_retention_candidate(
            root, candidate_path, candidate_id, candidate_sha, store,
            result["store_sha256"], result["head"], ledger, ledger_sha,
            head, paths, ids)
    with pytest.raises(ValueError, match="scope or zero-authority"):
        links.verify_retention_candidate(
            root, candidate_path, json.loads(candidate_path.read_bytes())["identity"],
            file_digest(candidate_path), store, result["store_sha256"],
            result["head"], ledger, ledger_sha, head, paths, ids)


def training_authority_fixture(tmp_path, monkeypatch):
    root, ledger, head, item, paths, ids = fixture(tmp_path, monkeypatch)
    store = root / "links.jsonl"
    verified = links.create_store(root, store, ledger, head, paths, ids)
    candidate = root / "retention.json"
    candidate_id, candidate_sha = links.create_retention_candidate(
        root, candidate, store, verified["store_sha256"], verified["head"],
        ledger, file_digest(ledger), head, paths, ids)
    link = verified["links"][item]
    source = dict(source_id=link["source_id"], revision=link["revision"])
    file = dict(path=link["path"], sha256=link["raw_sha256"])
    def artifact(path):
        return dict(path=path.relative_to(root).as_posix(),
                    sha256=file_digest(path))
    manifest_path = root / "production_manifest.json"
    manifest = seal(dict(mode="production", source=source, files=[file],
                         decision_scope=dict(source=source,
                             artifact_path=paths[0].relative_to(root).as_posix(),
                             files=[file]),
                         artifacts=[artifact(store), artifact(candidate),
                                    artifact(paths[0])]))
    manifest_path.write_bytes(canonical(manifest))
    signature = root / "signature.json"
    timestamp = root / "timestamp.tsr"
    tsa_ca = root / "tsa-ca.pem"
    trust_pin = root / "trust-pin.json"
    for path in (signature, timestamp, tsa_ca, trust_pin):
        path.write_bytes(b"test fixture; verification is mocked")
    pin_identity = "sha256:" + "9" * 64
    production_status = dict(signature="VERIFIED", timestamp="VERIFIED",
                             authority="NONE", files=1,
                             trust_pin_identity=pin_identity)

    def fake_bundle(_root, path, _identity, **_kwargs):
        if path == manifest_path:
            return production_status
        return dict(authority="NONE", files=1)

    monkeypatch.setattr(links, "verify_bundle", fake_bundle)
    arguments = dict(ledger_path=ledger, expected_ledger_sha256=file_digest(ledger),
        expected_ledger_head=head, store_path=store,
        expected_store_sha256=verified["store_sha256"],
        expected_store_head=verified["head"], material_paths=paths,
        expected_identities=ids, retention_candidate=candidate,
        expected_retention_identity=candidate_id,
        expected_retention_sha256=candidate_sha,
        channel_a_retained=True, channel_b_retained=True,
        production_manifest=manifest_path,
        expected_production_manifest_identity=manifest["identity"],
        expected_production_manifest_sha256=file_digest(manifest_path),
        signature_path=signature, timestamp_path=timestamp,
        tsa_ca_file=tsa_ca, expected_tsa_ca_sha256=file_digest(tsa_ca),
        trust_pin_path=trust_pin, expected_trust_pin_identity=pin_identity,
        trust_root=root, expected_source_id=source["source_id"],
        expected_revision=source["revision"])
    return root, item, arguments, production_status


def test_training_authority_combines_all_inputs_without_accounting(tmp_path,
                                                                    monkeypatch):
    root, item, arguments, _ = training_authority_fixture(tmp_path, monkeypatch)
    before = file_digest(arguments["ledger_path"])
    result = links.authorize_existing_historical_items(root, [item], **arguments)
    assert result["status"] == "TRAINING_AUTHORIZED"
    assert result["authority"] == "EXISTING_ITEM_USE_ONLY"
    assert result["item_identities"] == [item]
    assert result["evidence_authority"] == "NONE"
    assert result["accounting_delta_documents"] == 0
    assert result["accounting_delta_tokens"] == 0
    assert result["grants_training_run"] is False
    assert result["grants_shards"] is False
    assert result["grants_corpus_admission"] is False
    assert file_digest(arguments["ledger_path"]) == before


@pytest.mark.parametrize("field,value", [
    ("channel_a_retained", False),
    ("channel_b_retained", False),
    ("expected_store_sha256", "sha256:" + "0" * 64),
    ("expected_store_head", "sha256:" + "0" * 64),
    ("expected_retention_identity", "sha256:" + "0" * 64),
    ("expected_tsa_ca_sha256", "sha256:" + "0" * 64),
    ("expected_trust_pin_identity", "sha256:" + "0" * 64),
    ("expected_revision", "7" * 40),
])
def test_training_authority_rejects_missing_or_wrong_pins(tmp_path, monkeypatch,
                                                           field, value):
    root, item, arguments, _ = training_authority_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        links.authorize_existing_historical_items(
            root, [item], **dict(arguments, **{field: value}))


@pytest.mark.parametrize("field", ["signature", "timestamp"])
def test_training_authority_rejects_invalid_production_verification(
        tmp_path, monkeypatch, field):
    root, item, arguments, production_status = training_authority_fixture(
        tmp_path, monkeypatch)
    production_status[field] = "INVALID"
    with pytest.raises(ValueError, match="not verified"):
        links.authorize_existing_historical_items(root, [item], **arguments)


def test_training_authority_rejects_missing_production_evidence(tmp_path,
                                                                 monkeypatch):
    root, item, arguments, _ = training_authority_fixture(tmp_path, monkeypatch)
    arguments["signature_path"].unlink()
    with pytest.raises(ValueError, match="missing"):
        links.authorize_existing_historical_items(root, [item], **arguments)


def test_training_authority_rejects_unlinked_or_tampered_item(tmp_path,
                                                               monkeypatch):
    root, item, arguments, _ = training_authority_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="missing from retained"):
        links.authorize_existing_historical_items(
            root, ["sha256:" + "0" * 64], **arguments)
    store = arguments["store_path"]
    event = json.loads(store.read_bytes())
    store.write_bytes(canonical(seal(dict(event, raw_sha256="sha256:" + "0" * 64)))
                      + b"\n")
    with pytest.raises(ValueError):
        links.authorize_existing_historical_items(root, [item], **arguments)


def test_training_authority_rejects_wrong_production_file_scope(tmp_path,
                                                                 monkeypatch):
    root, item, arguments, _ = training_authority_fixture(tmp_path, monkeypatch)
    path = arguments["production_manifest"]
    manifest = json.loads(path.read_bytes())
    manifest["files"][0]["path"] = "wrong.py"
    manifest = seal(manifest)
    path.write_bytes(canonical(manifest))
    arguments["expected_production_manifest_identity"] = manifest["identity"]
    arguments["expected_production_manifest_sha256"] = file_digest(path)
    with pytest.raises(ValueError, match="exact historical decision scope"):
        links.authorize_existing_historical_items(root, [item], **arguments)


@pytest.mark.parametrize("missing", ["links.jsonl", "retention.json",
                                     "decision.json"])
def test_training_authority_rejects_unbound_input(tmp_path, monkeypatch,
                                                  missing):
    root, item, arguments, _ = training_authority_fixture(tmp_path, monkeypatch)
    path = arguments["production_manifest"]
    manifest = json.loads(path.read_bytes())
    manifest["artifacts"] = [row for row in manifest["artifacts"]
                              if row["path"] != missing]
    manifest = seal(manifest)
    path.write_bytes(canonical(manifest))
    arguments["expected_production_manifest_identity"] = manifest["identity"]
    arguments["expected_production_manifest_sha256"] = file_digest(path)
    with pytest.raises(ValueError, match="does not bind"):
        links.authorize_existing_historical_items(root, [item], **arguments)


def test_training_authority_rejects_verifier_failure(tmp_path, monkeypatch):
    root, item, arguments, _ = training_authority_fixture(tmp_path, monkeypatch)

    def fail_bundle(*_args, **_kwargs):
        raise ValueError("Invalid production signature")

    monkeypatch.setattr(links, "verify_bundle", fail_bundle)
    with pytest.raises(ValueError, match="Invalid production signature"):
        links.authorize_existing_historical_items(root, [item], **arguments)
