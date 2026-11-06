"""Append-only prospective-use links for already accounted historical items.

This store never admits content or changes corpus accounting. Its byte hash must
be pinned independently before a caller treats its readback as authority.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from training.cumulative_accounting import GENESIS
from training.evidence_bundle import verify_bundle
from training.production_corpus import canonical, file_digest, identity
from training.production_spec import TOKENIZER, need, seal, verify_seal

SCHEMA = "yug-historical-item-authority-link-v1"
TYPE = "PROSPECTIVE_TRAINING_USE_OF_EXISTING_ITEM"
RETENTION_SCHEMA = "yug-historical-authority-link-retention-candidate-v1"


def _time(value):
    need(type(value) is str and value.endswith("Z"), "UTC authority time required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid UTC authority time") from exc
    need(parsed.utcoffset().total_seconds() == 0, "Authority time must be UTC")
    return parsed


def _read(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            need(key not in value, "Duplicate authority input field")
            value[key] = item
        return value

    value = json.loads(Path(path).read_bytes(), object_pairs_hook=pairs)
    need(type(value) is dict, "Authority input must be an object")
    verify_seal(value)
    return value


def _materials(root, decision_path, approval_path, manifest_path,
               verification_path, expected):
    paths = [Path(p).resolve() for p in (decision_path, approval_path,
             manifest_path, verification_path)]
    need(all(p.is_relative_to(root) for p in paths), "Authority input outside root")
    decision, approval, manifest, verification = map(_read, paths)
    need([x["identity"] for x in (decision, approval, manifest, verification)]
         == list(expected), "Prospective authority identity mismatch")
    need(decision.get("schema") == "yug-source-training-use-decision-v2"
         and decision.get("decision") == "APPROVE_TRAINING_USE"
         and decision.get("intended_use") == "commercial_training"
         and decision.get("split") == "train", "Unsupported prospective decision")
    need(approval.get("schema") == "yug-human-prospective-use-approval-record-v1"
         and approval.get("authority_scope") == "prospective_intended_use_decision_only"
         and approval.get("intended_use") == "commercial_training"
         and approval.get("file_set_identity") == identity(decision["files"])
         and approval.get("decision", "").startswith("APPROVE_"),
         "Prospective human approval mismatch")
    need(decision["governance_decision_identity"] == approval["identity"]
         and decision["decided_at"] == approval["decided_at"]
         and all(decision[k] == approval[k] for k in
                 ("source_id", "revision", "locator", "split", "max_files",
                  "max_raw_bytes", "max_accepted_tokens", "attribution",
                  "notice_obligations", "obligations")),
         "Decision and human approval disagree")
    need(manifest.get("mode") == "development"
         and verification.get("authority") == "NONE"
         and verification.get("decision_identity") == decision["identity"]
         and verification.get("human_approval_record_identity") == approval["identity"]
         and verification.get("development_bundle_identity") == manifest["identity"]
         and verification.get("decision_sha256") == file_digest(paths[0])
         and verification.get("human_approval_record_sha256") == file_digest(paths[1])
         and verification.get("development_bundle_sha256") == file_digest(paths[2]),
         "Prospective evidence verification mismatch")
    result = verify_bundle(root, paths[2], manifest["identity"])
    need(result == verification["result"] and result["authority"] == "NONE",
         "Prospective evidence bundle changed")
    files = decision["files"]
    need(type(files) is list and len(files) == decision["max_files"]
         and len({f["path"] for f in files}) == len(files)
         and sum(f["bytes"] for f in files) <= decision["max_raw_bytes"],
         "Prospective file set changed")
    need(verification.get("file_count") == len(files)
         and verification.get("accepted_token_cap") == decision["max_accepted_tokens"]
         and verification.get("raw_byte_cap") == decision["max_raw_bytes"],
         "Prospective verification scope changed")
    _time(decision["decided_at"])
    return decision, approval, manifest, verification


def _ledger_item(ledger_path, expected_head, item):
    """Read a committed historical item through its sealed corpus commit."""
    path = Path(ledger_path).resolve()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        need(db.execute("PRAGMA quick_check").fetchone()[0] == "ok",
             "Corpus ledger corrupt")
        head = db.execute("SELECT value FROM meta WHERE key='head'").fetchone()
        need(head == (expected_head,), "Externally pinned corpus head mismatch")
        parent = GENESIS
        ordinal = 0
        for position, commit_head, encoded in db.execute(
                "SELECT ordinal,head,entry FROM commits ORDER BY ordinal"):
            commit = json.loads(encoded)
            need(position == ordinal + 1 and commit["parent"] == parent
                 and identity(commit) == commit_head,
                 "Corpus commit chain changed")
            parent = commit_head
            ordinal = position
        need(parent == expected_head, "Historical item outside trusted corpus chain")
        rows = list(db.execute(
            "SELECT e.build_id,e.evidence,b.state,c.ordinal,c.head,c.entry "
            "FROM item_evidence e JOIN builds b ON b.build_id=e.build_id "
            "JOIN commits c ON c.build_id=e.build_id WHERE e.item=?", (item,)))
        need(len(rows) == 1, "Existing historical item missing or duplicated")
        bid, encoded, state, ordinal, commit_head, entry_encoded = rows[0]
        entry = json.loads(entry_encoded)
        need(state == "COMMITTED" and ordinal >= 1
             and "historical_import_identity" in entry
             and identity(entry) == commit_head
             and entry["build_id"] == bid,
             "Item is not in a sealed historical commit")
        evidence_rows = [json.loads(row[0]) for row in db.execute(
            "SELECT evidence FROM item_evidence WHERE build_id=? ORDER BY item", (bid,))]
        need(identity(evidence_rows) == entry["evidence_identity"],
             "Historical commit item evidence changed")
        evidence = json.loads(encoded)
        record = evidence["record"]
        verify_seal(record)
        need(evidence["item"] == item
             and item == identity({k: record[k] for k in
                                  ("source_id", "revision", "path", "sha256")})
             and evidence["source_id"] == record["source_id"]
             and evidence["owner"] == record["owner"]
             and evidence["split"] == record["split"],
             "Historical item identity mismatch")
        return evidence, dict(build_id=bid, ordinal=ordinal, head=commit_head,
                              state=state)


def _fields(evidence, commit, decision, approval, manifest, verification,
            file, ledger_head):
    record = evidence["record"]
    need(record["source_id"] == decision["source_id"]
         and record["revision"] == decision["revision"]
         and record["path"] == file["path"]
         and record["sha256"] == file["sha256"]
         and record["bytes"] == file["bytes"]
         and record["owner"] == decision["owner"]
         and record["split"] == decision["split"] == "train"
         and evidence["tokens"] > 0,
         "Prospective scope differs from historical item")
    return dict(schema=SCHEMA, link_type=TYPE,
                item_identity=evidence["item"], source_id=record["source_id"],
                revision=record["revision"], path=record["path"],
                raw_sha256=record["sha256"],
                normalized_sha256=evidence["normalized_sha"],
                token_stream_sha256=evidence["token_sha"],
                tokenizer_identity=TOKENIZER, token_count=evidence["tokens"],
                historical_state_sha256=evidence["historical_state_sha256"],
                category=evidence["owner"], split=evidence["split"],
                historical_ingestion_at="UNKNOWN",
                historical_accounting_state=commit["state"],
                historical_accounting_commit_ordinal=commit["ordinal"],
                historical_accounting_commit_head=commit["head"],
                corpus_head_at_link=ledger_head,
                authority_decision_identity=decision["identity"],
                human_approval_identity=approval["identity"],
                evidence_manifest_identity=manifest["identity"],
                verification_identity=verification["identity"],
                authority_decided_at=decision["decided_at"],
                accounting_delta_documents=0, accounting_delta_tokens=0)


def verify_store(root, store_path, expected_store_sha256, ledger_path,
                 expected_ledger_head, material_paths, expected_identities):
    """Verify an externally byte-pinned link log and its exact historical items."""
    root = Path(root).resolve()
    store = Path(store_path).resolve()
    need(store.is_relative_to(root) and Path(ledger_path).resolve().is_relative_to(root),
         "Authority or ledger path outside root")
    need(file_digest(store) == expected_store_sha256,
         "External authority-link store pin mismatch")
    decision, approval, manifest, verification = _materials(
        root, *material_paths, expected_identities)
    scope = {f["path"]: f for f in decision["files"]}
    links = {}
    previous = None
    data = store.read_bytes()
    need(data and data.endswith(b"\n"), "Incomplete authority-link log")
    for line in data.splitlines(keepends=True):
        event = json.loads(line)
        need(canonical(event) + b"\n" == line, "Noncanonical authority-link event")
        verify_seal(event)
        need(event.get("schema") == SCHEMA and event.get("link_type") == TYPE
             and event.get("previous_link_identity") == previous
             and event.get("accounting_delta_documents") == 0
             and event.get("accounting_delta_tokens") == 0
             and event.get("historical_ingestion_at") == "UNKNOWN",
             "Invalid authority-link event or accounting delta")
        item = event["item_identity"]
        need(item not in links and event["path"] in scope,
             "Duplicate or out-of-scope authority link")
        evidence, commit = _ledger_item(ledger_path, expected_ledger_head, item)
        fields = _fields(evidence, commit, decision, approval, manifest,
                         verification, scope[event["path"]], event["corpus_head_at_link"])
        need(set(event) == set(fields) | {"link_created_at",
             "previous_link_identity", "identity"},
             "Unexpected authority-link field")
        with sqlite3.connect(Path(ledger_path).resolve().as_uri() + "?mode=ro",
                             uri=True) as db:
            linked_head = db.execute("SELECT ordinal FROM commits WHERE head=?",
                                     (event["corpus_head_at_link"],)).fetchone()
        need(linked_head is not None and linked_head[0] >= commit["ordinal"]
             and all(event.get(k) == v for k, v in fields.items()),
             "Authority link differs from historical item or decision")
        need(_time(event["link_created_at"]) >= _time(event["authority_decided_at"]),
             "Authority link backdated before decision")
        links[item] = event
        previous = event["identity"]
    need({x["path"] for x in links.values()} == set(scope),
         "Incomplete prospective authority-link file set")
    need(sum(x["token_count"] for x in links.values())
         <= decision["max_accepted_tokens"], "Prospective token ceiling exceeded")
    return dict(head=previous, store_sha256=expected_store_sha256,
                links=links, authority="PROSPECTIVE_USE_ONLY",
                accounting_delta_documents=0, accounting_delta_tokens=0)


def create_store(root, store_path, ledger_path, expected_ledger_head,
                 material_paths, expected_identities, *, created_at=None):
    """Create one immutable, complete log; existing paths are never overwritten."""
    root = Path(root).resolve()
    store = Path(store_path).resolve()
    need(store.is_relative_to(root) and not store.exists(),
         "Authority-link store already exists")
    decision, approval, manifest, verification = _materials(
        root, *material_paths, expected_identities)
    now = created_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    need(_time(now) >= _time(decision["decided_at"]),
         "Authority link cannot predate decision")
    events = []
    previous = None
    for file in decision["files"]:
        item = identity({"source_id": decision["source_id"],
                         "revision": decision["revision"],
                         "path": file["path"], "sha256": file["sha256"]})
        evidence, commit = _ledger_item(ledger_path, expected_ledger_head, item)
        fields = _fields(evidence, commit, decision, approval, manifest,
                         verification, file, expected_ledger_head)
        event = seal(dict(fields, link_created_at=now,
                          previous_link_identity=previous))
        events.append(event)
        previous = event["identity"]
    need(len(events) == decision["max_files"]
         and sum(x["token_count"] for x in events)
         <= decision["max_accepted_tokens"], "Prospective scope cap exceeded")
    store.parent.mkdir(parents=True, exist_ok=True)
    with store.open("xb") as output:
        output.write(b"".join(canonical(x) + b"\n" for x in events))
        output.flush()
        os.fsync(output.fileno())
    digest = file_digest(store)
    result = verify_store(root, store, digest, ledger_path, expected_ledger_head,
                          material_paths, expected_identities)
    return result


def require_linked_historical_training_use(root, item, store_path,
                                            expected_store_sha256, ledger_path,
                                            expected_ledger_head, material_paths,
                                            expected_identities):
    """Integrity readback only; no independent retention or use authority."""
    result = verify_store(root, store_path, expected_store_sha256, ledger_path,
                          expected_ledger_head, material_paths, expected_identities)
    need(item in result["links"], "Historical item has no prospective-use link")
    return result["links"][item]


def _corpus_counts(ledger_path):
    with sqlite3.connect(Path(ledger_path).resolve().as_uri() + "?mode=ro",
                         uri=True) as db:
        documents = db.execute("SELECT count(*) FROM item_evidence").fetchone()[0]
        tokens = db.execute(
            "SELECT sum(tokens) FROM totals WHERE kind='category'").fetchone()[0]
    return documents, tokens


def _retention_body(root, store_path, store_sha256, store_head, ledger_path,
                    ledger_sha256, ledger_head, material_paths,
                    expected_identities, created_at):
    root = Path(root).resolve()
    store = Path(store_path).resolve()
    ledger = Path(ledger_path).resolve()
    need(store.is_relative_to(root) and ledger.is_relative_to(root),
         "Retention inputs outside repository root")
    need(file_digest(ledger) == ledger_sha256,
         "Corpus ledger byte identity changed")
    result = verify_store(root, store, store_sha256, ledger, ledger_head,
                          material_paths, expected_identities)
    need(result["head"] == store_head, "Authority-link chained head mismatch")
    links = list(result["links"].values())
    need(links and len({x["item_identity"] for x in links}) == len(links),
         "Missing or duplicate authority links")
    source = links[0]["source_id"]
    revision = links[0]["revision"]
    need(all(x["source_id"] == source and x["revision"] == revision
             and x["accounting_delta_documents"] == 0
             and x["accounting_delta_tokens"] == 0 for x in links),
         "Mixed retention scope or accounting delta")
    documents, tokens = _corpus_counts(ledger)
    _time(created_at)
    return dict(schema=RETENTION_SCHEMA, authority="NONE",
                status="CANDIDATE_FOR_INDEPENDENT_RETENTION_ONLY",
                store_path=store.relative_to(root).as_posix(),
                store_sha256=store_sha256, store_head=store_head,
                link_count=len(links), linked_tokens=sum(x["token_count"] for x in links),
                source_id=source, revision=revision,
                decision_identity=expected_identities[0],
                human_approval_identity=expected_identities[1],
                evidence_manifest_identity=expected_identities[2],
                verification_identity=expected_identities[3],
                ledger_path=ledger.relative_to(root).as_posix(),
                ledger_sha256=ledger_sha256, ledger_head=ledger_head,
                corpus_documents=documents, corpus_train_tokens=tokens,
                accounting_delta_documents=0, accounting_delta_tokens=0,
                grants_training=False, grants_shards=False,
                grants_corpus_admission=False, created_at=created_at)


def create_retention_candidate(root, candidate_path, store_path, store_sha256,
                               store_head, ledger_path, ledger_sha256, ledger_head,
                               material_paths, expected_identities, *, created_at=None):
    """Create one zero-authority canonical candidate without replacing old bytes."""
    root = Path(root).resolve()
    candidate_path = Path(candidate_path).resolve()
    need(candidate_path.is_relative_to(root) and not candidate_path.exists(),
         "Retention candidate already exists or outside root")
    now = created_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = _retention_body(root, store_path, store_sha256, store_head,
                           ledger_path, ledger_sha256, ledger_head,
                           material_paths, expected_identities, now)
    candidate = seal(body)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    with candidate_path.open("xb") as output:
        output.write(canonical(candidate))
        output.flush()
        os.fsync(output.fileno())
    return candidate["identity"], file_digest(candidate_path)


def verify_retention_candidate(root, candidate_path, expected_identity,
                               expected_candidate_sha256, store_path,
                               expected_store_sha256, expected_store_head,
                               ledger_path, expected_ledger_sha256,
                               expected_ledger_head, material_paths,
                               expected_identities):
    """Check independently supplied identity pins; candidate remains zero authority."""
    need(all(type(value) is str and value.startswith("sha256:")
             and len(value) == 71 for value in
             (expected_identity, expected_candidate_sha256,
              expected_store_sha256, expected_store_head,
              expected_ledger_sha256, expected_ledger_head)),
         "Independent retention and ledger pins required")
    root = Path(root).resolve()
    path = Path(candidate_path).resolve()
    need(path.is_relative_to(root), "Retention candidate outside root")
    raw = path.read_bytes()
    need(file_digest(path) == expected_candidate_sha256,
         "Retention candidate byte pin mismatch")
    candidate = _read(path)
    need(canonical(candidate) == raw and candidate["identity"] == expected_identity,
         "Retention candidate identity mismatch")
    body = _retention_body(root, store_path, expected_store_sha256,
                           expected_store_head, ledger_path, expected_ledger_sha256,
                           expected_ledger_head, material_paths,
                           expected_identities, candidate["created_at"])
    need(set(candidate) == set(body) | {"identity"}
         and all(candidate.get(k) == value for k, value in body.items()),
         "Retention candidate scope or zero-authority status changed")
    return candidate


def authorize_existing_historical_items(root, item_identities, *, ledger_path,
        expected_ledger_sha256, expected_ledger_head, store_path,
        expected_store_sha256, expected_store_head, material_paths,
        expected_identities, retention_candidate, expected_retention_identity,
        expected_retention_sha256, channel_a_retained, channel_b_retained,
        production_manifest, expected_production_manifest_identity,
        expected_production_manifest_sha256, signature_path, timestamp_path,
        tsa_ca_file, expected_tsa_ca_sha256, trust_pin_path,
        expected_trust_pin_identity, trust_root,
        expected_source_id, expected_revision, openssl="openssl"):
    """Authorize prospective use of exact existing items, without corpus mutation.

    The caller supplies independent retention attestations and identity pins.
    This does not authorize a training run, export, shards, or new admission.
    """
    root = Path(root).resolve()

    def relative(path):
        resolved = Path(path).resolve()
        need(resolved.is_relative_to(root), "Training authority input outside root")
        return resolved.relative_to(root).as_posix()

    need(channel_a_retained is True and channel_b_retained is True,
         "Independent retention channels required")
    need(type(item_identities) is list and item_identities
         and len(item_identities) == len(set(item_identities)),
         "Unique historical item identities required")
    need(file_digest(ledger_path) == expected_ledger_sha256,
         "Corpus ledger byte pin mismatch")
    need(file_digest(production_manifest) == expected_production_manifest_sha256,
         "Production manifest byte pin mismatch")
    need(all(path is not None and Path(path).is_file() for path in
             (signature_path, timestamp_path, tsa_ca_file, trust_pin_path))
         and trust_root is not None
         and file_digest(tsa_ca_file) == expected_tsa_ca_sha256,
         "Production signature, timestamp, or pinned TSA CA missing")
    verify_retention_candidate(
        root, retention_candidate, expected_retention_identity,
        expected_retention_sha256, store_path, expected_store_sha256,
        expected_store_head, ledger_path, expected_ledger_sha256,
        expected_ledger_head, material_paths, expected_identities)
    store = verify_store(root, store_path, expected_store_sha256, ledger_path,
                         expected_ledger_head, material_paths, expected_identities)
    need(store["head"] == expected_store_head
         and store["accounting_delta_documents"] == 0
         and store["accounting_delta_tokens"] == 0
         and all(item in store["links"] for item in item_identities),
         "Historical item missing from retained authority-link store")

    manifest = _read(production_manifest)
    links = list(store["links"].values())
    scope = {(link["path"], link["raw_sha256"]) for link in links}
    need(manifest.get("mode") == "production"
         and manifest["identity"] == expected_production_manifest_identity
         and manifest["source"]["source_id"] == expected_source_id
         and manifest["source"]["revision"] == expected_revision
         and all(link["source_id"] == expected_source_id
                 and link["revision"] == expected_revision
                 and link["split"] == "train"
                 and link["accounting_delta_documents"] == 0
                 and link["accounting_delta_tokens"] == 0 for link in links),
         "Production evidence source or zero-accounting scope mismatch")
    need(len(manifest["files"]) == len(links)
         and {(row["path"], row["sha256"]) for row in manifest["files"]} == scope
         and manifest["decision_scope"]["artifact_path"] == relative(material_paths[0])
         and len(manifest["decision_scope"]["files"]) == len(links)
         and {(row["path"], row["sha256"])
              for row in manifest["decision_scope"]["files"]} == scope,
         "Production evidence differs from exact historical decision scope")
    artifacts = {(row["path"], row["sha256"])
                 for row in manifest["artifacts"]}
    need(all((relative(path), file_digest(path)) in artifacts for path in
             (store_path, retention_candidate, material_paths[0])),
         "Production evidence does not bind retained links and decision")

    result = verify_bundle(
        root, production_manifest, expected_production_manifest_identity,
        signature_path=signature_path, timestamp_path=timestamp_path,
        tsa_ca_file=tsa_ca_file, trust_pin_path=trust_pin_path,
        expected_trust_pin_identity=expected_trust_pin_identity,
        trust_root=trust_root, openssl=openssl)
    need(result["signature"] == "VERIFIED"
         and result["timestamp"] == "VERIFIED"
         and result["authority"] == "NONE"
         and result["files"] == len(links)
         and result["trust_pin_identity"] == expected_trust_pin_identity,
         "Production signature, timestamp, or trust pin not verified")
    need(file_digest(ledger_path) == expected_ledger_sha256
         and file_digest(store_path) == expected_store_sha256
         and file_digest(retention_candidate) == expected_retention_sha256
         and file_digest(production_manifest) == expected_production_manifest_sha256
         and file_digest(tsa_ca_file) == expected_tsa_ca_sha256,
         "Training authority inputs changed during verification")
    return dict(status="TRAINING_AUTHORIZED", authority="EXISTING_ITEM_USE_ONLY",
                item_identities=item_identities, source_id=expected_source_id,
                revision=expected_revision, store_head=expected_store_head,
                production_manifest_identity=expected_production_manifest_identity,
                evidence_authority=result["authority"],
                accounting_delta_documents=0, accounting_delta_tokens=0,
                grants_training_run=False, grants_shards=False,
                grants_corpus_admission=False)
