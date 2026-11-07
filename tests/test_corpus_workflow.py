"""The coordinator reuses existing gates and never grants review data itself."""

import base64
import hashlib
import json
import shutil
import sqlite3

import pytest

from tests.test_corpus_gate import receipt
from tests.test_production_acquisition import (
    BASE,
    approval_kwargs,
    fixture_plan,
    fixture_reference,
    prior_inventory,
)
from tests.test_source_prepare import command as git_command
from tests.test_source_prepare import fixture as source_prepare_fixture
from training import corpus_workflow as workflow
from training.corpus_gate import BUNDLE
from training.cumulative_accounting import GENESIS, CorpusLedger
from training.license_evidence import MIT_TERMS
from training.production_corpus import PipelineError, atomic_json, canonical, file_digest
from training.production_spec import TOKENIZER, seal
from training.source_prepare import POLICY as CACHE_POLICY
from training.token_shards import TokenShardStream
from yug.tokenizer.bpe import ByteLevelBpeTokenizer


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)
    return path


@pytest.fixture
def local_git_fixture(tmp_path):
    return source_prepare_fixture.__wrapped__(tmp_path)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path
    (root / "artifacts").mkdir()
    write(root / "registry.json", {"sources": []})
    write(root / "spec.json", {"categories": []})
    (root / "references.jsonl").write_bytes(b"")
    seeds = [dict(locator=f"https://github.com/team/repo-{n}", branch="main",
                  category="code", subcategory="python") for n in range(2)]
    batch = write(root / "batch.json", dict(seeds=seeds, registry="registry.json",
                                            spec="spec.json", references="references.jsonl"))
    output = root / "artifacts/workflow"
    calls = []

    def discovery(registry, spec, directory, **kwargs):
        calls.append("discovery")
        directory.mkdir(parents=True)
        sources = [dict(source_id=f"gh-team-repo-{n}", revision=f"{n + 1:040x}")
                   for n in range(2)]
        write(directory / "catalog.json", seal(dict(sources=sources)))
        decisions = [dict(seed, source_id=source["source_id"],
                          revision=source["revision"], status="ACCEPTED")
                     for seed, source in zip(seeds, sources, strict=True)]
        write(directory / "decisions.json", dict(sources=decisions))
        write(directory / "complete.json", seal(dict(request_identity="test",
                                                       artifacts={})))
        return {}

    def source_prepare(registry, spec, directory, **kwargs):
        calls.append("prepare")
        assert kwargs["fetcher"] is workflow._offline_fetch
        rows = [dict(source_id=f"gh-team-repo-{n}", status=status,
                     result=dict(counts={"REVIEW": 1}) if status == "CACHED" else {},
                     error="test fetch failure" if status == "FAILED" else None)
                for n, status in enumerate(("CACHED", "FAILED"))]
        write(directory / "status.json", dict(sources=rows))
        return rows

    def clearance(cache, directory, **kwargs):
        calls.append("clearance")
        assert kwargs["review_only"] is True
        return dict(complete=True, counts={"EVIDENCED": 0},
                    per_source={"gh-team-repo-0": {"REVIEW": 1}})

    def proposals(cache, catalog, evidence, references, directory, **kwargs):
        calls.append("proposals")
        return {}

    monkeypatch.setattr(workflow.source_discovery, "build", discovery)
    monkeypatch.setattr(workflow.source_prepare, "run", source_prepare)
    monkeypatch.setattr(workflow.clearance_evidence, "build", clearance)
    monkeypatch.setattr(workflow.source_grant_proposals, "compile_proposals", proposals)
    return root, batch, output, calls


def test_prepare_reuses_modules_and_isolates_failed_cache_source(prepared):
    root, batch, output, calls = prepared
    report = workflow.prepare(batch, output, root=root)
    assert calls == ["discovery", "prepare", "clearance", "proposals"]
    assert report["phase"] == "HUMAN_REVIEW" and report["training_allowed"] is False
    assert [row["state"] for row in report["sources"]] == ["HUMAN_REVIEW", "BLOCKED"]
    assert all(row["duplicate_state"] == "NOT_RUN" for row in report["sources"])
    assert not (output / "admission").exists()
    assert not (output / "production").exists()


def test_changed_batch_cannot_resume(prepared):
    root, batch, output, _ = prepared
    workflow.prepare(batch, output, root=root)
    value = json.loads(batch.read_bytes())
    value["seeds"][0]["branch"] = "other"
    write(batch, value)
    with pytest.raises(Exception, match="changed"):
        workflow.prepare(batch, output, root=root, resume=True)


def test_real_discovery_rejects_source_and_verifies_offline_resume(tmp_path,
                                                                  monkeypatch):
    root = tmp_path
    (root / "artifacts").mkdir()
    write(root / "registry.json", seal(dict(schema="yug-production-source-registry-v1",
                                             sources=[])))
    write(root / "spec.json", seal(dict(categories=[dict(id="code",
        subcategories=["python"])], filters=dict(deny_paths=[]))))
    (root / "references.jsonl").write_bytes(b"")
    seed = dict(locator="https://github.com/team/rejected", branch="main",
                category="code", subcategory="python")
    batch = write(root / "batch.json", dict(seeds=[seed], registry="registry.json",
        spec="spec.json", references="references.jsonl"))
    calls = []

    def fetch(url):
        calls.append(url)
        if "/git/ref/heads/" in url:
            return dict(ref="refs/heads/main", object=dict(type="commit", sha="a" * 40))
        payload = b"GPL example"
        return dict(license=dict(spdx_id="GPL-3.0"), encoding="base64", type="file",
                    content=base64.b64encode(payload).decode(), size=len(payload),
                    path="LICENSE", sha=hashlib.sha1(
                        f"blob {len(payload)}\0".encode() + payload).hexdigest(),
                    html_url=seed["locator"] + "/blob/" + "a" * 40 + "/LICENSE")

    monkeypatch.setattr(workflow.source_discovery, "fetch_json", fetch)
    output = root / "artifacts/workflow"
    first = workflow.prepare(batch, output, root=root, fetch=True)
    assert first["sources"][0]["state"] == "BLOCKED" and len(calls) == 2
    assert workflow.verify(output, root=root) == first
    assert workflow.prepare(batch, output, root=root, resume=True) == first
    assert len(calls) == 2
    catalog = output / "discovery/catalog.json"
    catalog.write_bytes(catalog.read_bytes() + b" ")
    with pytest.raises(Exception, match="Discovery artifact changed"):
        workflow.verify(output, root=root)


@pytest.mark.parametrize("noassertion", [False, True])
def test_local_git_review_cache_verifies_without_grant(tmp_path, local_git_fixture,
                                                      monkeypatch, noassertion):
    """Exercise the real review stages with a local fixture Git repository."""
    source, spec, fetch_git, calls = local_git_fixture
    if noassertion:
        license_text = ("MIT License\nCopyright (c) 2026 Fixture\n\n" +
                        MIT_TERMS).encode()
        upstream = tmp_path / "upstream"
        (upstream / "LICENSE").write_bytes(license_text)
        git_command("add", "LICENSE", cwd=upstream)
        git_command("-c", "core.hooksPath="+str(tmp_path/"no-hooks"),
                    "commit", "-m", "license fixture", cwd=upstream)
        source["revision"] = git_command("rev-parse", "HEAD", cwd=upstream)
        tree_sha = git_command("rev-parse", "HEAD^{tree}", cwd=upstream)
        blob_sha = hashlib.sha1(f"blob {len(license_text)}\0".encode() +
                                license_text).hexdigest()
    root = tmp_path
    (root / "artifacts").mkdir()
    specifications = root / "training/specifications"
    specifications.mkdir(parents=True)
    registry = write(specifications / "source_registry_v1.json",
                     seal(dict(schema="yug-production-source-registry-v1", sources=[])))
    for entry in spec["evidence"]:
        target = root / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workflow.ROOT / entry["path"], target)
    spec_path = write(specifications / "production_corpus_v1.json",
                      seal(dict(spec, registry_identity=json.loads(
                          registry.read_bytes())["identity"])))
    references = root / "references.jsonl"
    references.write_bytes(canonical(fixture_reference(
        "Separate held-out observations of planetary astronomy.",
        provenance={"source_id": "unrelated-fixture"})) + b"\n")
    seed = dict(locator=source["locator"], branch="main", category="code",
                subcategory="python")
    batch = write(root / "batch.json", dict(seeds=[seed],
        registry=registry.relative_to(root).as_posix(),
        spec=spec_path.relative_to(root).as_posix(),
        references=references.relative_to(root).as_posix()))
    license_bytes = license_text if noassertion else b"Fixture MIT license evidence\n"

    def metadata(url):
        if "/git/ref/heads/" in url:
            return dict(ref="refs/heads/main",
                        object=dict(type="commit", sha=source["revision"]))
        if noassertion and "/git/commits/" in url:
            return dict(sha=source["revision"], tree=dict(sha=tree_sha))
        if noassertion and "/git/trees/" in url:
            return dict(sha=tree_sha, truncated=False, tree=[
                dict(path="LICENSE", type="blob", mode="100644", sha=blob_sha)])
        if noassertion and "/git/blobs/" in url:
            return dict(sha=blob_sha, encoding="base64", size=len(license_bytes),
                        content=base64.b64encode(license_bytes).decode())
        return dict(license=dict(spdx_id="NOASSERTION" if noassertion else "MIT"),
                    encoding="base64", type="file",
                    content=base64.b64encode(license_bytes).decode(),
                    size=len(license_bytes), path="LICENSE",
                    sha=hashlib.sha1(f"blob {len(license_bytes)}\0".encode()
                                     + license_bytes).hexdigest(),
                    html_url=source["locator"] + "/blob/" + source["revision"]
                    + "/LICENSE")

    monkeypatch.setattr(workflow.source_discovery, "fetch_json", metadata)
    monkeypatch.setattr(workflow.source_prepare, "fetch", fetch_git)
    output = root / "artifacts/workflow"
    report = workflow.prepare(batch, output, root=root, fetch=True)
    assert report["phase"] == "HUMAN_REVIEW"
    assert report["training_allowed"] is False
    assert report["sources"][0]["state"] == "HUMAN_REVIEW"
    assert report["sources"][0]["file_count"] > 0
    if noassertion:
        assert (output / "discovery/source-00/canonical_evidence.json").is_file()
        catalog = json.loads((output / "discovery/catalog.json").read_bytes())
        assert catalog["sources"][0]["license_id"] == "MIT"
    assert len(calls) == 1
    assert (output / "proposals/complete.json").is_file()
    assert workflow.verify(output, root=root) == report
    assert workflow.prepare(batch, output, root=root, resume=True) == report
    assert len(calls) == 1
    assert not (output / "admission").exists()
    assert not (output / "production").exists()


def test_registered_source_is_not_prepared_or_admitted(tmp_path, monkeypatch):
    root = tmp_path
    (root / "artifacts").mkdir()
    locator = "https://github.com/team/existing"
    write(root / "registry.json", seal(dict(schema="yug-production-source-registry-v1",
        sources=[dict(source_id="gh-team-existing", locator=locator)])))
    write(root / "spec.json", seal(dict(categories=[dict(id="code",
        subcategories=["python"])], filters=dict(deny_paths=[]))))
    (root / "references.jsonl").write_bytes(b"")
    seed = dict(locator=locator, branch="main", category="code", subcategory="python")
    batch = write(root / "batch.json", dict(seeds=[seed], registry="registry.json",
        spec="spec.json", references="references.jsonl"))
    monkeypatch.setattr(workflow.source_discovery, "fetch_json",
                        lambda _: pytest.fail("duplicate must not fetch"))
    output = root / "artifacts/workflow"
    report = workflow.prepare(batch, output, root=root)
    assert report["sources"][0]["state"] == "BLOCKED"
    assert report["sources"][0]["file_count"] == 0
    assert not (output / "admission").exists()
    assert not (output / "production").exists()


def approved(root, output, *, allow=False, production_pin=None):
    names = workflow.REQUIRED_FILES
    files = {}
    for name in names:
        path = root / f"{name}.json"
        write(path, {"name": name})
        files[name] = dict(path=path.name, sha256=file_digest(path))
    pins = {name: "sha256:" + "1" * 64 for name in workflow.REQUIRED_PINS}
    pins["approved_plan_sha256"] = files["approved_plan"]["sha256"]
    pins["approved_ownership_approval_sha256"] = files["ownership_approval"]["sha256"]
    pins["approved_governance_sha256"] = files["governance_log"]["sha256"]
    if production_pin is not None:
        pins["approved_production_plan_sha256"] = production_pin
    path = write(root / "approved-inputs.json", dict(schema=workflow.APPROVAL_POLICY,
        files=files, pins=pins, intended_use="commercial_training", target_tokens=100,
        execute_production=allow))
    return path, file_digest(path)


def compiled_plan(monkeypatch, output):
    def compile_inventory(spec, registry, root, directory, plan):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "plan.jsonl"
        path.write_bytes(plan.read_bytes())
        return path, dict(scheduled=1, plan_sha256=file_digest(path))

    monkeypatch.setattr(workflow.production_build, "compile_inventory", compile_inventory)


def test_missing_or_wrong_external_pin_fails_closed(prepared, monkeypatch):
    root, batch, output, _ = prepared
    workflow.prepare(batch, output, root=root)
    monkeypatch.setattr(workflow, "verify", lambda *a, **k: {"identity": "test"})
    path, identity = approved(root, output)
    with pytest.raises(Exception, match="changed"):
        workflow.resume(output, path, "sha256:" + "0" * 64, root=root)
    path.unlink()
    with pytest.raises(PipelineError):
        workflow.resume(output, path, identity, root=root)
    assert not (output / "admission").exists()


def test_preflight_stops_before_production_and_requires_candidate_match(prepared, monkeypatch):
    root, batch, output, _ = prepared
    workflow.prepare(batch, output, root=root)
    monkeypatch.setattr(workflow, "verify", lambda *a, **k: {"identity": "test"})
    calls = []

    def admission(*args, **kwargs):
        calls.append("admission")
        write(output / "admission/complete.json", {"matched": 1})
        (output / "admission/plan.jsonl").write_bytes(b"approved file\n")
        return dict(cleared_cache_matches=1,
                    per_source={"gh-team-repo-0": {"CLEARED": 1}})

    monkeypatch.setattr(workflow.source_admission, "build", admission)
    compiled_plan(monkeypatch, output)
    monkeypatch.setattr(workflow.production_build, "execute",
                        lambda _: pytest.fail("production must not run during preflight"))
    path, identity = approved(root, output)
    report = workflow.resume(output, path, identity, root=root)
    assert calls == ["admission"] and report["phase"] == "PREFLIGHT"
    assert report["training_allowed"] is False
    assert report["sources"][0]["state"] == "READY"
    assert report["sources"][1]["state"] == "BLOCKED"

    monkeypatch.setattr(workflow.source_admission, "build",
                        lambda *a, **k: dict(cleared_cache_matches=0))
    with pytest.raises(Exception, match="No exact candidate grant"):
        workflow.resume(output, path, identity, root=root)


def test_production_handoff_requires_second_pinned_plan(prepared, monkeypatch):
    root, batch, output, _ = prepared
    workflow.prepare(batch, output, root=root)
    monkeypatch.setattr(workflow, "verify", lambda *a, **k: {"identity": "test"})
    plan = output / "admission/plan.jsonl"

    def admission(*args, **kwargs):
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_bytes(b"approved file\n")
        write(output / "admission/complete.json", {"matched": 1})
        return dict(cleared_cache_matches=1,
                    per_source={"gh-team-repo-0": {"CLEARED": 1}})

    monkeypatch.setattr(workflow.source_admission, "build", admission)
    compiled_plan(monkeypatch, output)
    monkeypatch.setattr(workflow.production_build, "execute",
                        lambda _: pytest.fail("unapproved production call"))
    path, identity = approved(root, output, allow=True)
    with pytest.raises(Exception, match="SHA256 pin"):
        workflow.resume(output, path, identity, root=root, execute=True)
    path, identity = approved(root, output, allow=True,
                              production_pin="sha256:" + "0" * 64)
    with pytest.raises(Exception, match="own external pin"):
        workflow.resume(output, path, identity, root=root, execute=True)


def test_authorized_handoff_uses_existing_runner_and_resume(prepared, monkeypatch):
    root, batch, output, _ = prepared
    workflow.prepare(batch, output, root=root)
    monkeypatch.setattr(workflow, "verify", lambda *a, **k: {"identity": "test"})
    plan = output / "admission/plan.jsonl"

    def admission(*args, **kwargs):
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_bytes(b"approved file\n")
        write(output / "admission/complete.json", {"matched": 1})
        return dict(cleared_cache_matches=1,
                    per_source={"gh-team-repo-0": {"CLEARED": 1}})

    monkeypatch.setattr(workflow.source_admission, "build", admission)
    compiled_plan(monkeypatch, output)
    seen = []

    def production(args):
        seen.append(args)
        directory = output / "production"
        directory.mkdir(exist_ok=True)
        with sqlite3.connect(directory / "state.sqlite") as db:
            db.execute("CREATE TABLE IF NOT EXISTS items("
                       "record TEXT,token_count INTEGER,stage TEXT)")
            db.execute("DELETE FROM items")
            db.execute("INSERT INTO items VALUES (?,?,?)",
                       (json.dumps({"source_id": "gh-team-repo-0"}), 7, "ACCEPTED"))
        write(directory / "runner/progress.json", {"state": "COMPLETE"})
        return 0

    monkeypatch.setattr(workflow.production_build, "execute", production)
    admission()
    path, identity = approved(root, output, allow=True, production_pin=file_digest(plan))
    first = workflow.resume(output, path, identity, root=root, execute=True)
    second = workflow.resume(output, path, identity, root=root, execute=True)
    assert len(seen) == 2 and seen[0].resume is False and seen[1].resume is True
    assert all(args.intended_use == "commercial_training" for args in seen)
    assert first["sources"][0]["state"] == second["sources"][0]["state"] == "ADMITTED"
    assert second["sources"][0]["token_estimate"] == 7


def test_approved_input_tamper_rejected(prepared):
    root, _, output, _ = prepared
    path, identity = approved(root, output)
    workflow._approved(root, path, identity)
    write(root / "approved_registry.json", {"changed": True})
    with pytest.raises(Exception, match="byte pin mismatch"):
        workflow._approved(root, path, identity)


def _synthetic_resume_fixture(root):
    """Join existing production fixtures to a sealed review cache at resume."""
    spec, registry, plan, records = fixture_plan(root, texts=[BASE, BASE])
    reg_path = write(root / "registry.json", registry)
    spec_path = write(root / "spec.json", spec)
    references = root / "references.jsonl"
    references.write_bytes(canonical(fixture_reference(
        "Unrelated observations of planetary spectroscopy and stellar spectra.",
        provenance={"source": "synthetic holdout"})) + b"\n")
    batch = write(root / "batch.json", dict(seeds=[], registry=reg_path.name,
                                            spec=spec_path.name,
                                            references=references.name))
    output = root / "artifacts/workflow"
    output.mkdir(parents=True)
    request, _ = workflow._request(root, json.loads(batch.read_bytes()))
    write(output / "batch.json", json.loads(batch.read_bytes()))
    write(output / "request.json", request)

    catalog = write(output / "discovery/catalog.json", seal(dict(
        schema="yug-production-source-registry-v1", sources=registry["sources"])))
    decisions = write(output / "discovery/decisions.json", dict(sources=[
        dict(source_id=s["source_id"], revision=s["revision"], status="ACCEPTED")
        for s in registry["sources"]]))
    discovery_request = write(output / "discovery/request.json", seal(dict(
        policy="synthetic-review-fixture", batch=request["identity"])))
    write(output / "discovery/complete.json", seal(dict(
        request_identity=json.loads(discovery_request.read_bytes())["identity"],
        artifacts={p.name: file_digest(p) for p in (catalog, decisions)})))

    statuses = []
    for source, record in zip(registry["sources"], records, strict=True):
        directory = output / "cache" / source["source_id"]
        blob = directory / "files/0"
        blob.parent.mkdir(parents=True)
        blob.write_bytes((root / record["input_path"]).read_bytes())
        row = dict(source_id=source["source_id"], locator=source["locator"],
                   revision=source["revision"], path=record["path"],
                   bytes=blob.stat().st_size, sha256=file_digest(blob),
                   git_mode="100644", local_blob="files/0",
                   screening="UNRESOLVED", category=source["category"],
                   subcategory=source["subcategory"])
        inventory = directory / "inventory.jsonl"
        inventory.write_bytes(canonical(row) + b"\n")
        cache_request = write(directory / "request.json", seal(dict(
            source=source, filters=spec["filters"])))
        done = write(directory / "complete.json", seal(dict(
            request_identity=json.loads(cache_request.read_bytes())["identity"],
            revision=source["revision"], inventory_sha256=file_digest(inventory),
            counts={"UNRESOLVED": 1}, training_allowed=False)))
        statuses.append(dict(source_id=source["source_id"], status="CACHED",
                             classification="CACHE_READY",
                             result=json.loads(done.read_bytes())))
    status = write(output / "cache/status.json", seal(dict(
        policy=CACHE_POLICY, registry_identity=json.loads(catalog.read_bytes())["identity"],
        spec_identity=spec["identity"], dry_run=False,
        classification_counts={"CACHE_READY": len(statuses)},
        sources=statuses, training_allowed=False)))
    write(output / "report.json", seal(dict(
        policy=workflow.POLICY, request_identity=request["identity"],
        phase="HUMAN_REVIEW", training_allowed=False,
        sources=workflow._source_rows(json.loads(decisions.read_bytes()), statuses),
        artifacts={"discovery/complete.json": file_digest(output / "discovery/complete.json"),
                   "discovery/catalog.json": file_digest(catalog),
                   "cache/status.json": file_digest(status)})))
    return spec, registry, plan, records, output


def _approved_transport(root, plan, references, ledger,
                        prior, tokenizer, ownership, governance, decisions,
                        *, production_pin=None, execute=False):
    paths = dict(approved_registry=root / "registry.json",
                 approved_spec=root / "spec.json", approved_plan=plan,
                 references=references, ownership_approval=ownership,
                 governance_log=governance, prior_production=prior,
                 corpus_ledger=ledger, decision_receipts=decisions,
                 tokenizer=tokenizer)
    files = {name: dict(path=path.relative_to(root).as_posix(),
                        sha256=file_digest(path)) for name, path in paths.items()}
    pins = dict(approved_plan_sha256=files["approved_plan"]["sha256"],
                approved_ownership_approval_sha256=files["ownership_approval"]["sha256"],
                approved_governance_sha256=files["governance_log"]["sha256"],
                approved_corpus_head=GENESIS,
                approved_historical_import_identity=GENESIS,
                approved_historical_manifest_identity=GENESIS,
                approved_historical_manifest_sha256=GENESIS)
    if production_pin is not None:
        pins["approved_production_plan_sha256"] = production_pin
    transport = dict(schema=workflow.APPROVAL_POLICY, files=files, pins=pins,
                     intended_use="commercial_training", target_tokens=100000,
                     execute_production=execute)
    name = "approved-execute.json" if execute else "approved-preflight.json"
    path = write(root / name, transport)
    return path, file_digest(path)


def test_uninterrupted_approved_resume_to_production_export_and_readback(tmp_path):
    root = tmp_path
    spec, registry, plan, records, output = _synthetic_resume_fixture(root)
    assert workflow.verify(output, root=root)["phase"] == "HUMAN_REVIEW"
    references = root / "references.jsonl"
    prior = prior_inventory(root)
    tokenizer = root / "tokenizer.json"
    shutil.copyfile(workflow.ROOT / (
        "training/resources/phase1/bpe_fitter_optimization_gate_v1/"
        "comparison/run-4096-0/tokenizer.json"), tokenizer)
    ledger = root / "corpus-ledger.sqlite"
    with CorpusLedger(ledger, root, GENESIS):
        pass
    ledger_before = file_digest(ledger)
    placeholder = write(root / "preflight-only.json", {"fixture": "not an approval"})
    preflight_path, preflight_sha = _approved_transport(
        root, plan, references, ledger, prior, tokenizer,
        placeholder, placeholder, placeholder)
    with pytest.raises(Exception, match="Approved inputs file changed"):
        workflow.resume(output, preflight_path, "sha256:" + "0" * 64, root=root)
    assert not (output / "admission").exists()
    assert file_digest(ledger) == ledger_before
    with pytest.raises(Exception, match="Missing approved input file"):
        missing = write(root / "missing-approval.json", dict(
            schema=workflow.APPROVAL_POLICY, files={}, pins={}))
        workflow.resume(output, missing, file_digest(missing), root=root)
    assert not (output / "admission").exists()
    assert file_digest(ledger) == ledger_before

    preflight = workflow.resume(output, preflight_path, preflight_sha, root=root)
    assert preflight["phase"] == "PREFLIGHT" and preflight["training_allowed"] is False
    compiled = output / "production/runner/plan.jsonl"
    compiled_sha = file_digest(compiled)
    admitted = [json.loads(line) for line in
                (output / "admission/plan.jsonl").read_bytes().splitlines()]
    sources = {s["source_id"]: s for s in registry["sources"]}
    decisions = write(root / "decisions.json", seal(dict(
        schema=BUNDLE, receipts=[receipt(r, sources[r["source_id"]], spec, root)
                                  for r in admitted])))
    authority = approval_kwargs(root, compiled_sha)
    production_path, production_sha = _approved_transport(
        root, plan, references, ledger, prior, tokenizer,
        authority["ownership_approval"], authority["governance_log"], decisions,
        production_pin=compiled_sha, execute=True)
    assert file_digest(ledger) == json.loads(production_path.read_bytes())[
        "files"]["corpus_ledger"]["sha256"]
    report = workflow.resume(output, production_path, production_sha,
                             root=root, execute=True)
    assert report["phase"] == "PRODUCTION" and report["training_allowed"] is False
    assert report["scheduled_files"] == 2
    progress = json.loads((output / "production/runner/progress.json").read_bytes())
    assert progress["state"] == "COMPLETE"
    assert progress["network_files_this_process"] == 0
    assert progress["stages"]["ACCEPTED"] == 1
    assert progress["stages"]["DUPLICATE"] == 1
    export = output / "production" / "exports"
    destinations = list(export.glob("tokens-*"))
    assert len(destinations) == 1
    assert (destinations[0] / "prepared.json").is_file()
    assert (destinations[0] / "admission.jsonl").is_file()
    notice = json.loads((destinations[0] / "notice_attribution.json").read_bytes())
    assert notice["accepted_items"] == 1
    assert len(notice["entries"]) == 1
    assert notice["entries"][0]["attribution"] == "Fixture"
    assert notice["entries"][0]["license_id"] == "MIT"
    shards = destinations[0] / "shards"
    manifest = json.loads((shards / "manifest.json").read_bytes())
    assert manifest["manifest_identity"] == progress["shard_manifest_identity"]
    stream = TokenShardStream(shards, split="train",
                              tokenizer_identity=TOKENIZER,
                              prepared_corpus_identity=manifest["prepared_corpus_identity"])
    expected = ByteLevelBpeTokenizer.load(tokenizer).encode(BASE)
    assert list(stream) == expected
    assert list(stream) == expected
    with sqlite3.connect(ledger) as db:
        evidence = [json.loads(row[0]) for row in
                    db.execute("SELECT evidence FROM item_evidence")]
        assert len(evidence) == 1
        assert evidence[0]["tokens"] == len(expected)
        assert evidence[0]["record"]["sha256"] == records[0]["sha256"]
    assert progress["accepted_clean_train_tokens"] == len(expected)
