"""Connect existing Git-source review and approved production stages.

``prepare BATCH OUTPUT --fetch`` accepts JSON with ``seeds`` (explicit GitHub
seed objects), and repository-relative ``registry``, ``spec``, ``references``
paths. Omit ``--fetch`` for offline replay with ``--resume``. Preparation
publishes review outputs and stops; it never grants rights or use.

``resume OUTPUT --approved-inputs FILE --expected-approved-inputs-sha256 HASH``
accepts an externally pinned JSON transport with schema
``yug-corpus-workflow-approved-inputs-v1``. Its ``files`` map contains each
REQUIRED_FILES name mapped to ``{path, sha256}``; ``pins`` contains REQUIRED_PINS.
It also specifies ``intended_use: commercial_training`` and ``target_tokens``.
The first resume runs admission/preflight only and reports the compiled plan
hash. Separate owner approval of that hash adds
``pins.approved_production_plan_sha256`` and ``execute_production: true`` to a
new externally pinned transport. Only then may ``--execute-production`` call
the existing production runner. The transport is not an approval by itself.
``verify OUTPUT`` checks retained workflow identities without admission.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from training import (
    clearance_evidence,
    production_build,
    source_admission,
    source_discovery,
    source_grant_proposals,
    source_prepare,
)
from training.production_corpus import atomic_json, file_digest
from training.production_spec import need, seal, verify_seal

ROOT = Path(__file__).resolve().parents[1]
POLICY = "yug-corpus-workflow-v1"
APPROVAL_POLICY = "yug-corpus-workflow-approved-inputs-v1"
REQUIRED_FILES = ("approved_registry", "approved_spec", "approved_plan",
                  "references", "ownership_approval", "governance_log",
                  "prior_production", "corpus_ledger", "decision_receipts",
                  "tokenizer")
REQUIRED_PINS = ("approved_plan_sha256", "approved_ownership_approval_sha256",
                 "approved_governance_sha256", "approved_corpus_head",
                 "approved_historical_import_identity",
                 "approved_historical_manifest_identity",
                 "approved_historical_manifest_sha256")


def _json(path):
    return json.loads(Path(path).read_bytes())


def _inside(root, path):
    result = (root / path).resolve()
    need(result.is_relative_to(root), "Workflow path outside repository")
    return result


def _sha(value):
    need(isinstance(value, str) and len(value) == 71
         and value.startswith("sha256:")
         and all(ch in "0123456789abcdef" for ch in value[7:]),
         "Invalid SHA256 pin")
    return value


def _offline_fetch(*_):
    raise RuntimeError("Git cache missing; explicit --fetch required")


def _request(root, batch):
    need(isinstance(batch, dict) and set(batch) == {"seeds", "registry", "spec", "references"},
         "Batch must name seeds, registry, spec, and references")
    need(isinstance(batch["seeds"], list), "Explicit seed list required")
    files = {name: _inside(root, batch[name]) for name in ("registry", "spec", "references")}
    need(all(path.is_file() for path in files.values()), "Missing batch input")
    return seal(dict(policy=POLICY, seeds=batch["seeds"],
                     inputs={name: file_digest(path) for name, path in files.items()})), files


def _workflow(root, output):
    root = Path(root).resolve()
    output = _inside(root, output)
    need(output.is_relative_to(root / "artifacts") and output != root / "artifacts",
         "Workflow output must be under artifacts")
    return root, output


def _check_request(root, output):
    request = _json(output / "request.json")
    verify_seal(request)
    need(request["policy"] == POLICY, "Wrong workflow policy")
    for name, expected in request["inputs"].items():
        batch = _json(output / "batch.json")
        need(file_digest(_inside(root, batch[name])) == expected,
             "Workflow input changed")
    saved, _ = _request(root, _json(output / "batch.json"))
    need(saved == request, "Workflow request changed")
    return request


def _source_rows(discovery, prepared, clearance=None, admitted=None):
    statuses = {row["source_id"]: row for row in prepared}
    evidence = (clearance or {}).get("per_source", {})
    allowed = (admitted or {}).get("per_source", {})
    rows = []
    for decision in discovery["sources"]:
        sid = decision.get("source_id") or ("gh-" + decision["locator"].split("github.com/")[-1]
                                         .lower().replace("/", "-"))
        cache = statuses.get(sid, {})
        counts = cache.get("result", {}).get("counts", {})
        blocked = decision["status"] in ("REJECTED", "REVIEW") or cache.get("status") == "FAILED"
        cleared = allowed.get(sid, {}).get("CLEARED", 0)
        state = "BLOCKED" if blocked else ("READY" if cleared else "HUMAN_REVIEW")
        rows.append(dict(source_id=sid, revision=decision.get("revision"),
                         file_count=sum(counts.values()),
                         stage=cache.get("status", decision["status"]),
                         rights_state="EXACT_GRANT_MATCHED" if cleared else "HUMAN_REVIEW",
                         training_use_state="HUMAN_REVIEW", screening_state=evidence.get(sid, {}),
                         duplicate_state="NOT_RUN", token_estimate=None, state=state,
                         reason=cache.get("error") or decision.get("reason")
                         or ("Exact rights/use approval required" if not cleared else None)))
    return rows


def prepare(batch_path, output, *, root=ROOT, resume=False, fetch=False):
    root, output = _workflow(root, output)
    batch_path = _inside(root, batch_path)
    batch = _json(batch_path)
    request, paths = _request(root, batch)
    need(not batch_path.is_relative_to(output), "Batch input cannot be workflow output")
    if (output / "request.json").exists():
        need(resume, "Existing workflow requires --resume")
        need(_json(output / "request.json") == request and _json(output / "batch.json") == batch,
             "Workflow request changed")
        _check_request(root, output)
        if resume and (output / "report.json").is_file():
            return verify(output, root=root)
    else:
        need(not output.exists() or not any(output.iterdir()), "Unknown workflow output")
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "batch.json", batch)
        atomic_json(output / "request.json", request)

    discovery_dir = output / "discovery"
    source_discovery.build(paths["registry"], paths["spec"], discovery_dir,
                           seeds=batch["seeds"], root=root, resume=resume,
                           fetcher=source_discovery.fetch_json if fetch else None)
    catalog = _json(discovery_dir / "catalog.json")
    source_prepare.run(catalog, _json(paths["spec"]), output / "cache", resume=resume,
                       fetcher=source_prepare.fetch if fetch else _offline_fetch)
    status_path = output / "cache/status.json"
    need(status_path.is_file() or not catalog["sources"],
         "Source preparation did not publish a batch status")
    prepared = _json(status_path)["sources"] if status_path.is_file() else []
    cached = [row for row in prepared if row["status"] == "CACHED"]
    clearance = None
    if cached:
        clearance = clearance_evidence.build(output / "cache", output / "clearance",
            root=root, resume=resume, review_only=True,
            cache_registry=discovery_dir / "catalog.json", references=paths["references"])
        need(clearance["complete"] and clearance["counts"]["EVIDENCED"] == 0,
             "Review clearance incomplete or unexpectedly granted")
        source_grant_proposals.compile_proposals(output / "cache",
            discovery_dir / "catalog.json", output / "clearance", paths["references"],
            output / "proposals", root=root)
    report = seal(dict(policy=POLICY, request_identity=request["identity"],
                       phase="HUMAN_REVIEW", training_allowed=False,
                       sources=_source_rows(_json(discovery_dir / "decisions.json"),
                                            prepared, clearance),
                       artifacts={name: file_digest(output / name) for name in
                                  ("discovery/complete.json", "discovery/catalog.json",
                                   "cache/status.json") if (output / name).is_file()}))
    atomic_json(output / "report.json", report)
    return report


def verify(output, *, root=ROOT):
    root, output = _workflow(root, output)
    request = _check_request(root, output)
    discovery = _json(output / "discovery/complete.json")
    verify_seal(discovery)
    need(discovery["request_identity"] == _json(output / "discovery/request.json")["identity"],
         "Discovery request mismatch")
    for name, sha in discovery["artifacts"].items():
        need(file_digest(_inside(output / "discovery", name)) == sha,
             "Discovery artifact changed")
    catalog = _json(output / "discovery/catalog.json")
    verify_seal(catalog)
    status_path = output / "cache/status.json"
    need(status_path.is_file() or not catalog["sources"],
         "Source preparation status missing")
    status = _json(status_path) if status_path.is_file() else {"sources": []}
    if status_path.is_file():
        verify_seal(status)
        need(status["registry_identity"] == catalog["identity"], "Cache catalog mismatch")
    spec = _json(_inside(root, _json(output / "batch.json")["spec"]))
    need(not status_path.is_file() or status["spec_identity"] == spec["identity"],
         "Cache specification mismatch")
    caches = list(source_admission.cached_sources(output / "cache", catalog, spec, root))
    need(len(caches) == sum(row["status"] == "CACHED" for row in status["sources"]),
         "Cache status/receipts mismatch")
    for directory, _, inventory, _ in caches:
        with inventory.open("rb") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("local_blob"):
                    blob = _inside(directory, row["local_blob"])
                    need(blob.is_file() and blob.stat().st_size == row["bytes"]
                         and file_digest(blob) == row["sha256"],
                         "Cached source payload changed")
    if (output / "clearance/complete.json").exists():
        clearance_evidence.verify_bundle(output / "clearance", root)
        proposal_request = _json(output / "proposals/request.json")
        verify_seal(proposal_request)
        proposal = _json(output / "proposals/complete.json")
        verify_seal(proposal)
        need(proposal["request_identity"] == proposal_request["identity"],
             "Proposal request mismatch")
        need(file_digest(output / "proposals/proposals.jsonl") == proposal["proposals_sha256"],
             "Proposal payload changed")
    report = _json(output / "report.json")
    verify_seal(report)
    need(report["request_identity"] == request["identity"], "Workflow report mismatch")
    for name, sha in report["artifacts"].items():
        need(file_digest(_inside(output, name)) == sha, "Workflow artifact changed")
    if "approved_inputs_path" in report:
        need(file_digest(_inside(root, report["approved_inputs_path"]))
             == report["approved_inputs_sha256"], "Approved inputs changed")
    for key, name in (("admission_complete_sha256", "admission/complete.json"),
                      ("compiled_production_plan_sha256", "production/runner/plan.jsonl"),
                      ("production_progress_sha256", "production/runner/progress.json")):
        if key in report:
            need(file_digest(output / name) == report[key],
                 "Workflow stage output changed")
    return report


def _approved(root, path, expected_sha):
    path = _inside(root, path)
    need(file_digest(path) == _sha(expected_sha), "Approved inputs file changed")
    approval = _json(path)
    need(approval.get("schema") == APPROVAL_POLICY, "Wrong approved-inputs schema")
    files = approval.get("files", {})
    need(all(name in files for name in REQUIRED_FILES), "Missing approved input file")
    resolved = {}
    for name, ref in files.items():
        need(isinstance(ref, dict) and set(ref) == {"path", "sha256"},
             "Approved file reference must have path and SHA256")
        resolved[name] = _inside(root, ref["path"])
        need(file_digest(resolved[name]) == _sha(ref["sha256"]),
             "Approved input byte pin mismatch")
    pins = approval.get("pins", {})
    need(all(name in pins for name in REQUIRED_PINS), "Missing external production pin")
    for name in REQUIRED_PINS:
        _sha(pins[name])
    need(pins["approved_ownership_approval_sha256"] == files["ownership_approval"]["sha256"]
         and pins["approved_governance_sha256"] == files["governance_log"]["sha256"]
         and pins["approved_plan_sha256"] == files["approved_plan"]["sha256"],
         "Approved path/pin mismatch")
    need(approval.get("intended_use") == "commercial_training"
         and type(approval.get("target_tokens")) is int and approval["target_tokens"] > 0,
         "Explicit commercial-training use and positive target required")
    return approval, resolved, pins


def resume(output, approved_inputs, expected_sha, *, root=ROOT, execute=False):
    root, output = _workflow(root, output)
    previous = verify(output, root=root)
    approval, files, pins = _approved(root, approved_inputs, expected_sha)
    batch = _json(output / "batch.json")
    admission = source_admission.build(output / "cache", _inside(root, batch["registry"]),
        _inside(root, batch["spec"]), files["approved_registry"], files["approved_spec"],
        files["approved_plan"], files["references"], output / "admission", root=root,
        cache_registry=output / "discovery/catalog.json")
    need(admission["cleared_cache_matches"] > 0, "No exact candidate grant matched")
    compiled_plan, inventory = production_build.compile_inventory(
        _json(files["approved_spec"]), _json(files["approved_registry"]), root,
        output / "production/runner", output / "admission/plan.jsonl")
    need(inventory["scheduled"] > 0 and file_digest(compiled_plan) == inventory["plan_sha256"],
         "No verified production plan")
    discovered = _json(output / "discovery/decisions.json")
    prepared = _json(output / "cache/status.json")["sources"]
    rows = _source_rows(discovered, prepared, admitted=admission)
    approved_path = _inside(root, approved_inputs).relative_to(root).as_posix()
    admission_sha = file_digest(output / "admission/complete.json")
    if not execute:
        report = seal(dict(previous, phase="PREFLIGHT", training_allowed=False,
                           approved_inputs_sha256=expected_sha,
                           approved_inputs_path=approved_path,
                           admission_complete_sha256=admission_sha,
                           compiled_production_plan_sha256=inventory["plan_sha256"],
                           scheduled_files=inventory["scheduled"], sources=rows))
        atomic_json(output / "report.json", report)
        return report

    need(approval.get("execute_production") is True,
         "Externally pinned approval does not authorize workflow execution")
    approved_production_plan = _sha(pins.get("approved_production_plan_sha256"))
    need(inventory["plan_sha256"] == approved_production_plan,
         "Post-admission production plan needs its own external pin")
    args = argparse.Namespace(root=str(root), spec=files["approved_spec"],
        registry=files["approved_registry"], plan=output / "admission/plan.jsonl",
        directory=output / "production", approved_plan_sha256=approved_production_plan,
        ownership_approval=files["ownership_approval"],
        approved_ownership_approval_sha256=pins["approved_ownership_approval_sha256"],
        governance_log=files["governance_log"],
        approved_governance_sha256=pins["approved_governance_sha256"],
        references=files["references"], prior_production=files["prior_production"],
        corpus_ledger=files["corpus_ledger"], approved_corpus_head=pins["approved_corpus_head"],
        approved_historical_import_identity=pins["approved_historical_import_identity"],
        approved_historical_manifest_identity=pins["approved_historical_manifest_identity"],
        approved_historical_manifest_sha256=pins["approved_historical_manifest_sha256"],
        intended_use="commercial_training", decision_receipts=files["decision_receipts"],
        tokenizer=files["tokenizer"], target_tokens=approval["target_tokens"],
        resume=(output / "production/state.sqlite").exists(), dry_run=False,
        chunk_items=50, reserve_mib=512, request_interval=.25)
    code = production_build.execute(args)
    need(code in (0, 3), "Production build failed")
    with sqlite3.connect((output / "production/state.sqlite").as_uri() + "?mode=ro",
                         uri=True) as db:
        accepted = {}
        for encoded, count in db.execute(
                "SELECT record,token_count FROM items WHERE stage='ACCEPTED'"):
            sid = json.loads(encoded)["source_id"]
            n, tokens = accepted.get(sid, (0, 0))
            accepted[sid] = (n + 1, tokens + count)
    for row in rows:
        n, tokens = accepted.get(row["source_id"], (0, 0))
        if n:
            row.update(state="ADMITTED", stage="PRODUCTION", training_use_state="VERIFIED",
                       token_estimate=tokens, duplicate_state="SETTLED", reason=None)
    report = seal(dict(previous, phase="PRODUCTION", training_allowed=False,
                       use_receipts_verified_by_production=True,
                       approved_inputs_sha256=expected_sha,
                       approved_inputs_path=approved_path,
                       admission_complete_sha256=admission_sha,
                       compiled_production_plan_sha256=inventory["plan_sha256"],
                       scheduled_files=inventory["scheduled"], production_exit_code=code,
                       production_progress_sha256=file_digest(
                           output / "production/runner/progress.json"),
                       sources=rows))
    atomic_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("batch", type=Path)
    prep.add_argument("output", type=Path)
    prep.add_argument("--fetch", action="store_true", help="Allow bounded metadata/Git fetch")
    prep.add_argument("--resume", action="store_true")
    continued = sub.add_parser("resume")
    continued.add_argument("output", type=Path)
    continued.add_argument("--approved-inputs", type=Path, required=True)
    continued.add_argument("--expected-approved-inputs-sha256", required=True)
    continued.add_argument("--execute-production", action="store_true")
    check = sub.add_parser("verify")
    check.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.batch, args.output, resume=args.resume, fetch=args.fetch)
        elif args.command == "resume":
            result = resume(args.output, args.approved_inputs,
                args.expected_approved_inputs_sha256, execute=args.execute_production)
        else:
            result = verify(args.output)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Workflow stopped: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
