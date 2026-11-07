"""Bounded, zero-authority publication dossiers for human exact-file review.

This compiler consumes YUG's verified review cache. It never issues a source
clearance, a file grant, a training-use decision, or an admission plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath

from training.clearance_evidence import verify_bundle
from training.production_acquisition import read_plan
from training.production_corpus import atomic_json, canonical, file_digest
from training.production_spec import local_path, need, seal, verify_seal
from training.source_admission import cached_sources, holdout_groups
from training.source_grant_proposals import _payload, _source_evidence
from training.source_prepare import ROOT, git

POLICY = "bounded-publication-candidates-v1"
BATCH_POLICY = "bounded-publication-decision-batch-v1"
LICENSES = frozenset({"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"})
DOC_SUFFIXES = frozenset({".md", ".rst"})
CODE_SUFFIXES = frozenset({".py", ".cs", ".c", ".cc", ".cpp", ".h", ".hpp", ".rs"})
DOC_ROOTS = frozenset({"doc", "docs", "documentation"})
CODE_ROOTS = frozenset({"src", "lib"})
UNSAFE_PARTS = frozenset(
    {
        "vendor",
        "vendored",
        "third_party",
        "thirdparty",
        "external",
        "dependencies",
        "generated",
        "build",
        "dist",
        "out",
        "target",
        "node_modules",
        "submodules",
        "benchmark",
        "benchmarks",
        "evaluation",
        "tests",
        "test",
        "testing",
        "fixtures",
        "examples",
        "_static",
        "_templates",
    }
)
PROVENANCE = re.compile(r"\b(copied|adapted|derived from|borrowed|vendored|third.party)\b", re.I)
GENERATED = re.compile(r"\b(generated|auto.generated|do not edit)\b", re.I)
SPDX = re.compile(r"SPDX-License-Identifier\s*:\s*([^\r\n*#]+)", re.I)
RIGHTS_NAMES = re.compile(r"^(?:licen[sc]e|copying|notice|copyright)(?:[._-]|$)", re.I)


def _type(path):
    parts = PurePosixPath(path).parts
    suffix = PurePosixPath(path).suffix.lower()
    if len(parts) > 1 and parts[0].casefold() in DOC_ROOTS and suffix in DOC_SUFFIXES:
        return "documentation"
    if len(parts) > 1 and parts[0].casefold() in CODE_ROOTS and suffix in CODE_SUFFIXES:
        return "code"
    return None


def _companion(rows, names, directory, root, *, allow_docs=False):
    matches = [
        rows[p]
        for p in sorted(rows)
        if (
            ("/" not in p)
            or (allow_docs and p.casefold().startswith("docs/") and p.count("/") == 1)
        )
        and any(PurePosixPath(p).name.casefold().startswith(n) for n in names)
    ]
    return [
        dict(
            path=(directory / r["local_blob"]).relative_to(root).as_posix(),
            upstream_path=r["path"],
            sha256=r["sha256"],
            git_blob=r["git_object"],
        )
        for r in matches
        if r["sha256"] is not None and r["local_blob"] and r["git_mode"] in ("100644", "100755")
    ]


def _history_map(directory, source, *, fetch_history=False, history_depth=32):
    """Acquire a bounded history window, then map exact changed paths once."""
    repo = directory / "git"
    need(
        type(history_depth) is int and 1 <= history_depth <= 256,
        "History depth must stay within 1..256 commits",
    )
    shallow = git(repo, "rev-parse", "--is-shallow-repository").stdout.strip() == b"true"
    if fetch_history and shallow:
        try:
            count = int(git(repo, "rev-list", "--count", source["revision"]).stdout)
            additional = max(0, history_depth + 1 - count)
            if additional:
                git(
                    repo,
                    "fetch",
                    "--deepen=" + str(additional),
                    "--no-tags",
                    "--no-recurse-submodules",
                    source["locator"],
                    source["revision"],
                )
            git(repo, "fsck", "--full", "--strict", "--no-reflogs")
            shallow = git(repo, "rev-parse", "--is-shallow-repository").stdout.strip() == b"true"
        except subprocess.SubprocessError:
            # Network/history unavailability never promotes a file.
            pass
    raw = git(
        repo,
        "log",
        "--max-count=" + str(history_depth + 1),
        "--format=commit:%H",
        "--name-only",
        source["revision"],
    ).stdout.decode("utf-8", "surrogateescape")
    commits = []
    by_path = {}
    current = None
    for line in raw.splitlines():
        if line.startswith("commit:"):
            current = line[7:]
            need(re.fullmatch("[0-9a-f]{40}", current), "Malformed retained commit")
            commits.append(current)
        elif line and current is not None:
            by_path.setdefault(line, []).append(current)
    # The last shallow commit is a synthetic boundary: Git reports all files
    # as introduced there. It cannot establish an exact path's origin.
    if shallow and commits:
        boundary = set((repo / "shallow").read_text().splitlines())
        for path in by_path:
            by_path[path] = [oid for oid in by_path[path] if oid not in boundary]
    return by_path, shallow


def _history(source, row, history_map, shallow):
    commits = history_map.get(row["path"], [])[:3]
    return seal(
        dict(
            policy=POLICY,
            source_id=source["source_id"],
            revision=source["revision"],
            path=row["path"],
            git_blob=row["git_object"],
            commits=commits,
            shallow=shallow,
            adequate=bool(commits),
            authority="NONE",
        )
    )


def _classify(row, observed, source, rows, data, history, publication, contribution, protected):
    if observed["status"] == "EXCLUDED":
        return "EXCLUDE", observed["reason"], None
    parts = [p.casefold() for p in PurePosixPath(row["path"]).parts]
    if protected:
        return "EXCLUDE", "protected source group", None
    if any(p in UNSAFE_PARTS for p in parts):
        return "EXCLUDE", "third-party/test/generated/benchmark path requires separate scope", None
    kind = _type(row["path"])
    if kind is None:
        return "REVIEW", "outside bounded documentation/source-code formats and subtrees", None
    if source["status"] != "ACCEPT" or source["license_id"] not in LICENSES:
        return "REVIEW", "unsupported or conditional source license", kind
    if not data or history is None:
        return "REVIEW", "empty or unavailable bounded text payload", kind
    if not publication or not contribution:
        return "REVIEW", "publication or contribution-process evidence absent", kind
    if not history["adequate"]:
        return "REVIEW", "bounded exact-path Git history unavailable", kind
    if row["git_mode"] not in ("100644", "100755"):
        return "EXCLUDE", "nonregular Git entry", kind
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return "REVIEW", "unsupported text encoding", kind
    if GENERATED.search(text):
        return "REVIEW", "generated-file signal", kind
    if PROVENANCE.search(text):
        return "REVIEW", "copied/adapted/third-party provenance signal", kind
    labels = [s.strip() for s in SPDX.findall(text)]
    if labels and labels != [source["license_id"]]:
        return "REVIEW", "conflicting or mixed per-file license", kind
    path = PurePosixPath(row["path"])
    if any(
        PurePosixPath(other).parent in path.parents
        and RIGHTS_NAMES.search(PurePosixPath(other).name)
        for other in rows
        if "/" in other
    ):
        return "REVIEW", "nested license/NOTICE/COPYRIGHT requires separate review", kind
    if (
        kind == "documentation"
        and path.suffix.lower() == ".rst"
        and re.search(
            r"^\s*\.\.\s*(?:include|literalinclude|image|figure)::|:doc:|:download:",
            text,
            re.I | re.M,
        )
    ):
        return "REVIEW", "RST local include/reference needs exact scope review", kind
    return "ROUTINE_CANDIDATE", "exact evidence dossier ready for bounded human review", kind


def _encode_lines(rows):
    return b"".join(canonical(row) + b"\n" for row in rows)


def decision_batches(routine, scope, request_identity):
    """Stable exact sets; any scope/file change changes the sealed batch identity."""
    routine = sorted(routine, key=lambda d: d["path"])
    need(
        len({d["path"].casefold() for d in routine}) == len(routine),
        "Duplicate/ambiguous publication candidate path",
    )
    return [
        seal(
            dict(
                policy=BATCH_POLICY,
                request_identity=request_identity,
                source_id=scope["source_id"],
                source_scope_candidate_identity=scope["identity"],
                inventory_identity=scope["inventory_identity"],
                ordinal=start // 64,
                paths=[d["path"] for d in routine[start : start + 64]],
                candidate_identities=[d["identity"] for d in routine[start : start + 64]],
                authority="NONE",
                training_allowed=False,
            )
        )
        for start in range(0, len(routine), 64)
    ]


def compile_candidates(
    cache,
    catalog,
    evidence,
    references,
    proposals,
    output,
    *,
    root=ROOT,
    fetch_history=False,
    history_depth=32,
):
    """Generate deterministic dossiers, exception queue, and <=64-file batches."""
    root = Path(root).resolve()
    cache, catalog, evidence, references, proposals, output = (
        Path(p).resolve() for p in (cache, catalog, evidence, references, proposals, output)
    )
    need(
        output.is_relative_to(root / "artifacts")
        and all(
            not output.is_relative_to(p) and not p.is_relative_to(output)
            for p in (cache, evidence, proposals)
        ),
        "Unsafe candidate output",
    )
    source_catalog = json.loads(catalog.read_bytes())
    verify_seal(source_catalog)
    review = verify_bundle(evidence, root)
    need(
        review["complete"]
        and review["training_allowed"] is False
        and review["counts"]["EVIDENCED"] == 0,
        "Candidate compilation needs review-only evidence",
    )
    proposal = json.loads((proposals / "complete.json").read_bytes())
    verify_seal(proposal)
    need(
        proposal["training_allowed"] is False
        and proposal["legitimately_grantable"] == 0
        and file_digest(proposals / "proposals.jsonl") == proposal["proposals_sha256"],
        "Unverified proposal input",
    )
    inputs = {
        p.relative_to(root).as_posix(): file_digest(p)
        for p in (
            catalog,
            references,
            evidence / "complete.json",
            evidence / "evidence.jsonl",
            proposals / "request.json",
            proposals / "complete.json",
            proposals / "proposals.jsonl",
            cache / "status.json",
        )
    }
    request = seal(
        dict(
            policy=POLICY,
            inputs=inputs,
            catalog_identity=source_catalog["identity"],
            review_identity=review["identity"],
            proposal_identity=proposal["identity"],
        )
    )
    observed = {}
    for item in read_plan(evidence / "evidence.jsonl"):
        verify_seal(item)
        key = item["source_id"], item["path"]
        need(key not in observed, "Duplicate file evidence")
        observed[key] = item
    heldout = holdout_groups(references, source_catalog)
    dossiers = []
    histories = []
    scopes = []
    batches = []
    counts = Counter()
    per_source = {}
    for directory, source, inventory, done in cached_sources(
        cache,
        source_catalog,
        json.loads((root / "training/specifications/production_corpus_v1.json").read_bytes()),
        root,
    ):
        rows_list = list(read_plan(inventory))
        rows = {r["path"]: r for r in rows_list}
        need(
            len(rows) == len(rows_list) == sum(done["counts"].values()),
            "Duplicate or incomplete source inventory",
        )
        _source_evidence(source, root)
        history_map, shallow = _history_map(
            directory, source, fetch_history=fetch_history, history_depth=history_depth
        )
        publication = _companion(rows, ("readme",), directory, root)
        contribution = _companion(rows, ("contributing",), directory, root, allow_docs=True)
        notices = _companion(rows, ("notice", "copyright"), directory, root)
        scope = seal(
            dict(
                policy=POLICY,
                source_id=source["source_id"],
                locator=source["locator"],
                revision=source["revision"],
                inventory_identity=done["inventory_sha256"],
                license_id=source["license_id"],
                root_license_evidence=source.get("license_evidence"),
                publication_evidence=publication,
                contribution_evidence=contribution,
                notice_copyright_evidence=notices,
                candidate_subtrees=["doc/", "docs/", "documentation/", "src/", "lib/"],
                excluded_path_parts=sorted(UNSAFE_PARTS),
                protected_source=source["split_group"] in heldout,
                human_scope_approved=False,
                authority="NONE",
                training_allowed=False,
            )
        )
        scopes.append(scope)
        source_counts = Counter()
        routine = []
        for row in sorted(rows_list, key=lambda r: r["path"]):
            key = source["source_id"], row["path"]
            need(
                key in observed
                and observed[key]["cache_inventory_sha256"] == done["inventory_sha256"]
                and all(
                    observed[key][k] == row[k]
                    for k in ("source_id", "revision", "path", "sha256", "bytes")
                ),
                "File review/cache identity mismatch",
            )
            item = observed.pop(key)
            data = _payload(directory, row) if item["status"] == "REVIEW" else b""
            history = _history(source, row, history_map, shallow) if data else None
            if history is not None:
                histories.append(history)
            state, reason, kind = _classify(
                row,
                item,
                source,
                rows,
                data,
                history,
                publication,
                contribution,
                source["split_group"] in heldout,
            )
            decoded = data.decode("utf-8-sig", errors="replace") if data else ""
            dossier = seal(
                dict(
                    policy=POLICY,
                    request_identity=request["identity"],
                    source_scope_candidate_identity=scope["identity"],
                    source_id=source["source_id"],
                    locator=source["locator"],
                    revision=source["revision"],
                    path=row["path"],
                    bytes=row["bytes"],
                    sha256=row["sha256"],
                    git_blob=row["git_object"],
                    inventory_identity=done["inventory_sha256"],
                    root_license_evidence=source.get("license_evidence"),
                    publication_evidence=publication,
                    contribution_evidence=contribution,
                    notice_copyright_evidence=notices,
                    file_history_identity=history["identity"] if history else None,
                    file_type=kind,
                    file_license_headers=SPDX.findall(decoded),
                    third_party_indicator=bool(PROVENANCE.search(decoded)),
                    generated_indicator=bool(GENERATED.search(decoded)),
                    protected_source=source["split_group"] in heldout,
                    clearance_status=item["status"],
                    state=state,
                    reason=reason,
                    human_scope_approved=False,
                    authority="NONE",
                    grantable=False,
                    training_allowed=False,
                )
            )
            dossiers.append(dossier)
            counts[state] += 1
            source_counts[state] += 1
            if state == "ROUTINE_CANDIDATE":
                routine.append(dossier)
        per_source[source["source_id"]] = dict(source_counts)
        batches.extend(decision_batches(routine, scope, request["identity"]))
    need(not observed, "Unmatched file review evidence")
    files = {
        "dossiers.jsonl": _encode_lines(dossiers),
        "scopes.jsonl": _encode_lines(scopes),
        "histories.jsonl": _encode_lines(histories),
        "batches.jsonl": _encode_lines(batches),
    }
    for history in histories:
        name = hashlib.sha256((history["source_id"] + "\0" + history["path"]).encode()).hexdigest()
        files["histories/" + name + ".json"] = canonical(history)
    for dossier in dossiers:
        if dossier["state"] == "ROUTINE_CANDIDATE":
            name = hashlib.sha256(
                (dossier["source_id"] + "\0" + dossier["path"]).encode()
            ).hexdigest()
            files["dossiers/" + name + ".json"] = canonical(dossier)
    for batch in batches:
        name = hashlib.sha256(
            (batch["source_id"] + "\0" + str(batch["ordinal"])).encode()
        ).hexdigest()
        files["batches/" + name + ".json"] = canonical(batch)
    receipt = seal(
        dict(
            policy=POLICY,
            request_identity=request["identity"],
            counts=dict(counts),
            per_source=per_source,
            total=len(dossiers),
            batches=len(batches),
            files={
                name: "sha256:" + hashlib.sha256(data).hexdigest() for name, data in files.items()
            },
            authority="NONE",
            training_allowed=False,
        )
    )
    if output.exists():
        need(
            json.loads((output / "request.json").read_bytes()) == request
            and json.loads((output / "complete.json").read_bytes()) == receipt
            and all((output / name).read_bytes() == data for name, data in files.items()),
            "Candidate replay differs",
        )
        return receipt
    output.mkdir(parents=True)
    atomic_json(output / "request.json", request)
    for name, data in files.items():
        (output / name).parent.mkdir(parents=True, exist_ok=True)
        (output / name).write_bytes(data)
    atomic_json(output / "complete.json", receipt)
    return receipt


def verify_candidates(output, *, root=ROOT):
    output = Path(output).resolve()
    root = Path(root).resolve()
    request = json.loads((output / "request.json").read_bytes())
    verify_seal(request)
    receipt = json.loads((output / "complete.json").read_bytes())
    verify_seal(receipt)
    need(
        request["policy"] == receipt["policy"] == POLICY
        and receipt["request_identity"] == request["identity"]
        and receipt["authority"] == "NONE"
        and receipt["training_allowed"] is False,
        "Candidate authority/identity mismatch",
    )
    for path, sha in request["inputs"].items():
        need(file_digest(local_path(root, path)) == sha, "Candidate input changed")
    for name, sha in receipt["files"].items():
        need(file_digest(output / name) == sha, "Candidate output changed")
    dossiers = list(read_plan(output / "dossiers.jsonl"))
    scopes = {s["identity"]: s for s in read_plan(output / "scopes.jsonl")}
    for scope in scopes.values():
        verify_seal(scope)
        need(
            scope["authority"] == "NONE" and scope["human_scope_approved"] is False,
            "Mechanical scope cannot grant rights",
        )
    need(len(dossiers) == receipt["total"], "Candidate count changed")
    routine = {}
    for d in dossiers:
        verify_seal(d)
        need(
            d["source_scope_candidate_identity"] in scopes
            and d["authority"] == "NONE"
            and d["training_allowed"] is False
            and d["grantable"] is False,
            "Candidate promoted or scope missing",
        )
        if d["state"] == "ROUTINE_CANDIDATE":
            routine[d["source_id"], d["path"]] = d
    seen = set()
    for batch in read_plan(output / "batches.jsonl"):
        verify_seal(batch)
        need(
            batch["policy"] == BATCH_POLICY
            and 0 < len(batch["paths"]) <= 64
            and len(batch["paths"]) == len(batch["candidate_identities"])
            and batch["paths"] == sorted(batch["paths"])
            and batch["authority"] == "NONE"
            and batch["training_allowed"] is False,
            "Invalid decision batch",
        )
        for path, identity in zip(batch["paths"], batch["candidate_identities"],
                                  strict=True):
            key = batch["source_id"], path
            need(
                key not in seen
                and key in routine
                and routine[key]["identity"] == identity
                and routine[key]["source_scope_candidate_identity"]
                == batch["source_scope_candidate_identity"],
                "Batch path moved or changed",
            )
            seen.add(key)
    need(seen == set(routine), "Unbatched routine candidate")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "catalog", "evidence", "references", "proposals", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--fetch-history", action="store_true", help="Allow bounded pinned Git history fetch"
    )
    parser.add_argument(
        "--history-depth",
        type=int,
        default=32,
        help="Maximum retained commit window (1..256; default 32)",
    )
    args = parser.parse_args(argv)
    try:
        result = compile_candidates(
            args.cache,
            args.catalog,
            args.evidence,
            args.references,
            args.proposals,
            args.output,
            fetch_history=args.fetch_history,
            history_depth=args.history_depth,
        )
        verify_candidates(args.output)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Publication candidates stopped: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
