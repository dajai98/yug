"""Offline, zero-authority evidence packaging and verification.

The caller must supply a trusted manifest pin. A local hash or local public key
is not an approval. This module never calls corpus admission or governance APIs.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from training.evidence_trust import TrustError, verify_trust_pin
from training.production_corpus import canonical, file_digest, identity

SCHEMA = "yug-evidence-bundle-v1"
SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
REV = re.compile(r"[0-9a-f]{40}\Z")
ROLES = frozenset({"source_file", "source_inventory", "license", "notice", "rights_review",
                   "screening", "protected_references", "intended_use_decision",
                   "ownership_decision", "production_admission", "production_build",
                   "corpus_ledger", "dataset_lineage", "model_lineage", "other"})
MAX_MANIFEST = 4 * 1024 * 1024
MAX_ARTIFACTS = 10000
DECISION_V1 = "yug-source-training-use-decision-v1"
DECISION_V2 = "yug-source-training-use-decision-v2"


class EvidenceError(ValueError):
    pass


def need(ok, message):
    if not ok:
        raise EvidenceError(message)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def _read_json(path, limit=MAX_MANIFEST):
    try:
        raw = Path(path).read_bytes()
        need(0 < len(raw) <= limit, "Missing or oversized JSON")
        value = json.loads(raw, object_pairs_hook=_unique)
        need(type(value) is dict and raw == canonical(value), "Noncanonical JSON")
        return value
    except (OSError, UnicodeError, ValueError) as exc:
        raise EvidenceError(f"Invalid evidence JSON: {exc}") from exc


def _path(root, name):
    need(type(name) is str and name and "\\" not in name and ":" not in name
         and not name.startswith("/") and all(x not in ("", ".", "..")
                                               for x in name.split("/")),
         "Invalid artifact path")
    root = Path(root).resolve()
    path = (root / name).resolve()
    need(path.is_relative_to(root) and path.is_file(), "Artifact missing or escapes root")
    return path


def _manifest(value):
    need(type(value) is dict and set(value) == {
        "schema", "mode", "source", "files", "artifacts", "decision_scope",
        "lineage", "identity"}, "Incomplete evidence manifest")
    need(value["schema"] == SCHEMA and value["mode"] in ("development", "production"),
         "Unsupported evidence schema or mode")
    source = value["source"]
    need(type(source) is dict and set(source) == {"source_id", "locator", "revision"}
         and all(type(source[k]) is str and source[k] for k in source)
         and (bool(REV.fullmatch(source["revision"])) or source["revision"] == "UNKNOWN"),
         "Invalid source identity")
    locator = urlsplit(source["locator"])
    need(locator.scheme == "https" and locator.hostname and
         locator.username is None and locator.password is None and
         not locator.query and not locator.fragment and locator.path.strip("/"),
         "Noncanonical source locator")
    need(value["mode"] != "production" or REV.fullmatch(source["revision"]),
         "Production source requires exact revision")
    files = value["files"]
    need(type(files) is list and len(files) <= MAX_ARTIFACTS, "Invalid file scope")
    need(value["mode"] != "production" or files,
         "Production source requires exact nonempty file scope")
    paths = []
    for row in files:
        need(type(row) is dict and set(row) == {"path", "sha256", "git_blob", "artifact_path"}
             and type(row["sha256"]) is str and SHA.fullmatch(row["sha256"])
             and (row["git_blob"] == "UNKNOWN" or
                  (type(row["git_blob"]) is str and
                   re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", row["git_blob"]))),
             "Invalid exact file record")
        _path_name(row["path"])
        _path_name(row["artifact_path"])
        paths.append(row["path"])
    need(paths == sorted(set(paths)) and len({p.casefold() for p in paths}) == len(paths),
         "Duplicate or unsorted file scope")
    decision = value["decision_scope"]
    need(decision == "MISSING" or decision == "UNKNOWN" or
         (type(decision) is dict and set(decision) == {"source", "files", "artifact_path"}
          and decision["source"] == source and decision["files"] == files),
         "Decision/source/file scope mismatch")
    need(value["mode"] != "production" or type(decision) is dict,
         "Production evidence requires exact decision scope")
    artifacts = value["artifacts"]
    need(type(artifacts) is list and len(artifacts) <= MAX_ARTIFACTS,
         "Invalid artifact inventory")
    names = []
    for row in artifacts:
        need(type(row) is dict and set(row) == {"role", "path", "sha256", "bytes"}
             and type(row["role"]) is str and row["role"] in ROLES
             and type(row["sha256"]) is str
             and SHA.fullmatch(row["sha256"]) and type(row["bytes"]) is int
             and row["bytes"] >= 0, "Invalid artifact record")
        _path_name(row["path"])
        names.append(row["path"])
    need(names == sorted(set(names)) and len({p.casefold() for p in names}) == len(names),
         "Duplicate or unsorted artifact paths")
    if type(decision) is dict:
        need(decision["artifact_path"] in names and any(
            a["path"] == decision["artifact_path"] and
            a["role"] == "intended_use_decision" for a in artifacts),
             "Decision artifact is not bound")
    if value["mode"] == "production":
        need({"source_inventory", "license", "intended_use_decision"}.issubset(
            {a["role"] for a in artifacts}), "Mandatory production evidence missing")
        need(all(any(a["role"] == "source_file" and
                         a["path"] == f["artifact_path"] and
                         a["sha256"] == f["sha256"] for a in artifacts)
                 for f in files), "Exact source file bytes not bound")
    lineage = value["lineage"]
    need(type(lineage) is dict and set(lineage) == {"dataset_versions", "model_versions"}
         and all(type(lineage[k]) is list and
                 all(type(x) is str and x for x in lineage[k])
                 for k in lineage), "Invalid lineage")
    need(type(value["identity"]) is str and SHA.fullmatch(value["identity"])
         and value["identity"] == identity({k: v for k, v in value.items()
                                             if k != "identity"}),
         "Manifest identity mismatch")


def _path_name(name):
    need(type(name) is str and name and "\\" not in name and ":" not in name
         and not name.startswith("/") and all(x not in ("", ".", "..")
                                               for x in name.split("/")),
         "Invalid artifact/file path")


def _git_blob(path, length):
    digest = hashlib.sha1() if length == 40 else hashlib.sha256()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(root, specification):
    """Build a reviewable manifest; never issues a rights or use decision."""
    need(type(specification) is dict and set(specification) == {
        "mode", "source", "files", "artifacts", "decision_scope", "lineage"},
         "Invalid bundle specification")
    artifacts = []
    for row in specification["artifacts"]:
        need(type(row) is dict and set(row) == {"role", "path"},
             "Invalid artifact request")
        path = _path(root, row["path"])
        artifacts.append(dict(role=row["role"], path=row["path"],
                              sha256=file_digest(path), bytes=path.stat().st_size))
    unsigned = dict(schema=SCHEMA, mode=specification["mode"],
                    source=specification["source"], files=specification["files"],
                    artifacts=sorted(artifacts, key=lambda x: x["path"]),
                    decision_scope=specification["decision_scope"],
                    lineage=specification["lineage"])
    manifest = dict(unsigned, identity=identity(unsigned))
    _manifest(manifest)
    _verify_decision_scope(root, manifest)
    return manifest


def _verify_decision_scope(root, manifest):
    """Compare complete supported decision scopes; never grant authority."""
    decision_scope = manifest["decision_scope"]
    if type(decision_scope) is not dict:
        return
    path = _path(root, decision_scope["artifact_path"])
    decision = _read_json(path)
    need(decision.get("schema") in (DECISION_V1, DECISION_V2)
         and decision.get("decision") == "APPROVE_TRAINING_USE",
         "Unsupported decision shape for exact-scope comparison")
    need(decision.get("identity") == identity({k: v for k, v in decision.items()
                                              if k != "identity"}),
         "Decision identity mismatch")
    source = manifest["source"]
    need(all(decision.get(k) == source[k] for k in ("source_id", "locator", "revision")),
         "Decision source/revision mismatch")
    files = manifest["files"]
    if decision["schema"] == DECISION_V1:
        need(len(files) == 1 and all(decision.get(k) == files[0][k]
                                     for k in ("path", "sha256", "git_blob")),
             "Decision exact file scope mismatch")
    else:
        _verify_multifile_decision(decision, files, manifest)


def _verify_multifile_decision(decision, files, manifest):
    required = {"schema", "decision", "source_id", "locator", "revision",
                "intended_use", "split", "owner", "subcategory", "files",
                "rights_review_identity", "attribution", "notice_obligations",
                "obligations", "max_files", "max_raw_bytes", "max_accepted_tokens",
                "reviewer", "decided_at", "governance_decision_identity", "identity"}
    need(set(decision) == required, "Incomplete multi-file intended-use decision")
    need((decision["intended_use"], decision["split"]) in
         (("commercial_training", "train"), ("evaluation", "validation")),
         "Intended use/split mismatch")
    need(all(type(decision[k]) is str and decision[k].strip()
             for k in ("owner", "subcategory", "attribution", "reviewer", "decided_at")),
         "Incomplete multi-file decision authority/attribution")
    need(decision["decided_at"].endswith("Z"), "UTC decision timestamp required")
    try:
        datetime.fromisoformat(decision["decided_at"][:-1] + "+00:00")
    except ValueError as exc:
        raise EvidenceError("Malformed decision timestamp") from exc
    need(all(type(decision[k]) is str and SHA.fullmatch(decision[k])
             for k in ("rights_review_identity", "governance_decision_identity")),
         "Missing independent review/governance identity")
    for key in ("notice_obligations", "obligations"):
        values = decision[key]
        need(type(values) is list and
             all(type(item) is str and item.strip() for item in values) and
             values == sorted(set(values)),
             "Malformed or duplicate obligations")
    need(type(decision["max_files"]) is int and decision["max_files"] == len(files)
         and type(decision["max_raw_bytes"]) is int and decision["max_raw_bytes"] > 0
         and type(decision["max_accepted_tokens"]) is int and
         decision["max_accepted_tokens"] > 0,
         "Invalid multi-file caps")
    rows = decision["files"]
    need(type(rows) is list and len(rows) == len(files),
         "Decision exact file count mismatch")
    for row, bound in zip(rows, files, strict=True):
        need(type(row) is dict and set(row) == {
            "path", "sha256", "git_blob", "bytes", "source_clearance_identity",
            "file_clearance_identity"}, "Incomplete decision file record")
        need(all(row[k] == bound[k] for k in ("path", "sha256", "git_blob")),
             "Decision exact file scope mismatch")
        need(type(row["bytes"]) is int and row["bytes"] > 0 and
             all(type(row[k]) is str and SHA.fullmatch(row[k]) for k in
                 ("source_clearance_identity", "file_clearance_identity")),
             "Missing exact file clearance/byte evidence")
        need(any(a["role"] == "source_file" and
                 a["path"] == bound["artifact_path"] and
                 a["bytes"] == row["bytes"] and a["sha256"] == row["sha256"]
                 for a in manifest["artifacts"]),
             "Decision file bytes are not bound")
    need(sum(row["bytes"] for row in rows) <= decision["max_raw_bytes"],
         "Decision raw-byte cap exceeded")


def _verify_signature(manifest_identity, signature_path, trusted_public_key,
                      openssl, key_id=None):
    need(signature_path is not None and trusted_public_key is not None,
         "Signature and externally trusted public key required")
    signature = _read_json(signature_path, 8192)
    v1 = {"schema", "algorithm", "manifest_identity", "public_key_sha256",
          "signature_base64"}
    v2 = v1 | {"key_id"}
    need((signature.get("schema") == "yug-evidence-signature-v1" and
          set(signature) == v1) or
         (signature.get("schema") == "yug-evidence-signature-v2" and
          set(signature) == v2 and key_id is not None and
          signature["key_id"] == key_id),
         "Unsupported or mismatched signature identity")
    need(signature["algorithm"] == "Ed25519" and
         signature["manifest_identity"] == manifest_identity and
         signature["public_key_sha256"] == file_digest(trusted_public_key),
         "Signature binding mismatch")
    try:
        raw = base64.b64decode(signature["signature_base64"], validate=True)
        need(len(raw) == 64, "Invalid Ed25519 signature length")
        # OpenSSL reads the signed message and signature from bounded temporary files.
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            message = Path(tmp) / "message"
            sig = Path(tmp) / "signature"
            message.write_bytes(b"YUG-EVIDENCE-MANIFEST-V1\0" + manifest_identity.encode("ascii"))
            sig.write_bytes(raw)
            run = subprocess.run([openssl, "pkeyutl", "-verify", "-pubin",
                                  "-inkey", str(trusted_public_key), "-rawin", "-in",
                                  str(message), "-sigfile", str(sig)],
                                 capture_output=True, timeout=10, check=False)
            need(run.returncode == 0, "Invalid manifest signature")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise EvidenceError(f"Signature verification unavailable/invalid: {exc}") from exc
    return "VERIFIED"


def _verify_timestamp(manifest_identity, timestamp_path, tsa_ca_file, openssl):
    need(timestamp_path is not None and tsa_ca_file is not None,
         "Timestamp receipt and externally trusted TSA CA required")
    need(Path(timestamp_path).is_file() and Path(tsa_ca_file).is_file(),
         "Timestamp receipt/trust anchor missing")
    # Timestamp the manifest identity bytes, never the corpus or evidence bytes.
    digest = hashlib.sha256(manifest_identity.encode("ascii")).hexdigest()
    try:
        run = subprocess.run([openssl, "ts", "-verify", "-digest", digest,
                              "-in", str(timestamp_path), "-CAfile", str(tsa_ca_file)],
                             capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvidenceError(f"Timestamp verification unavailable: {exc}") from exc
    need(run.returncode == 0, "Invalid or mismatched RFC 3161 timestamp")
    return "VERIFIED"


def verify_bundle(root, manifest_path, expected_identity, *, signature_path=None,
                  trusted_public_key=None, timestamp_path=None, tsa_ca_file=None,
                  trust_pin_path=None, expected_trust_pin_identity=None,
                  trust_root=None, archival_only=False, openssl="openssl"):
    """Offline verification; result is NEVER an authorization to use content."""
    need(type(expected_identity) is str and SHA.fullmatch(expected_identity),
         "Externally trusted manifest identity required")
    manifest = _read_json(manifest_path)
    _manifest(manifest)
    need(manifest["identity"] == expected_identity, "External manifest pin mismatch")
    for row in manifest["artifacts"]:
        path = _path(root, row["path"])
        need(path.stat().st_size == row["bytes"] and file_digest(path) == row["sha256"],
             "Bound evidence changed")
    for row in manifest["files"]:
        if row["git_blob"] != "UNKNOWN":
            path = _path(root, row["artifact_path"])
            need(_git_blob(path, len(row["git_blob"])) == row["git_blob"],
                 "Git blob identity mismatch")
    _verify_decision_scope(root, manifest)
    required = manifest["mode"] == "production"
    pin = None
    if required:
        need(trusted_public_key is None,
             "Direct public-key path cannot replace production trust pin")
        need(trust_pin_path is not None and expected_trust_pin_identity is not None
             and trust_root is not None, "Externally pinned production trust record required")
    if trust_pin_path is not None or expected_trust_pin_identity is not None:
        need(trust_pin_path is not None and expected_trust_pin_identity is not None
             and trust_root is not None and trusted_public_key is None,
             "Incomplete or conflicting trust-pin inputs")
        try:
            pin, trusted_public_key = verify_trust_pin(
                trust_pin_path, expected_trust_pin_identity, trust_root,
                archival_only=archival_only)
        except TrustError as exc:
            raise EvidenceError(str(exc)) from exc
        need(required, "Production trust pin cannot verify development evidence")
    if required or signature_path is not None or trusted_public_key is not None:
        signature_status = _verify_signature(expected_identity, signature_path,
                                             trusted_public_key, openssl,
                                             key_id=pin["key_id"] if pin else None)
        if pin and pin["status"] == "RETIRED":
            signature_status = "VERIFIED_RETIRED_ARCHIVAL_ONLY"
    else:
        signature_status = "UNSIGNED_DEVELOPMENT_ONLY"
    # Production policy is TIMESTAMP_REQUIRED. Do not let omitted timestamp
    # inputs turn a signed production bundle into a successful verification.
    if required or timestamp_path is not None or tsa_ca_file is not None:
        if required:
            need(timestamp_path is not None and tsa_ca_file is not None,
                 "TIMESTAMP_REQUIRED production evidence needs RFC 3161 receipt and TSA CA")
        timestamp_status = _verify_timestamp(expected_identity, timestamp_path,
                                             tsa_ca_file, openssl)
    else:
        timestamp_status = "MISSING"
    return dict(bundle_identity=expected_identity, artifacts=len(manifest["artifacts"]),
                files=len(manifest["files"]), signature=signature_status,
                timestamp=timestamp_status,
                trust_pin_identity=pin["identity"] if pin else "MISSING",
                key_id=pin["key_id"] if pin else "MISSING",
                dataset_versions=manifest["lineage"]["dataset_versions"],
                model_versions=manifest["lineage"]["model_versions"],
                authority="NONE")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline zero-authority evidence verification")
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-identity", required=True)
    parser.add_argument("--signature")
    parser.add_argument("--trusted-public-key")
    parser.add_argument("--trust-pin")
    parser.add_argument("--expected-trust-pin-identity")
    parser.add_argument("--trust-root")
    parser.add_argument("--archival-only", action="store_true")
    parser.add_argument("--timestamp")
    parser.add_argument("--tsa-ca-file")
    parser.add_argument("--openssl", default="openssl")
    args = parser.parse_args(argv)
    result = verify_bundle(args.root, args.manifest, args.expected_identity,
                           signature_path=args.signature,
                           trusted_public_key=args.trusted_public_key,
                           trust_pin_path=args.trust_pin,
                           expected_trust_pin_identity=args.expected_trust_pin_identity,
                           trust_root=args.trust_root, archival_only=args.archival_only,
                           timestamp_path=args.timestamp, tsa_ca_file=args.tsa_ca_file,
                           openssl=args.openssl)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
