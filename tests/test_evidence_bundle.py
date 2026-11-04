"""Evidence packaging is integrity only, never an approval path."""
import base64
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from training.evidence_bundle import EvidenceError, build_manifest, verify_bundle
from training.evidence_test_ceremony import verify_test_bundle
from training.evidence_trust import SCHEMA as TRUST_SCHEMA
from training.evidence_trust import build_test_trust_pin
from training.production_corpus import canonical, file_digest, identity


def _openssl():
    git_install = Path("C:/Program Files/Git/usr/bin/openssl.exe")
    return shutil.which("openssl") or (str(git_install) if git_install.is_file() else None)


def _trust_pin(tmp_path, public, *, status="ACTIVE"):
    payload = dict(schema=TRUST_SCHEMA, key_id="YUG-EVIDENCE-KEY-001",
                   algorithm="Ed25519", public_key_sha256=file_digest(public),
                   public_key_path=public.name, status=status,
                   valid_from="2020-01-01T00:00:00Z", valid_until="UNKNOWN",
                   predecessor_key_id="NONE",
                   governance_decision_identity="sha256:" + "c" * 64)
    pin = dict(payload, identity=identity(payload))
    path = tmp_path / "trust-pin.json"
    path.write_bytes(canonical(pin))
    return path, pin


def _test_signature(tmp_path, manifest, *, schema="yug-evidence-signature-v1",
                    key_id="YUG-EVIDENCE-KEY-001"):
    openssl = _openssl()
    if openssl is None:
        pytest.skip("OpenSSL verification tool unavailable")
    private = tmp_path / "ephemeral-test-only.pem"
    public = tmp_path / "public.pem"
    message = tmp_path / "message.bin"
    raw_sig = tmp_path / "signature.bin"
    try:
        subprocess.run([openssl, "genpkey", "-algorithm", "ed25519", "-out", str(private)],
                       check=True, capture_output=True)
        subprocess.run([openssl, "pkey", "-in", str(private), "-pubout", "-out", str(public)],
                       check=True, capture_output=True)
        message.write_bytes(b"YUG-EVIDENCE-MANIFEST-V1\0" + manifest["identity"].encode())
        subprocess.run([openssl, "pkeyutl", "-sign", "-inkey", str(private), "-rawin",
                        "-in", str(message), "-out", str(raw_sig)], check=True,
                       capture_output=True)
    finally:
        private.unlink(missing_ok=True)
    signature = dict(schema=schema, algorithm="Ed25519",
                     manifest_identity=manifest["identity"],
                     public_key_sha256=file_digest(public),
                     signature_base64=base64.b64encode(raw_sig.read_bytes()).decode())
    if schema.endswith("v2"):
        signature["key_id"] = key_id
    path = tmp_path / "signature.json"
    path.write_bytes(canonical(signature))
    return public, path, openssl


def _test_timestamp(tmp_path, manifest, openssl):
    """Create an offline RFC 3161 response with disposable test-only CA/TSA keys."""
    ca_key = tmp_path / "test-tsa-ca.key"
    ca_cert = tmp_path / "test-tsa-ca.pem"
    tsa_key = tmp_path / "test-tsa.key"
    tsa_cert = tmp_path / "test-tsa.pem"
    csr = tmp_path / "test-tsa.csr"
    extensions = tmp_path / "test-tsa.ext"
    extensions.write_text("basicConstraints=critical,CA:FALSE\n"
                          "keyUsage=critical,digitalSignature\n"
                          "extendedKeyUsage=critical,timeStamping\n")
    commands = [
        ["req", "-new", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(ca_key), "-out", str(ca_cert), "-days", "1",
         "-subj", "/CN=YUG local test TSA CA",
         "-addext", "basicConstraints=critical,CA:TRUE",
         "-addext", "keyUsage=critical,keyCertSign,cRLSign"],
        ["req", "-new", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(tsa_key), "-out", str(csr),
         "-subj", "/CN=YUG local test TSA"],
        ["x509", "-req", "-in", str(csr), "-CA", str(ca_cert),
         "-CAkey", str(ca_key), "-set_serial", "0x1001", "-out", str(tsa_cert),
         "-days", "1", "-extfile", str(extensions)],
    ]
    for args in commands:
        subprocess.run([openssl, *args], check=True, capture_output=True)
    serial = tmp_path / "test-tsa-serial"
    serial.write_text("01\n")
    config = tmp_path / "test-tsa.cnf"
    config.write_text(
        "[tsa]\ndefault_tsa=local_tsa\n[local_tsa]\n"
        f"serial={serial.as_posix()}\n"
        f"signer_cert={tsa_cert.as_posix()}\n"
        f"signer_key={tsa_key.as_posix()}\n"
        f"certs={ca_cert.as_posix()}\n"
        "signer_digest=sha256\ndefault_policy=1.3.6.1.4.1.55555.1\n"
        "digests=sha256\naccuracy=secs:1\nordering=yes\ntsa_name=yes\n")
    query = tmp_path / "test-timestamp.tsq"
    receipt = tmp_path / "test-timestamp.tsr"
    digest = hashlib.sha256(manifest["identity"].encode("ascii")).hexdigest()
    subprocess.run([openssl, "ts", "-query", "-digest", digest, "-sha256",
                    "-cert", "-out", str(query)], check=True, capture_output=True)
    subprocess.run([openssl, "ts", "-reply", "-config", str(config),
                    "-queryfile", str(query), "-out", str(receipt)],
                   check=True, capture_output=True)
    return receipt, ca_cert


def fixture(tmp_path, *, mode="development", decision=True):
    root = tmp_path / "retained"
    root.mkdir()
    (root / "license.txt").write_bytes(b"MIT license evidence\n")
    (root / "inventory.jsonl").write_bytes(b"pinned source inventory\n")
    (root / "payload.py").write_bytes(b"print('exact source')\n")
    payload = (root / "payload.py").read_bytes()
    git_blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    scope = [{"path": "src/example.py", "sha256": file_digest(root / "payload.py"),
              "git_blob": git_blob, "artifact_path": "payload.py"}]
    source = {"source_id": "example", "locator": "https://example.test/example",
              "revision": "c" * 40}
    decision_record = dict(schema="yug-source-training-use-decision-v1",
                           decision="APPROVE_TRAINING_USE", **source, **scope[0])
    (root / "decision.json").write_bytes(canonical(dict(
        decision_record, identity=identity(decision_record))))
    specification = {
        "mode": mode, "source": source, "files": scope,
        "artifacts": [{"role": "license", "path": "license.txt"},
                      {"role": "source_inventory", "path": "inventory.jsonl"},
                      {"role": "source_file", "path": "payload.py"},
                      {"role": "intended_use_decision", "path": "decision.json"}],
        "decision_scope": {"source": source, "files": scope,
                           "artifact_path": "decision.json"} if decision else "UNKNOWN",
        "lineage": {"dataset_versions": [], "model_versions": []},
    }
    manifest = build_manifest(root, specification)
    saved = tmp_path / "manifest.json"
    saved.write_bytes(canonical(manifest))
    return root, saved, manifest, specification


def multifile_fixture(tmp_path):
    root, saved, _, spec = fixture(tmp_path, mode="production")
    (root / "payload2.py").write_bytes(b"print('second exact source')\n")
    second = (root / "payload2.py").read_bytes()
    blob = hashlib.sha1(f"blob {len(second)}\0".encode() + second).hexdigest()
    files = sorted(spec["files"] + [
        dict(path="src/second.py", sha256=file_digest(root / "payload2.py"),
             git_blob=blob, artifact_path="payload2.py")], key=lambda x: x["path"])
    spec["files"] = files
    spec["artifacts"].append({"role": "source_file", "path": "payload2.py"})
    spec["decision_scope"] = dict(source=spec["source"], files=files,
                                  artifact_path="decision.json")
    rows = []
    for row in files:
        rows.append(dict(path=row["path"], sha256=row["sha256"],
                         git_blob=row["git_blob"],
                         bytes=(root / row["artifact_path"]).stat().st_size,
                         source_clearance_identity="sha256:" + "d" * 64,
                         file_clearance_identity="sha256:" + "e" * 64))
    decision = dict(schema="yug-source-training-use-decision-v2",
                    decision="APPROVE_TRAINING_USE", **spec["source"],
                    intended_use="commercial_training", split="train",
                    owner="code", subcategory="curated_subset", files=rows,
                    rights_review_identity="sha256:" + "f" * 64,
                    attribution="Retain publisher notice", notice_obligations=["retain NOTICE"],
                    obligations=["retain license"], max_files=len(rows),
                    max_raw_bytes=sum(row["bytes"] for row in rows),
                    max_accepted_tokens=1000, reviewer="test-only reviewer",
                    decided_at="2026-01-01T00:00:00Z",
                    governance_decision_identity="sha256:" + "1" * 64)
    (root / "decision.json").write_bytes(canonical(dict(decision, identity=identity(decision))))
    manifest = build_manifest(root, spec)
    saved.write_bytes(canonical(manifest))
    return root, saved, manifest, spec, decision


def _rewrite_decision(root, decision):
    (root / "decision.json").write_bytes(canonical(dict(decision, identity=identity(decision))))


def check(root, saved, manifest, **kwargs):
    return verify_bundle(root, saved, manifest["identity"], **kwargs)


def test_unchanged_development_bundle_passes_with_zero_authority(tmp_path):
    root, saved, manifest, spec = fixture(tmp_path)
    assert build_manifest(root, spec) == manifest
    result = check(root, saved, manifest)
    assert result["authority"] == "NONE"
    assert result["signature"] == "UNSIGNED_DEVELOPMENT_ONLY"
    assert result["timestamp"] == "MISSING"


@pytest.mark.parametrize("path,data", [
    ("license.txt", b"changed license"),
    ("decision.json", b"changed intended use"),
])
def test_changed_bound_evidence_fails(tmp_path, path, data):
    root, saved, manifest, _ = fixture(tmp_path)
    (root / path).write_bytes(data)
    with pytest.raises(EvidenceError, match="Bound evidence changed"):
        check(root, saved, manifest)


def test_missing_mandatory_evidence_fails(tmp_path):
    root, saved, manifest, _ = fixture(tmp_path)
    (root / "license.txt").unlink()
    with pytest.raises(EvidenceError):
        check(root, saved, manifest)


def test_changed_source_file_bytes_fail(tmp_path):
    root, saved, manifest, _ = fixture(tmp_path, mode="production")
    (root / "payload.py").write_bytes(b"print('substituted')\n")
    with pytest.raises(EvidenceError, match="Bound evidence changed"):
        check(root, saved, manifest)


@pytest.mark.parametrize("mutation", [
    lambda m: m["source"].update(revision="d" * 40),
    lambda m: m["files"][0].update(sha256="sha256:" + "e" * 64),
    lambda m: m["files"].append({"path": "src/other.py", "sha256": "sha256:" + "f" * 64,
                                 "git_blob": "a" * 40, "artifact_path": "other.py"}),
    lambda m: m["decision_scope"]["files"][0].update(sha256="sha256:" + "e" * 64),
])
def test_revision_hash_scope_or_decision_mutation_fails(tmp_path, mutation):
    root, saved, manifest, _ = fixture(tmp_path)
    mutation(manifest)
    saved.write_bytes(canonical(manifest))
    with pytest.raises(EvidenceError):
        check(root, saved, manifest)


def test_resealed_local_manifest_cannot_replace_external_pin(tmp_path):
    root, saved, manifest, _ = fixture(tmp_path)
    old = manifest["identity"]
    manifest["source"]["revision"] = "d" * 40
    manifest["decision_scope"]["source"]["revision"] = "d" * 40
    from training.production_corpus import identity
    manifest["identity"] = identity({k: v for k, v in manifest.items() if k != "identity"})
    saved.write_bytes(canonical(manifest))
    with pytest.raises(EvidenceError, match="External manifest pin mismatch"):
        verify_bundle(root, saved, old)


def test_missing_and_invalid_signature_fail_closed(tmp_path):
    root, saved, manifest, spec = fixture(tmp_path, mode="production")
    with pytest.raises(EvidenceError, match="trust record"):
        check(root, saved, manifest)
    pub = tmp_path / "public.pem"
    pub.write_text("invalid public key")
    pin_path, pin = _trust_pin(tmp_path, pub)
    with pytest.raises(EvidenceError, match="Signature"):
        check(root, saved, manifest, trust_pin_path=pin_path,
              expected_trust_pin_identity=pin["identity"], trust_root=tmp_path)
    signature = tmp_path / "signature.json"
    signature.write_bytes(canonical(dict(schema="yug-evidence-signature-v1",
        algorithm="Ed25519", manifest_identity=manifest["identity"],
        public_key_sha256=file_digest(pub), signature_base64=base64.b64encode(b"x" * 64).decode())))
    with pytest.raises(EvidenceError):
        check(root, saved, manifest, signature_path=signature,
              trust_pin_path=pin_path, expected_trust_pin_identity=pin["identity"],
              trust_root=tmp_path, openssl=_openssl() or "openssl")


def test_production_requires_external_trust_pin_not_bundled_public_key(tmp_path):
    root, saved, manifest, _ = fixture(tmp_path, mode="production")
    public = tmp_path / "bundled-public.pem"
    public.write_bytes(b"locally substituted key")
    with pytest.raises(EvidenceError, match="Direct public-key path"):
        check(root, saved, manifest, trusted_public_key=public)
    pin_path, pin = _trust_pin(tmp_path, public)
    with pytest.raises(EvidenceError, match="External trust-pin identity mismatch"):
        check(root, saved, manifest, trust_pin_path=pin_path,
              expected_trust_pin_identity="sha256:" + "0" * 64, trust_root=tmp_path)
    public.write_bytes(b"changed key")
    with pytest.raises(EvidenceError, match="fingerprint mismatch"):
        check(root, saved, manifest, trust_pin_path=pin_path,
              expected_trust_pin_identity=pin["identity"], trust_root=tmp_path)


def test_valid_multifile_scope_and_zero_authority(tmp_path):
    root, saved, manifest, spec, decision = multifile_fixture(tmp_path)
    assert len(manifest["files"]) == 2
    assert build_manifest(root, spec) == manifest
    # Production signature is still mandatory; no scope decision alone grants use.
    with pytest.raises(EvidenceError, match="Externally pinned production trust"):
        check(root, saved, manifest)


@pytest.mark.parametrize("mutation", [
    lambda d: d["files"][0].update(path="src/changed.py"),
    lambda d: d["files"].pop(),
    lambda d: d["files"].append(dict(d["files"][0], path="src/third.py")),
    lambda d: d["files"][1].update(sha256="sha256:" + "2" * 64),
    lambda d: d["files"][1].update(git_blob="3" * 40),
    lambda d: d.update(revision="4" * 40),
])
def test_multifile_decision_substitution_fails(tmp_path, mutation):
    root, saved, manifest, spec, decision = multifile_fixture(tmp_path)
    mutation(decision)
    _rewrite_decision(root, decision)
    with pytest.raises(EvidenceError):
        build_manifest(root, spec)
    with pytest.raises(EvidenceError, match="Bound evidence changed"):
        check(root, saved, manifest)


def test_multifile_clearance_identity_change_requires_new_external_pin(tmp_path):
    root, saved, manifest, spec, decision = multifile_fixture(tmp_path)
    decision["files"][1]["file_clearance_identity"] = "sha256:" + "5" * 64
    _rewrite_decision(root, decision)
    rebuilt = build_manifest(root, spec)
    assert rebuilt["identity"] != manifest["identity"]
    saved.write_bytes(canonical(rebuilt))
    with pytest.raises(EvidenceError, match="External manifest pin mismatch"):
        verify_bundle(root, saved, manifest["identity"])


@pytest.mark.parametrize("changes", [
    {"split": "validation"},
    {"intended_use": "evaluation"},
    {"attribution": "replacement notice"},
    {"notice_obligations": []},
])
def test_multifile_use_split_or_obligation_change_invalidates_pin(tmp_path, changes):
    root, saved, manifest, spec, decision = multifile_fixture(tmp_path)
    decision.update(changes)
    _rewrite_decision(root, decision)
    with pytest.raises(EvidenceError, match="Bound evidence changed"):
        check(root, saved, manifest)


@pytest.mark.parametrize("duplicate", ["src/example.py", "SRC/EXAMPLE.PY"])
def test_multifile_duplicate_or_case_colliding_path_fails(tmp_path, duplicate):
    root, _, _, spec, _ = multifile_fixture(tmp_path)
    spec["files"][1] = dict(spec["files"][1], path=duplicate)
    spec["decision_scope"] = dict(spec["decision_scope"], files=spec["files"])
    with pytest.raises(EvidenceError, match="Duplicate or unsorted"):
        build_manifest(root, spec)


def test_multifile_unsorted_or_unrelated_file_fails(tmp_path):
    root, _, _, spec, _ = multifile_fixture(tmp_path)
    spec["files"] = list(reversed(spec["files"]))
    spec["decision_scope"] = dict(spec["decision_scope"], files=spec["files"])
    with pytest.raises(EvidenceError, match="Duplicate or unsorted"):
        build_manifest(root, spec)
    spec["files"] = sorted(spec["files"], key=lambda r: r["path"])
    spec["files"].append(dict(spec["files"][1], path="src/unrelated.py"))
    spec["decision_scope"] = dict(spec["decision_scope"], files=spec["files"])
    with pytest.raises(EvidenceError):
        build_manifest(root, spec)


def test_multifile_source_bytes_and_git_blob_checked(tmp_path):
    root, saved, manifest, spec, decision = multifile_fixture(tmp_path)
    (root / "payload2.py").write_bytes(b"changed second file\n")
    with pytest.raises(EvidenceError, match="Bound evidence changed"):
        check(root, saved, manifest)
    (root / "payload2.py").unlink()
    with pytest.raises(EvidenceError):
        check(root, saved, manifest)
    (root / "payload2.py").write_bytes(b"print('second exact source')\n")
    spec["files"][1]["git_blob"] = "a" * 40
    decision["files"][1]["git_blob"] = "a" * 40
    _rewrite_decision(root, decision)
    rebuilt = build_manifest(root, spec)
    saved.write_bytes(canonical(rebuilt))
    with pytest.raises(EvidenceError, match="Git blob identity mismatch"):
        check(root, saved, rebuilt)


def test_valid_test_only_ed25519_signature_and_wrong_pin(tmp_path):
    root, saved, manifest, _ = fixture(tmp_path, mode="production")
    public, signature, openssl = _test_signature(tmp_path, manifest)
    pin_path, pin = _trust_pin(tmp_path, public)
    receipt, ca = _test_timestamp(tmp_path, manifest, openssl)
    assert check(root, saved, manifest, signature_path=signature,
                 trust_pin_path=pin_path, expected_trust_pin_identity=pin["identity"],
                 trust_root=tmp_path, timestamp_path=receipt, tsa_ca_file=ca,
                 openssl=openssl)["signature"] == "VERIFIED"
    with pytest.raises(EvidenceError):
        verify_bundle(root, saved, "sha256:" + "0" * 64, signature_path=signature,
                      trust_pin_path=pin_path, expected_trust_pin_identity=pin["identity"],
                      trust_root=tmp_path, openssl=openssl)


def test_production_timestamp_required_rejects_missing_receipt(tmp_path):
    root, saved, manifest, _ = fixture(tmp_path, mode="production")
    public, signature, openssl = _test_signature(tmp_path, manifest)
    pin_path, pin = _trust_pin(tmp_path, public)
    args = dict(signature_path=signature, trust_pin_path=pin_path,
                expected_trust_pin_identity=pin["identity"], trust_root=tmp_path,
                openssl=openssl)
    with pytest.raises(EvidenceError, match="TIMESTAMP_REQUIRED"):
        check(root, saved, manifest, **args)
    receipt, _ = _test_timestamp(tmp_path, manifest, openssl)
    with pytest.raises(EvidenceError, match="TIMESTAMP_REQUIRED"):
        check(root, saved, manifest, timestamp_path=receipt, **args)


def test_production_timestamp_required_rejects_wrong_imprint(tmp_path):
    root, saved, manifest, _ = fixture(tmp_path, mode="production")
    public, signature, openssl = _test_signature(tmp_path, manifest)
    pin_path, pin = _trust_pin(tmp_path, public)
    wrong = {"identity": "sha256:" + "0" * 64}
    receipt, ca = _test_timestamp(tmp_path, wrong, openssl)
    with pytest.raises(EvidenceError, match="Invalid or mismatched RFC 3161 timestamp"):
        check(root, saved, manifest, signature_path=signature,
              trust_pin_path=pin_path, expected_trust_pin_identity=pin["identity"],
              trust_root=tmp_path, timestamp_path=receipt, tsa_ca_file=ca,
              openssl=openssl)


def test_production_timestamp_required_accepts_verified_local_receipt(tmp_path):
    root, saved, manifest, _ = fixture(tmp_path, mode="production")
    public, signature, openssl = _test_signature(tmp_path, manifest)
    pin_path, pin = _trust_pin(tmp_path, public)
    receipt, ca = _test_timestamp(tmp_path, manifest, openssl)
    result = check(root, saved, manifest, signature_path=signature,
                   trust_pin_path=pin_path, expected_trust_pin_identity=pin["identity"],
                   trust_root=tmp_path, timestamp_path=receipt, tsa_ca_file=ca,
                   openssl=openssl)
    assert result["timestamp"] == "VERIFIED"
    assert result["authority"] == "NONE"


def test_development_cannot_use_production_trust_pin_to_skip_timestamp(tmp_path):
    root, saved, manifest, _ = fixture(tmp_path)
    public, signature, openssl = _test_signature(tmp_path, manifest)
    pin_path, pin = _trust_pin(tmp_path, public)
    with pytest.raises(EvidenceError, match="Production trust pin cannot verify development"):
        check(root, saved, manifest, signature_path=signature,
              trust_pin_path=pin_path, expected_trust_pin_identity=pin["identity"],
              trust_root=tmp_path, openssl=openssl)


def _test_ceremony(tmp_path):
    root, saved, _, spec, decision = multifile_fixture(tmp_path)
    spec["mode"] = "development"
    spec["source"]["source_id"] = "TEST_ONLY-example"
    decision["source_id"] = spec["source"]["source_id"]
    decision["attribution"] = "TEST_ONLY NON_PRODUCTION authority=NONE"
    _rewrite_decision(root, decision)
    manifest = build_manifest(root, spec)
    saved.write_bytes(canonical(manifest))
    key_id = "YUG-EVIDENCE-TEST-KEY-001"
    public, signature, openssl = _test_signature(
        tmp_path, manifest, schema="yug-evidence-signature-v2", key_id=key_id)
    pin = build_test_trust_pin(key_id, public.name, tmp_path)
    pin_path = tmp_path / "test-trust-pin.json"
    pin_path.write_bytes(canonical(pin))
    args = dict(signature_path=signature, test_trust_pin_path=pin_path,
                expected_test_trust_pin_identity=pin["identity"],
                trust_root=tmp_path, openssl=openssl)
    assert not (tmp_path / "ephemeral-test-only.pem").exists()
    return root, saved, manifest, spec, public, pin_path, pin, args


def test_distinct_test_only_multifile_verifier_is_zero_authority(tmp_path):
    root, saved, manifest, _, _, _, pin, args = _test_ceremony(tmp_path)
    result = verify_test_bundle(root, saved, manifest["identity"], **args)
    assert result["files"] == 2
    assert result["signature"] == "VERIFIED"
    assert result["key_id"] == "YUG-EVIDENCE-TEST-KEY-001"
    assert result["trust_pin_identity"] == pin["identity"]
    assert result["environment"] == "TEST_ONLY"
    assert result["production_eligible"] is False
    assert result["authority"] == "NONE"
    assert result["timestamp"] == "MISSING"


def test_test_pin_and_key_cannot_enter_production_verifier(tmp_path):
    root, saved, manifest, spec, _, _, _, args = _test_ceremony(tmp_path)
    with pytest.raises(EvidenceError, match="Unsupported trust-pin schema"):
        verify_bundle(root, saved, manifest["identity"],
                      signature_path=args["signature_path"],
                      trust_pin_path=args["test_trust_pin_path"],
                      expected_trust_pin_identity=args["expected_test_trust_pin_identity"],
                      trust_root=tmp_path, openssl=args["openssl"])
    spec["mode"] = "production"
    production = build_manifest(root, spec)
    saved.write_bytes(canonical(production))
    with pytest.raises(EvidenceError, match="Unsupported trust-pin schema"):
        verify_bundle(root, saved, production["identity"],
                      signature_path=args["signature_path"],
                      trust_pin_path=args["test_trust_pin_path"],
                      expected_trust_pin_identity=args["expected_test_trust_pin_identity"],
                      trust_root=tmp_path, openssl=args["openssl"])
    with pytest.raises(EvidenceError, match="TEST_ONLY development"):
        verify_test_bundle(root, saved, production["identity"], **args)


@pytest.mark.parametrize("field,value", [
    ("schema", TRUST_SCHEMA),
    ("environment", "PRODUCTION"),
    ("production_eligible", True),
    ("authority", "TRAINING"),
    ("key_id", "YUG-EVIDENCE-KEY-001"),
    ("public_key_sha256", "sha256:" + "0" * 64),
    ("status", "ACTIVE"),
])
def test_resealed_test_pin_promotion_or_substitution_fails(tmp_path, field, value):
    root, saved, manifest, _, _, pin_path, pin, args = _test_ceremony(tmp_path)
    pin[field] = value
    pin["identity"] = identity({k: v for k, v in pin.items() if k != "identity"})
    pin_path.write_bytes(canonical(pin))
    args["expected_test_trust_pin_identity"] = pin["identity"]
    with pytest.raises(EvidenceError):
        verify_test_bundle(root, saved, manifest["identity"], **args)


def test_test_pin_requires_external_identity_and_exact_public_bytes(tmp_path):
    root, saved, manifest, _, public, _, _, args = _test_ceremony(tmp_path)
    with pytest.raises(EvidenceError, match="External test trust-pin identity"):
        verify_test_bundle(root, saved, manifest["identity"],
                           **dict(args, expected_test_trust_pin_identity="sha256:" + "0" * 64))
    with pytest.raises(EvidenceError, match="Test trust pin"):
        verify_test_bundle(root, saved, manifest["identity"],
                           **dict(args, test_trust_pin_path=None))
    public.write_bytes(b"different public key")
    with pytest.raises(EvidenceError, match="fingerprint mismatch"):
        verify_test_bundle(root, saved, manifest["identity"], **args)


def test_test_verifier_rejects_production_pin_and_signature_key_swap(tmp_path):
    root, saved, manifest, _, public, _, _, args = _test_ceremony(tmp_path)
    production_pin_path, production_pin = _trust_pin(tmp_path, public)
    with pytest.raises(EvidenceError, match="Unsupported test trust-pin fields"):
        verify_test_bundle(root, saved, manifest["identity"],
                           **dict(args, test_trust_pin_path=production_pin_path,
                                  expected_test_trust_pin_identity=production_pin["identity"]))
    signature_path = args["signature_path"]
    signature = json.loads(signature_path.read_bytes())
    signature["key_id"] = "YUG-EVIDENCE-KEY-001"
    signature_path.write_bytes(canonical(signature))
    with pytest.raises(EvidenceError, match="test-only key ID"):
        verify_test_bundle(root, saved, manifest["identity"], **args)


@pytest.mark.parametrize("change", ["remove", "extra", "hash", "blob", "revision"])
def test_test_multifile_scope_mutations_fail(tmp_path, change):
    root, saved, manifest, _, _, _, _, args = _test_ceremony(tmp_path)
    changed = json.loads(saved.read_bytes())
    if change == "remove":
        changed["files"].pop()
    elif change == "extra":
        changed["files"].append(dict(changed["files"][-1], path="src/third.py"))
    elif change == "hash":
        changed["files"][0]["sha256"] = "sha256:" + "0" * 64
    elif change == "blob":
        changed["files"][0]["git_blob"] = "0" * 40
    else:
        changed["source"]["revision"] = "0" * 40
    changed["identity"] = identity({k: v for k, v in changed.items() if k != "identity"})
    saved.write_bytes(canonical(changed))
    with pytest.raises(EvidenceError):
        verify_test_bundle(root, saved, changed["identity"], **args)


def test_test_verifier_rejects_wrong_manifest_pin_and_decision(tmp_path):
    root, saved, manifest, _, _, _, _, args = _test_ceremony(tmp_path)
    with pytest.raises(EvidenceError, match="External manifest pin mismatch"):
        verify_test_bundle(root, saved, "sha256:" + "0" * 64, **args)
    (root / "decision.json").write_bytes(b"wrong decision")
    with pytest.raises(EvidenceError):
        verify_test_bundle(root, saved, manifest["identity"], **args)


def test_multifile_signature_key_id_and_key_lifecycle(tmp_path):
    root, saved, manifest, _, _ = multifile_fixture(tmp_path)
    public, signature, openssl = _test_signature(
        tmp_path, manifest, schema="yug-evidence-signature-v2")
    pin_path, pin = _trust_pin(tmp_path, public)
    receipt, ca = _test_timestamp(tmp_path, manifest, openssl)
    args = dict(signature_path=signature, trust_pin_path=pin_path,
                expected_trust_pin_identity=pin["identity"],
                trust_root=tmp_path, timestamp_path=receipt,
                tsa_ca_file=ca, openssl=openssl)
    result = check(root, saved, manifest, **args)
    assert result["authority"] == "NONE"
    assert result["key_id"] == "YUG-EVIDENCE-KEY-001"
    assert result["signature"] == "VERIFIED"
    signature_record = json.loads(signature.read_bytes())
    signature_record["key_id"] = "YUG-EVIDENCE-KEY-002"
    signature.write_bytes(canonical(signature_record))
    with pytest.raises(EvidenceError, match="signature identity"):
        check(root, saved, manifest, **args)
    signature_record["key_id"] = "YUG-EVIDENCE-KEY-001"
    signature.write_bytes(canonical(signature_record))
    pin_path, pin = _trust_pin(tmp_path, public, status="RETIRED")
    args.update(trust_pin_path=pin_path, expected_trust_pin_identity=pin["identity"])
    with pytest.raises(EvidenceError, match="not active"):
        check(root, saved, manifest, **args)
    assert check(root, saved, manifest, archival_only=True,
                 **args)["signature"] == "VERIFIED_RETIRED_ARCHIVAL_ONLY"
    pin_path, pin = _trust_pin(tmp_path, public, status="COMPROMISED")
    args.update(trust_pin_path=pin_path, expected_trust_pin_identity=pin["identity"])
    with pytest.raises(EvidenceError, match="not active"):
        check(root, saved, manifest, archival_only=True, **args)


def test_duplicate_trust_pin_fields_fail_closed(tmp_path):
    root, saved, manifest, _ = fixture(tmp_path, mode="production")
    public = tmp_path / "public.pem"
    public.write_bytes(b"test-only key placeholder")
    pin_path, pin = _trust_pin(tmp_path, public)
    pin_path.write_bytes(b'{"schema":"x","schema":"y"}')
    with pytest.raises(EvidenceError, match="Duplicate trust-pin field"):
        check(root, saved, manifest, trust_pin_path=pin_path,
              expected_trust_pin_identity=pin["identity"], trust_root=tmp_path)


def test_mismatched_timestamp_fails_and_does_not_contact_network(tmp_path):
    root, saved, manifest, _ = fixture(tmp_path)
    receipt = tmp_path / "unrelated.tsr"
    receipt.write_bytes(b"not a receipt for this identity")
    ca = tmp_path / "ca.pem"
    ca.write_bytes(b"not a TSA CA")
    with pytest.raises(EvidenceError, match="timestamp"):
        check(root, saved, manifest, timestamp_path=receipt, tsa_ca_file=ca,
              openssl=_openssl() or "openssl")


def test_legacy_unknown_remains_unknown_and_historical_artifacts_unchanged(tmp_path):
    root, saved, manifest, spec = fixture(tmp_path, decision=False)
    spec["source"]["revision"] = "UNKNOWN"
    spec["files"] = []
    spec["artifacts"] = [{"role": "license", "path": "license.txt"}]
    spec["decision_scope"] = "UNKNOWN"
    before = hashlib.sha256((root / "license.txt").read_bytes()).hexdigest()
    manifest = build_manifest(root, spec)
    saved.write_bytes(canonical(manifest))
    assert check(root, saved, manifest)["authority"] == "NONE"
    assert hashlib.sha256((root / "license.txt").read_bytes()).hexdigest() == before
    with pytest.raises(EvidenceError, match="Production source requires"):
        build_manifest(root, dict(spec, mode="production"))


def test_production_unknown_revision_and_empty_scope_fail(tmp_path):
    root, saved, manifest, spec = fixture(tmp_path, mode="production")
    spec["source"] = dict(spec["source"], revision="UNKNOWN")
    spec["decision_scope"] = dict(spec["decision_scope"], source=spec["source"])
    with pytest.raises(EvidenceError, match="exact revision"):
        build_manifest(root, spec)
    spec["source"] = manifest["source"]
    spec["files"] = []
    spec["decision_scope"] = dict(spec["decision_scope"], source=spec["source"], files=[])
    with pytest.raises(EvidenceError, match="nonempty file scope"):
        build_manifest(root, spec)


def test_duplicate_json_keys_rejected(tmp_path):
    root, saved, manifest, _ = fixture(tmp_path)
    saved.write_bytes(b'{"schema":"x","schema":"y"}')
    with pytest.raises(EvidenceError, match="Duplicate JSON key"):
        check(root, saved, manifest)
