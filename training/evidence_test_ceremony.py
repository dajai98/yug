"""Offline TEST_ONLY evidence verification; never a production trust path.

This module does not generate, decrypt, import, or sign with private keys.
It reuses the ordinary manifest, artifact, decision, and Ed25519 checks, but
accepts only development manifests and separately pinned test trust records.
"""
from __future__ import annotations

import argparse
import json

from training.evidence_bundle import (
    EvidenceError,
    _read_json,
    _verify_signature,
    need,
    verify_bundle,
)
from training.evidence_trust import TrustError, verify_test_trust_pin


def verify_test_bundle(root, manifest_path, expected_identity, *, signature_path,
                       test_trust_pin_path, expected_test_trust_pin_identity,
                       trust_root, openssl="openssl"):
    """Check a synthetic multi-file bundle and return no content authority."""
    try:
        pin, public_key = verify_test_trust_pin(
            test_trust_pin_path, expected_test_trust_pin_identity, trust_root)
    except TrustError as exc:
        raise EvidenceError(str(exc)) from exc
    manifest = _read_json(manifest_path)
    need(manifest.get("mode") == "development" and
         type(manifest.get("source")) is dict and
         type(manifest["source"].get("source_id")) is str and
         manifest["source"]["source_id"].startswith("TEST_ONLY-") and
         type(manifest.get("files")) is list and len(manifest["files"]) >= 2 and
         type(manifest.get("decision_scope")) is dict,
         "Test verifier requires a TEST_ONLY development multi-file scope")
    result = verify_bundle(root, manifest_path, expected_identity)
    artifacts = manifest.get("artifacts")
    need(type(artifacts) is list and
         {"source_inventory", "license", "intended_use_decision"}.issubset(
             {row.get("role") for row in artifacts if type(row) is dict}) and
         all(any(type(row) is dict and row.get("role") == "source_file" and
                     row.get("path") == file["artifact_path"] and
                     row.get("sha256") == file["sha256"]
                 for row in artifacts) for file in manifest["files"]),
         "Test scope lacks exact source-file evidence")
    signature = _read_json(signature_path, 8192)
    need(signature.get("schema") == "yug-evidence-signature-v2" and
         signature.get("key_id") == pin["key_id"],
         "Test signature must bind the test-only key ID")
    status = _verify_signature(expected_identity, signature_path, public_key,
                               openssl, key_id=pin["key_id"])
    return dict(result, signature=status, key_id=pin["key_id"],
                trust_pin_identity=pin["identity"], environment="TEST_ONLY",
                production_eligible=False, authority="NONE")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline TEST_ONLY evidence check")
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-identity", required=True)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--test-trust-pin", required=True)
    parser.add_argument("--expected-test-trust-pin-identity", required=True)
    parser.add_argument("--trust-root", required=True)
    parser.add_argument("--openssl", default="openssl")
    args = parser.parse_args(argv)
    result = verify_test_bundle(
        args.root, args.manifest, args.expected_identity,
        signature_path=args.signature, test_trust_pin_path=args.test_trust_pin,
        expected_test_trust_pin_identity=args.expected_test_trust_pin_identity,
        trust_root=args.trust_root, openssl=args.openssl)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
