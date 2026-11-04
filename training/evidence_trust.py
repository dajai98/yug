"""Externally pinned, zero-authority public-key trust records for evidence bundles.

This module cannot create or import private keys. A locally created record is
not trusted unless its exact identity comes from an independent authority.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from training.production_corpus import canonical, file_digest, identity

SCHEMA = "yug-evidence-trust-pin-v1"
KEY_ID = re.compile(r"YUG-EVIDENCE-KEY-[0-9]{3,}\Z")
TEST_SCHEMA = "yug-evidence-test-trust-pin-v1"
TEST_KEY_ID = re.compile(r"YUG-EVIDENCE-TEST-KEY-[0-9]{3,}\Z")
SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
STATUSES = frozenset({"ACTIVE", "RETIRED", "REVOKED", "COMPROMISED"})
MAX_PIN_BYTES = 8192


class TrustError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise TrustError(message)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate trust-pin field")
        result[key] = value
    return result


def _utc(value):
    require(type(value) is str and value.endswith("Z"), "UTC trust-pin time required")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise TrustError("Invalid trust-pin time") from exc
    require(parsed.utcoffset().total_seconds() == 0, "UTC trust-pin time required")
    return parsed


def _public_key(root, name):
    require(type(name) is str and name and "\\" not in name and ":" not in name
            and not name.startswith("/") and all(part not in ("", ".", "..")
                                                 for part in name.split("/")),
            "Invalid public-key reference")
    root = Path(root).resolve()
    path = (root / name).resolve()
    require(path.is_relative_to(root) and path.is_file(),
            "Trusted public key missing or outside trust root")
    return path


def verify_trust_pin(pin_path, externally_trusted_identity, trust_root,
                     *, archival_only=False):
    """Return a pinned public key; never grants source or training authority."""
    require(type(externally_trusted_identity) is str and
            SHA.fullmatch(externally_trusted_identity),
            "External trust-pin identity required")
    try:
        raw = Path(pin_path).read_bytes()
        require(0 < len(raw) <= MAX_PIN_BYTES, "Missing or oversized trust pin")
        pin = json.loads(raw, object_pairs_hook=_unique)
    except (OSError, ValueError, UnicodeError) as exc:
        raise TrustError(f"Invalid trust pin: {exc}") from exc
    require(type(pin) is dict and raw == canonical(pin), "Noncanonical trust pin")
    require(set(pin) == {"schema", "key_id", "algorithm", "public_key_sha256",
                         "public_key_path", "status", "valid_from", "valid_until",
                         "predecessor_key_id", "governance_decision_identity", "identity"}
            and pin["schema"] == SCHEMA and pin["algorithm"] == "Ed25519",
            "Unsupported trust-pin schema or algorithm")
    require(type(pin["key_id"]) is str and KEY_ID.fullmatch(pin["key_id"])
            and type(pin["public_key_sha256"]) is str and
            SHA.fullmatch(pin["public_key_sha256"])
            and type(pin["governance_decision_identity"]) is str and
            SHA.fullmatch(pin["governance_decision_identity"]),
            "Invalid trust-pin identity fields")
    require(pin["predecessor_key_id"] == "NONE" or
            (type(pin["predecessor_key_id"]) is str and
             KEY_ID.fullmatch(pin["predecessor_key_id"]) and
             pin["predecessor_key_id"] != pin["key_id"]),
            "Invalid predecessor key")
    require(type(pin["status"]) is str and pin["status"] in STATUSES,
            "Unsupported key status")
    start = _utc(pin["valid_from"])
    end = None if pin["valid_until"] == "UNKNOWN" else _utc(pin["valid_until"])
    require(end is None or end > start, "Invalid trust-pin validity interval")
    payload = {key: value for key, value in pin.items() if key != "identity"}
    require(pin["identity"] == identity(payload), "Trust-pin seal mismatch")
    require(pin["identity"] == externally_trusted_identity,
            "External trust-pin identity mismatch")
    key = _public_key(trust_root, pin["public_key_path"])
    require(file_digest(key) == pin["public_key_sha256"],
            "Trusted public-key fingerprint mismatch")
    require(pin["status"] == "ACTIVE" or
            (archival_only and pin["status"] == "RETIRED"),
            "Key is not active for production verification")
    if pin["status"] == "ACTIVE":
        now = datetime.now(timezone.utc)
        require(start <= now and (end is None or now < end),
                "Key is outside active validity interval")
    return pin, key


def build_test_trust_pin(key_id, public_key_path, trust_root):
    """Construct an in-memory TEST_ONLY pin; never creates production trust."""
    require(type(key_id) is str and TEST_KEY_ID.fullmatch(key_id),
            "Test-only key ID required")
    key = _public_key(trust_root, public_key_path)
    payload = dict(schema=TEST_SCHEMA, environment="TEST_ONLY",
                   production_eligible=False, authority="NONE", key_id=key_id,
                   algorithm="Ed25519", public_key_sha256=file_digest(key),
                   public_key_path=public_key_path, status="TEST_ONLY")
    return dict(payload, identity=identity(payload))


def verify_test_trust_pin(pin_path, externally_trusted_identity, trust_root):
    """Verify an externally pinned disposable key; no production status exists."""
    require(pin_path is not None and trust_root is not None,
            "Test trust pin and public-key root required")
    require(type(externally_trusted_identity) is str and
            SHA.fullmatch(externally_trusted_identity),
            "External test trust-pin identity required")
    try:
        raw = Path(pin_path).read_bytes()
        require(0 < len(raw) <= MAX_PIN_BYTES, "Missing or oversized test trust pin")
        pin = json.loads(raw, object_pairs_hook=_unique)
    except (OSError, ValueError, UnicodeError) as exc:
        raise TrustError(f"Invalid test trust pin: {exc}") from exc
    require(type(pin) is dict and raw == canonical(pin), "Noncanonical test trust pin")
    require(set(pin) == {"schema", "environment", "production_eligible",
                         "authority", "key_id", "algorithm", "public_key_sha256",
                         "public_key_path", "status", "identity"},
            "Unsupported test trust-pin fields")
    require(pin["schema"] == TEST_SCHEMA and pin["environment"] == "TEST_ONLY"
            and pin["production_eligible"] is False and pin["authority"] == "NONE"
            and pin["status"] == "TEST_ONLY" and pin["algorithm"] == "Ed25519"
            and type(pin["key_id"]) is str and TEST_KEY_ID.fullmatch(pin["key_id"])
            and type(pin["public_key_sha256"]) is str and
            SHA.fullmatch(pin["public_key_sha256"]),
            "Invalid or production-like test trust pin")
    payload = {key: value for key, value in pin.items() if key != "identity"}
    require(pin["identity"] == identity(payload), "Test trust-pin seal mismatch")
    require(pin["identity"] == externally_trusted_identity,
            "External test trust-pin identity mismatch")
    key = _public_key(trust_root, pin["public_key_path"])
    require(file_digest(key) == pin["public_key_sha256"],
            "Test public-key fingerprint mismatch")
    return pin, key
