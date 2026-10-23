"""Tests for training/experiments/rep_corpus_sources.py (Stage-1 registry).

Stdlib-only tests run anywhere. The three repo-coupled tests (policy mirror, pins
vs the GitHub manifest, ids/statuses vs the prior registry) and the architecture
boundary test skip when those files are absent; a skip is NOT a pass.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import re
from pathlib import Path

import pytest

from training.experiments import rep_corpus_sources as rs

REGISTRY = rs.build_registry()
BY_ID = {x.source_id: x for x in REGISTRY}


def _replace(record, **kw):
    return dataclasses.replace(record, **kw)


def test_registry_validates_and_is_nonempty():
    rs.validate_registry(REGISTRY)
    assert len(REGISTRY) >= 40
    assert {x.category for x in REGISTRY} == set(rs.CATEGORIES)


def test_deterministic_order_and_fingerprint():
    assert rs.build_registry() == REGISTRY
    fp = rs.registry_fingerprint()
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", fp)
    assert fp == rs.registry_fingerprint()
    order = [(rs.CATEGORIES.index(x.category), x.source_id) for x in REGISTRY]
    assert order == sorted(order)
    changed = list(REGISTRY)
    changed[0] = _replace(changed[0], status_reason=changed[0].status_reason + " x")
    assert rs.registry_fingerprint(changed) != fp


def test_allowlist_is_frozen():
    assert rs.ALLOWED_LICENSE_IDS == frozenset(
        {"CC-BY-SA-4.0", "CC-BY-SA-3.0", "CC-BY-4.0", "CC-BY-3.0", "MIT", "Apache-2.0", "MS-PL"})


def test_accept_only_with_allowlisted_license_pin_and_evidence():
    accepted = [x for x in REGISTRY if x.status == rs.ACCEPT]
    assert len(accepted) == 12
    for x in accepted:
        assert x.license_id in rs.ALLOWED_LICENSE_IDS
        assert x.license_verification != rs.NOT_VERIFIED
        assert x.pin and x.pin_kind != rs.PIN_NONE and x.max_category_share > 0
        assert x.license_evidence.strip()


def test_non_accept_is_never_eligible_and_blocked_has_zero_cap():
    for x in REGISTRY:
        flags = (x.tokenizer_fit_eligible, x.model_training_eligible, x.validation_eligible)
        if x.status != rs.ACCEPT:
            assert not any(flags), x.source_id
            assert x.effective_cap == 0.0
        if x.status == rs.BLOCKED:
            assert x.max_category_share == 0.0
    for purpose in ("fit", "train", "validation"):
        assert all(x.status == rs.ACCEPT for x in rs.eligible_sources(REGISTRY, purpose))
    with pytest.raises(rs.SourceRegistryError):
        rs.eligible_sources(REGISTRY, "bogus")


def test_unity_restrictions():
    for sid in ("unity-docs", "unity-asset-store-academy", "unity-learn", "gh-unity-csreference"):
        assert BY_ID[sid].status == rs.BLOCKED
    assert BY_ID["gh-unity-ml-agents"].status == rs.REVIEW
    assert all(x.status != rs.ACCEPT for x in REGISTRY if x.unity_authored)
    unity_authored = {x.source_id for x in REGISTRY if x.unity_authored}
    assert unity_authored == {"gh-unity-ml-agents", "gh-unity-csreference", "unity-docs",
                              "unity-asset-store-academy", "unity-learn"}


def test_policy_gated_licenses_are_not_accepted():
    for sid in ("gh-pallets-flask", "gh-encode-httpx", "python-docs-cpython-doc", "mdn-content"):
        assert BY_ID[sid].status == rs.REVIEW and BY_ID[sid].owner_decision_ids
    assert BY_ID["docker-docs"].status == rs.UNKNOWN
    for x in REGISTRY:
        if x.status == rs.REVIEW and x.license_verification == rs.NOT_VERIFIED:
            assert not x.tokenizer_fit_eligible


def test_synthetic_and_natural_codeswitch():
    syn = BY_ID["synthetic-codeswitch-diagnostic"]
    assert syn.diagnostic_only and not (syn.tokenizer_fit_eligible or syn.model_training_eligible or syn.validation_eligible)
    nat = BY_ID["natural-codeswitch-none"]
    assert nat.status == rs.UNKNOWN and nat.effective_cap == 0.0
    assert rs.coverage_ceiling(REGISTRY)[rs.CAT_CODESWITCH] == 0.0


def test_existing_sources_keep_prior_status():
    existing_accept = {"gh-psf-requests", "gh-expressjs-express", "gh-gin-gonic-gin", "gh-burntsushi-ripgrep",
                       "gh-cysharp-unitask", "gh-mirrornetworking-mirror", "gh-monogame-monogame",
                       "gh-google-eng-practices", "gh-microsoft-api-guidelines", "gh-rust-lang-book",
                       "cp-wikimedia-filtered", "hf-wikipedia-ne-20231101"}
    assert {x.source_id for x in REGISTRY if x.status == rs.ACCEPT} == existing_accept
    for x in REGISTRY:
        if x.prior_status in (rs.ACCEPT, rs.REVIEW, rs.BLOCKED):
            assert x.status == x.prior_status, x.source_id
        elif x.prior_status is not None:
            assert (x.status_change_note or "").strip(), x.source_id


def test_github_pins_are_40_hex():
    for x in REGISTRY:
        if x.status == rs.ACCEPT and x.pin_kind == rs.PIN_COMMIT:
            assert re.fullmatch(r"[0-9a-f]{40}", x.pin)


def test_owner_decisions_are_referenced_and_valid():
    ids = {d.decision_id for d in rs.OWNER_DECISIONS}
    assert len(ids) == len(rs.OWNER_DECISIONS)
    for x in REGISTRY:
        assert set(x.owner_decision_ids) <= ids


def test_validator_rejects_bad_records():
    flask = BY_ID["gh-pallets-flask"]
    requests_ = BY_ID["gh-psf-requests"]
    ml = BY_ID["gh-unity-ml-agents"]
    bad_cases = [
        _replace(flask, status=rs.ACCEPT),                                    # BSD not allow-listed
        _replace(flask, tokenizer_fit_eligible=True),                         # REVIEW but eligible
        _replace(requests_, license_verification=rs.NOT_VERIFIED),            # unverified ACCEPT
        _replace(requests_, pin=None, pin_kind=rs.PIN_NONE),                  # unpinned ACCEPT
        _replace(requests_, pin="abc"),                                       # malformed pin
        _replace(requests_, max_category_share=0.0),                          # ACCEPT with no cap
        _replace(ml, status=rs.ACCEPT),                                       # Unity-authored ACCEPT
        _replace(BY_ID["unity-docs"], max_category_share=0.1),                # BLOCKED with cap
        _replace(BY_ID["synthetic-codeswitch-diagnostic"], model_training_eligible=True),
        _replace(requests_, status_reason=" "),                               # empty reason
        _replace(BY_ID["web-crawl-nepali"], status=rs.UNKNOWN),               # silent status change
        _replace(flask, owner_decision_ids=("OD-99",)),                       # unknown decision
        _replace(flask, category="nonsense"),
    ]
    rs.validate_registry([requests_, flask, ml], check_decision_coverage=False)  # unmodified records pass
    for bad in bad_cases:
        with pytest.raises(rs.SourceRegistryError):
            rs.validate_registry([bad], check_decision_coverage=False)
    with pytest.raises(rs.SourceRegistryError):
        rs.validate_registry([requests_, requests_], check_decision_coverage=False)  # duplicate id
    with pytest.raises(rs.SourceRegistryError):
        rs.validate_registry([requests_])                                     # decisions unreferenced


def test_coverage_ceiling_accept_only_is_below_target_for_most_categories():
    accept = rs.coverage_ceiling(REGISTRY)
    assert accept[rs.CAT_CODE] == pytest.approx(0.40)
    assert accept[rs.CAT_CSHARP] == pytest.approx(0.30)
    assert accept[rs.CAT_DOCS] == pytest.approx(0.45)
    assert accept[rs.CAT_ENGLISH] == pytest.approx(0.50)
    assert accept[rs.CAT_NEPALI] == pytest.approx(1.0)
    both = rs.coverage_ceiling(REGISTRY, include=(rs.ACCEPT, rs.REVIEW))
    assert both[rs.CAT_CODE] == pytest.approx(1.0)
    assert both[rs.CAT_ENGLISH] == pytest.approx(0.85)
    assert all(v <= 1.0 for v in both.values())


def test_concentration_proposal_is_labelled():
    ids = [c.rule_id for c in rs.CONCENTRATION_PROPOSAL]
    assert len(ids) == len(set(ids))
    assert all(c.label in {"PROPOSAL", "CALCULATED", "ASSUMPTION"} for c in rs.CONCENTRATION_PROPOSAL)


def test_cli_roundtrip(tmp_path):
    out = tmp_path / "registry.json"
    assert rs.main(["--output", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["not_a_milestone"] is True and data["nb13_assigned"] is False
    assert data["fingerprint"] == rs.registry_fingerprint()
    assert len(data["records"]) == len(REGISTRY)


def test_module_is_stdlib_only():
    tree = ast.parse(Path(rs.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert not ({"yug", "training", "torch"} & roots)


# ---- repo-coupled tests: skip (not pass) when the repo files are absent ----

def test_allowlist_matches_mini_corpus_policy():
    mod = pytest.importorskip("training.experiments.representative_mini_corpus")
    policy = mod.MINI_CORPUS_LICENSE_POLICY
    allowed = {k for k, v in policy.decisions.items() if v.name == "ALLOW"}
    assert allowed == set(rs.ALLOWED_LICENSE_IDS)
    assert policy.default.name == "REVIEW"


def test_existing_records_match_prior_registry_and_manifest():
    mod = pytest.importorskip("training.experiments.representative_mini_corpus")
    prior = {s.source_id: s for s in mod.build_source_registry()}
    manifest = json.loads(Path(mod._GITHUB_MANIFEST_PATH).read_text(encoding="utf-8"))
    commits = {s["source_id"]: s["commit"] for s in manifest["sources"]}
    for x in REGISTRY:
        if x.source_id in prior:
            p = prior[x.source_id]
            assert x.prior_status == p.status, x.source_id
            if p.status == "ACCEPT" and x.source_id in commits:
                assert x.pin == commits[x.source_id], x.source_id
                assert x.license_id == p.license_id, x.source_id
    assert commits.keys() <= {x.source_id for x in REGISTRY}


def test_production_yug_does_not_import_training():
    yug_dir = Path(__file__).resolve().parents[1] / "yug"
    if not yug_dir.is_dir():
        pytest.skip("yug/ not found relative to tests/")
    pattern = re.compile(r"^\s*(from|import)\s+training(\.|\s|$)", re.MULTILINE)
    offenders = [str(p) for p in yug_dir.rglob("*.py") if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


# ---------------- Stage 1B additions ----------------

_ARTIFACT = Path(__file__).resolve().parents[1] / "training" / "resources" / "phase1" / "rep_corpus_10m_v1" / "00_source_license_evidence.json"


def test_stage1b_changed_no_statuses():
    stage1a_status = {
        "gh-pallets-flask": "REVIEW", "gh-encode-httpx": "REVIEW", "gh-lodash-lodash": "REVIEW", "gh-spf13-cobra": "REVIEW",
        "gh-tokio-rs-tokio": "REVIEW", "gh-serde-rs-serde": "REVIEW", "gh-google-guava": "REVIEW", "gh-nlohmann-json": "REVIEW",
        "gh-flame-engine-flame": "REVIEW", "gh-dotnet-runtime": "REVIEW", "gh-dotnet-roslyn": "REVIEW",
        "gh-stride3d-stride": "REVIEW", "gh-fna-xna-fna": "REVIEW", "gh-neuecc-unirx": "REVIEW",
        "gh-messagepack-csharp": "REVIEW", "gh-unity-ml-agents": "REVIEW", "gh-rust-lang-reference": "REVIEW",
        "gh-rust-lang-rust-by-example": "REVIEW", "gh-rust-lang-rustc-dev-guide": "REVIEW",
        "python-docs-cpython-doc": "REVIEW", "gh-dotnet-docs": "REVIEW", "mdn-content": "REVIEW",
        "gh-kubernetes-website": "REVIEW", "docker-docs": "UNKNOWN", "hf-wikipedia-en-20231101": "REVIEW",
    }
    for sid, status in stage1a_status.items():
        assert BY_ID[sid].status == status, sid
    assert len([x for x in REGISTRY if x.status == rs.ACCEPT]) == 12
    assert "gh-cysharp-messagepack-csharp" not in BY_ID  # Stage 1A named a repository that does not exist


def test_stage1b_facts_cover_every_github_source_and_match_helper_table():
    from training.experiments import rep_corpus_license_verify as lv
    assert set(rs._STAGE1B_FACTS) == set(lv.GITHUB_SOURCES)
    for sid in lv.GITHUB_SOURCES:
        assert sid in BY_ID, sid


def test_stage1b_existing_accepts_are_real_fetch_verified_at_their_pins():
    for x in REGISTRY:
        if x.status == rs.ACCEPT and x.source_id in rs._STAGE1B_FACTS:
            assert x.license_verification == rs.REAL_FETCH
            assert x.evidence_commit == x.pin
            assert x.license_text_sha256.startswith("sha256:") and x.repository_license_verified
            assert x.license_id in rs.ALLOWED_LICENSE_IDS


def test_stage1b_proposals_are_not_promotions():
    proposed = [x for x in REGISTRY if x.proposed_status == rs.ACCEPT and x.status != rs.ACCEPT]
    assert {x.source_id for x in proposed} == {
        "gh-lodash-lodash", "gh-spf13-cobra", "gh-tokio-rs-tokio", "gh-serde-rs-serde", "gh-google-guava",
        "gh-flame-engine-flame", "gh-dotnet-roslyn", "gh-neuecc-unirx", "gh-rust-lang-reference",
        "gh-rust-lang-rust-by-example", "gh-rust-lang-rustc-dev-guide"}
    for x in proposed:
        assert x.status == rs.REVIEW and not (x.tokenizer_fit_eligible or x.model_training_eligible or x.validation_eligible)
        assert x.license_id in rs.ALLOWED_LICENSE_IDS and x.pin and re.fullmatch(r"[0-9a-f]{40}", x.pin)
    for sid in ("gh-pallets-flask", "gh-encode-httpx", "python-docs-cpython-doc", "mdn-content", "gh-unity-ml-agents",
                "gh-messagepack-csharp", "gh-kubernetes-website", "gh-dotnet-docs", "gh-fna-xna-fna"):
        assert BY_ID[sid].proposed_status == rs.REVIEW, sid


def test_stage1b_licenses_found_by_fetch_not_assumed():
    assert BY_ID["gh-messagepack-csharp"].license_id == "Apache-2.0"
    assert BY_ID["gh-kubernetes-website"].license_id == "CC-BY-4.0"
    assert BY_ID["gh-rust-lang-book"].license_id == "MIT"
    assert BY_ID["mdn-content"].license_id == "CC-BY-SA-2.5"
    assert BY_ID["python-docs-cpython-doc"].license_id == "PSF-2.0"
    assert BY_ID["gh-pallets-flask"].license_id == "BSD-3-Clause"
    assert "OR" in BY_ID["gh-burntsushi-ripgrep"].license_expression
    assert BY_ID["docker-docs"].status == rs.UNKNOWN


def test_stage1b_unity_rules_unchanged():
    assert BY_ID["gh-unity-ml-agents"].status == rs.REVIEW and BY_ID["gh-unity-ml-agents"].proposed_status == rs.REVIEW
    for sid in ("unity-docs", "unity-asset-store-academy", "unity-learn", "gh-unity-csreference"):
        assert BY_ID[sid].status == rs.BLOCKED


def test_validator_rejects_bad_promotion_proposals():
    flask = BY_ID["gh-pallets-flask"]
    lodash = BY_ID["gh-lodash-lodash"]
    bad = [
        _replace(flask, proposed_status=rs.ACCEPT),                      # BSD not allow-listed
        _replace(lodash, license_verification=rs.NOT_VERIFIED),          # no real-fetch evidence
        _replace(lodash, pin=None),                                      # no pin
        _replace(BY_ID["gh-unity-ml-agents"], proposed_status=rs.ACCEPT),
        _replace(lodash, proposed_status="MAYBE"),
        _replace(lodash, file_level_license_uniformity="TOTALLY"),
        _replace(lodash, license_text_sha256="abc"),
    ]
    rs.validate_registry([lodash], check_decision_coverage=False)
    for b in bad:
        with pytest.raises(rs.SourceRegistryError):
            rs.validate_registry([b], check_decision_coverage=False)


def test_proposed_coverage_ceiling_is_planning_only():
    c = rs.proposed_coverage_ceiling(REGISTRY)
    assert c[rs.CAT_CODE] == pytest.approx(1.0)
    assert c[rs.CAT_CSHARP] == pytest.approx(0.50)
    assert c[rs.CAT_DOCS] == pytest.approx(0.90)
    assert c[rs.CAT_ENGLISH] == pytest.approx(0.50)
    assert c[rs.CAT_NEPALI] == pytest.approx(1.0)
    assert rs.coverage_ceiling(REGISTRY)[rs.CAT_CODE] == pytest.approx(0.40)  # unchanged: nothing was promoted


def test_artifact_matches_registry_facts():
    if not _ARTIFACT.is_file():
        pytest.skip(f"evidence artifact not present at {_ARTIFACT} (it may be gitignored)")
    from training.experiments import rep_corpus_license_verify as lv
    data = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    assert data["fingerprint"] == lv.evidence_fingerprint(data) == rs.STAGE1B_EVIDENCE_FINGERPRINT
    assert data["not_a_milestone"] is True and data["nb13_assigned"] is False
    by = {s["source_id"]: s for s in data["sources"]}
    for sid, fact in rs._STAGE1B_FACTS.items():
        s = by[sid]
        assert s["commit"] == fact["commit"]
        shas = {f["path"]: f["sha256"] for f in s["license_files"]}
        assert shas[fact["primary_path"]] == fact["sha256"]