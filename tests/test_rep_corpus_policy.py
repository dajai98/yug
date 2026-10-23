"""Tests for training/experiments/rep_corpus_policy.py (Stage 1C source-policy freeze).

Offline and stdlib-only. Tests that need repository files absent from a checkout skip
(a skip is NOT a pass).
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

from training.experiments import rep_corpus_policy as pol
from training.experiments import rep_corpus_sources as rs

REGISTRY = rs.build_registry()
BY_ID = {x.source_id: x for x in REGISTRY}
ENTRIES = pol.build_entries(REGISTRY)
E = {e.source_id: e for e in ENTRIES}
_ROOT = Path(__file__).resolve().parents[1]
_POLICY_ARTIFACT = _ROOT / pol.POLICY_ARTIFACT
_EVIDENCE_ARTIFACT = _ROOT / pol.EVIDENCE_ARTIFACT


# ---------- OD-08 and category targets ----------

def test_od08_totals_exactly_100_and_values_are_frozen():
    assert pol.OD08_LANGUAGE_TARGET_PCT == {"python": 35, "rust": 25, "c_cpp": 15, "javascript_typescript": 15, "go": 10}
    assert sum(pol.OD08_LANGUAGE_TARGET_PCT.values()) == 100
    pol.validate_language_targets()
    for bad in ({"python": 35, "rust": 25, "c_cpp": 15, "javascript_typescript": 15, "go": 9},
                {"python": 36, "rust": 25, "c_cpp": 15, "javascript_typescript": 15, "go": 10},
                {"python": 0, "rust": 60, "c_cpp": 15, "javascript_typescript": 15, "go": 10}):
        with pytest.raises(pol.PolicyError):
            pol.validate_language_targets(bad)


def test_category_targets_are_the_owner_working_mix():
    assert pol.CATEGORY_TARGET_PCT == {rs.CAT_CODE: 24, rs.CAT_CSHARP: 22, rs.CAT_ENGLISH: 19, rs.CAT_DOCS: 18, rs.CAT_NEPALI: 17}
    assert sum(pol.CATEGORY_TARGET_PCT.values()) == 100
    total = sum(pol.category_target_bytes(c) for c in pol.CATEGORY_TARGET_PCT)
    assert total == pol.PLANNING_TOTAL_TRAIN_BYTES == 41_000_000


def test_csharp_cannot_enter_the_general_code_mix():
    with pytest.raises(pol.PolicyError):
        pol.od08_language_for(rs.CAT_CSHARP, "general_csharp_dotnet")
    with pytest.raises(pol.PolicyError):
        pol.od08_language_for(rs.CAT_CODE, "csharp")
    with pytest.raises(pol.PolicyError):
        pol.validate_language_targets({"python": 50, "csharp": 50})
    assert all(e.od08_language is None for e in ENTRIES if e.category == rs.CAT_CSHARP)
    assert not any((e.subcategory or "").lower().startswith("csharp") for e in ENTRIES if e.category == rs.CAT_CODE)


def test_unknown_languages_cannot_silently_enter():
    for lang in ("zig", "java", "dart_game", "kotlin", "", None):
        with pytest.raises(pol.PolicyError):
            pol.od08_language_for(rs.CAT_CODE, lang)
    assert pol.od08_language_for(rs.CAT_CODE, "cpp") == "c_cpp"
    for sid in ("gh-google-guava", "gh-flame-engine-flame"):
        assert E[sid].stage1c_status == rs.REVIEW and E[sid].od08_language is None
    for e in ENTRIES:
        if e.stage1c_status == rs.ACCEPT and e.category == rs.CAT_CODE:
            assert e.od08_language in pol.OD08_LANGUAGE_TARGET_PCT
    bad = dataclasses.replace(E["gh-psf-requests"], od08_language=None)
    with pytest.raises(pol.PolicyError):
        pol.validate_policy([bad])
    bad = dataclasses.replace(E["gh-psf-requests"], od08_language="java")
    with pytest.raises(pol.PolicyError):
        pol.validate_policy([bad])


# ---------- resolution of the 11-vs-10 discrepancy ----------

def test_stage1b_proposed_accept_set_is_exactly_eleven_and_flame_is_the_extra():
    proposed = {x.source_id for x in REGISTRY if x.proposed_status == rs.ACCEPT and x.status != rs.ACCEPT}
    assert len(proposed) == 11
    assert "gh-flame-engine-flame" in proposed
    owner_list_of_ten = {"gh-lodash-lodash", "gh-spf13-cobra", "gh-tokio-rs-tokio", "gh-serde-rs-serde", "gh-google-guava",
                         "gh-dotnet-roslyn", "gh-neuecc-unirx", "gh-rust-lang-reference", "gh-rust-lang-rust-by-example",
                         "gh-rust-lang-rustc-dev-guide"}
    assert proposed - owner_list_of_ten == {"gh-flame-engine-flame"}
    assert owner_list_of_ten <= proposed
    assert set(pol.STAGE1C_PROMOTIONS) | set(pol.STAGE1C_HELD_PROPOSALS) == proposed
    assert not (set(pol.STAGE1C_PROMOTIONS) & set(pol.STAGE1C_HELD_PROPOSALS))


def test_minimal_promotion_set():
    assert set(pol.STAGE1C_PROMOTIONS) == {"gh-lodash-lodash", "gh-spf13-cobra", "gh-tokio-rs-tokio", "gh-serde-rs-serde",
                                           "gh-rust-lang-rust-by-example", "gh-dotnet-roslyn", "gh-neuecc-unirx"}
    for sid in pol.STAGE1C_PROMOTIONS:
        assert E[sid].stage1c_status == rs.ACCEPT and E[sid].status_authority == "STAGE1C_PROMOTION"
        assert E[sid].registry_status == rs.REVIEW
    for sid in pol.STAGE1C_HELD_PROPOSALS:
        assert E[sid].stage1c_status == rs.REVIEW and E[sid].status_authority == "NOT_PROMOTED"
        assert E[sid].owner_decision_ids


def test_every_promoted_source_has_stage1b_evidence_and_a_frozen_commit():
    for sid in pol.STAGE1C_PROMOTIONS:
        x = BY_ID[sid]
        assert x.license_verification == rs.REAL_FETCH
        assert (x.license_text_sha256 or "").startswith("sha256:")
        assert re.fullmatch(r"[0-9a-f]{40}", x.pin or "") and x.pin == x.evidence_commit
        assert x.license_id in rs.ALLOWED_LICENSE_IDS
        assert pol.promotion_problems(x) == []
        assert E[sid].evidence_ref and E[sid].license_text_sha256 == x.license_text_sha256


def test_promotion_checks_reject_unsuitable_sources():
    for sid in ("gh-pallets-flask", "gh-encode-httpx", "gh-unity-ml-agents", "gh-nlohmann-json", "gh-dotnet-runtime",
                "gh-stride3d-stride", "gh-fna-xna-fna", "gh-messagepack-csharp", "python-docs-cpython-doc", "gh-dotnet-docs",
                "mdn-content", "gh-kubernetes-website", "docker-docs", "gh-google-guava", "gh-flame-engine-flame",
                "gh-gin-gonic-gin"):
        assert pol.promotion_problems(BY_ID[sid]), sid
        with pytest.raises(pol.PolicyError):
            pol.build_entries(REGISTRY, promotions={sid: "test"})
    with pytest.raises(pol.PolicyError):
        pol.build_entries(REGISTRY, promotions={"no-such-source": "x"})


# ---------- ACCEPT-only selection and invariants ----------

def test_accept_only_selector_excludes_review_unknown_blocked():
    selected = set(pol.accepted_source_ids(ENTRIES))
    assert len(selected) == 19
    for e in ENTRIES:
        assert (e.source_id in selected) == (e.stage1c_status == rs.ACCEPT)
        if e.stage1c_status != rs.ACCEPT:
            assert not (e.tokenizer_fit_eligible or e.model_training_eligible or e.validation_eligible)
            assert e.repo_cap_bp == 0 and e.license_used_for_eligibility is None
    assert selected >= {x.source_id for x in REGISTRY if x.status == rs.ACCEPT}
    assert set(pol.accepted_source_ids(ENTRIES, rs.CAT_CODE)) == {
        "gh-psf-requests", "gh-expressjs-express", "gh-gin-gonic-gin", "gh-burntsushi-ripgrep", "gh-lodash-lodash",
        "gh-spf13-cobra", "gh-tokio-rs-tokio", "gh-serde-rs-serde"}


def test_every_accept_has_allowed_license_and_concrete_pin():
    for e in ENTRIES:
        if e.stage1c_status != rs.ACCEPT:
            continue
        assert e.license_used_for_eligibility in rs.ALLOWED_LICENSE_IDS
        if e.pin_kind in (rs.PIN_COMMIT, rs.PIN_DATASET_REVISION):
            assert re.fullmatch(r"[0-9a-f]{40}", e.commit)
        else:
            assert e.source_id == "hf-wikipedia-ne-20231101" and e.pin_kind == rs.PIN_SNAPSHOT_LABEL
        assert e.repo_cap_bp > 0
    for e in ENTRIES:
        if e.source_id.startswith("gh-") and e.stage1c_status == rs.ACCEPT:
            assert e.pin_kind == rs.PIN_COMMIT
            assert e.evidence_ref == f"{pol.EVIDENCE_ARTIFACT}#{e.source_id}"
            assert e.commit == BY_ID[e.source_id].pin == BY_ID[e.source_id].evidence_commit


def test_allowlist_not_changed_by_stage1c():
    assert rs.ALLOWED_LICENSE_IDS == frozenset({"CC-BY-SA-4.0", "CC-BY-SA-3.0", "CC-BY-4.0", "CC-BY-3.0", "MIT", "Apache-2.0", "MS-PL"})
    policy = pol.build_policy()
    assert policy["owner_decisions"]["license_allowlist_changed_by_stage1c"] is False
    assert policy["owner_decisions"]["license_allowlist"] == sorted(rs.ALLOWED_LICENSE_IDS)


def test_unity_sources_remain_blocked_and_ml_agents_review():
    for sid in ("gh-unity-csreference", "unity-docs", "unity-asset-store-academy", "unity-learn"):
        assert E[sid].stage1c_status == rs.BLOCKED and E[sid].registry_status == rs.BLOCKED
        assert sid not in pol.accepted_source_ids(ENTRIES)
    assert E["gh-unity-ml-agents"].stage1c_status == rs.REVIEW
    assert sorted(pol.build_policy()["owner_decisions"]["unity_blocked_source_ids"]) == [
        "gh-unity-csreference", "unity-asset-store-academy", "unity-docs", "unity-learn"]


def test_no_status_silently_defaults_to_accept():
    field = {f.name: f for f in dataclasses.fields(pol.PolicyEntry)}["stage1c_status"]
    assert field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
    # a REVIEW record that Stage 1B proposed for ACCEPT stays REVIEW unless explicitly promoted
    lodash = BY_ID["gh-lodash-lodash"]
    entries = pol.build_entries(REGISTRY, promotions={})
    assert {e.source_id: e for e in entries}["gh-lodash-lodash"].stage1c_status == rs.REVIEW
    # an unknown/UNKNOWN record never becomes ACCEPT
    assert E["docker-docs"].stage1c_status == rs.UNKNOWN
    fake = dataclasses.replace(lodash, source_id="gh-fake-source", proposed_status=rs.ACCEPT)
    out = {e.source_id: e for e in pol.build_entries(list(REGISTRY) + [fake], promotions=pol.STAGE1C_PROMOTIONS)}
    assert out["gh-fake-source"].stage1c_status == rs.REVIEW
    with pytest.raises(pol.PolicyError):
        pol.validate_policy([dataclasses.replace(E["gh-lodash-lodash"], stage1c_status="MAYBE")])
    with pytest.raises(pol.PolicyError):
        pol.validate_policy([dataclasses.replace(E["gh-pallets-flask"], stage1c_status=rs.REVIEW, tokenizer_fit_eligible=True)])
    with pytest.raises(pol.PolicyError):
        pol.validate_policy([dataclasses.replace(E["gh-lodash-lodash"], license_used_for_eligibility="BSD-3-Clause")])
    with pytest.raises(pol.PolicyError):
        pol.validate_policy([dataclasses.replace(E["gh-lodash-lodash"], commit="abc")])


# ---------- caps ----------

def _files(n: int, base: int = 1000):
    return [(f"src/dir{i % 17}/file{i}.rs", base + (i * 37) % 900) for i in range(n)]


def test_cap_arithmetic():
    assert pol.repo_cap_bytes(1000, rs.CAT_CODE) == 984_000
    assert pol.repo_cap_bytes(1000, rs.CAT_CSHARP) == 902_000
    assert pol.repo_cap_bytes(1500, rs.CAT_DOCS) == 1_107_000
    for bad in (0, -1, 10_001):
        with pytest.raises(pol.PolicyError):
            pol.repo_cap_bytes(bad, rs.CAT_CODE)


def test_source_caps_are_deterministic_and_order_independent():
    files = _files(1500)
    a = pol.select_within_cap(files, 500_000, "seed-1")
    b = pol.select_within_cap(files, 500_000, "seed-1")
    c = pol.select_within_cap(list(reversed(files)), 500_000, "seed-1")
    d = pol.select_within_cap(files, 500_000, "seed-2")
    assert a == b == c
    assert a != d
    assert sum(s for _, s in a) <= 500_000


def test_large_repositories_cannot_bypass_caps():
    files = _files(6_000) + [("huge/generated_blob.rs", 900_000_000)]
    cap = pol.repo_cap_bytes(1000, rs.CAT_CODE)
    chosen = pol.select_within_cap(files, cap, "s", source_id="gh-tokio-rs-tokio")
    assert sum(s for _, s in chosen) <= cap
    assert all(p != "huge/generated_blob.rs" for p, _ in chosen)
    assert pol.select_within_cap(files, 0, "s") == []
    with pytest.raises(pol.PolicyError):
        pol.select_within_cap(files, -1, "s")
    with pytest.raises(pol.PolicyError):
        pol.select_within_cap([("a", -5)], 10, "s")


def test_excluded_paths_never_selected_but_projects_own_plugin_source_is_kept():
    assert pol.is_excluded("testdata/protoexample/test.pb.go")
    assert pol.is_excluded("MonoGame.Framework/ThirdParty/zlib/x.c", "gh-monogame-monogame")
    assert pol.is_excluded("web/node_modules/a/b.js")
    assert pol.is_excluded("src/Foo.Designer.cs") and pol.is_excluded("dist/app.min.js")
    assert pol.is_excluded("lodash.js", "gh-lodash-lodash") and not pol.is_excluded("lodash.js", "gh-tokio-rs-tokio")
    assert pol.is_excluded("src/Compilers/Test/Foo.cs", "gh-dotnet-roslyn")
    # VERIFIED IN REPOSITORY: these accepted-sample paths must remain eligible
    for path, sid in (("src/UniTask/Assets/Plugins/UniTask/Runtime/AsyncLazy.cs", "gh-cysharp-unitask"),
                      ("Assets/Mirror/Core/Batching/Batcher.cs", "gh-mirrornetworking-mirror"),
                      ("MonoGame.Framework/GameWindow.cs", "gh-monogame-monogame"),
                      ("src/requests/api.py", "gh-psf-requests"), ("crates/cli/src/lib.rs", "gh-burntsushi-ripgrep")):
        assert not pol.is_excluded(path, sid), path
    assert not any("Plugins" in p for p in pol.GLOBAL_EXCLUDED_PATTERNS)
    chosen = pol.select_within_cap([("a/b.pb.go", 10), ("a/ok.go", 10)], 100, "s")
    assert chosen == [("a/ok.go", 10)]


def test_realized_shares_are_reporting_only():
    shares = pol.realized_shares({"a": 30, "b": 10})
    assert shares == {"a": 0.75, "b": 0.25}
    assert pol.realized_shares({}) == {}


def test_cap_audit_shows_the_approved_set_cannot_fill_the_targets():
    audit = pol.cap_audit(ENTRIES)
    at10 = audit["by_cap_bp"]["1000"]
    assert at10["category_ceilings"][rs.CAT_CODE]["accepted_sources"] == 8
    assert at10["category_ceilings"][rs.CAT_CODE]["ceiling_pct"] == pytest.approx(80.0)
    assert at10["category_ceilings"][rs.CAT_CSHARP]["ceiling_pct"] == pytest.approx(50.0)
    assert at10["category_ceilings"][rs.CAT_DOCS]["ceiling_pct"] == pytest.approx(40.0)
    langs = at10["code_language_ceilings"]
    assert langs["c_cpp"]["accepted_repos"] == 0 and langs["c_cpp"]["ceiling_bytes"] == 0
    assert langs["python"]["accepted_repos"] == 1 and langs["python"]["ceiling_pct_of_language_target"] == pytest.approx(28.6)
    assert langs["_total"]["ceiling_pct_of_code_category"] == pytest.approx(60.0)
    assert audit["by_cap_bp"]["500"]["code_language_ceilings"]["_total"]["ceiling_pct_of_code_category"] == pytest.approx(40.0)
    assert audit["min_cap_bp_to_reach_category_target_if_all_sources_supply_fully"] == {
        rs.CAT_CODE: 1250, rs.CAT_CSHARP: 2000, rs.CAT_DOCS: 2500}


# ---------- serialization / artifact ----------

def test_policy_serialization_is_deterministic():
    a, b = pol.policy_json(), pol.policy_json()
    assert a == b
    body = json.loads(a)
    assert body["fingerprint"] == pol._fingerprint(body)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", body["fingerprint"])
    assert "retrieved_at" not in a and "timestamp" not in a


def test_policy_sections_are_kept_apart_and_nothing_is_authorized():
    body = pol.build_policy()
    for key in ("owner_decisions", "facts", "proposals", "unresolved", "source_policy"):
        assert key in body
    assert "od08_general_code_language_target_pct" in body["owner_decisions"]
    assert "od08_general_code_language_target_pct" not in body["proposals"]
    assert "repo_cap_bp_of_category_target" in body["proposals"] and "repo_cap_bp_of_category_target" not in body["owner_decisions"]
    assert body["facts"]["stage1b_proposed_accept_count"] == 11
    assert body["not_a_milestone"] is True and body["nb13_assigned"] is False
    assert body["acquisition_authorized"] is False
    assert body["tokenizer_training_authorized"] is False and body["model_training_authorized"] is False
    dp = body["owner_decisions"]["design_principles"][0]
    assert dp["status"] == "RECORDED_NOT_IMPLEMENTED" and len(dp["flow"]) == 10
    row = next(r for r in body["source_policy"] if r["source_id"] == "gh-lodash-lodash")
    assert set(row) >= {"stage1c_status", "facts", "proposal", "owner_decision_ids", "reason", "eligibility", "od08_language"}
    assert row["facts"]["commit"] == BY_ID["gh-lodash-lodash"].pin
    assert {d["id"] for d in body["unresolved"]["open_owner_decisions"]} == {"OD-13", "OD-14", "OD-15"}


def test_policy_artifact_file_matches_the_builder():
    # The immutable snapshot retains its original provenance URLs and fingerprint.
    # Only the builder's evidence location moved; decisions must match exactly.
    historical = json.loads(_POLICY_ARTIFACT.read_text(encoding="utf-8"))
    assert historical["fingerprint"] == pol._fingerprint(historical)
    current = json.loads(pol.policy_json())
    assert current["fingerprint"] == pol._fingerprint(current)
    original = "artifacts/rep_corpus_10m_v1/00_source_license_evidence.json"
    assert current["facts"]["evidence_artifact"] == pol.EVIDENCE_ARTIFACT
    current["facts"]["evidence_artifact"] = original
    for row in current["source_policy"]:
        ref = row["facts"]["evidence_ref"]
        if ref is not None:
            assert ref == f"{pol.EVIDENCE_ARTIFACT}#{row['source_id']}"
            row["facts"]["evidence_ref"] = f"{original}#{row['source_id']}"
    current["fingerprint"] = pol._fingerprint(current)
    assert current == historical


def test_file_flags_match_the_evidence_artifact():
    data = json.loads(_EVIDENCE_ARTIFACT.read_text(encoding="utf-8"))
    for s in data["sources"]:
        flags = pol._STAGE1B_FILE_FLAGS[s["source_id"]]
        assert flags["gitmodules_present"] == s["gitmodules_present"]
        assert tuple(flags["notice_files"]) == tuple(n["path"] for n in s["notice_files"])
        assert flags["license_file_count"] == len(s["license_files"])
    assert set(pol._STAGE1B_FILE_FLAGS) == {s["source_id"] for s in data["sources"]}


def test_self_improvement_principle_document_if_present():
    doc = _ROOT / "ARCHITECTURE" / "YUG_SELF_IMPROVEMENT_PRINCIPLE.md"
    if not doc.is_file():
        pytest.skip("principle document not present")
    text = doc.read_text(encoding="utf-8")
    assert "NOT IMPLEMENTED" in text and "explicit owner approval" in text.lower()


def test_policy_module_is_stdlib_only_and_does_not_touch_yug():
    import ast
    tree = ast.parse(Path(pol.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert "yug" not in roots and "torch" not in roots


def test_production_yug_does_not_import_training():
    yug_dir = _ROOT / "yug"
    if not yug_dir.is_dir():
        pytest.skip("yug/ not found relative to tests/")
    pattern = re.compile(r"^\s*(from|import)\s+training(\.|\s|$)", re.MULTILINE)
    offenders = [str(p) for p in yug_dir.rglob("*.py") if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []
