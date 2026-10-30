"""Frozen synthetic calibration set for the Phase 2 bounded quality policy.

Expectations below were declared before selecting new metric thresholds. These
diagnostics are not source clearance or training data.
"""

import json
import sqlite3

import pytest

from training.corpus_prep import QualityConfig, normalize_text
from training.quality_filtering import (check_quality, quality_policy,
                                        QUALITY_LIMITS, QUALITY_POLICY_VERSION)
from training.experiments.representative_mini_corpus import normalize_structured_text
from training.production_acquisition import ProductionBuild, digest
from training.production_corpus import canonical, identity, file_digest
from training.production_spec import seal, SpecificationError
from tests.test_production_acquisition import Tokenizer, fixture_plan, prior_inventory, approval_kwargs
from tests.test_protected_references import inventory, row
from tests.test_corpus_gate import bundle


PARAGRAPH = (
    "The renderer caches geometry before applying material overrides. "
    "This avoids rebuilding identical meshes for each preview frame."
)
BLOCK_A = (
    "The asset loader checks coordinate units and axis orientation before import. "
    "It reports every converted transform."
)
BLOCK_B = (
    "The scene exporter preserves texture references and material bindings. "
    "It validates each output path."
)
FUNCTION = (
    "def transform_vertices(points):\n"
    "    result = [point * 2 for point in points]\n"
    "    return result\n"
)
PYTHON = (
    "def load_mesh(path):\n"
    "    if not path:\n"
    "        raise ValueError('path required')\n"
    "    data = read_mesh(path)\n"
    "    return data\n"
)
CSHARP = (
    "public sealed class MeshLoader {\n"
    "    public Mesh Load(string path) {\n"
    "        if (string.IsNullOrEmpty(path)) throw new ArgumentException();\n"
    "        return Import(path);\n"
    "    }\n"
    "}\n"
)

# (name, declared content type, text, predeclared quality expectation)
CASES = (
    ("paragraph_20", "prose", "\n\n".join([PARAGRAPH] * 20), "REJECT"),
    ("alternating_20", "prose", "\n\n".join([BLOCK_A, BLOCK_B] * 20), "REJECT"),
    ("function_15", "code", "\n".join([FUNCTION] * 15), "REJECT"),
    ("boilerplate_docs", "markdown", "# Guide\n\n" + "Home | Next | Previous\n" * 30
     + "The renderer maps scene coordinates to engine units.\n", "REJECT"),
    ("filler_prose", "prose", " ".join(["test test test test useful"] * 30), "REJECT"),
    ("punctuation", "prose", "!?;:," * 100, "REJECT"),
    ("invisible_padding", "prose", "A" + "\u200b" * 5000 + "B", "REJECT"),
    ("placeholder_prose", "prose", " ".join(["placeholder value pending"] * 35), "REJECT"),
    ("normal_technical", "prose", PARAGRAPH + " The source geometry is retained.", "ACCEPT"),
    ("concise_docs", "markdown", "# Import steps\n\nCheck input scale. Convert coordinates. "
     "Verify material paths.\n", "ACCEPT"),
    ("normal_python", "code", PYTHON, "ACCEPT"),
    ("normal_csharp", "code", CSHARP, "ACCEPT"),
    ("unity_csharp", "code", "using UnityEngine;\npublic class MeshPreview : MonoBehaviour {\n"
     "    void Update() {\n        if (Input.GetKeyDown(KeyCode.Space)) RefreshMesh();\n"
     "    }\n}\n", "ACCEPT"),
    ("shell_config", "code", "#!/bin/sh\nset -eu\nSOURCE_DIR=assets\n"
     "OUTPUT_DIR=build\nmkdir -p \"$OUTPUT_DIR\"\ncp \"$SOURCE_DIR\"/*.obj \"$OUTPUT_DIR\"\n", "ACCEPT"),
    ("small_enum", "code", "enum ImportMode {\n    Strict,\n    Repair,\n    Preview,\n}\n", "ACCEPT"),
    ("markdown_table", "markdown", "| Format | Import | Export |\n"
     "| --- | --- | --- |\n| OBJ | Yes | Yes |\n| GLB | Yes | Yes |\n", "ACCEPT"),
    ("markdown_list", "markdown", "# Checks\n- Verify scene scale\n"
     "- Verify material bindings\n- Verify texture paths\n- Verify mesh topology\n", "ACCEPT"),
    ("repeated_imports", "code", "import os\nimport sys\nimport json\nimport os\n"
     "import pathlib\nimport hashlib\nimport os\n"
     "def read_config(path):\n    return pathlib.Path(path).read_text()\n", "ACCEPT"),
    ("repeated_braces", "code", "class Renderer {\n"
     + "\n".join(f"    void Step{i}() {{ return; }}" for i in range(12)) + "\n}\n", "ACCEPT"),
    ("common_keywords", "code", "def valid(value):\n"
     "    if value is None:\n        return False\n"
     "    if value == 0:\n        return False\n"
     "    if value < 0:\n        return False\n"
     "    return True\n", "ACCEPT"),
    ("short_explanation", "prose", "RoPE rotates query and key vectors by position.", "ACCEPT"),
)


def normalized(kind, text):
    return normalize_structured_text(text) if kind in ("code", "markdown") else normalize_text(text)


@pytest.mark.parametrize("name,kind,text,expected", CASES, ids=[c[0] for c in CASES])
def test_calibrated_quality_decision(name, kind, text, expected):
    reason = check_quality(normalized(kind, text), QualityConfig(), content_type=kind)
    assert ("ACCEPT" if reason is None else "REJECT") == expected, name


def production_case(tmp_path, text, kind="prose"):
    spec, registry, plan, records = fixture_plan(tmp_path, texts=[text])
    if kind != "prose":
        manifest_path = tmp_path / "files0.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["files"][0]["content_type"] = kind
        manifest = seal({k: v for k, v in manifest.items() if k != "identity"})
        manifest_path.write_bytes(canonical(manifest))
        clearance_path = tmp_path / "clear0.json"
        clearance = json.loads(clearance_path.read_bytes())
        clearance["file_manifest_identity"] = manifest["identity"]
        clearance = seal({k: v for k, v in clearance.items() if k != "identity"})
        clearance_path.write_bytes(canonical(clearance))
        record = dict(records[0])
        record["content_type"] = kind
        record["source_clearance_identity"] = clearance["identity"]
        if kind == "code":
            record["language"] = "unknown"  # Quality fixture has no asserted programming language.
        record = seal({k: v for k, v in record.items() if k != "identity"})
        plan.write_bytes(canonical(record) + b"\n")
    return spec, registry, plan


@pytest.mark.parametrize("name,kind,text,expected", CASES, ids=[c[0] for c in CASES])
def test_production_quality_decision(tmp_path, name, kind, text, expected):
    spec, registry, plan = production_case(tmp_path, text, kind)
    tokenizer = Tokenizer()
    with ProductionBuild(tmp_path / "out", spec, registry, repository_root=tmp_path,
                         diagnostic=True) as build:
        report = build.run(plan, tokenizer)
        stage, reason = build.db.execute("SELECT stage,reason FROM items").fetchone()
        if expected == "REJECT":
            assert stage == "FILTERED" and reason is not None, name
            assert tokenizer.calls == 0 and report["accepted_clean_train_tokens"] == 0
        else:
            assert stage == "ACCEPTED" and reason is None, name
            assert tokenizer.calls == 1 and report["accepted_clean_train_tokens"] > 0


def test_late_generated_marker_is_filtered_before_encoding(tmp_path):
    text = "".join(f"Technical note {i}: validate units and texture mapping.\n"
                   for i in range(110)) + "DO NOT EDIT\n"
    assert text.index("DO NOT EDIT") > 4096
    spec, registry, plan = production_case(tmp_path, text)
    tokenizer = Tokenizer()
    with ProductionBuild(tmp_path / "out", spec, registry, repository_root=tmp_path,
                         diagnostic=True) as build:
        report = build.run(plan, tokenizer)
        assert build.db.execute("SELECT stage,reason FROM items").fetchone() == ("FILTERED", "generated")
        assert tokenizer.calls == 0 and report["accepted_clean_train_tokens"] == 0
        decision = build.db.execute("SELECT outcome,reason,normalized_sha FROM quality_decisions").fetchone()
        assert decision == ("NOT_EVALUATED", "generated", None)


def test_quality_policy_contract_is_deterministic_and_covers_configuration():
    markers = ["DO NOT EDIT"]
    policy = quality_policy(QualityConfig(), markers)
    assert policy == quality_policy(QualityConfig(), markers)
    assert policy["version"] == QUALITY_POLICY_VERSION
    assert identity(policy) != identity(quality_policy(QualityConfig(min_length=21), markers))
    assert identity(policy) != identity(quality_policy(QualityConfig(), markers + ["@generated"]))


def test_partial_resume_refuses_changed_quality_policy(tmp_path, monkeypatch):
    import training.quality_filtering as quality_filtering

    spec, registry, plan = production_case(tmp_path, PARAGRAPH + " Unique technical detail.")
    with ProductionBuild(tmp_path / "out", spec, registry, repository_root=tmp_path,
                         diagnostic=True) as build:
        report = build.run(plan, Tokenizer(), max_items=1)
        assert report["state"] == "PARTIAL"
        old_identity = build.get("request")["quality_policy"]
        monkeypatch.setattr(quality_filtering, "QUALITY_POLICY_VERSION", "changed-quality-policy-v2")
        assert build.quality_policy_identity() != old_identity
        with pytest.raises(SpecificationError, match="Quality policy changed"):
            build.run(plan, Tokenizer())
    with pytest.raises(SpecificationError, match="Incompatible production build resume"):
        ProductionBuild(tmp_path / "out", spec, registry, repository_root=tmp_path,
                        diagnostic=True)


def test_completed_export_and_authorized_replay_refuse_stale_quality_policy(tmp_path, monkeypatch):
    import training.quality_filtering as quality_filtering

    spec, registry, plan = production_case(tmp_path, PARAGRAPH + " Unique technical detail.")
    references = inventory(tmp_path / "references.jsonl", [row(text="Separate diagnostic holdout material.")])
    receipts = bundle(tmp_path, (spec, registry, references, [plan]))
    with ProductionBuild(tmp_path / "out", spec, registry, repository_root=tmp_path,
                         reference_plan=references, prior_production=prior_inventory(tmp_path),
                         expected_plan_sha256=file_digest(plan),**approval_kwargs(tmp_path,file_digest(plan))) as build:
        report = build.run(plan, Tokenizer(), intended_use="commercial_training",
                           decision_receipts=receipts)
        assert report["accepted_clean_train_tokens"] > 0
        destination = tmp_path / "out" / "export"
        build.export_shards(destination, intended_use="commercial_training", decision_receipts=receipts)
        prepared = json.loads((destination / "prepared.json").read_bytes())
        assert prepared["quality_policy"] == build.current_quality_policy()
        assert list(build.replay_for_use(destination, split="train",
                                         intended_use="commercial_training",
                                         decision_receipts=receipts))
        monkeypatch.setattr(quality_filtering, "QUALITY_POLICY_VERSION", "changed-quality-policy-v2")
        with pytest.raises(SpecificationError, match="Quality policy changed"):
            build.replay_for_use(destination, split="train", intended_use="commercial_training",
                                 decision_receipts=receipts)
        with pytest.raises(SpecificationError, match="Quality policy changed"):
            build.export_shards(destination, intended_use="commercial_training",
                                decision_receipts=receipts)


def test_quality_decision_binds_evaluated_normalized_content(tmp_path):
    spec, registry, plan = production_case(tmp_path, PARAGRAPH + " Unique technical detail.")
    with ProductionBuild(tmp_path / "out", spec, registry, repository_root=tmp_path,
                         diagnostic=True) as build:
        build.run(plan, Tokenizer())
        decision = build.db.execute("SELECT policy_identity,normalized_sha,outcome,reason FROM quality_decisions").fetchone()
        assert decision[0] == build.quality_policy_identity()
        assert decision[1] == build.db.execute("SELECT normalized_sha FROM items").fetchone()[0]
        assert decision[2:] == ("PASS", None)
        build.db.execute("UPDATE quality_decisions SET normalized_sha=?", ("sha256:" + "0" * 64,))
        build.db.commit()
        with pytest.raises(SpecificationError, match="Quality decision identity mismatch"):
            build.verify()


def test_filtered_quality_reason_retains_evaluated_content_identity(tmp_path):
    bad = next(c for c in CASES if c[0] == "paragraph_20")
    spec, registry, plan = production_case(tmp_path, bad[2], bad[1])
    with ProductionBuild(tmp_path / "out", spec, registry, repository_root=tmp_path,
                         diagnostic=True) as build:
        build.run(plan, Tokenizer())
        decision = build.db.execute("SELECT policy_identity,normalized_sha,outcome,reason FROM quality_decisions").fetchone()
        assert decision[0] == build.quality_policy_identity()
        assert decision[1] == digest(normalized(bad[1], bad[2]).encode("utf-8"))
        assert decision[2:] == ("FILTERED", "excessive_repeated_structure")
        build.verify()


def test_legacy_incompatible_build_is_not_migrated(tmp_path):
    spec, registry, plan = production_case(tmp_path, PARAGRAPH + " Unique technical detail.")
    directory = tmp_path / "out"
    with ProductionBuild(directory, spec, registry, repository_root=tmp_path,
                         diagnostic=True) as build:
        build.run(plan, Tokenizer())
        old_request = dict(build.get("request"))
        old_request.pop("quality_policy")
        build.put("request", old_request)
        build.db.execute("DROP TABLE quality_decisions")
        build.db.commit()
    with pytest.raises(SpecificationError, match="Incompatible production build resume"):
        ProductionBuild(directory, spec, registry, repository_root=tmp_path,
                        diagnostic=True)
    with sqlite3.connect(directory / "state.sqlite") as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE name='quality_decisions'").fetchone() is None


def test_prior_quality_table_without_decision_identity_is_not_migrated(tmp_path):
    spec, registry, plan = production_case(tmp_path, PARAGRAPH + " Unique technical detail.")
    directory = tmp_path / "out"
    with ProductionBuild(directory, spec, registry, repository_root=tmp_path,
                         diagnostic=True) as build:
        build.run(plan, Tokenizer())
    with sqlite3.connect(directory / "state.sqlite") as db:
        db.execute("ALTER TABLE quality_decisions RENAME TO quality_decisions_new")
        db.execute("CREATE TABLE quality_decisions(item TEXT PRIMARY KEY,policy_identity TEXT NOT NULL,"
                   "normalized_sha TEXT,outcome TEXT NOT NULL,reason TEXT)")
        db.execute("INSERT INTO quality_decisions SELECT item,policy_identity,normalized_sha,"
                   "outcome,reason FROM quality_decisions_new")
        db.execute("DROP TABLE quality_decisions_new")
    with pytest.raises(SpecificationError, match="Incompatible quality decision evidence"):
        ProductionBuild(directory, spec, registry, repository_root=tmp_path,
                        diagnostic=True, expected_plan_sha256=file_digest(plan))
    with sqlite3.connect(directory / "state.sqlite") as db:
        assert [row[1] for row in db.execute("PRAGMA table_info(quality_decisions)")] == [
            "item", "policy_identity", "normalized_sha", "outcome", "reason"]


def test_tiny_valid_code_remains_documented_length_residual(tmp_path):
    spec, registry, plan = production_case(tmp_path, 'print("Hi")\n', "code")
    tokenizer = Tokenizer()
    with ProductionBuild(tmp_path / "out", spec, registry, repository_root=tmp_path,
                         diagnostic=True) as build:
        report = build.run(plan, tokenizer)
        assert build.db.execute("SELECT stage,reason FROM items").fetchone() == (
            "FILTERED", "below_min_length")
        assert tokenizer.calls == 0 and report["accepted_clean_train_tokens"] == 0


@pytest.mark.parametrize("boundary", ["verify", "resume", "fresh_export", "existing_export", "replay"])
def test_relabelled_quality_policy_cannot_authorize_old_decision(tmp_path, monkeypatch, boundary):
    content = PARAGRAPH + " Unique technical detail."
    spec, registry, plan = production_case(tmp_path, content)
    references = inventory(tmp_path / "references.jsonl", [row(text="Separate diagnostic holdout material.")])
    receipts = bundle(tmp_path, (spec, registry, references, [plan]))
    directory = tmp_path / "out"
    with ProductionBuild(directory, spec, registry, repository_root=tmp_path,
                         reference_plan=references, prior_production=prior_inventory(tmp_path),
                         expected_plan_sha256=file_digest(plan),**approval_kwargs(tmp_path,file_digest(plan))) as build:
        report = build.run(plan, Tokenizer(), intended_use="commercial_training",
                           decision_receipts=receipts)
        assert report["accepted_clean_train_tokens"] == 153
        old_export = directory / "export-A"
        build.export_shards(old_export, intended_use="commercial_training",
                            decision_receipts=receipts)
        monkeypatch.setitem(QUALITY_LIMITS, "meaningful_min_prose_markdown", 0.99)
        reason = check_quality(normalized("prose", content), QualityConfig(), content_type="prose")
        assert reason.value == "low_information"
        request = dict(build.get("request"))
        request["quality_policy"] = build.quality_policy_identity()
        build.put("request", request)
        build.db.execute("UPDATE quality_decisions SET policy_identity=?",
                         (build.quality_policy_identity(),))
        build.db.commit()
        with pytest.raises(SpecificationError, match="Quality decision identity mismatch"):
            if boundary == "verify":
                build.verify()
            elif boundary == "resume":
                build.run(plan, Tokenizer(), intended_use="commercial_training",
                          decision_receipts=receipts)
            elif boundary == "fresh_export":
                build.export_shards(directory / "export-B", intended_use="commercial_training",
                                    decision_receipts=receipts)
            elif boundary == "existing_export":
                build.export_shards(old_export, intended_use="commercial_training",
                                    decision_receipts=receipts)
            else:
                build.replay_for_use(old_export, split="train", intended_use="commercial_training",
                                     decision_receipts=receipts)
        assert not (directory / "export-B" / "prepared.json").exists()


@pytest.mark.parametrize("mutation", ["request", "decision_policy", "status", "reason",
                                      "normalized_hash", "decision_identity"])
def test_quality_decision_metadata_tampering_fails_closed(tmp_path, mutation):
    spec, registry, plan = production_case(tmp_path, PARAGRAPH + " Unique technical detail.")
    with ProductionBuild(tmp_path / "out", spec, registry, repository_root=tmp_path,
                         diagnostic=True) as build:
        build.run(plan, Tokenizer())
        if mutation == "request":
            request = dict(build.get("request"))
            request["quality_policy"] = "sha256:" + "0" * 64
            build.put("request", request)
            expected = "Quality policy changed"
        else:
            assignments = {
                "decision_policy": ("policy_identity", "sha256:" + "0" * 64),
                "status": ("outcome", "FILTERED"),
                "reason": ("reason", "low_information"),
                "normalized_hash": ("normalized_sha", "sha256:" + "0" * 64),
                "decision_identity": ("decision_identity", "sha256:" + "0" * 64),
            }
            column, value = assignments[mutation]
            build.db.execute(f"UPDATE quality_decisions SET {column}=?", (value,))
            expected = "Quality decision identity mismatch"
        build.db.commit()
        with pytest.raises(SpecificationError, match=expected):
            build.verify()


def test_recomputed_quality_rejects_resealed_decision_under_new_policy(tmp_path, monkeypatch):
    content = PARAGRAPH + " Unique technical detail."
    spec, registry, plan = production_case(tmp_path, content)
    with ProductionBuild(tmp_path / "out", spec, registry, repository_root=tmp_path,
                         diagnostic=True) as build:
        build.run(plan, Tokenizer())
        monkeypatch.setitem(QUALITY_LIMITS, "meaningful_min_prose_markdown", 0.99)
        assert check_quality(normalized("prose", content), QualityConfig(),
                             content_type="prose").value == "low_information"
        request = dict(build.get("request"))
        request["quality_policy"] = build.quality_policy_identity()
        build.put("request", request)
        item, nsha = build.db.execute("SELECT item,normalized_sha FROM items").fetchone()
        body = dict(item=item, policy_identity=build.quality_policy_identity(),
                    normalized_sha256=nsha, outcome="PASS", reason=None)
        build.db.execute("UPDATE quality_decisions SET policy_identity=?,decision_identity=?",
                         (body["policy_identity"], identity(body)))
        build.db.commit()
        with pytest.raises(SpecificationError, match="Quality decision does not match current policy"):
            build.verify()


def test_honest_new_policy_build_is_evaluated_and_replayable(tmp_path, monkeypatch):
    monkeypatch.setitem(QUALITY_LIMITS, "meaningful_min_prose_markdown", 0.40)
    spec, registry, plan = production_case(tmp_path, PARAGRAPH + " Unique technical detail.")
    references = inventory(tmp_path / "references.jsonl", [row(text="Separate diagnostic holdout material.")])
    receipts = bundle(tmp_path, (spec, registry, references, [plan]))
    with ProductionBuild(tmp_path / "out", spec, registry, repository_root=tmp_path,
                         reference_plan=references, prior_production=prior_inventory(tmp_path),
                         expected_plan_sha256=file_digest(plan),**approval_kwargs(tmp_path,file_digest(plan))) as build:
        report = build.run(plan, Tokenizer(), intended_use="commercial_training",
                           decision_receipts=receipts)
        assert report["accepted_clean_train_tokens"] > 0
        build.verify()
        destination = tmp_path / "out" / "export"
        build.export_shards(destination, intended_use="commercial_training",
                            decision_receipts=receipts)
        assert sum(1 for _ in build.replay_for_use(destination, split="train",
                                                   intended_use="commercial_training",
                                                   decision_receipts=receipts)) == report["accepted_clean_train_tokens"]
