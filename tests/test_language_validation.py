"""Fixed synthetic language controls; declared before the v1 thresholds.

These texts are diagnostic fixtures, never source clearance or training data.
Script evidence cannot identify Nepali as distinct from Hindi or reliably
recognize Latin-transliterated Nepali.
"""
import json

import pytest

from tests.test_quality_filtering import production_case
from tests.test_production_acquisition import Tokenizer
from training.production_acquisition import ProductionBuild, item_key
from training.production_corpus import canonical
from training.production_spec import SpecificationError, seal
from training.language_validation import POLICY, script_evidence, validate_language


ENGLISH = (
    "The renderer caches geometry before applying material overrides. "
    "This avoids rebuilding identical meshes for each preview frame. "
    "The importer validates coordinate units and records converted transforms."
)
DOCS = "# Asset guide\n\nCheck the scene scale before import. Preserve materials and verify texture paths after export."
NEPALI = (
    "यो प्राविधिक विवरणले दृश्यको आकार र दिशा जाँच्ने तरिका बताउँछ। "
    "आयात गर्नुअघि वस्तुको मापन मिलाउनुहोस् र निर्यात भएपछि सामग्री तथा बनावटका मार्गहरू परीक्षण गर्नुहोस्।"
)
NEPALI_TERMS = (
    "वस्तुको मापन जाँच्दा mesh, UV, material र texture का मार्ग सुरक्षित राख्नुहोस्। "
    "त्रुटि भेटिएमा मूल फाइलसँग तुलना गर्नुहोस् र अर्को निर्यातको परिणाम परीक्षण गर्नुहोस्।"
)
BALANCED = (
    "The importer checks unit scale and mesh orientation. "
    "त्यसपछि सामग्रीको मार्ग जाँचेर हराएका texture फाइलहरूको सूची तयार गर्नुहोस्।"
)
MOSTLY_EN = (
    "The asset pipeline validates every texture reference before export. "
    "नेपाली विवरणले बाँकी सामग्रीका त्रुटिहरू स्पष्ट पार्छ।"
)
PYTHON = "def load_mesh(path):\n    # Check source path before import\n    if not path:\n        raise ValueError('path required')\n    return read_mesh(path)\n"
PYTHON_NE = "def load_mesh(path):\n    # यो मार्ग सही छ कि छैन जाँच्नुहोस्\n    if not path:\n        raise ValueError('path required')\n    return read_mesh(path)\n"
RUST = "fn load_mesh(path: &str) -> Result<Mesh, Error> {\n    if path.is_empty() { return Err(Error::MissingPath); }\n    parse_mesh(path)\n}\n"
GO = "func LoadMesh(path string) (*Mesh, error) {\n    if path == \"\" { return nil, ErrMissingPath }\n    return ParseMesh(path)\n}\n"
CSHARP = "using UnityEngine;\npublic class MeshPreview : MonoBehaviour {\n    void Update() { if (Input.GetKeyDown(KeyCode.Space)) RefreshMesh(); }\n}\n"
HINDI = "यह तकनीकी विवरण दृश्य का आकार और दिशा जाँचने की विधि बताता है। आयात से पहले वस्तु का माप मिलाएँ और निर्यात के बाद सामग्री के मार्गों की पुष्टि करें।"
TRANSLITERATED = "Yo pranali le drishya ko aakar ra disha janchha. Aayat garnu aghi mesh ko maapan milaunuhos ra niryaat pachi samagri ko path jaanch garnuhos."

# (name, content type, text, label, expected supported-and-consistent outcome)
CALIBRATION = (
    ("english_prose", "prose", ENGLISH, "en", True),
    ("english_docs", "markdown", DOCS, "en", True),
    ("nepali_prose", "prose", NEPALI, "ne", True),
    ("nepali_technical_terms", "prose", NEPALI_TERMS, "ne", True),
    ("balanced", "prose", BALANCED, "mixed", True),
    ("mostly_nepali", "prose", NEPALI_TERMS, "mixed", True),
    ("mostly_english", "prose", MOSTLY_EN, "mixed", True),
    ("python", "code", PYTHON, "python", True),
    ("rust", "code", RUST, "rust", True),
    ("go", "code", GO, "go", True),
    ("csharp_unity", "code", CSHARP, "csharp", True),
    ("code_english_comment", "code", PYTHON, "unknown", True),
    ("code_nepali_comment", "code", PYTHON_NE, "mixed", True),
    ("json_config", "code", '{"mesh":"asset.glb","scale":1.0,"enabled":true}', "unknown", True),
    ("url_path", "prose", "https://example.invalid/assets/mesh.glb /assets/materials/base.png", "unknown", True),
    ("identifiers", "prose", "meshScale textureAtlas uvCoordinates skeletonRoot", "unknown", True),
    ("numbers", "prose", "12345 : 67890 / 24680 ; 13579 * 98765", "unknown", True),
    ("english_as_ne", "prose", ENGLISH, "ne", False),
    ("nepali_as_en", "prose", NEPALI, "en", False),
    ("mixed_as_en", "prose", BALANCED, "en", False),
    ("mixed_as_ne", "prose", BALANCED, "ne", False),
    ("unsupported", "prose", ENGLISH, "xx-fake", False),
    ("empty", "prose", ENGLISH, "", False),
)


@pytest.mark.parametrize("name,kind,text,label,expected", CALIBRATION, ids=[r[0] for r in CALIBRATION])
def test_fixed_calibration(name, kind, text, label, expected):
    result = lambda: validate_language(label, kind, text)
    if expected:
        result()
    else:
        with pytest.raises(SpecificationError, match="[Ll]anguage"):
            result()


def test_script_evidence_is_deterministic_and_bounded():
    assert script_evidence(NEPALI) == script_evidence(NEPALI)
    assert script_evidence(NEPALI)["devanagari_letters"] > 80
    assert script_evidence(ENGLISH)["latin_letters"] > 80
    assert POLICY["version"] == "script-consistency-v1"
    with pytest.raises(SpecificationError, match="Language document byte limit"):
        script_evidence("a" * (POLICY["max_document_bytes"] + 1))


def test_expected_limits_are_explicit():
    # Hindi shares Devanagari; transliteration shares Latin. These are not
    # semantic identification results, even if the script rule accepts/rejects.
    assert script_evidence(HINDI)["devanagari_letters"] > 0
    assert script_evidence(TRANSLITERATED)["devanagari_letters"] == 0


@pytest.mark.parametrize("kind,text,label", [
    ("prose", ENGLISH, "en"),
    ("prose", NEPALI, "ne"),
    ("prose", BALANCED, "mixed"),
    ("prose", "12345 : 67890 / 24680 ; 13579 * 98765", "unknown"),
    ("code", PYTHON, "python"),
    ("code", RUST, "rust"),
    ("code", GO, "go"),
    ("code", CSHARP, "csharp"),
])
def test_honest_production_paths_and_deterministic_export(tmp_path, kind, text, label):
    spec, registry, plan = production_case(tmp_path, text, kind)
    record = json.loads(plan.read_bytes())
    plan.write_bytes(canonical(seal(dict(record, language=label))) + b"\n")
    with ProductionBuild(tmp_path / "out", spec, registry, repository_root=tmp_path, diagnostic=True) as build:
        report = build.run(plan, Tokenizer())
        assert report["stages"] == {"ACCEPTED": 1}
        assert report["accepted_clean_train_tokens"] > 0
        assert build.export_shards() == build.export_shards()
        build.verify()


@pytest.mark.parametrize("text,label", [
    (ENGLISH, "ne"), (NEPALI, "en"),
    (BALANCED, "en"), (BALANCED, "ne"),
])
def test_content_contradiction_never_accounts_tokens(tmp_path, text, label):
    spec, registry, plan = production_case(tmp_path, text)
    record = json.loads(plan.read_bytes())
    plan.write_bytes(canonical(seal(dict(record, language=label))) + b"\n")
    with ProductionBuild(tmp_path / "out", spec, registry, repository_root=tmp_path, diagnostic=True) as build:
        with pytest.raises(SpecificationError, match="Language"):
            build.run(plan, Tokenizer())
        assert build.db.execute("SELECT count(*) FROM totals").fetchone()[0] == 0
        assert not (tmp_path / "out" / "prepared.json").exists()


@pytest.mark.parametrize("replacement", ["ne", "unknown", "xx-fake", "python"])
def test_resealed_accepted_language_relabel_fails_every_boundary(tmp_path, replacement):
    spec, registry, plan = production_case(tmp_path, ENGLISH)
    with ProductionBuild(tmp_path / "out", spec, registry, repository_root=tmp_path, diagnostic=True) as build:
        build.run(plan, Tokenizer())
        item, encoded = build.db.execute("SELECT item,record FROM items WHERE stage='ACCEPTED'").fetchone()
        assert json.loads(encoded)["language"] == "en"
        changed = seal(dict(json.loads(encoded), language=replacement))
        build.db.execute("UPDATE items SET record=? WHERE item=?", (canonical(changed).decode(), item))
        build.db.commit()
        build.path("clearances", item, ".json").write_bytes(canonical(changed))
        with pytest.raises(SpecificationError, match="[Pp]lan|[Ll]anguage"):
            build.verify()
        with pytest.raises(SpecificationError, match="[Pp]lan|[Ll]anguage"):
            build.export_shards()
        with pytest.raises(SpecificationError, match="[Pp]lan|[Ll]anguage"):
            build.run(plan, Tokenizer())
        assert not (tmp_path / "out" / "prepared.json").exists()


def test_programming_language_relabel_fails(tmp_path):
    spec, registry, plan = production_case(tmp_path, PYTHON, "code")
    record = json.loads(plan.read_bytes())
    plan.write_bytes(canonical(seal(dict(record, language="python"))) + b"\n")
    with ProductionBuild(tmp_path / "out", spec, registry, repository_root=tmp_path, diagnostic=True) as build:
        build.run(plan, Tokenizer())
        item = item_key(json.loads(plan.read_bytes()))
        row = json.loads(build.db.execute("SELECT record FROM items WHERE item=?", (item,)).fetchone()[0])
        changed = seal(dict(row, language="unknown"))
        build.db.execute("UPDATE items SET record=? WHERE item=?", (canonical(changed).decode(), item))
        build.db.commit()
        build.path("clearances", item, ".json").write_bytes(canonical(changed))
        with pytest.raises(SpecificationError, match="[Pp]lan|[Ll]anguage"):
            build.verify()


def test_dedup_provenance_must_equal_pinned_language_record(tmp_path):
    spec, registry, plan = production_case(tmp_path, PYTHON, "code")
    record = json.loads(plan.read_bytes())
    plan.write_bytes(canonical(seal(dict(record, language="python"))) + b"\n")
    with ProductionBuild(tmp_path / "out", spec, registry, repository_root=tmp_path, diagnostic=True) as build:
        build.run(plan, Tokenizer())
        item = item_key(json.loads(plan.read_bytes()))
        altered = seal(dict(json.loads(plan.read_bytes()), language="ne"))
        build.db.execute("UPDATE dd SET provenance=? WHERE item=?", (canonical(altered).decode(), item))
        build.db.commit()
        with pytest.raises(SpecificationError, match="[Dd]edup|[Ll]anguage"):
            build.verify()


def test_missing_language_has_deterministic_error(tmp_path):
    spec, registry, plan = production_case(tmp_path, ENGLISH)
    record = json.loads(plan.read_bytes())
    record.pop("language")
    plan.write_bytes(canonical(seal(record)) + b"\n")
    with ProductionBuild(tmp_path / "out", spec, registry, repository_root=tmp_path, diagnostic=True) as build:
        with pytest.raises(SpecificationError, match="[Ll]anguage"):
            build.run(plan, Tokenizer())


def test_missing_language_policy_in_old_request_fails_closed(tmp_path):
    spec, registry, plan = production_case(tmp_path, ENGLISH)
    with ProductionBuild(tmp_path / "out", spec, registry, repository_root=tmp_path, diagnostic=True) as build:
        build.run(plan, Tokenizer())
        request = build.get("request")
        request.pop("language_policy")
        build.put("request", request)
        build.db.commit()
        with pytest.raises(SpecificationError, match="[Ll]anguage policy"):
            build.verify()
    with pytest.raises(SpecificationError, match="Incompatible production build resume"):
        ProductionBuild(tmp_path / "out", spec, registry, repository_root=tmp_path, diagnostic=True)
