"""Tests for training.experiments.representative_mini_corpus.

No network, no PyTorch. All fixtures are tiny, original, deterministic
and synthetic (including the Nepali/code fixtures below: they are NOT
natural corpus data); they only prove the experiment CODE is correct.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from training.corpus_prep import LicenseDecision, RawDocument, normalize_text, prepare_corpus
from training.experiments import representative_mini_corpus as mc
from yug.tokenizer.bpe import ByteLevelBpeTokenizer

SIZES = (270, 300, 340)

_EN = "river valley trade routes ancient cities farmers soil library manuscripts stars plates musicians rhythm".split()
_NE = "नेपाल काठमाडौं पहाड हिमाल नदी गाउँ शहर विद्यालय किताब मानिस समय काम घर पानी".split()
_CODE_LINES = [
    "def compute(values):",
    "    total = 0",
    "    for v in values:",
    "        if v > 0:",
    "            total += v",
    "    return total",
    "class Player : MonoBehaviour {",
    "\tvoid Update() {",
    "\t\ttransform.Translate(speed * Time.deltaTime);",
    "\t}",
    "}",
]


def _mk(i, cat, text, license_id="MIT", source="fx", language="en"):
    return RawDocument(
        document_id=f"{cat}-{i:03d}", text=text, source_id=source, source_name="Fixture",
        license=license_id, language=language, primary_domain=cat,
    )


def _fixture_docs(per_cat: int = 8) -> list[RawDocument]:
    rng = random.Random(3)
    docs: list[RawDocument] = []
    for i in range(per_cat):
        docs.append(_mk(i, mc.CAT_ENGLISH, f"Entry {i}. " + " ".join(rng.choice(_EN) for _ in range(30)) + ".", "CC-BY-SA-4.0"))
        code = "\n".join(rng.choice(_CODE_LINES) for _ in range(14)) + f"\n// unit {i}\n"
        docs.append(_mk(i, mc.CAT_CODE, code, "MIT", language="zxx"))
        docs.append(_mk(i, mc.CAT_CSHARP, "using UnityEngine;\n" + code, "MIT", language="zxx"))
        docs.append(_mk(i, mc.CAT_DOCS, f"# Guide {i}\n\n- item one\n  - nested item\n\n```\n    indented block {i}\n```\n" + " ".join(rng.choice(_EN) for _ in range(15)), "CC-BY-4.0"))
        docs.append(_mk(i, mc.CAT_NEPALI, f"लेख {i}। " + " ".join(rng.choice(_NE) for _ in range(30)) + "।", "CC-BY-SA-3.0", language="ne"))
    return docs


@pytest.fixture(scope="module")
def result():
    return mc.run_representative_mini_corpus(_fixture_docs(), vocab_sizes=SIZES, verify_prefix_up_to=340)


def _tok(res, size):
    return {t["requested_vocab_size"]: t for t in res["tokenizers"]}[size]


# --- categories / provenance -------------------------------------------------


def test_categories_and_content_types():
    assert set(mc.CATEGORIES) == set(mc.CONTENT_TYPE)
    assert mc.CONTENT_TYPE[mc.CAT_NEPALI] == "nepali_prose"
    assert mc.CONTENT_TYPE[mc.CAT_CODESWITCH] == "code_switched_text"
    assert len(mc.CATEGORIES) == 6


def test_unknown_category_rejected_before_processing():
    bad = _fixture_docs(2) + [_mk(0, "hindi", "यह हिंदी है और यह पर्याप्त लंबा वाक्य है।", "MIT")]
    with pytest.raises(mc.MiniCorpusError):
        mc.build_corpus(bad)
    with pytest.raises(mc.MiniCorpusError):
        mc.run_representative_mini_corpus(bad, vocab_sizes=SIZES)


def test_category_and_provenance_preserved_through_pipeline():
    built = mc.build_corpus(_fixture_docs())
    by_id = {d.document_id: d for d in _fixture_docs()}
    assert built["prep"].accepted
    for a in built["prep"].accepted:
        raw = by_id[a.document_id]
        assert (a.primary_domain, a.source_id, a.license, a.language) == (raw.primary_domain, raw.source_id, raw.license, raw.language)
    assert {d.primary_domain for d in built["train"]} == {mc.CAT_ENGLISH, mc.CAT_CODE, mc.CAT_CSHARP, mc.CAT_DOCS, mc.CAT_NEPALI}


# --- source registry ------------------------------------------------------------


def test_registry_statuses_and_license_consistency():
    reg = {s.source_id: s for s in mc.build_source_registry()}
    assert reg["unity-docs"].status == "BLOCKED" and reg["unity-asset-store-academy"].status == "BLOCKED"
    assert reg["gh-unity-csreference"].status == "BLOCKED"
    assert reg["gh-unity-ml-agents"].status == "REVIEW"
    assert reg["natural-codeswitch-none"].status == "NATURAL SOURCE NOT YET VERIFIED"
    assert reg["synthetic-codeswitch-diagnostic"].status == "SYNTHETIC_DIAGNOSTIC_ONLY"
    for s in reg.values():
        decision = mc.MINI_CORPUS_LICENSE_POLICY.decide(s.license_id)
        if s.status == "ACCEPT":
            assert decision is LicenseDecision.ALLOW, s.source_id
        elif s.status != "SYNTHETIC_DIAGNOSTIC_ONLY":
            # A held source may carry an ALLOW-able license id (e.g. ml-agents Apache-2.0),
            # so the SOURCE-level gate must be "no acquisition path", not the license id.
            assert decision is not LicenseDecision.ALLOW or s.automated is False, s.source_id
        else:
            assert decision is not LicenseDecision.ALLOW
        assert s.category in mc.CATEGORIES
        assert s.reason and s.acquisition_method and s.license_evidence


def test_github_manifest_is_pinned_and_covered_by_registry():
    manifest = mc._load_github_manifest()
    reg = {s.source_id: s for s in mc.build_source_registry()}
    for src in manifest["sources"]:
        assert len(src["commit"]) == 40
        assert src["source_id"] in reg and reg[src["source_id"]].revision == src["commit"]
        assert reg[src["source_id"]].status == "ACCEPT" and reg[src["source_id"]].automated
        assert src["files"]
        for f in src["files"]:
            assert len(f["git_blob_sha1"]) == 40


def test_unknown_and_synthetic_licenses_go_to_review_not_accepted():
    docs = _fixture_docs(3) + [
        _mk(90, mc.CAT_CODE, "def unlicensed():\n    return 1  # padding text here", "all-rights-reserved"),
        *mc.synthetic_codeswitch_documents(),
    ]
    prep = mc.build_corpus(docs)["prep"]
    review_ids = {r.document_id for r in prep.review}
    assert "code-090" in review_ids
    assert all(f"synthetic-codeswitch:{i:03d}" in review_ids for i in range(len(mc.SYNTHETIC_CODESWITCH_LINES)))
    assert not any(a.document_id.startswith("synthetic") for a in prep.accepted)


# --- normalization -------------------------------------------------------------------


def test_structured_normalization_preserves_code_formatting():
    src = "\r\n\r\ndef f(x):\r\n    if x:\r\n\t\treturn  x   \r\n    return 0  \r\n\r\n"
    out = mc.normalize_structured_text(src)
    assert out == "def f(x):\n    if x:\n\t\treturn  x\n    return 0"
    assert normalize_text(src) != out  # prose normalizer would destroy indentation / space runs
    assert "    if x:" in out and "\t\treturn  x" in out


def test_prose_normalization_still_collapses_whitespace():
    doc = _mk(0, mc.CAT_ENGLISH, "Hello    world\r\n   second   line  ", "CC-BY-SA-4.0")
    assert mc.normalize_document(doc) == "Hello world\nsecond line"


def test_code_indentation_survives_full_pipeline_but_prose_is_normalized():
    built = mc.build_corpus(_fixture_docs())
    code = [d for d in built["prep"].accepted if d.primary_domain == mc.CAT_CODE][0]
    assert "\n        if v > 0:" in code.text or "\n    for v in values:" in code.text or "\t" in code.text or "    total" in code.text
    docs = [d for d in built["prep"].accepted if d.primary_domain == mc.CAT_DOCS][0]
    assert "    indented block" in docs.text and "  - nested item" in docs.text
    eng = [d for d in built["prep"].accepted if d.primary_domain == mc.CAT_ENGLISH][0]
    assert "  " not in eng.text


def test_prepare_corpus_default_behaviour_unchanged_by_hook():
    docs = [d for d in _fixture_docs(4) if d.primary_domain in (mc.CAT_ENGLISH, mc.CAT_NEPALI)]
    kw = dict(license_policy=mc.MINI_CORPUS_LICENSE_POLICY, val_fraction=0.2, seed=1234)
    a = prepare_corpus(docs, **kw)
    b = prepare_corpus(docs, text_normalizer=lambda d: normalize_text(d.text), **kw)
    assert a.manifest.to_json() == b.manifest.to_json()
    assert [d.text for d in a.train] == [d.text for d in b.train]


# --- determinism / dedup / split ----------------------------------------------------------


def test_corpus_construction_is_deterministic():
    a, b = mc.build_corpus(_fixture_docs()), mc.build_corpus(_fixture_docs())
    assert a["prep"].manifest.accepted_content_hash == b["prep"].manifest.accepted_content_hash
    assert [d.document_id for d in a["train"]] == [d.document_id for d in b["train"]]
    assert [d.document_id for d in a["validation"]] == [d.document_id for d in b["validation"]]
    assert a["per_category"] == b["per_category"]


def test_exact_duplicates_rejected_and_counted_per_category():
    docs = _fixture_docs(3)
    dup = docs[1]
    docs.append(_mk(77, dup.primary_domain, dup.text, dup.license, language=dup.language))
    built = mc.build_corpus(docs)
    assert built["prep"].manifest.duplicate_count == 1
    assert built["per_category"][dup.primary_domain]["duplicates"] == 1
    assert built["per_category"][dup.primary_domain]["rejected"] == 1
    assert built["per_category"][dup.primary_domain]["accepted"] == 3


def test_cross_category_exact_duplicate_removed_once():
    text = "Shared line of text that is long enough to pass the quality filter."
    docs = [_mk(1, mc.CAT_ENGLISH, text, "CC-BY-SA-4.0"), _mk(2, mc.CAT_DOCS, text, "CC-BY-4.0")]
    built = mc.build_corpus(docs)
    assert built["prep"].manifest.duplicate_count == 1
    assert built["prep"].manifest.accepted_count == 1


def test_split_is_stratified_and_isolated():
    built = mc.build_corpus(_fixture_docs())
    for cat, v in built["per_category"].items():
        if v["accepted"] >= 2:
            assert v["train"] >= 1 and v["validation"] >= 1, cat
    tr = {d.document_id for d in built["train"]}
    va = {d.document_id for d in built["validation"]}
    assert tr.isdisjoint(va) and tr | va == {d.document_id for d in built["prep"].accepted}
    assert {d.content_hash for d in built["train"]}.isdisjoint({d.content_hash for d in built["validation"]})


def test_single_document_category_goes_to_train_only():
    docs = [_mk(0, mc.CAT_CODE, "def lonely():\n    return 42  # only doc in the category", "MIT", language="zxx")] + [
        d for d in _fixture_docs(4) if d.primary_domain == mc.CAT_ENGLISH
    ]
    built = mc.build_corpus(docs)
    assert built["per_category"][mc.CAT_CODE]["train"] == 1 and built["per_category"][mc.CAT_CODE]["validation"] == 0


# --- Nepali Unicode --------------------------------------------------------------------------


def test_nepali_unicode_round_trip_and_separate_metrics():
    lines = [
        "नेपाल हिमालयको काखमा बसेको सुन्दर देश हो।",
        "काठमाडौं उपत्यकामा धेरै ऐतिहासिक मन्दिरहरू छन्।",
        "क्ष त्र ज्ञ श्री र्‍ कर्‍ हिन्दू  ‌ZWNJ‍ZWJ",  # conjuncts + ZWNJ/ZWJ
    ]
    tok = ByteLevelBpeTokenizer.train(lines * 3, 300)
    for line in lines:
        assert tok.decode(tok.encode(line)) == line
    stats, _ = mc.measure_group(tok, lines)
    assert stats["encoding_failures"] == stats["decoding_failures"] == stats["round_trip_mismatches"] == 0
    assert stats["total_utf8_bytes"] == sum(len(x.encode("utf-8")) for x in lines) > stats["total_characters"]


def test_nepali_unchanged_by_prose_normalization_except_whitespace():
    text = "नेपाल   एक देश हो।\r\nकाठमाडौं राजधानी हो।"
    doc = _mk(0, mc.CAT_NEPALI, text, "CC-BY-SA-3.0", language="ne")
    assert mc.normalize_document(doc) == "नेपाल एक देश हो।\nकाठमाडौं राजधानी हो।"


def test_devanagari_fraction_is_diagnostic_only():
    assert mc.devanagari_fraction("नेपाल") == 1.0
    assert mc.devanagari_fraction("abc") == 0.0
    assert mc.devanagari_fraction("   ") is None


# --- acquisition (offline, injected) --------------------------------------------------------------


def test_git_blob_sha1_matches_git():
    assert mc.git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert mc.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def _tiny_manifest(contents: dict[str, bytes]):
    reg = {s.source_id: s for s in mc.build_source_registry()}
    src = "gh-psf-requests"
    return {
        "sources": [{
            "source_id": src, "repo_url": "x", "commit": "c" * 40,
            "files": [{"path": p, "git_blob_sha1": mc.git_blob_sha1(b), "size_bytes": len(b)} for p, b in contents.items()],
        }]
    }, reg[src]


def test_github_acquisition_verifies_blob_sha_and_records_failures():
    good = b"\xef\xbb\xbfdef ok():\n    return 1\n"
    contents = {"a.py": good, "b.py": b"print('x')\n", "c.py": b"y = 2\n", "d.py": b"\xff\xfe\x00bad utf8"}
    manifest, spec = _tiny_manifest(contents)
    served = dict(contents)
    served["b.py"] = b"print('TAMPERED')\n"  # drift -> SHA mismatch

    def fetch(url):
        name = url.rsplit("/", 1)[-1]
        if name == "c.py":
            raise OSError("network down")
        return served[name]

    docs, prov, errors = mc.acquire_github_files(manifest, open_bytes=fetch)
    assert [d.document_id.split(":")[1].split("@")[0] for d in docs] == ["a.py"]
    assert docs[0].text.startswith("def ok")  # BOM stripped after hash verification
    d = docs[0]
    assert (d.license, d.primary_domain, d.source_id) == (spec.license_id, spec.category, spec.source_id)
    assert prov[d.document_id]["bom_stripped"] is True and prov[d.document_id]["git_blob_sha1"]
    kinds = " | ".join(e["error"] for e in errors)
    assert "SHA-1 mismatch" in kinds and "fetch failed" in kinds and "not valid UTF-8" in kinds and len(errors) == 3


def test_english_acquisition_applies_bounds_and_relabels_category():
    class Rep:
        def to_dict(self):
            return {"dataset_id": "d", "provenance": {"x": 1}}

    base = [RawDocument(f"wikibooks:{i}", "x" * n, "wikibooks", "Wikimedia", "CC-BY-SA-4.0", "en", "wikibooks") for i, n in enumerate([100, 5000, 100, 100])]
    docs, info = mc.acquire_english(4, max_doc_chars=1000, max_total_chars=250, acquire=lambda n: (base, Rep()))
    assert [d.document_id for d in docs] == ["wikibooks:0", "wikibooks:2"]  # long doc skipped; 3rd stops at total cap
    assert all(d.primary_domain == mc.CAT_ENGLISH and d.license == "CC-BY-SA-4.0" for d in docs)
    assert info["skipped_over_max_doc_chars"] == 1 and "provenance" not in info["acquisition_report"]


def _rows(start, specs):
    return {"num_rows_total": 999, "rows": [
        {"row_idx": start + i, "row": {"id": str(start + i), "title": f"T{start + i}", "url": f"u{start + i}", "text": t},
         "truncated_cells": ["text"] if trunc else []}
        for i, (t, trunc) in enumerate(specs)
    ]}


def test_nepali_acquisition_filters_and_budget():
    long_ne = "नेपाल " * 100  # 600 chars
    pages = {0: _rows(0, [(long_ne, False), ("छोटो", False), (long_ne, True), (long_ne * 30, False), (long_ne, False), (long_ne, False)])}
    calls = []

    def fetch(url):
        calls.append(url)
        off = int(url.split("offset=")[1].split("&")[0])
        return pages.get(off, {"rows": []})

    docs, prov, info = mc.acquire_nepali_wikipedia(max_total_chars=1200, max_doc_chars=3000, min_doc_chars=100, open_json=fetch)
    assert len(docs) == 2 and info["total_chars_kept"] == 1200
    assert info["skipped"] == {"below_min_doc_chars": 1, "text_truncated_by_api": 1, "over_max_doc_chars": 1}
    assert all(d.language == "ne" and d.primary_domain == mc.CAT_NEPALI and d.license == "CC-BY-SA-3.0" for d in docs)
    assert prov[docs[0].document_id]["title"] and prov[docs[0].document_id]["devanagari_fraction_diagnostic"] == 1.0
    assert "20231101.ne" in calls[0] and "length=100" in calls[0]


def test_nepali_acquisition_failures_raise():
    with pytest.raises(mc.MiniCorpusAcquisitionError):
        mc.acquire_nepali_wikipedia(open_json=lambda u: (_ for _ in ()).throw(OSError("down")))
    with pytest.raises(mc.MiniCorpusAcquisitionError):
        mc.acquire_nepali_wikipedia(open_json=lambda u: {"rows": [{"nope": 1}]})
    with pytest.raises(mc.MiniCorpusAcquisitionError):
        mc.acquire_nepali_wikipedia(open_json=lambda u: {"rows": []})


def test_assemble_fails_when_required_category_missing(monkeypatch):
    monkeypatch.setattr(mc, "acquire_english", lambda n: ([d for d in _fixture_docs(2) if d.primary_domain == mc.CAT_ENGLISH], {}))
    monkeypatch.setattr(mc, "acquire_github_files", lambda: ([d for d in _fixture_docs(2) if d.primary_domain in (mc.CAT_CODE, mc.CAT_CSHARP, mc.CAT_DOCS)], {}, []))

    def boom(**kw):
        raise mc.MiniCorpusAcquisitionError("no network")

    monkeypatch.setattr(mc, "acquire_nepali_wikipedia", boom)
    with pytest.raises(mc.MiniCorpusAcquisitionError):
        mc.assemble_natural_documents()
    docs, info = mc.assemble_natural_documents(allow_missing_categories=True)
    assert info["missing_required_categories"] == [mc.CAT_NEPALI] and docs


# --- synthetic code-switching --------------------------------------------------------------------------


def test_synthetic_codeswitch_is_labelled_and_mixed_script():
    docs = mc.synthetic_codeswitch_documents()
    assert docs and all(d.license == mc.LICENSE_SYNTHETIC and "synthetic" in d.source_id and "SYNTHETIC" in d.source_name for d in docs)
    for d in docs:
        has_dev = any("\u0900" <= c <= "\u097f" for c in d.text)
        has_lat = any(c.isascii() and c.isalpha() for c in d.text)
        assert has_dev and has_lat, d.text


def test_synthetic_documents_must_carry_synthetic_license():
    bad = [_mk(0, mc.CAT_CODESWITCH, "मेरो code मा bug छ र म debug गर्दै छु", "MIT", source="synthetic-x")]
    with pytest.raises(mc.MiniCorpusError):
        mc.run_representative_mini_corpus(_fixture_docs(3), synthetic_documents=bad, vocab_sizes=SIZES)


# --- full run ----------------------------------------------------------------------------------------------


def test_result_flags_and_required_sections(result):
    assert result["experiment"] == "representative_mini_corpus_investigation"
    assert result["not_a_milestone"] is True and result["nb13_assigned"] is False and result["trains_transformer"] is False
    for key in ("configuration", "source_manifest", "source_license_status", "category_distribution", "language_distribution", "split",
                "corpus_counts", "corpus_hashes", "leakage_checks", "tokenizers", "global_metrics_by_vocab",
                "per_category_metrics_by_vocab", "comparison_to_baseline", "errors", "limitations", "timings_seconds_non_deterministic"):
        assert key in result, key
    assert result["errors"] == []
    assert result["configuration"]["no_winner_selected"] is True
    text = json.dumps(result).lower()
    assert "winner" not in text.replace("no_winner_selected", "") and '"best' not in text
    json.dumps(result)


def test_module_never_touches_transformer_training():
    src = Path(mc.__file__).read_text(encoding="utf-8")
    assert "import torch" not in src and "Trainer" not in src and "run_real_corpus_experiment" not in src


def test_per_category_and_global_metrics_present_and_consistent(result):
    for t in result["tokenizers"]:
        pc = t["per_category"]
        assert set(pc) == set(mc.CATEGORIES)
        measured = [c for c, v in pc.items() if v["status"] == "measured"]
        assert set(measured) == {mc.CAT_ENGLISH, mc.CAT_CODE, mc.CAT_CSHARP, mc.CAT_DOCS, mc.CAT_NEPALI}
        for field in ("total_tokens", "total_characters", "total_words", "document_count", "total_utf8_bytes"):
            assert t["global"]["combined"][field] == sum(pc[c]["combined"][field] for c in measured)
        for c in measured:
            for part in ("train", "validation"):
                s = pc[c][part]
                assert {"tokens_per_word", "tokens_per_character", "encoding_failures", "decoding_failures", "round_trip_mismatches"} <= set(s)
            assert pc[c]["utilization"]["train"]["tokens_observed"] > 0
        g = t["global"]["combined"]
        assert g["tokens_per_character"] == pytest.approx(g["total_tokens"] / g["total_characters"])
        assert t["category_dispersion"]["tokens_per_character"]["max_over_min_ratio"] >= 1.0


def test_nepali_costs_more_tokens_per_character_than_english_at_small_vocab(result):
    pc = _tok(result, 270)["per_category"]
    assert pc[mc.CAT_NEPALI]["combined"]["tokens_per_character"] > pc[mc.CAT_ENGLISH]["combined"]["tokens_per_character"]


def test_codeswitch_natural_status_and_synthetic_reported_separately(result):
    for t in result["tokenizers"]:
        assert t["per_category"][mc.CAT_CODESWITCH]["status"] == "NATURAL SOURCE NOT YET VERIFIED"
        syn = t["synthetic_diagnostic"]
        assert syn["is_synthetic"] is True and syn["natural_data"] is False and syn["used_for_tokenizer_fit"] is False
        assert syn["metrics"]["document_count"] == len(mc.SYNTHETIC_CODESWITCH_LINES)
        assert syn["metrics"]["round_trip_mismatches"] == 0
    assert result["category_distribution"][mc.CAT_CODESWITCH]["accepted"] == 0


def test_correctness_checks_and_leakage(result):
    for t in result["tokenizers"]:
        assert all(t["checks"].values()), t["checks"]
        assert t["fit"]["prefix_equals_independent_training"] is True
    leak = result["leakage_checks"]
    assert leak["document_id_overlap_train_validation"] == 0
    assert leak["content_hash_overlap_train_validation"] == 0
    assert leak["synthetic_documents_in_fit_corpus"] == 0
    assert leak["tokenizer_fit_partition"] == "train_only"


def test_tokenizer_fit_receives_only_train_texts_never_validation_or_synthetic(monkeypatch):
    seen = []
    real = ByteLevelBpeTokenizer.train.__func__

    def spy(cls, corpus, vocab_size, special_tokens=("bos", "eos")):
        seen.append(list(corpus))
        return real(cls, corpus, vocab_size, special_tokens)

    monkeypatch.setattr(ByteLevelBpeTokenizer, "train", classmethod(spy))
    docs = _fixture_docs()
    mc.run_representative_mini_corpus(docs, vocab_sizes=SIZES, verify_prefix_up_to=340)
    built = mc.build_corpus(docs)
    train_texts = {d.text for d in built["train"]}
    val_texts = {d.text for d in built["validation"]}
    synth = {d.text for d in mc.synthetic_codeswitch_documents()}
    assert seen
    for corpus in seen:
        assert set(corpus) <= train_texts
        assert set(corpus).isdisjoint(val_texts) and set(corpus).isdisjoint(synth)


def test_run_is_deterministic(result):
    again = mc.run_representative_mini_corpus(_fixture_docs(), vocab_sizes=SIZES, verify_prefix_up_to=340)
    strip = mc.tvs.strip_nondeterministic
    assert json.dumps(strip(result), sort_keys=True) == json.dumps(strip(again), sort_keys=True)


def test_baseline_comparison_per_category_and_global(result):
    base = 270
    for s, entry in result["comparison_to_baseline"].items():
        assert entry["baseline_vocab_size"] == base
    zero = result["comparison_to_baseline"]["270"]
    assert zero["global"]["token_count_reduction_pct"] == 0.0
    big = result["comparison_to_baseline"]["340"]
    t270, t340 = _tok(result, 270), _tok(result, 340)
    for cat in (mc.CAT_ENGLISH, mc.CAT_NEPALI, mc.CAT_CODE):
        a, b = t270["per_category"][cat]["combined"]["total_tokens"], t340["per_category"][cat]["combined"]["total_tokens"]
        assert big["per_category"][cat]["token_count_reduction_pct"] == pytest.approx((a - b) / a * 100)
    assert big["per_category"][mc.CAT_CODESWITCH] is None


def test_source_and_category_counts_reported(result):
    cc = result["corpus_counts"]
    assert cc["input"] == cc["accepted"] + cc["rejected"] + cc["review"]
    dist = result["category_distribution"]
    assert sum(v["accepted"] for v in dist.values()) == cc["accepted"]
    assert result["language_distribution"] == {"en": 16, "ne": 8, "zxx": 16}
    assert result["source_license_status"]["BLOCKED"] >= 3
    assert result["train_composition"][mc.CAT_NEPALI]["share_of_train_bytes"] > 0


# --- failure handling / configuration ----------------------------------------------------------------------------


@pytest.mark.parametrize("sizes,baseline", [([], None), ([270, 270], 270), ([100, 300], 100), ([300, 400], 270)])
def test_invalid_vocab_configuration_rejected(sizes, baseline):
    with pytest.raises((mc.tvs.VocabScalingConfigError, ValueError)):
        mc.run_representative_mini_corpus(_fixture_docs(3), vocab_sizes=sizes, baseline_vocab_size=baseline)


def test_fit_failure_is_recorded_not_raised(monkeypatch):
    def boom(cls, *a, **k):
        raise RuntimeError("synthetic fit failure")

    monkeypatch.setattr(ByteLevelBpeTokenizer, "train", classmethod(boom))
    res = mc.run_representative_mini_corpus(_fixture_docs(3), vocab_sizes=SIZES, verify_prefix_up_to=0)
    assert res["tokenizers"] == [] and res["errors"] and res["errors"][0]["stage"] == "fit"
    assert res["not_a_milestone"] is True


def test_missing_category_documents_reported_not_faked():
    docs = [d for d in _fixture_docs(6) if d.primary_domain != mc.CAT_NEPALI]
    res = mc.run_representative_mini_corpus(docs, vocab_sizes=SIZES, verify_prefix_up_to=0)
    assert _tok(res, 270)["per_category"][mc.CAT_NEPALI]["status"] == "NO DOCUMENTS ACCEPTED"
    assert res["errors"] == []


# --- architecture -------------------------------------------------------------------------------------------------


def test_yug_package_does_not_import_training():
    root = Path(__file__).resolve().parent.parent / "yug"
    offenders = [
        f"{p}: {ln.strip()}" for p in root.rglob("*.py") for ln in p.read_text(encoding="utf-8").splitlines()
        if ln.strip().startswith(("import training", "from training"))
    ]
    assert offenders == []