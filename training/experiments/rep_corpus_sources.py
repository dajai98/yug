"""Representative-corpus expansion, STAGE 1: source & license registry.

NOT a milestone (no NB-13). Data + validation only. This module does NOT
acquire anything, fetch anything, deduplicate, filter, split, fit a
tokenizer, or train a model. It is stdlib-only and imports nothing from
``yug`` or ``training``.

WHAT IT IS
  A deterministic, validated, fingerprinted table of candidate sources:
  license, pin, acquisition mechanism, status (ACCEPT / REVIEW / BLOCKED /
  UNKNOWN), reason, risks, a proposed concentration cap, and eligibility for
  tokenizer fitting / model training / validation.

POLICY BOUNDARIES (enforced by ``validate_registry``)
  - The license allow-list is a frozen mirror of
    ``training.experiments.representative_mini_corpus.MINI_CORPUS_LICENSE_POLICY``.
    It is NOT extended here. A repo-side test compares the two.
  - ACCEPT requires: allow-listed license id, license evidence that is not
    ``NOT_VERIFIED``, a pin, a reason, and a positive cap.
  - Anything not ACCEPT is ineligible for fitting, training and validation.
  - Unity-authored material can never be ACCEPT.
  - Diagnostic-only sources (synthetic code-switching) can never be eligible.
  - A status that differs from the prior mini-corpus registry requires an
    explicit ``status_change_note`` (no silent re-labelling).
  - Policy questions (BSD, PSF, ...) are OWNER DECISIONS listed separately;
    the registry never resolves them.

STAGE 1B (real-fetch license verification)
  For GitHub-hosted sources, license files were fetched at the recorded commit by
  ``rep_corpus_license_verify`` and their SHA-256 recorded in
  ``artifacts/rep_corpus_10m_v1/00_source_license_evidence.json``. Facts copied into this
  module are in ``_STAGE1B_FACTS``. NO status was changed by Stage 1B: where evidence
  makes a REVIEW source technically eligible, ``proposed_status`` says so and the owner
  decides. The allow-list is untouched.

LICENSE VERIFICATION LEVELS
  PRIOR_RECORD  carried from the earlier registry text; not re-fetched here
  NOT_VERIFIED  claimed/expected license only; never sufficient for ACCEPT
  REAL_FETCH    LICENSE read at a pinned revision (reserved for the next pass;
                no record currently carries it)

Caps are PROPOSALS expressed as a fraction of the category's accepted bytes.
They apply only if the record is ACCEPT. They are not tuned on evidence of
availability, which is unknown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import dataclasses
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REGISTRY_VERSION = "rep-corpus-sources-v1"

CAT_CODE = "code"
CAT_CSHARP = "csharp_gamedev"
CAT_ENGLISH = "general_english"
CAT_DOCS = "software_docs"
CAT_NEPALI = "nepali"
CAT_CODESWITCH = "nepali_english_codeswitch"
CATEGORIES: tuple[str, ...] = (CAT_CODE, CAT_CSHARP, CAT_ENGLISH, CAT_DOCS, CAT_NEPALI, CAT_CODESWITCH)

ACCEPT, REVIEW, BLOCKED, UNKNOWN = "ACCEPT", "REVIEW", "BLOCKED", "UNKNOWN"
STATUSES: tuple[str, ...] = (ACCEPT, REVIEW, BLOCKED, UNKNOWN)

PRIOR_RECORD, NOT_VERIFIED, REAL_FETCH = "PRIOR_RECORD", "NOT_VERIFIED", "REAL_FETCH"
LICENSE_VERIFICATIONS: tuple[str, ...] = (PRIOR_RECORD, NOT_VERIFIED, REAL_FETCH)

PIN_COMMIT, PIN_DATASET_REVISION, PIN_SNAPSHOT_LABEL, PIN_NONE = (
    "COMMIT", "DATASET_REVISION", "SNAPSHOT_LABEL", "NONE")
PIN_KINDS: tuple[str, ...] = (PIN_COMMIT, PIN_DATASET_REVISION, PIN_SNAPSHOT_LABEL, PIN_NONE)

# Frozen mirror of MINI_CORPUS_LICENSE_POLICY's ALLOW entries (default is REVIEW).
UNIFORMITY_VALUES: tuple[str, ...] = ("VERIFIED", "PARTIAL", "UNKNOWN")
STAGE1B_EVIDENCE_FINGERPRINT = "sha256:bbc2fe55f3f9b2942ae9a3f1c38f6464cfa80e2465b02939b49aefa4df4315fb"
ALLOWED_LICENSE_IDS: frozenset[str] = frozenset(
    {"CC-BY-SA-4.0", "CC-BY-SA-3.0", "CC-BY-4.0", "CC-BY-3.0", "MIT", "Apache-2.0", "MS-PL"}
)

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")

_DEFAULT_CAP = {CAT_CODE: 0.10, CAT_CSHARP: 0.10, CAT_ENGLISH: 0.20, CAT_DOCS: 0.15,
                CAT_NEPALI: 0.20, CAT_CODESWITCH: 0.0}


class SourceRegistryError(ValueError):
    """The registry (or a record) violates a Stage-1 invariant."""


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    category: str
    subcategory: str | None
    project: str
    license_id: str
    license_verification: str
    license_evidence: str
    pin: str | None
    pin_kind: str
    acquisition: str
    status: str
    status_reason: str
    risks: tuple[str, ...]
    prior_status: str | None
    status_change_note: str | None
    max_category_share: float  # proposed cap; effective only if ACCEPT
    tokenizer_fit_eligible: bool
    model_training_eligible: bool
    validation_eligible: bool
    diagnostic_only: bool
    unity_authored: bool
    owner_decision_ids: tuple[str, ...]
    evidence_labels: tuple[str, ...]
    # Stage 1B additions (all defaulted so Stage 1A construction still works)
    license_expression: str | None = None
    license_text_sha256: str | None = None
    license_files_detected: tuple[str, ...] = ()
    evidence_commit: str | None = None
    repository_license_verified: bool = False
    file_level_license_uniformity: str = "UNKNOWN"  # VERIFIED / PARTIAL / UNKNOWN
    proposed_status: str | None = None
    proposed_status_note: str | None = None

    @property
    def effective_cap(self) -> float:
        return self.max_category_share if self.status == ACCEPT else 0.0


@dataclass(frozen=True)
class OwnerDecision:
    decision_id: str
    title: str
    question: str
    options: tuple[str, ...]
    impact: str


@dataclass(frozen=True)
class ConcentrationRule:
    rule_id: str
    scope: str
    parameter: str
    value: Any
    basis: str
    label: str  # PROPOSAL / CALCULATED / ASSUMPTION


def _rec(source_id: str, category: str, project: str, status: str, reason: str, *,
         subcategory: str | None = None, license_id: str = "UNKNOWN", verification: str = NOT_VERIFIED,
         evidence: str = "", pin: str | None = None, pin_kind: str = PIN_NONE, acquisition: str = "",
         risks: Sequence[str] = (), prior_status: str | None = None, change_note: str | None = None,
         cap: float | None = None, diagnostic_only: bool = False, unity_authored: bool = False,
         decisions: Sequence[str] = (), labels: Sequence[str] = ()) -> SourceRecord:
    if status == BLOCKED:
        cap = 0.0
    elif cap is None:
        cap = _DEFAULT_CAP[category]
    eligible = status == ACCEPT and not diagnostic_only
    return SourceRecord(
        source_id=source_id, category=category, subcategory=subcategory, project=project,
        license_id=license_id, license_verification=verification, license_evidence=evidence,
        pin=pin, pin_kind=pin_kind, acquisition=acquisition, status=status, status_reason=reason,
        risks=tuple(risks), prior_status=prior_status, status_change_note=change_note,
        max_category_share=float(cap), tokenizer_fit_eligible=eligible, model_training_eligible=eligible,
        validation_eligible=eligible, diagnostic_only=diagnostic_only, unity_authored=unity_authored,
        owner_decision_ids=tuple(decisions), evidence_labels=tuple(labels))


_GH_ACQ = ("explicit file list (or tree enumeration) at the pinned commit via raw.githubusercontent.com, "
           "git blob SHA-1 verified per file (mini-corpus method); method at scale is chosen by the Stage-2 probe")
_GH_NEW_ACQ = "same method as existing GitHub sources; pin chosen at Stage 1 license read; NOT yet pinned"
_PRIOR_LICENSE = "LICENSE at pinned commit per prior registry text; not re-fetched in this task"
_L_PIN = "[VERIFIED IN REPOSITORY] pin in mini_corpus_github_files.json"
_L_PRIOR = "[SOURCE-DERIVED] license/status from prior registry text, not re-verified here"
_L_ASSUME = "[ASSUMPTION] expected license from general knowledge; unverified"
_L_UNKNOWN = "[UNKNOWN] no evidence in repository"
_NO_EVIDENCE = "NOT YET READ: LICENSE at repo root (Stage-1 pass)"
_REVIEW_UNVERIFIED = ("Expected license is allow-listed but no LICENSE evidence has been read; "
                      "REVIEW until it is read at a pinned commit.")


def _existing_gh(sid, cat, sub, repo, lic, ev, commit, risks, *, decisions=(), cap=None):
    return _rec(sid, cat, repo, ACCEPT,
                "Allow-listed license, pinned commit, per-file blob SHAs verified in a prior real run; carried over unchanged.",
                subcategory=sub, license_id=lic, verification=PRIOR_RECORD, evidence=ev, pin=commit,
                pin_kind=PIN_COMMIT, acquisition=_GH_ACQ, risks=risks, prior_status=ACCEPT, cap=cap,
                decisions=decisions, labels=(_L_PIN, _L_PRIOR))


def _build_records() -> list[SourceRecord]:
    r: list[SourceRecord] = []
    repo_lvl = "repo-level license only; bundled third-party code not audited"

    # ---- CODE: existing ----
    r += [
        _existing_gh("gh-psf-requests", CAT_CODE, "python", "psf/requests", "Apache-2.0",
                     "LICENSE = Apache License 2.0 (" + _PRIOR_LICENSE + ")", "611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60",
                     (repo_lvl, "small repository: volume needs many more repos")),
        _existing_gh("gh-expressjs-express", CAT_CODE, "javascript", "expressjs/express", "MIT",
                     "LICENSE = MIT (" + _PRIOR_LICENSE + ")", "7ef98448f8b38099ab1ded55e458538ad47a51e7",
                     (repo_lvl, "small repository")),
        _existing_gh("gh-gin-gonic-gin", CAT_CODE, "go", "gin-gonic/gin", "MIT",
                     "LICENSE = MIT (" + _PRIOR_LICENSE + ")", "1bd0ecf2d3e08da37efa270bb7be20c1eddb0d68",
                     (repo_lvl, "existing file list includes testdata/protoexample/test.pb.go (filename suggests protoc-generated code; "
                      "no generated-file filter existed)")),
        _existing_gh("gh-burntsushi-ripgrep", CAT_CODE, "rust", "BurntSushi/ripgrep", "MIT",
                     "COPYING: dual-licensed Unlicense OR MIT; MIT chosen (" + _PRIOR_LICENSE + ")",
                     "3fce3b5bb0236da2df6d99672afb8a719642eca7",
                     (repo_lvl, "relies on choosing the MIT option of a dual license"), decisions=("OD-10",)),
    ]
    # ---- CODE: candidates (REVIEW; none pinned) ----
    def cand_code(sid, sub, repo, lic, reason, risks=(), decisions=()):
        return _rec(sid, CAT_CODE, repo, REVIEW, reason, subcategory=sub, license_id=lic, evidence=_NO_EVIDENCE,
                    acquisition=_GH_NEW_ACQ, risks=risks, decisions=decisions, labels=(_L_ASSUME,))
    bsd_reason = ("Expected BSD-3-Clause is not in the allow-list; REVIEW until the owner decides OD-01 and the LICENSE is read.")
    r += [
        cand_code("gh-pallets-flask", "python", "pallets/flask", "BSD-3-Clause", bsd_reason, ("BSD not allow-listed",), ("OD-01",)),
        cand_code("gh-encode-httpx", "python", "encode/httpx", "BSD-3-Clause", bsd_reason, ("BSD not allow-listed",), ("OD-01",)),
        cand_code("gh-lodash-lodash", "javascript", "lodash/lodash", "MIT", _REVIEW_UNVERIFIED,
                  ("large generated/modular build artifacts likely; filter needed",)),
        cand_code("gh-spf13-cobra", "go", "spf13/cobra", "Apache-2.0", _REVIEW_UNVERIFIED),
        cand_code("gh-tokio-rs-tokio", "rust", "tokio-rs/tokio", "MIT", _REVIEW_UNVERIFIED, ("large; per-repo cap",)),
        cand_code("gh-serde-rs-serde", "rust", "serde-rs/serde", "MIT OR Apache-2.0", _REVIEW_UNVERIFIED,
                  ("dual license: depends on OD-10",), ("OD-10",)),
        cand_code("gh-google-guava", "java", "google/guava", "Apache-2.0", _REVIEW_UNVERIFIED,
                  ("very large; per-repo cap", "generated/test code share to measure")),
        cand_code("gh-nlohmann-json", "cpp", "nlohmann/json", "MIT", _REVIEW_UNVERIFIED,
                  ("single huge amalgamated header duplicates the split sources",)),
        cand_code("gh-flame-engine-flame", "dart_game", "flame-engine/flame", "MIT", _REVIEW_UNVERIFIED,
                  ("only relevant if the owner wants Dart in the code mix (OD-08)",), ("OD-08",)),
    ]

    # ---- C#/GAME-DEV: existing ----
    r += [
        _existing_gh("gh-cysharp-unitask", CAT_CSHARP, "unity_ecosystem_third_party", "Cysharp/UniTask", "MIT",
                     "LICENSE = MIT (" + _PRIOR_LICENSE + ")", "ceac8d6946b1125fe782cd171fbcb245b567dbf9",
                     (repo_lvl, "third-party Unity-ecosystem source, NOT Unity documentation", "small")),
        _existing_gh("gh-mirrornetworking-mirror", CAT_CSHARP, "unity_ecosystem_third_party", "MirrorNetworking/Mirror", "MIT",
                     "LICENSE = MIT; copyright Unity Technologies 2015 and Mirror contributors 2019 (" + _PRIOR_LICENSE + ")",
                     "c4f3739966e151f405be1762d33502794fd034ff",
                     (repo_lvl, "includes code with Unity Technologies copyright released under MIT by that license; "
                      "owner may want to confirm this is acceptable under OD-07")),
        _existing_gh("gh-monogame-monogame", CAT_CSHARP, "oss_game_framework", "MonoGame/MonoGame", "MS-PL",
                     "LICENSE.txt = Microsoft Public License (" + _PRIOR_LICENSE + ")", "9cce5c73ad5869a8abe54dad9ed201d0678fa8fe",
                     (repo_lvl, "restricted to MonoGame.Framework/ (ThirdParty/ excluded); large, so cap")),
    ]
    # ---- C#: candidates ----
    def cand_cs(sid, sub, repo, lic, risks, reason=_REVIEW_UNVERIFIED, decisions=()):
        return _rec(sid, CAT_CSHARP, repo, REVIEW, reason, subcategory=sub, license_id=lic, evidence=_NO_EVIDENCE,
                    acquisition=_GH_NEW_ACQ, risks=risks, decisions=decisions, labels=(_L_ASSUME,))
    r += [
        cand_cs("gh-dotnet-runtime", "general_csharp_dotnet", "dotnet/runtime", "MIT",
                ("general C#/.NET, NOT Unity data", "huge: must be capped", "likely vendored native code and generated files")),
        cand_cs("gh-dotnet-roslyn", "general_csharp_dotnet", "dotnet/roslyn", "MIT",
                ("general C#, NOT Unity data", "large generated test and syntax files likely")),
        cand_cs("gh-stride3d-stride", "oss_game_framework", "stride3d/stride", "MIT",
                ("game engine; third-party/vendored content to exclude",)),
        cand_cs("gh-fna-xna-fna", "oss_game_framework", "FNA-XNA/FNA", "MS-PL",
                ("license mixture of FNA and bundled third-party components unverified",)),
        cand_cs("gh-neuecc-unirx", "unity_ecosystem_third_party", "neuecc/UniRx", "MIT",
                ("project is archived; small",)),
        cand_cs("gh-messagepack-csharp", "general_csharp_dotnet", "MessagePack-CSharp/MessagePack-CSharp", "Apache-2.0",
                ("general C# with Unity support; do not count as Unity-specific",)),
        _rec("gh-unity-ml-agents", CAT_CSHARP, "Unity-Technologies/ml-agents", REVIEW,
             "Unity-authored; prior registry records Apache-2.0 but the owner's Unity policy is not visible; status held unchanged.",
             subcategory="unity_authored", license_id="Apache-2.0", verification=PRIOR_RECORD,
             evidence="LICENSE.md = Apache License 2.0 (prior registry says VERIFIED BY REAL FETCH; not re-fetched here)",
             acquisition="not automated", risks=("Unity-authored material", "owner policy unseen"),
             prior_status=REVIEW, unity_authored=True, decisions=("OD-07",), labels=(_L_PRIOR,)),
        _rec("gh-unity-csreference", CAT_CSHARP, "Unity-Technologies/UnityCsReference", BLOCKED,
             "Unity Reference-Only License is not a training-use grant.", subcategory="unity_official",
             license_id="Unity Reference-Only License", verification=PRIOR_RECORD,
             evidence="LICENSE.md points to the Unity Reference-Only License (prior registry: VERIFIED BY REAL FETCH)",
             acquisition="none", risks=("reference-only terms",), prior_status=BLOCKED, unity_authored=True,
             labels=(_L_PRIOR,)),
        _rec("unity-docs", CAT_CSHARP, "Unity Manual / Scripting API documentation", BLOCKED,
             "Not established as training-safe; must not be scraped.", subcategory="unity_official",
             evidence="not verified as training-safe (prior registry)", acquisition="none", prior_status=BLOCKED,
             unity_authored=True, labels=(_L_PRIOR,)),
        _rec("unity-asset-store-academy", CAT_CSHARP, "Unity Asset Store / Unity Academy content", BLOCKED,
             "Directive: must not be used.", subcategory="unity_official", acquisition="none",
             prior_status=BLOCKED, unity_authored=True, labels=(_L_PRIOR,)),
        _rec("unity-learn", CAT_CSHARP, "Unity Learn content", BLOCKED,
             "New record (no prior entry): Unity Learn content is not acquired unless explicit training-use rights are independently established.",
             subcategory="unity_official", acquisition="none", unity_authored=True,
             risks=("not separately registered before; previously covered only by the Academy entry",),
             labels=(_L_UNKNOWN,)),
    ]

    # ---- DOCS: existing ----
    r += [
        _existing_gh("gh-google-eng-practices", CAT_DOCS, "engineering_practices", "google/eng-practices", "CC-BY-3.0",
                     "LICENSE = Creative Commons Attribution 3.0 Unported (" + _PRIOR_LICENSE + ")",
                     "3bb3ec25b3b0199f4940b1aa75f0ac5c5753301c", (repo_lvl, "attribution required", "small")),
        _existing_gh("gh-microsoft-api-guidelines", CAT_DOCS, "api_design", "microsoft/api-guidelines", "CC-BY-4.0",
                     "license.txt = Creative Commons Attribution 4.0 International (" + _PRIOR_LICENSE + ")",
                     "a7022a299442a8352431874e63ec4dff548a1b81", (repo_lvl, "attribution required", "small")),
        _existing_gh("gh-rust-lang-book", CAT_DOCS, "rust", "rust-lang/book", "MIT",
                     "LICENSE-MIT + LICENSE-APACHE, dual MIT OR Apache-2.0; MIT chosen (" + _PRIOR_LICENSE + ")",
                     "1500248d8f230566e4ec9f27fcbb8fe9e2898ab1",
                     (repo_lvl, "relies on choosing the MIT option of a dual license"), decisions=("OD-10",)),
    ]
    # ---- DOCS: candidates ----
    def cand_doc(sid, sub, proj, lic, status, reason, risks, decisions=(), label=_L_ASSUME):
        return _rec(sid, CAT_DOCS, proj, status, reason, subcategory=sub, license_id=lic,
                    evidence=_NO_EVIDENCE, acquisition=_GH_NEW_ACQ, risks=risks, decisions=decisions, labels=(label,))
    docs_note = "Documentation license is evaluated separately from code license; "
    r += [
        cand_doc("gh-rust-lang-reference", "rust", "rust-lang/reference", "MIT OR Apache-2.0", REVIEW,
                 _REVIEW_UNVERIFIED, ("dual license: OD-10",), ("OD-10",)),
        cand_doc("gh-rust-lang-rust-by-example", "rust", "rust-lang/rust-by-example", "MIT OR Apache-2.0", REVIEW,
                 _REVIEW_UNVERIFIED, ("dual license: OD-10",), ("OD-10",)),
        cand_doc("gh-rust-lang-rustc-dev-guide", "rust", "rust-lang/rustc-dev-guide", "MIT OR Apache-2.0", REVIEW,
                 _REVIEW_UNVERIFIED, ("dual license: OD-10", "compiler-internals specialization"), ("OD-10",)),
        cand_doc("python-docs-cpython-doc", "python", "CPython Doc/ (Python documentation)", "PSF-2.0", REVIEW,
                 "PSF license is not in the allow-list; REVIEW until the owner decides OD-02. " + docs_note + "not assumed.",
                 ("PSF license not allow-listed",), ("OD-02",)),
        cand_doc("gh-dotnet-docs", "dotnet", "dotnet/docs (Microsoft/.NET documentation)", "CC-BY-4.0 (docs) + MIT (code samples)",
                 REVIEW, "Mixed docs/code licensing expected; path-level license handling needed (OD-11). " + docs_note + "not assumed.",
                 ("separate docs and code licenses", "Microsoft Learn site content itself is not covered by the repo license",
                  "includes localized and templated content"), ("OD-11",)),
        cand_doc("mdn-content", "web_platform", "mdn/content (MDN)", "CC-BY-SA-2.5 (prose) + CC0-1.0 (code samples)", REVIEW,
                 "Older CC-BY-SA version and CC0 are not in the allow-list; REVIEW until OD-03/OD-04.",
                 ("older CC version", "heavy markup and templating"), ("OD-03", "OD-04")),
        cand_doc("gh-kubernetes-website", "kubernetes", "kubernetes/website (Kubernetes documentation)", "CC-BY-4.0 (docs) + Apache-2.0 (code)",
                 REVIEW, _REVIEW_UNVERIFIED + " " + docs_note + "not assumed.",
                 ("non-English translation directories must be excluded", "generated reference pages", "Hugo shortcodes"),
                 ("OD-11",)),
        cand_doc("docker-docs", "containers", "docker/docs (Docker documentation)", "UNKNOWN", UNKNOWN,
                 "No license evidence for the documentation content; do not infer from the repository code license.",
                 ("docs vs code license unknown",), (), _L_UNKNOWN),
    ]

    # ---- ENGLISH ----
    r += [
        _rec("cp-wikimedia-filtered", CAT_ENGLISH, "common-pile/wikimedia_filtered", ACCEPT,
             "Pinned dataset revision; per-record license strings exact-match mapped to CC-BY-SA-4.0 (other strings pass through to REVIEW); carried over unchanged.",
             subcategory="wikibooks (the only wiki observed in shard 0)", license_id="CC-BY-SA-4.0", verification=PRIOR_RECORD,
             evidence="per-record license string mapped by exact match in real_corpus_experiment.py (WIKIMEDIA_LICENSE_SOURCE_STRING)",
             pin="0641bb84bd9b7162bcddf8be7836822161a9a342", pin_kind=PIN_DATASET_REVISION,
             acquisition="stdlib urllib streamed shard prefix at the pinned revision; needs per-shard quotas to diversify",
             risks=("Wikibooks-only domain observed; filling 19% from it alone would be a diversity failure",
                    "dataset card warns of possible upstream license misassignment",
                    "share-alike + attribution obligations", "no raw-stream hash is kept"),
             prior_status=ACCEPT, cap=0.50, labels=("[VERIFIED IN REPOSITORY] pin and mapping", _L_PRIOR)),
        _rec("hf-wikipedia-en-20231101", CAT_ENGLISH, "wikimedia/wikipedia (20231101.en)", REVIEW,
             "Same Wikipedia license basis as the Nepali record; REVIEW until the owner settles OD-06.",
             subcategory="encyclopedia", license_id="CC-BY-SA-3.0", verification=NOT_VERIFIED,
             evidence="dataset card text (seen via a mirror's copy, not the original): GFDL and CC BY-SA 3.0",
             pin="20231101", pin_kind=PIN_SNAPSHOT_LABEL, acquisition="not designed; rows API or parquet, decided at the Stage-2 probe",
             risks=("snapshot label only, no revision pin", "card says 3.0 + GFDL; newer Wikipedia text may be CC BY-SA 4.0", "very large: cap"),
             decisions=("OD-06",), cap=0.35, labels=("[SOURCE-DERIVED] mirror copy of card text",)),
        _rec("cp-other-subsets", CAT_ENGLISH, "common-pile (non-wikimedia subsets)", UNKNOWN,
             "Subsets and per-subset licenses not inspected.", subcategory="mixed", acquisition="not designed",
             risks=("per-subset license varies",), labels=(_L_UNKNOWN,)),
        _rec("public-domain-books", CAT_ENGLISH, "public-domain books (e.g. via a library collection)", UNKNOWN,
             "Public-domain status varies by jurisdiction and edition; no evidence gathered.", subcategory="public_domain",
             acquisition="not designed", risks=("per-work status", "platform terms"), labels=(_L_UNKNOWN,)),
        _rec("common-corpus", CAT_ENGLISH, "Common Corpus", UNKNOWN, "Not inspected.", acquisition="not designed",
             risks=("per-document licensing",), labels=(_L_UNKNOWN,)),
    ]

    # ---- NEPALI ----
    r += [
        _rec("hf-wikipedia-ne-20231101", CAT_NEPALI, "wikimedia/wikipedia (20231101.ne)", ACCEPT,
             "Only source with prior evidence; prior registry marks ACCEPT as a PROPOSAL with the final legal call left to the owner; status carried over unchanged.",
             subcategory="wikipedia", license_id="CC-BY-SA-3.0", verification=PRIOR_RECORD,
             evidence="dataset card: GFDL and CC BY-SA 3.0, some text may carry additional licenses (prior registry; the card text was also seen via a mirror)",
             pin="20231101", pin_kind=PIN_SNAPSHOT_LABEL,
             acquisition="HF dataset-viewer /rows API, <=100 rows per request (prior probe: 2,000 rows scanned, 23 documents kept); at scale to be re-decided at Stage 2",
             risks=("32,885 rows reported but yield is unproven", "API truncates long text cells so long articles are skipped, biasing to short or templated pages",
                    "single source: no source diversification possible", "snapshot label only, no revision pin",
                    "Devanagari also covers Hindi/Marathi: script alone does not verify language"),
             prior_status=ACCEPT, cap=1.0, decisions=("OD-06", "OD-09"),
             labels=("[VERIFIED BY PRIOR REAL EXECUTION] rows reported / scanned / kept supplied by owner", _L_PRIOR)),
        _rec("ne-wikimedia-other-projects", CAT_NEPALI, "Nepali Wikisource / Wikibooks / Wikinews", UNKNOWN,
             "Volume and licenses unknown.", subcategory="other_wikimedia", acquisition="not designed",
             risks=("per-project license", "small volume"), labels=(_L_UNKNOWN,)),
        _rec("web-crawl-nepali", CAT_NEPALI, "Web-crawl-derived Nepali corpora (CC-100 / OSCAR / mC4 ne)", REVIEW,
             "Unclear per-document licensing; status unchanged. Proposal: no contribution even if cleared, until quality evidence exists.",
             subcategory="web_crawl", acquisition="none", risks=("licensing", "quality", "republication of Wikipedia text"),
             prior_status=REVIEW, cap=0.0, labels=(_L_PRIOR,)),
        _rec("ne-news-gov-books", CAT_NEPALI, "Nepali news, government and book sources", UNKNOWN,
             "No license evidence; not proposed.", subcategory="news_gov_books", acquisition="none",
             risks=("copyright", "no explicit license"), labels=(_L_UNKNOWN,)),
    ]

    # ---- CODE-SWITCHING ----
    r += [
        _rec("natural-codeswitch-none", CAT_CODESWITCH, "Natural English-Nepali code-switching corpus", UNKNOWN,
             "No source with verified license and provenance exists.", acquisition="none",
             prior_status="NATURAL SOURCE NOT YET VERIFIED",
             change_note="Relabelled to UNKNOWN to fit the four-status scheme; meaning unchanged: no natural source, 0% share.",
             labels=(_L_PRIOR,)),
        _rec("synthetic-codeswitch-diagnostic", CAT_CODESWITCH, "Hand-written synthetic code-switching lines", BLOCKED,
             "Diagnostic only: excluded from tokenizer fitting, training and primary scoring; may be measured separately as a labelled synthetic set.",
             license_id="synthetic-diagnostic", verification=PRIOR_RECORD,
             evidence="original text written for diagnostics (prior registry); not native-speaker reviewed",
             acquisition="in-repo constant", prior_status="SYNTHETIC_DIAGNOSTIC_ONLY",
             change_note="Relabelled to BLOCKED + diagnostic_only to fit the four-status scheme; it remains measurable as a diagnostic.",
             diagnostic_only=True, labels=(_L_PRIOR,)),
    ]
    return r


# ---------------------------------------------------------------------------
# Stage 1B: facts from the real license fetch (see rep_corpus_license_verify.py).
# "expression" and "proposed" are human-reviewed; the rest is copied from the evidence file.
# ---------------------------------------------------------------------------
_STAGE1B_FACTS: dict[str, dict[str, Any]] = {'docker-docs': {'commit': '4d3cbcd0f78327cfc6ec5f357e3af512fcbe53f3',
                 'extra_risks': (),
                 'files': ('LICENSE=Apache-2.0',),
                 'license_expression': 'Apache-2.0 (repository LICENSE; applicability to documentation content not established)',
                 'license_id': 'Apache-2.0',
                 'primary_path': 'LICENSE',
                 'proposed_note': 'Repository-level license does not establish the documentation license (rule: do not assume).',
                 'proposed_status': 'UNKNOWN',
                 'sha256': 'sha256:089009c932d3cc296355e429fa9d6faf80ac89597da866e2424b4ea09add64b1',
                 'uniformity': 'UNKNOWN'},
 'gh-burntsushi-ripgrep': {'commit': '3fce3b5bb0236da2df6d99672afb8a719642eca7',
                           'extra_risks': (),
                           'files': ('COPYING=UNRECOGNISED', 'LICENSE-MIT=MIT', 'UNLICENSE=Unlicense'),
                           'license_expression': 'MIT OR Unlicense',
                           'license_id': 'MIT',
                           'primary_path': 'LICENSE-MIT',
                           'proposed_note': 'Unchanged. [SOURCE-DERIVED] COPYING and README state dual licensing (Unlicense and '
                                            'MIT; either license). [PROPOSAL] rely on the MIT branch (OD-10).',
                           'proposed_status': 'ACCEPT',
                           'sha256': 'sha256:0f96a83840e146e43c0ec96a22ec1f392e0680e6c1226e6f3ba87e0740af850f',
                           'uniformity': 'UNKNOWN'},
 'gh-cysharp-unitask': {'commit': 'ceac8d6946b1125fe782cd171fbcb245b567dbf9',
                        'extra_risks': (),
                        'files': ('LICENSE=MIT',),
                        'license_expression': 'MIT',
                        'license_id': 'MIT',
                        'primary_path': 'LICENSE',
                        'proposed_note': 'Unchanged.',
                        'proposed_status': 'ACCEPT',
                        'sha256': 'sha256:b0e6aada5cdff8e90865b150498b81af955343f1d8f091e5b609fd94c4d4b3d0',
                        'uniformity': 'UNKNOWN'},
 'gh-dotnet-docs': {'commit': '1443883303ca8a9e515a13c1c39c7224bb2cfa33',
                    'extra_risks': (),
                    'files': ('LICENSE=CC-BY-4.0', 'LICENSE-CODE=MIT'),
                    'license_expression': 'CC-BY-4.0 (LICENSE) + MIT (LICENSE-CODE); path scope not stated in the fetched text',
                    'license_id': 'CC-BY-4.0',
                    'primary_path': 'LICENSE',
                    'proposed_note': 'Both licenses are allow-listed, but which paths each covers is not evidenced: OD-11.',
                    'proposed_status': 'REVIEW',
                    'sha256': 'sha256:243f341367f0ee35271c8924aaf5d075157c62538042e2f3c11ff9a1fcc5539c',
                    'uniformity': 'PARTIAL'},
 'gh-dotnet-roslyn': {'commit': 'a1bbcb8c7337ff9c51e642bdb9564577a3d928b9',
                      'extra_risks': (),
                      'files': ('License.txt=MIT',),
                      'license_expression': 'MIT',
                      'license_id': 'MIT',
                      'primary_path': 'License.txt',
                      'proposed_note': 'Technically eligible. Not promoted.',
                      'proposed_status': 'ACCEPT',
                      'sha256': 'sha256:ae48df11a335dc1a615f4f938b69cba73bcf4485c4f97af49b38efb0f216353b',
                      'uniformity': 'UNKNOWN'},
 'gh-dotnet-runtime': {'commit': 'f823145d31bb750f7c91d1a25c191c6933df8938',
                       'extra_risks': (),
                       'files': ('LICENSE.TXT=MIT',),
                       'license_expression': 'MIT',
                       'license_id': 'MIT',
                       'primary_path': 'LICENSE.TXT',
                       'proposed_note': 'THIRD-PARTY-NOTICES.TXT present and README says the repo is MIT: vendored and '
                                        'third-party paths must be excluded first.',
                       'proposed_status': 'REVIEW',
                       'sha256': 'sha256:cfc21f5e8bd655ae997eec916138b707b1d290b83272c02a95c9f821b8c87310',
                       'uniformity': 'PARTIAL'},
 'gh-encode-httpx': {'commit': 'b5addb64f0161ff6bfe94c124ef76f6a1fba5254',
                     'extra_risks': (),
                     'files': ('LICENSE.md=BSD-3-Clause',),
                     'license_expression': 'BSD-3-Clause',
                     'license_id': 'BSD-3-Clause',
                     'primary_path': 'LICENSE.md',
                     'proposed_note': 'BSD-3-Clause is not allow-listed: OD-01.',
                     'proposed_status': 'REVIEW',
                     'sha256': 'sha256:4ec59d544f12b5f539a3a716fd321ac58ccd8030b465221f2c880200cdf28d8d',
                     'uniformity': 'UNKNOWN'},
 'gh-expressjs-express': {'commit': '7ef98448f8b38099ab1ded55e458538ad47a51e7',
                          'extra_risks': (),
                          'files': ('LICENSE=MIT',),
                          'license_expression': 'MIT',
                          'license_id': 'MIT',
                          'primary_path': 'LICENSE',
                          'proposed_note': 'Unchanged.',
                          'proposed_status': 'ACCEPT',
                          'sha256': 'sha256:95a5762890e5c1c9808921cef095661fc482c5e1f0bba31446ac85595df6237c',
                          'uniformity': 'UNKNOWN'},
 'gh-flame-engine-flame': {'commit': '13c34f301b370d90ac8dcb2abb5ff04ce093681e',
                           'extra_risks': (),
                           'files': ('LICENSE=MIT',),
                           'license_expression': 'MIT',
                           'license_id': 'MIT',
                           'primary_path': 'LICENSE',
                           'proposed_note': 'Technically eligible; only relevant if Dart is in the code mix (OD-08). Not '
                                            'promoted.',
                           'proposed_status': 'ACCEPT',
                           'sha256': 'sha256:a53bdfb9afc14859e8d0526dfc8880831de8cfa599b300ddbbe122441dec30e3',
                           'uniformity': 'UNKNOWN'},
 'gh-fna-xna-fna': {'commit': '651c0ffdf4db43cd112865ad368c5d6cc3dc7558',
                    'extra_risks': ('README: LzxDecoder.cs dual MSPL/LGPL',),
                    'files': ('licenses/LICENSE=MS-PL',
                              'licenses/lzxdecoder.LICENSE=UNRECOGNISED',
                              'licenses/monoxna.LICENSE=MIT'),
                    'license_expression': 'MS-PL (licenses/LICENSE); licenses/monoxna.LICENSE = MIT; licenses/lzxdecoder.LICENSE '
                                          'unrecognised',
                    'license_id': 'MS-PL',
                    'primary_path': 'licenses/LICENSE',
                    'proposed_note': '[SOURCE-DERIVED] README says one component is under a dual MSPL/LGPL license; submodules '
                                     'present. LGPL-covered files must be excluded and the license read by a human.',
                    'proposed_status': 'REVIEW',
                    'sha256': 'sha256:9f59f590c7697491a8846391b73eea7cd6f47a7e3b7e512b17813e85f94964f6',
                    'uniformity': 'PARTIAL'},
 'gh-gin-gonic-gin': {'commit': '1bd0ecf2d3e08da37efa270bb7be20c1eddb0d68',
                      'extra_risks': (),
                      'files': ('LICENSE=MIT',),
                      'license_expression': 'MIT',
                      'license_id': 'MIT',
                      'primary_path': 'LICENSE',
                      'proposed_note': 'Unchanged. Generated-file filter needed (existing list has a .pb.go file).',
                      'proposed_status': 'ACCEPT',
                      'sha256': 'sha256:b104efb2c7700691650f27034e8541c5ae0ed9af54d884287f19a7467ca2fe7f',
                      'uniformity': 'UNKNOWN'},
 'gh-google-eng-practices': {'commit': '3bb3ec25b3b0199f4940b1aa75f0ac5c5753301c',
                             'extra_risks': (),
                             'files': ('LICENSE=CC-BY-3.0',),
                             'license_expression': 'CC-BY-3.0',
                             'license_id': 'CC-BY-3.0',
                             'primary_path': 'LICENSE',
                             'proposed_note': 'Unchanged.',
                             'proposed_status': 'ACCEPT',
                             'sha256': 'sha256:11dbf7126ae9794e4a7dc54334f3c82a67efcf22df888d396dc2ed707b2a3f9b',
                             'uniformity': 'UNKNOWN'},
 'gh-google-guava': {'commit': '5dc4cd1ad452811629cb2ebe72c2c21142c22c1b',
                     'extra_risks': (),
                     'files': ('LICENSE=Apache-2.0',),
                     'license_expression': 'Apache-2.0',
                     'license_id': 'Apache-2.0',
                     'primary_path': 'LICENSE',
                     'proposed_note': 'Technically eligible; very large, needs the per-repo cap. Not promoted.',
                     'proposed_status': 'ACCEPT',
                     'sha256': 'sha256:cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30',
                     'uniformity': 'UNKNOWN'},
 'gh-kubernetes-website': {'commit': '7a1c866f252e2d62a0bcdd3ea23a0052e6674679',
                           'extra_risks': ('Stage 1A assumed an Apache-2.0 code license that no fetched file shows',),
                           'files': ('LICENSE=CC-BY-4.0',),
                           'license_expression': 'CC-BY-4.0 (LICENSE); no code license file found; submodules present',
                           'license_id': 'CC-BY-4.0',
                           'primary_path': 'LICENSE',
                           'proposed_note': 'Only a CC-BY-4.0 license was evidenced; the claimed Apache-2.0 code license was NOT '
                                            'found. Submodules present.',
                           'proposed_status': 'REVIEW',
                           'sha256': 'sha256:9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411',
                           'uniformity': 'PARTIAL'},
 'gh-lodash-lodash': {'commit': '2b5e6f7399a7b48005140b5d5c6bc6c0e62919a8',
                      'extra_risks': (),
                      'files': ('LICENSE=MIT',),
                      'license_expression': 'MIT',
                      'license_id': 'MIT',
                      'primary_path': 'LICENSE',
                      'proposed_note': 'Technically eligible: verified allow-listed license, no conflicting evidence found. Not '
                                       'promoted.',
                      'proposed_status': 'ACCEPT',
                      'sha256': 'sha256:f71e8ed126b46346494aad5486874cd8f0aafe95092ed67d2e3cb6110f939abc',
                      'uniformity': 'UNKNOWN'},
 'gh-messagepack-csharp': {'commit': 'f2bfa97ef56c3e0eda92d7df34856467afcda865',
                           'extra_risks': ("Stage 1A listed a non-existent repository 'Cysharp/MessagePack-CSharP'; corrected "
                                           'here to MessagePack-CSharp/MessagePack-CSharp',),
                           'files': ('LICENSE=Apache-2.0',),
                           'license_expression': 'Apache-2.0 (primary); file text also contains BSD-2-Clause and MIT notices',
                           'license_id': 'Apache-2.0',
                           'primary_path': 'LICENSE',
                           'proposed_note': 'LICENSE also contains BSD-2-Clause text, which is not allow-listed (OD-01).',
                           'proposed_status': 'REVIEW',
                           'sha256': 'sha256:580852521f6e48cb1d210fcb228e7179a21433a32ebe3fbb46ec8c10255e7451',
                           'uniformity': 'PARTIAL'},
 'gh-microsoft-api-guidelines': {'commit': 'a7022a299442a8352431874e63ec4dff548a1b81',
                                 'extra_risks': (),
                                 'files': ('license.txt=CC-BY-4.0',),
                                 'license_expression': 'CC-BY-4.0',
                                 'license_id': 'CC-BY-4.0',
                                 'primary_path': 'license.txt',
                                 'proposed_note': 'Unchanged.',
                                 'proposed_status': 'ACCEPT',
                                 'sha256': 'sha256:e5795f98ce65fbcdb6311f8ae468038bb58d0049d6b913493752aad2099eea04',
                                 'uniformity': 'UNKNOWN'},
 'gh-mirrornetworking-mirror': {'commit': 'c4f3739966e151f405be1762d33502794fd034ff',
                                'extra_risks': (),
                                'files': ('LICENSE=MIT',),
                                'license_expression': 'MIT',
                                'license_id': 'MIT',
                                'primary_path': 'LICENSE',
                                'proposed_note': 'Unchanged.',
                                'proposed_status': 'ACCEPT',
                                'sha256': 'sha256:74819ab7f08d5e3b91f0392fc41a3146e27bbefe4af90dfafbf71b9c470a9560',
                                'uniformity': 'UNKNOWN'},
 'gh-monogame-monogame': {'commit': '9cce5c73ad5869a8abe54dad9ed201d0678fa8fe',
                          'extra_risks': ('LICENSE.txt text also mentions MIT terms (third-party notices)',),
                          'files': ('LICENSE.txt=MS-PL',),
                          'license_expression': 'MS-PL',
                          'license_id': 'MS-PL',
                          'primary_path': 'LICENSE.txt',
                          'proposed_note': 'Unchanged. .gitmodules present and LICENSE.txt also mentions MIT: keep ThirdParty/ '
                                           'and submodule paths excluded.',
                          'proposed_status': 'ACCEPT',
                          'sha256': 'sha256:67301fd0f95a7e4ffb3cccde3dafed6ada739f32585c0f7b70adeb2fb904504b',
                          'uniformity': 'PARTIAL'},
 'gh-neuecc-unirx': {'commit': '6baeccf6c544c155497164327cca72f28163a578',
                     'extra_risks': (),
                     'files': ('LICENSE=MIT',),
                     'license_expression': 'MIT',
                     'license_id': 'MIT',
                     'primary_path': 'LICENSE',
                     'proposed_note': 'Technically eligible; archived project. Not promoted.',
                     'proposed_status': 'ACCEPT',
                     'sha256': 'sha256:8d12196d9689ef80cb99e5ca166ceb9613ff0fe62098de7ba110c413bd4c7b14',
                     'uniformity': 'UNKNOWN'},
 'gh-nlohmann-json': {'commit': '11e75a54280b3474cb18c2f24d42660dc385596d',
                      'extra_risks': ('README names bundled third-party code (UTF-8 decoder, Grisu2) under MIT',),
                      'files': ('LICENSE.MIT=MIT',),
                      'license_expression': 'MIT',
                      'license_id': 'MIT',
                      'primary_path': 'LICENSE.MIT',
                      'proposed_note': 'README states bundled third-party components under MIT; exclusions needed before '
                                       'promotion.',
                      'proposed_status': 'REVIEW',
                      'sha256': 'sha256:d64d1b4d948aa7751cfb613c02dc246aab7b358d760053e6580f01affe946906',
                      'uniformity': 'PARTIAL'},
 'gh-pallets-flask': {'commit': 'd73fa1cdcbd8b1465c151db8924ba58b1dd14e35',
                      'extra_risks': (),
                      'files': ('LICENSE.txt=BSD-3-Clause',),
                      'license_expression': 'BSD-3-Clause',
                      'license_id': 'BSD-3-Clause',
                      'primary_path': 'LICENSE.txt',
                      'proposed_note': 'BSD-3-Clause is not allow-listed: OD-01.',
                      'proposed_status': 'REVIEW',
                      'sha256': 'sha256:489a8e1108509ed98a37bb983e11e0f7e1d31f0bd8f99a79c8448e7ff37d07ea',
                      'uniformity': 'UNKNOWN'},
 'gh-psf-requests': {'commit': '611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60',
                     'extra_risks': (),
                     'files': ('LICENSE=Apache-2.0',),
                     'license_expression': 'Apache-2.0',
                     'license_id': 'Apache-2.0',
                     'primary_path': 'LICENSE',
                     'proposed_note': 'Unchanged. NOTICE file present (standard attribution).',
                     'proposed_status': 'ACCEPT',
                     'sha256': 'sha256:09e8a9bcec8067104652c168685ab0931e7868f9c8284b66f5ae6edae5f1130b',
                     'uniformity': 'PARTIAL'},
 'gh-rust-lang-book': {'commit': '1500248d8f230566e4ec9f27fcbb8fe9e2898ab1',
                       'extra_risks': (),
                       'files': ('LICENSE-MIT=MIT', 'LICENSE-APACHE=Apache-2.0'),
                       'license_expression': 'Apache-2.0 + MIT (two license files; relationship not stated in the fetched '
                                             'README)',
                       'license_id': 'MIT',
                       'primary_path': 'LICENSE-MIT',
                       'proposed_note': 'Unchanged. Both licenses are allow-listed, so ACCEPT holds whether the files combine as '
                                        "OR or AND. The earlier 'MIT OR Apache-2.0' wording is not evidenced by fetched text.",
                       'proposed_status': 'ACCEPT',
                       'sha256': 'sha256:0621878e61f0d0fda054bcbe02df75192c28bde1ecc8289cbd86aeba2dd72720',
                       'uniformity': 'UNKNOWN'},
 'gh-rust-lang-reference': {'commit': 'a286e1ecb8ce193d50056e4bb042eea9b23876ef',
                            'extra_risks': (),
                            'files': ('LICENSE-MIT=MIT', 'LICENSE-APACHE=Apache-2.0'),
                            'license_expression': 'Apache-2.0 + MIT (two license files; relationship not stated in the fetched '
                                                  'README)',
                            'license_id': 'MIT',
                            'primary_path': 'LICENSE-MIT',
                            'proposed_note': 'Technically eligible: both licenses allow-listed. Not promoted.',
                            'proposed_status': 'ACCEPT',
                            'sha256': 'sha256:0621878e61f0d0fda054bcbe02df75192c28bde1ecc8289cbd86aeba2dd72720',
                            'uniformity': 'UNKNOWN'},
 'gh-rust-lang-rust-by-example': {'commit': 'dcc14f912bf8790792a2df082eddc82ab2539229',
                                  'extra_risks': (),
                                  'files': ('LICENSE-MIT=MIT', 'LICENSE-APACHE=Apache-2.0'),
                                  'license_expression': 'Apache-2.0 OR MIT',
                                  'license_id': 'MIT',
                                  'primary_path': 'LICENSE-MIT',
                                  'proposed_note': '[SOURCE-DERIVED] README: licensed under either license at your option. Not '
                                                   'promoted.',
                                  'proposed_status': 'ACCEPT',
                                  'sha256': 'sha256:4594f18866be94a3ce309ed8f94d92699f5b87739b76e1216addca75f5253fa7',
                                  'uniformity': 'UNKNOWN'},
 'gh-rust-lang-rustc-dev-guide': {'commit': 'be8854f66df1b5450349061e45374ac072abd109',
                                  'extra_risks': (),
                                  'files': ('LICENSE-MIT=MIT', 'LICENSE-APACHE=Apache-2.0'),
                                  'license_expression': 'Apache-2.0 + MIT (two license files; relationship not stated in the '
                                                        'fetched README)',
                                  'license_id': 'MIT',
                                  'primary_path': 'LICENSE-MIT',
                                  'proposed_note': 'Technically eligible. Not promoted.',
                                  'proposed_status': 'ACCEPT',
                                  'sha256': 'sha256:23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3',
                                  'uniformity': 'UNKNOWN'},
 'gh-serde-rs-serde': {'commit': '6693a89cca77e0151437da1c7f890090b9ebf04c',
                       'extra_risks': (),
                       'files': ('LICENSE-MIT=MIT', 'LICENSE-APACHE=Apache-2.0'),
                       'license_expression': 'Apache-2.0 OR MIT',
                       'license_id': 'MIT',
                       'primary_path': 'LICENSE-MIT',
                       'proposed_note': '[SOURCE-DERIVED] README: licensed under either license at your option. [PROPOSAL] rely '
                                        'on either allow-listed branch; OD-10 still open. Not promoted.',
                       'proposed_status': 'ACCEPT',
                       'sha256': 'sha256:23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3',
                       'uniformity': 'UNKNOWN'},
 'gh-spf13-cobra': {'commit': 'adbc8813901bba65827259daa8e22ff94ec1f30e',
                    'extra_risks': (),
                    'files': ('LICENSE.txt=Apache-2.0',),
                    'license_expression': 'Apache-2.0',
                    'license_id': 'Apache-2.0',
                    'primary_path': 'LICENSE.txt',
                    'proposed_note': 'Technically eligible. Not promoted.',
                    'proposed_status': 'ACCEPT',
                    'sha256': 'sha256:5e3400b93bbb099e83e52bab885e7441750673c21f97988ca3f1240639b63283',
                    'uniformity': 'UNKNOWN'},
 'gh-stride3d-stride': {'commit': 'a7fa31ced680c7d4a919f1fe233051cf508f6060',
                        'extra_risks': ('README states some files carry other terms',),
                        'files': ('LICENSE.md=MIT',),
                        'license_expression': 'MIT',
                        'license_id': 'MIT',
                        'primary_path': 'LICENSE.md',
                        'proposed_note': 'README says MIT unless stated otherwise (some files copied from other projects, with a '
                                         'third-party list): exclusions needed.',
                        'proposed_status': 'REVIEW',
                        'sha256': 'sha256:46d821d4ede6fab226ec02b0392f8c2633ad3d9d2cc2cbd709e54a7a8b0a79f1',
                        'uniformity': 'PARTIAL'},
 'gh-tokio-rs-tokio': {'commit': 'e6e84a4d4a5e11d5d0f419444a634ee07203b3e5',
                       'extra_risks': (),
                       'files': ('LICENSE=MIT',),
                       'license_expression': 'MIT',
                       'license_id': 'MIT',
                       'primary_path': 'LICENSE',
                       'proposed_note': 'Technically eligible. Not promoted.',
                       'proposed_status': 'ACCEPT',
                       'sha256': 'sha256:253cd04c6714889df2d32f3f64d669179a1c95c76ac43c40882c52eb06bc3552',
                       'uniformity': 'UNKNOWN'},
 'gh-unity-ml-agents': {'commit': '3e2237ef0c8bca470d8a0277b17e78e1411166d4',
                        'extra_risks': (),
                        'files': ('LICENSE.md=Apache-2.0',),
                        'license_expression': 'Apache-2.0 (short notice, not full text)',
                        'license_id': 'Apache-2.0',
                        'primary_path': 'LICENSE.md',
                        'proposed_note': 'License evidence is allow-listed, but this is Unity-authored material: status held '
                                         'under OD-07; submodules present.',
                        'proposed_status': 'REVIEW',
                        'sha256': 'sha256:21169a6cd5de3af010f89d5bf5e679c0cc33e5ab1163464b6507b3634e3ca6e4',
                        'uniformity': 'PARTIAL'},
 'mdn-content': {'commit': 'ac295ae8d3435f587b5233a9fe46bb55d92c1a5d',
                 'extra_risks': (),
                 'files': ('LICENSE.md=CC-BY-SA-2.5',),
                 'license_expression': 'CC-BY-SA-2.5 (primary); LICENSE.md also mentions CC0-1.0 and MIT',
                 'license_id': 'CC-BY-SA-2.5',
                 'primary_path': 'LICENSE.md',
                 'proposed_note': 'CC-BY-SA-2.5 is not allow-listed (OD-04); CC0 and MIT mentions concern OD-03.',
                 'proposed_status': 'REVIEW',
                 'sha256': 'sha256:97a45d9a5c27c90ad7f058b20a870f11a24717f0eb4585f141948d25b74c4d89',
                 'uniformity': 'PARTIAL'},
 'python-docs-cpython-doc': {'commit': 'ab01d184746b3ea383f69317c16fc726d885c968',
                             'extra_risks': (),
                             'files': ('LICENSE=PSF-2.0', 'Doc/license.rst=PSF-2.0'),
                             'license_expression': 'PSF-2.0 (LICENSE and Doc/license.rst); text also mentions other licenses',
                             'license_id': 'PSF-2.0',
                             'primary_path': 'LICENSE',
                             'proposed_note': 'PSF license is not allow-listed: OD-02. Doc/license.rst lists third-party '
                                              'notices.',
                             'proposed_status': 'REVIEW',
                             'sha256': 'sha256:b0e25a78cffb43f4d92de8b61ccfa1f1f98ecbc22330b54b5251e7b6ba010231',
                             'uniformity': 'PARTIAL'}}

_DATASET_STAGE1B: dict[str, dict[str, Any]] = {
    "cp-wikimedia-filtered": dict(
        license_expression="per-record string; sampled records carry CC BY-SA 4.0",
        evidence="Stage 1B web_fetch of the dataset card: no dataset-level license field; per-record metadata.license strings; "
                 "card warns about license laundering. The pin was exercised by a prior real run.",
        repository_license_verified=False, uniformity="PARTIAL", proposed="ACCEPT",
        note="Keep ACCEPT only for records whose license string maps exactly to CC-BY-SA-4.0 (existing behaviour); no blanket license is claimed.",
        risks=("card says March 2025 dumps while sample records show an added date of 2024-07-27: provenance statements are inconsistent",
               "eight wikis are listed in the card, but only Wikibooks was observed in the sampled shard prefix")),
    "hf-wikipedia-ne-20231101": dict(
        license_expression="GFDL AND/OR CC-BY-SA-3.0 as stated by the card (card metadata: cc-by-sa-3.0, gfdl)",
        evidence="Stage 1B web_fetch of the dataset card and tree page: card states GFDL and CC BY-SA 3.0, with caveats that some text may be "
                 "Creative-Commons-only and some authors may use additional licenses or public domain; 20231101.ne has about 32.9k rows.",
        repository_license_verified=True, uniformity="PARTIAL", proposed="ACCEPT",
        note="Supported by the card; unresolved: the card does not mention CC BY-SA 4.0, the Wikimedia legal page could not be fetched, "
             "and the /rows path has no revision parameter in this repository's code (main-branch commit seen: b04c8d1c...). OD-06 stays open.",
        risks=("immutable dataset commit exists but the acquisition path does not use it",
               "Wikimedia legal page not fetchable here: newer licensing statements UNKNOWN")),
    "hf-wikipedia-en-20231101": dict(
        license_expression="same card as Nepali Wikipedia (GFDL and CC-BY-SA-3.0 as stated)",
        evidence="Same dataset card as the Nepali record (all language configs share it).",
        repository_license_verified=True, uniformity="PARTIAL", proposed="REVIEW",
        note="Same OD-06 basis as Nepali Wikipedia; also very large.", risks=()),
}


def _apply_stage1b(records: list[SourceRecord]) -> list[SourceRecord]:
    out: list[SourceRecord] = []
    for x in records:
        fact = _STAGE1B_FACTS.get(x.source_id)
        ds = _DATASET_STAGE1B.get(x.source_id)
        if fact is not None:
            if x.status == ACCEPT and x.pin != fact["commit"]:
                raise SourceRegistryError(f"{x.source_id}: evidence commit {fact['commit']} differs from the registry pin {x.pin}")
            accepted = x.status == ACCEPT
            reason = x.status_reason if accepted else (
                x.status_reason + " Stage 1B: " + fact["proposed_note"] + " Status unchanged (no silent promotion).")
            x = dataclasses.replace(
                x, license_verification=REAL_FETCH,
                license_id=x.license_id if accepted else fact["license_id"],
                license_evidence=("Stage 1B real fetch: %s at commit %s, sha256 %s; files: %s; full record in 00_source_license_evidence.json"
                                  % (fact["primary_path"], fact["commit"][:12], fact["sha256"], ", ".join(fact["files"]))),
                pin=x.pin if accepted else fact["commit"], pin_kind=PIN_COMMIT if (accepted or x.pin is None) else x.pin_kind,
                status_reason=reason, risks=x.risks + tuple(fact["extra_risks"]),
                evidence_labels=x.evidence_labels + ("[VERIFIED BY REAL FETCH] license file(s) at the recorded commit",),
                license_expression=fact["license_expression"], license_text_sha256=fact["sha256"],
                license_files_detected=tuple(fact["files"]), evidence_commit=fact["commit"],
                repository_license_verified=fact["proposed_status"] != UNKNOWN,
                file_level_license_uniformity=fact["uniformity"],
                proposed_status=fact["proposed_status"], proposed_status_note=fact["proposed_note"])
        elif ds is not None:
            x = dataclasses.replace(
                x, license_verification=REAL_FETCH, license_evidence=ds["evidence"], license_expression=ds["license_expression"],
                risks=x.risks + tuple(ds["risks"]), repository_license_verified=ds["repository_license_verified"],
                file_level_license_uniformity=ds["uniformity"], proposed_status=ds["proposed"], proposed_status_note=ds["note"],
                evidence_labels=x.evidence_labels + ("[VERIFIED BY REAL FETCH] dataset card text (rendered page; no byte hash)",))
        out.append(x)
    return out


OWNER_DECISIONS: tuple[OwnerDecision, ...] = (
    OwnerDecision("OD-01", "BSD-family licenses for code", "Add BSD-2-Clause / BSD-3-Clause to the allow-list?",
                  ("keep excluded", "allow BSD-2/3-Clause", "allow per repository after review"),
                  "Gates flask and httpx, and most of the Python ecosystem."),
    OwnerDecision("OD-02", "PSF license", "Allow the PSF license for Python documentation?",
                  ("keep excluded", "allow for documentation only", "allow"),
                  "Gates the Python documentation."),
    OwnerDecision("OD-03", "CC0 / public-dedication and other permissive licenses (ISC, 0BSD, Unlicense, MPL-2.0)",
                  "Add any of these to the allow-list?", ("keep excluded", "allow specific ones"),
                  "Gates MDN code samples and any ISC/Unlicense/MPL repository."),
    OwnerDecision("OD-04", "Older Creative Commons versions", "Allow CC-BY-SA-2.5 (and similar older versions)?",
                  ("keep excluded", "allow"), "Gates MDN prose."),
    OwnerDecision("OD-05", "Copyleft code (GPL/LGPL/AGPL) and source-available licenses",
                  "Confirm these stay excluded?", ("keep excluded", "allow specific cases"),
                  "No candidate is listed; recorded so the default is an explicit decision."),
    OwnerDecision("OD-06", "Wikipedia license basis",
                  "Confirm CC-BY-SA-3.0 (with GFDL) as the applicable option for Wikipedia text, and whether newer text under CC BY-SA 4.0 is acceptable?",
                  ("confirm 3.0 basis", "treat as 3.0/4.0 mixture", "hold Wikipedia sources at REVIEW"),
                  "Governs Nepali Wikipedia (currently ACCEPT) and English Wikipedia."),
    OwnerDecision("OD-07", "Unity-authored code", "Keep Unity-authored repositories (ml-agents) excluded, or allow after review?",
                  ("keep excluded", "allow ml-agents under Apache-2.0"),
                  "Unity-authored material currently contributes nothing; Mirror also embeds Unity-copyrighted code under MIT."),
    OwnerDecision("OD-08", "Programming-language set for the code category",
                  "Which languages should the 24% code share cover?",
                  ("current four (Python, JavaScript, Go, Rust)", "add Java, C++", "add Dart/Flame", "other"),
                  "Determines which candidate repositories matter."),
    OwnerDecision("OD-09", "Nepali shortfall rule",
                  "If quality-gated Nepali falls short of its target, which rule applies?",
                  ("reduced explicit share approved by owner", "block Nepali-share claims"),
                  "Prevents padding Nepali with low-quality or duplicated text."),
    OwnerDecision("OD-10", "Dual-licensed projects",
                  "Confirm that choosing the allow-listed option of an OR dual license is acceptable?",
                  ("confirm", "require review per project"),
                  "Existing practice for ripgrep and rust-lang/book; also affects serde and Rust docs."),
    OwnerDecision("OD-11", "Mixed documentation/code licenses",
                  "Approve path-level license handling where a repository has separate docs and code licenses?",
                  ("approve", "exclude mixed repositories"), "Gates dotnet/docs and the Kubernetes documentation."),
    OwnerDecision("OD-12", "Concentration caps", "Approve the proposed caps in CONCENTRATION_PROPOSAL?",
                  ("approve", "change values"), "Caps decide how many distinct sources the corpus needs."),
)

CONCENTRATION_PROPOSAL: tuple[ConcentrationRule, ...] = (
    ConcentrationRule("CR-01", "code: per repository", "max_share_of_category_accepted_bytes", 0.10,
                      "Prior mini-corpus top single-source shares from manifest sizes: code 32.6% (ripgrep), C# 39.0% (Mirror), docs 47.2% (eng-practices). "
                      "0.10 implies at least 10 contributing repositories; the value is not tuned on any availability evidence.", "PROPOSAL"),
    ConcentrationRule("CR-02", "csharp_gamedev: per repository", "max_share_of_category_accepted_bytes", 0.10, "as CR-01", "PROPOSAL"),
    ConcentrationRule("CR-03", "software_docs: per source", "max_share_of_category_accepted_bytes", 0.15,
                      "implies at least 7 contributing sources", "PROPOSAL"),
    ConcentrationRule("CR-04", "general_english: per wiki/project", "max_share_of_category_accepted_bytes", 0.35,
                      "Wikibooks is the only wiki observed in the prior sample (100% of English); any single source is additionally capped at 0.50", "PROPOSAL"),
    ConcentrationRule("CR-05", "nepali: per article", "max_share_of_category_accepted_bytes", 0.005,
                      "Only one source exists, so a source cap is impossible; disclose single-source status and cap templated near-duplicate clusters", "PROPOSAL"),
    ConcentrationRule("CR-06", "code and csharp: test files", "max_share_of_bytes", 0.20, "no measurement exists", "ASSUMPTION"),
    ConcentrationRule("CR-07", "code and csharp: generated files", "max_share_of_bytes", 0.0,
                      "exclude; the existing gin file list already includes a .pb.go file", "PROPOSAL"),
    ConcentrationRule("CR-08", "all code: vendored or dependency copies", "excluded_path_patterns",
                      ("vendor/", "node_modules/", "third_party/", "ThirdParty/", "Plugins/", "*.min.js"),
                      "MonoGame ThirdParty/ is already excluded in the prior registry; the rest is convention", "PROPOSAL"),
    ConcentrationRule("CR-09", "minimum distinct sources", "per_category",
                      {CAT_CODE: 10, CAT_CSHARP: 10, CAT_DOCS: 7, CAT_ENGLISH: 3, CAT_NEPALI: 1}, "follows from CR-01..CR-05", "PROPOSAL"),
    ConcentrationRule("CR-10", "csharp_gamedev: Unity-ecosystem share", "reporting_only", "report the unity_ecosystem_third_party share separately",
                      "Unity-authored and Unity-official sources are BLOCKED or REVIEW, so the share will be small", "PROPOSAL"),
)


def build_registry() -> tuple[SourceRecord, ...]:
    """Deterministic registry, ordered by (category order, source_id)."""
    records = sorted(_apply_stage1b(_build_records()), key=lambda x: (CATEGORIES.index(x.category), x.source_id))
    validate_registry(records)
    return tuple(records)


def validate_registry(records: Iterable[SourceRecord], *, check_decision_coverage: bool = True) -> None:
    """Raise SourceRegistryError on the first violated invariant. ``check_decision_coverage``
    (default True, used for the full registry) also requires every owner decision to be
    referenced by some record; tests of single records turn it off."""
    recs = list(records)
    seen: set[str] = set()
    decision_ids = {d.decision_id for d in OWNER_DECISIONS}
    referenced: set[str] = set()
    for x in recs:
        where = f"record {x.source_id!r}"
        if not _ID_PATTERN.match(x.source_id):
            raise SourceRegistryError(f"{where}: invalid source_id")
        if x.source_id in seen:
            raise SourceRegistryError(f"{where}: duplicate source_id")
        seen.add(x.source_id)
        if x.category not in CATEGORIES:
            raise SourceRegistryError(f"{where}: unknown category {x.category!r}")
        if x.status not in STATUSES:
            raise SourceRegistryError(f"{where}: unknown status {x.status!r}")
        if x.license_verification not in LICENSE_VERIFICATIONS:
            raise SourceRegistryError(f"{where}: unknown license_verification {x.license_verification!r}")
        if x.pin_kind not in PIN_KINDS:
            raise SourceRegistryError(f"{where}: unknown pin_kind {x.pin_kind!r}")
        if not (0.0 <= x.max_category_share <= 1.0):
            raise SourceRegistryError(f"{where}: cap must be in [0, 1]")
        if not x.status_reason.strip():
            raise SourceRegistryError(f"{where}: status_reason is required")
        for d in x.owner_decision_ids:
            if d not in decision_ids:
                raise SourceRegistryError(f"{where}: unknown owner decision {d!r}")
            referenced.add(d)
        eligible = (x.tokenizer_fit_eligible, x.model_training_eligible, x.validation_eligible)
        if x.status != ACCEPT and any(eligible):
            raise SourceRegistryError(f"{where}: only ACCEPT sources may be eligible")
        if x.diagnostic_only and (x.status == ACCEPT or any(eligible)):
            raise SourceRegistryError(f"{where}: diagnostic-only sources can never be accepted or eligible")
        if x.status == BLOCKED and x.max_category_share != 0.0:
            raise SourceRegistryError(f"{where}: BLOCKED sources must have cap 0.0")
        if x.unity_authored and x.status == ACCEPT:
            raise SourceRegistryError(f"{where}: Unity-authored material can never be ACCEPT")
        if x.status == ACCEPT:
            if x.license_id not in ALLOWED_LICENSE_IDS:
                raise SourceRegistryError(f"{where}: license {x.license_id!r} is not in the allow-list")
            if x.license_verification == NOT_VERIFIED:
                raise SourceRegistryError(f"{where}: ACCEPT requires verified or prior-recorded license evidence")
            if not x.license_evidence.strip():
                raise SourceRegistryError(f"{where}: ACCEPT requires license evidence")
            if x.pin_kind == PIN_NONE or not x.pin:
                raise SourceRegistryError(f"{where}: ACCEPT requires a pin")
            if x.pin_kind in (PIN_COMMIT, PIN_DATASET_REVISION) and not _HEX40.match(x.pin):
                raise SourceRegistryError(f"{where}: pin must be a 40-hex revision")
            if x.max_category_share <= 0.0:
                raise SourceRegistryError(f"{where}: ACCEPT requires a positive cap")
        if x.file_level_license_uniformity not in UNIFORMITY_VALUES:
            raise SourceRegistryError(f"{where}: bad file_level_license_uniformity")
        if x.proposed_status is not None and x.proposed_status not in STATUSES:
            raise SourceRegistryError(f"{where}: bad proposed_status")
        if x.repository_license_verified and x.evidence_commit:
            if not _HEX40.match(x.evidence_commit) or not (x.license_text_sha256 or "").startswith("sha256:"):
                raise SourceRegistryError(f"{where}: verified evidence needs a 40-hex commit and a license sha256")
        if x.proposed_status == ACCEPT and x.status != ACCEPT:
            if (x.license_id not in ALLOWED_LICENSE_IDS or x.license_verification != REAL_FETCH or not x.pin
                    or x.unity_authored or x.diagnostic_only):
                raise SourceRegistryError(f"{where}: a proposed promotion needs an allow-listed license, real-fetch evidence and a pin")
        if x.prior_status is not None and x.prior_status != x.status and not (x.status_change_note or "").strip():
            raise SourceRegistryError(f"{where}: status differs from prior status {x.prior_status!r} without a status_change_note")
    if not check_decision_coverage:
        return
    unused = decision_ids - referenced
    unreferenced_ok = {"OD-05", "OD-12"}  # policy-level decisions with no single record
    if unused - unreferenced_ok:
        raise SourceRegistryError(f"owner decisions not referenced by any record: {sorted(unused - unreferenced_ok)}")


def eligible_sources(records: Iterable[SourceRecord], purpose: str) -> tuple[SourceRecord, ...]:
    field = {"fit": "tokenizer_fit_eligible", "train": "model_training_eligible", "validation": "validation_eligible"}.get(purpose)
    if field is None:
        raise SourceRegistryError(f"unknown purpose {purpose!r}")
    return tuple(x for x in records if getattr(x, field))


def coverage_ceiling(records: Iterable[SourceRecord], include: Sequence[str] = (ACCEPT,)) -> dict[str, float]:
    """PLANNING ONLY. Per category, the sum of caps of records whose status is in
    ``include``, capped at 1.0. It is the most a category could reach if every counted
    source supplied its full cap; it says nothing about how much data exists."""
    out = {c: 0.0 for c in CATEGORIES}
    for x in records:
        if x.status in include:
            out[x.category] += x.max_category_share
    return {c: min(1.0, v) for c, v in out.items()}


def proposed_coverage_ceiling(records: Iterable[SourceRecord]) -> dict[str, float]:
    """PLANNING ONLY. Like :func:`coverage_ceiling`, counting records that are ACCEPT now or whose
    Stage-1B ``proposed_status`` is ACCEPT. Not a decision: promotion needs the owner."""
    out = {c: 0.0 for c in CATEGORIES}
    for x in records:
        if x.status == ACCEPT or x.proposed_status == ACCEPT:
            out[x.category] += x.max_category_share
    return {c: min(1.0, v) for c, v in out.items()}


def registry_to_dict(records: Sequence[SourceRecord] | None = None) -> dict[str, Any]:
    recs = tuple(records) if records is not None else build_registry()
    body = {
        "registry_version": REGISTRY_VERSION,
        "not_a_milestone": True,
        "nb13_assigned": False,
        "allowed_license_ids": sorted(ALLOWED_LICENSE_IDS),
        "stage1b_evidence_fingerprint": STAGE1B_EVIDENCE_FINGERPRINT,
        "records": [asdict(x) for x in recs],
        "owner_decisions": [asdict(d) for d in OWNER_DECISIONS],
        "concentration_proposal": [asdict(c) for c in CONCENTRATION_PROPOSAL],
    }
    body["fingerprint"] = _fingerprint(body)
    return body


def _fingerprint(body: dict[str, Any]) -> str:
    payload = {k: v for k, v in body.items() if k != "fingerprint"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=list)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def registry_fingerprint(records: Sequence[SourceRecord] | None = None) -> str:
    return registry_to_dict(records)["fingerprint"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the Stage-1 source registry as JSON (no network, no acquisition).")
    parser.add_argument("--output", default="rep_corpus_source_registry.json")
    args = parser.parse_args(argv)
    try:
        data = registry_to_dict()
    except SourceRegistryError as exc:
        print(f"REGISTRY INVALID: {exc}", file=sys.stderr)
        return 2
    Path(args.output).write_text(json.dumps(data, indent=2, sort_keys=True, default=list), encoding="utf-8")
    print(f"registry written to {args.output} ({len(data['records'])} records, {data['fingerprint']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())