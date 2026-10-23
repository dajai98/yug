"""Representative-corpus expansion, STAGE 1C: conservative source-policy freeze.

NOT a milestone (no NB-13). Policy data + deterministic helpers only. It does NOT acquire,
fetch, filter, split, fit a tokenizer, or train anything. Stdlib-only; imports only the
Stage 1A/1B registry (``rep_corpus_sources``).

LAYERING
  The registry (Stage 1A/1B) stays untouched: it carries fetched evidence, the allow-list and
  the registry status. This module is the Stage 1C overlay that an acquisition stage must read:
  ``stage1c_status`` is the operative status, and ONLY ``ACCEPT`` entries are selectable.
  No status defaults to ACCEPT: an entry is ACCEPT only if the registry already says ACCEPT
  or the source is in ``STAGE1C_PROMOTIONS`` and passes every promotion check.

KINDS OF STATEMENT (kept apart in the artifact)
  facts           copied from the Stage 1A/1B registry and the fetched evidence
  owner_decisions values the owner stated (category targets, OD-08 language mix, Unity blocks)
  proposals       auditor values awaiting owner approval (caps, path exclusions, promotions)
  unresolved      held sources, capacity shortfalls and open owner decisions

NOTHING HERE AUTHORIZES acquisition, tokenizer fitting, or model training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from training.experiments import rep_corpus_sources as rs

POLICY_SCHEMA = "rep-corpus-source-policy-v1"
EVIDENCE_ARTIFACT = "training/resources/phase1/rep_corpus_10m_v1/00_source_license_evidence.json"
POLICY_ARTIFACT = "training/resources/phase1/rep_corpus_10m_v1/01_source_policy.json"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class PolicyError(ValueError):
    """The Stage 1C policy (or an input to it) violates an invariant."""


# ---------------------------------------------------------------------------
# OWNER DECISIONS (stated by the owner; frozen here, never changed by the auditor)
# ---------------------------------------------------------------------------

# Working BYTE-share targets for the expansion experiment (not final YUG-100M proportions).
CATEGORY_TARGET_PCT: dict[str, int] = {
    rs.CAT_CODE: 24, rs.CAT_CSHARP: 22, rs.CAT_ENGLISH: 19, rs.CAT_DOCS: 18, rs.CAT_NEPALI: 17,
}

# OD-08, FROZEN by the owner: byte-share targets WITHIN the general-code (24%) category.
OD08_LANGUAGE_TARGET_PCT: dict[str, int] = {
    "python": 35, "rust": 25, "c_cpp": 15, "javascript_typescript": 15, "go": 10,
}

# Registry subcategory -> OD-08 language key. Anything not listed is NOT allowed in the code mix.
_SUBCATEGORY_TO_OD08: dict[str, str] = {
    "python": "python", "rust": "rust", "c": "c_cpp", "cpp": "c_cpp", "c_cpp": "c_cpp",
    "javascript": "javascript_typescript", "typescript": "javascript_typescript",
    "javascript_typescript": "javascript_typescript", "go": "go",
}

# Owner-supplied PLANNING estimate (about 10M 4096-equivalent train tokens at ~4.12 bytes/token).
# It is not a fixed corpus requirement.
PLANNING_TOTAL_TRAIN_BYTES = 41_000_000

DESIGN_PRINCIPLES: tuple[dict[str, Any], ...] = (
    {
        "id": "DP-01", "title": "Controlled YUG self-improvement", "status": "RECORDED_NOT_IMPLEMENTED",
        "authority": "OWNER",
        "flow": ["detect problem", "analyze repository", "propose change",
                 "apply change only inside an isolated sandbox/worktree", "run tests and checks",
                 "present diff + evidence to owner", "require explicit owner approval",
                 "apply approved change", "verify", "maintain rollback path"],
        "constraint": "YUG must not receive unrestricted authority to rewrite or upgrade itself.",
        "reference": "ARCHITECTURE/YUG_SELF_IMPROVEMENT_PRINCIPLE.md",
    },
)

# ---------------------------------------------------------------------------
# PROPOSALS (auditor values; not frozen until the owner approves them)
# ---------------------------------------------------------------------------

# Stage 1C promotions: REVIEW -> ACCEPT. Each must pass promotion_problems(); keep this list minimal.
STAGE1C_PROMOTIONS: dict[str, str] = {
    "gh-lodash-lodash": "JavaScript (OD-08); single MIT license; no submodules or notice files; generated builds excluded by path.",
    "gh-spf13-cobra": "Go (OD-08); single Apache-2.0 license; clean evidence.",
    "gh-tokio-rs-tokio": "Rust (OD-08); MIT stated in README and license file; clean evidence.",
    "gh-serde-rs-serde": "Rust (OD-08); README states either Apache-2.0 or MIT at the user's option; both allow-listed.",
    "gh-rust-lang-rust-by-example": "Docs; README states either Apache-2.0 or MIT at the user's option; both allow-listed.",
    "gh-dotnet-roslyn": "C# category (general C#, not Unity data); single MIT license; generated and test paths excluded and capped.",
    "gh-neuecc-unirx": "C# category (Unity-ecosystem third-party); single MIT license; archived and small.",
}

# Sources Stage 1B proposed for ACCEPT that Stage 1C holds back: id -> (reason, owner decision ids).
STAGE1C_HELD_PROPOSALS: dict[str, tuple[str, tuple[str, ...]]] = {
    "gh-google-guava": ("Java is not in the frozen OD-08 language mix; unknown languages cannot enter the code category.", ("OD-13",)),
    "gh-flame-engine-flame": ("Dart is not in the frozen OD-08 language mix; unknown languages cannot enter the code category.", ("OD-13",)),
    "gh-rust-lang-reference": ("Two license files (MIT, Apache-2.0) with no statement of their relationship in the fetched README.", ("OD-14",)),
    "gh-rust-lang-rustc-dev-guide": ("Two license files (MIT, Apache-2.0) with no statement of their relationship in the fetched README.", ("OD-14",)),
}

# Per-repository byte caps in basis points of the CATEGORY TARGET bytes (PROPOSAL). They bound absolute
# contribution; they are not a guarantee about realized shares when a category is under-filled.
REPO_CAP_BP: dict[str, int] = {rs.CAT_CODE: 1000, rs.CAT_CSHARP: 1000, rs.CAT_DOCS: 1500}
CAP_AUDIT_BP: tuple[int, ...] = (500, 1000, 1250, 1500, 2000)

# Global path exclusions (PROPOSAL; the file trees were NOT inspected in Stage 1B, so Stage 2 must
# validate them). Deliberately NOT included: a blanket "Plugins/" rule -- UniTask's and UniRx's own
# source lives under Assets/Plugins/<Project>/ (VERIFIED IN REPOSITORY: manifest paths).
GLOBAL_EXCLUDED_PATTERNS: tuple[str, ...] = (
    "**/vendor/**", "**/node_modules/**", "**/third_party/**", "**/ThirdParty/**", "**/target/**",
    "**/*.min.js", "**/*.min.css", "**/package-lock.json", "**/yarn.lock", "**/Cargo.lock", "**/go.sum",
    "**/*.pb.go", "**/*_pb2.py", "**/*_pb2_grpc.py", "**/*.g.cs", "**/*.g.i.cs", "**/*.Designer.cs",
    "**/*.generated.*", "**/*.Generated.*",
)
SOURCE_EXCLUDED_PATTERNS: dict[str, tuple[str, ...]] = {
    "gh-lodash-lodash": ("dist/**", "lodash.js", "lodash.min.js"),
    "gh-dotnet-roslyn": ("**/Generated/**", "**/Test/**", "**/Tests/**", "**/*.Tests/**", "**/TestResources/**"),
    "gh-monogame-monogame": ("**/ThirdParty/**",),
}
GENERATED_FILE_MARKERS: tuple[str, ...] = (
    "Code generated", "DO NOT EDIT", "<auto-generated", "@generated", "This file was generated",
)

# Stage 1B file-level flags copied from the evidence artifact (a test compares them to the artifact).
_STAGE1B_FILE_FLAGS: dict[str, dict[str, Any]] = {'docker-docs': {'gitmodules_present': False, 'license_file_count': 1, 'notice_files': ()},
 'gh-burntsushi-ripgrep': {'gitmodules_present': False, 'license_file_count': 3, 'notice_files': ()},
 'gh-cysharp-unitask': {'gitmodules_present': False, 'license_file_count': 1, 'notice_files': ()},
 'gh-dotnet-docs': {'gitmodules_present': False, 'license_file_count': 2, 'notice_files': ()},
 'gh-dotnet-roslyn': {'gitmodules_present': False, 'license_file_count': 1, 'notice_files': ()},
 'gh-dotnet-runtime': {'gitmodules_present': False,
                       'license_file_count': 1,
                       'notice_files': ('THIRD-PARTY-NOTICES.TXT',)},
 'gh-encode-httpx': {'gitmodules_present': False, 'license_file_count': 1, 'notice_files': ()},
 'gh-expressjs-express': {'gitmodules_present': False, 'license_file_count': 1, 'notice_files': ()},
 'gh-flame-engine-flame': {'gitmodules_present': False, 'license_file_count': 1, 'notice_files': ()},
 'gh-fna-xna-fna': {'gitmodules_present': True, 'license_file_count': 3, 'notice_files': ()},
 'gh-gin-gonic-gin': {'gitmodules_present': False, 'license_file_count': 1, 'notice_files': ()},
 'gh-google-eng-practices': {'gitmodules_present': False, 'license_file_count': 1, 'notice_files': ()},
 'gh-google-guava': {'gitmodules_present': False, 'license_file_count': 1, 'notice_files': ()},
 'gh-kubernetes-website': {'gitmodules_present': True, 'license_file_count': 1, 'notice_files': ()},
 'gh-lodash-lodash': {'gitmodules_present': False, 'license_file_count': 1, 'notice_files': ()},
 'gh-messagepack-csharp': {'gitmodules_present': False, 'license_file_count': 1, 'notice_files': ()},
 'gh-microsoft-api-guidelines': {'gitmodules_present': False, 'license_file_count': 1, 'notice_files': ()},
 'gh-mirrornetworking-mirror': {'gitmodules_present': False, 'license_file_count': 1, 'notice_files': ()},
 'gh-monogame-monogame': {'gitmodules_present': True, 'license_file_count': 1, 'notice_files': ()},
 'gh-neuecc-unirx': {'gitmodules_present': False, 'license_file_count': 1, 'notice_files': ()},
 'gh-nlohmann-json': {'gitmodules_present': False, 'license_file_count': 1, 'notice_files': ()},
 'gh-pallets-flask': {'gitmodules_present': False, 'license_file_count': 1, 'notice_files': ()},
 'gh-psf-requests': {'gitmodules_present': False, 'license_file_count': 1, 'notice_files': ('NOTICE',)},
 'gh-rust-lang-book': {'gitmodules_present': False, 'license_file_count': 2, 'notice_files': ()},
 'gh-rust-lang-reference': {'gitmodules_present': False, 'license_file_count': 2, 'notice_files': ()},
 'gh-rust-lang-rust-by-example': {'gitmodules_present': False, 'license_file_count': 2, 'notice_files': ()},
 'gh-rust-lang-rustc-dev-guide': {'gitmodules_present': False, 'license_file_count': 2, 'notice_files': ()},
 'gh-serde-rs-serde': {'gitmodules_present': False, 'license_file_count': 2, 'notice_files': ()},
 'gh-spf13-cobra': {'gitmodules_present': False, 'license_file_count': 1, 'notice_files': ()},
 'gh-stride3d-stride': {'gitmodules_present': False, 'license_file_count': 1, 'notice_files': ()},
 'gh-tokio-rs-tokio': {'gitmodules_present': False, 'license_file_count': 1, 'notice_files': ()},
 'gh-unity-ml-agents': {'gitmodules_present': True, 'license_file_count': 1, 'notice_files': ()},
 'mdn-content': {'gitmodules_present': False, 'license_file_count': 1, 'notice_files': ()},
 'python-docs-cpython-doc': {'gitmodules_present': False, 'license_file_count': 2, 'notice_files': ()}}


# ---------------------------------------------------------------------------
# Owner decisions still open after Stage 1C (new ids; OD-01..OD-12 live in the registry)
# ---------------------------------------------------------------------------

STAGE1C_OPEN_DECISIONS: tuple[dict[str, Any], ...] = (
    {"id": "OD-13", "title": "Amend OD-08 language set?",
     "question": "Add languages (for example Java, Dart) to the frozen OD-08 mix, or keep them out?",
     "affects": ["gh-google-guava", "gh-flame-engine-flame"]},
    {"id": "OD-14", "title": "Sources with two license files and no stated relationship",
     "question": "Both licenses are allow-listed, so eligibility is the same however they combine. Promote rust-lang/reference and "
                 "rustc-dev-guide, and keep rust-lang/book (ACCEPT since before Stage 1B), or require an explicit relationship statement "
                 "(which would also affect rust-lang/book)?",
     "affects": ["gh-rust-lang-reference", "gh-rust-lang-rustc-dev-guide", "gh-rust-lang-book"]},
    {"id": "OD-15", "title": "Under-fill policy",
     "question": "When approved sources cannot fill a category or language target: report the shortfall (default here), backfill from other "
                 "languages or categories, or shrink the corpus?",
     "affects": ["all categories"]},
)


# ---------------------------------------------------------------------------
# OD-08 helpers
# ---------------------------------------------------------------------------

def validate_language_targets(targets: dict[str, int] | None = None) -> None:
    t = OD08_LANGUAGE_TARGET_PCT if targets is None else targets
    if sum(t.values()) != 100 or any((not isinstance(v, int)) or v <= 0 for v in t.values()):
        raise PolicyError(f"language targets must be positive integers summing to exactly 100, got {t}")
    if "csharp" in t or "c_sharp" in t:
        raise PolicyError("C# belongs to its own category, not to the general-code language mix")


def od08_language_for(category: str, subcategory: str | None) -> str:
    """OD-08 language key for a general-code source. Raises for any other category or any
    language that is not part of the frozen mix (no silent entry)."""
    if category == rs.CAT_CSHARP:
        raise PolicyError("C# does not belong to the 24% general-code language mix")
    if category != rs.CAT_CODE:
        raise PolicyError(f"category {category!r} has no OD-08 language")
    key = _SUBCATEGORY_TO_OD08.get((subcategory or "").lower())
    if key is None:
        raise PolicyError(f"language {subcategory!r} is not in the frozen OD-08 mix")
    return key


# ---------------------------------------------------------------------------
# Path exclusion + deterministic cap
# ---------------------------------------------------------------------------

def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def is_excluded(path: str, source_id: str | None = None) -> bool:
    p = path.replace("\\", "/").lstrip("./")
    patterns = GLOBAL_EXCLUDED_PATTERNS + (SOURCE_EXCLUDED_PATTERNS.get(source_id or "", ()))
    return any(_glob_to_regex(pat).match(p) for pat in patterns)


def category_target_bytes(category: str, total_bytes: int = PLANNING_TOTAL_TRAIN_BYTES) -> int:
    if category not in CATEGORY_TARGET_PCT:
        raise PolicyError(f"no byte target for category {category!r}")
    return total_bytes * CATEGORY_TARGET_PCT[category] // 100


def repo_cap_bytes(cap_bp: int, category: str, total_bytes: int = PLANNING_TOTAL_TRAIN_BYTES) -> int:
    if not 0 < cap_bp <= 10_000:
        raise PolicyError("cap must be in (0, 10000] basis points")
    return category_target_bytes(category, total_bytes) * cap_bp // 10_000


def select_within_cap(files: Iterable[tuple[str, int]], cap_bytes: int, seed: str,
                      source_id: str | None = None) -> list[tuple[str, int]]:
    """Deterministic selection of (path, size) files whose total never exceeds ``cap_bytes``.

    Excluded paths are dropped; the rest are ordered by SHA-256(seed, path) and taken greedily, skipping any
    file that would exceed the cap (so a single huge file can never bypass it). The result is independent of
    the input order and returned sorted by path."""
    if cap_bytes < 0:
        raise PolicyError("cap_bytes must be >= 0")
    unique: dict[str, int] = {}
    for path, size in files:
        if size < 0:
            raise PolicyError(f"negative size for {path!r}")
        if not is_excluded(path, source_id):
            unique[path] = size
    ordered = sorted(unique.items(), key=lambda kv: (hashlib.sha256(f"{seed}\0{kv[0]}".encode("utf-8")).hexdigest(), kv[0]))
    chosen: list[tuple[str, int]] = []
    total = 0
    for path, size in ordered:
        if total + size <= cap_bytes:
            chosen.append((path, size))
            total += size
    return sorted(chosen)


def realized_shares(bytes_by_source: dict[str, int]) -> dict[str, float]:
    """Each source's share of the REALIZED category bytes (a reporting metric, not a cap)."""
    total = sum(bytes_by_source.values())
    return {k: (v / total if total else 0.0) for k, v in sorted(bytes_by_source.items())}


# ---------------------------------------------------------------------------
# Policy entries
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyEntry:
    source_id: str
    category: str
    subcategory: str | None
    od08_language: str | None
    project: str
    commit: str | None
    pin_kind: str
    registry_status: str
    stage1c_status: str          # no default on purpose: status must always be explicit
    status_authority: str        # REGISTRY_CARRIED_OVER | STAGE1C_PROMOTION | NOT_PROMOTED
    license_used_for_eligibility: str | None
    license_expression: str | None
    license_text_sha256: str | None
    evidence_ref: str | None
    repo_cap_bp: int
    excluded_path_patterns: tuple[str, ...]
    submodule_handling: str
    generated_file_handling: str
    owner_decision_ids: tuple[str, ...]
    reason: str
    tokenizer_fit_eligible: bool
    model_training_eligible: bool
    validation_eligible: bool


def promotion_problems(x: rs.SourceRecord) -> list[str]:
    """Every reason a registry record may NOT be promoted (empty list = promotable)."""
    problems: list[str] = []
    if x.status != rs.REVIEW:
        problems.append(f"registry status is {x.status}, not REVIEW")
    if x.proposed_status != rs.ACCEPT:
        problems.append("Stage 1B did not propose ACCEPT")
    if x.license_id not in rs.ALLOWED_LICENSE_IDS:
        problems.append(f"license {x.license_id!r} is not allow-listed")
    if x.license_verification != rs.REAL_FETCH or not (x.license_text_sha256 or "").startswith("sha256:"):
        problems.append("no real-fetch license evidence")
    if not (x.pin and _HEX40.match(x.pin) and x.pin == x.evidence_commit and x.pin_kind == rs.PIN_COMMIT):
        problems.append("no frozen 40-hex commit matching the evidence")
    if x.unity_authored:
        problems.append("Unity-authored material")
    if x.owner_decision_ids and set(x.owner_decision_ids) - {"OD-10"}:
        problems.append(f"open owner decisions: {sorted(set(x.owner_decision_ids) - {'OD-10'})}")
    flags = _STAGE1B_FILE_FLAGS.get(x.source_id)
    if flags is None:
        problems.append("no Stage 1B file-level flags")
    elif flags["gitmodules_present"] or flags["notice_files"]:
        problems.append("submodules or third-party notice files present")
    if x.category == rs.CAT_CODE:
        try:
            od08_language_for(x.category, x.subcategory)
        except PolicyError as exc:
            problems.append(str(exc))
    if x.category not in CATEGORY_TARGET_PCT:
        problems.append("category has no byte target")
    return problems


def _entry(x: rs.SourceRecord, status: str, authority: str, reason: str) -> PolicyEntry:
    accepted = status == rs.ACCEPT
    language = None
    if x.category == rs.CAT_CODE:
        language = _SUBCATEGORY_TO_OD08.get((x.subcategory or "").lower())
    flags = _STAGE1B_FILE_FLAGS.get(x.source_id)
    if flags is None:
        sub = "not a GitHub-evidenced source; handled at acquisition design"
    elif flags["gitmodules_present"]:
        sub = "Stage 2 must read .gitmodules at the frozen commit and exclude every listed path (a .gitmodules file exists)"
    else:
        sub = "no .gitmodules at the frozen commit (Stage 1B evidence); Stage 2 must still verify at acquisition"
    cap_bp = 0
    if accepted:
        cap_bp = REPO_CAP_BP.get(x.category, round(x.max_category_share * 10_000))
    return PolicyEntry(
        source_id=x.source_id, category=x.category, subcategory=x.subcategory, od08_language=language, project=x.project,
        commit=x.evidence_commit or (x.pin if x.pin_kind in (rs.PIN_COMMIT, rs.PIN_DATASET_REVISION) else None),
        pin_kind=x.pin_kind, registry_status=x.status, stage1c_status=status, status_authority=authority,
        license_used_for_eligibility=x.license_id if accepted else None, license_expression=x.license_expression,
        license_text_sha256=x.license_text_sha256,
        evidence_ref=(f"{EVIDENCE_ARTIFACT}#{x.source_id}" if x.evidence_commit else None),
        repo_cap_bp=cap_bp,
        excluded_path_patterns=(GLOBAL_EXCLUDED_PATTERNS + SOURCE_EXCLUDED_PATTERNS.get(x.source_id, ())) if accepted else (),
        submodule_handling=sub,
        generated_file_handling=("exclude by path pattern now; Stage 3 must also drop files whose header contains: "
                                 + "; ".join(GENERATED_FILE_MARKERS)),
        owner_decision_ids=x.owner_decision_ids, reason=reason,
        tokenizer_fit_eligible=accepted and not x.diagnostic_only, model_training_eligible=accepted and not x.diagnostic_only,
        validation_eligible=accepted and not x.diagnostic_only)


def build_entries(records: Sequence[rs.SourceRecord] | None = None,
                  promotions: dict[str, str] | None = None) -> tuple[PolicyEntry, ...]:
    recs = tuple(records) if records is not None else rs.build_registry()
    promos = STAGE1C_PROMOTIONS if promotions is None else promotions
    by_id = {x.source_id: x for x in recs}
    for sid in promos:
        if sid not in by_id:
            raise PolicyError(f"promotion for unknown source {sid!r}")
        problems = promotion_problems(by_id[sid])
        if problems:
            raise PolicyError(f"{sid} cannot be promoted: {problems}")
    out: list[PolicyEntry] = []
    for x in recs:
        if x.status == rs.ACCEPT:
            out.append(_entry(x, rs.ACCEPT, "REGISTRY_CARRIED_OVER", x.status_reason))
        elif x.source_id in promos:
            out.append(_entry(x, rs.ACCEPT, "STAGE1C_PROMOTION", promos[x.source_id]))
        else:
            held = STAGE1C_HELD_PROPOSALS.get(x.source_id)
            reason = (f"Held at {x.status}: {held[0]}" if held else x.status_reason)
            entry = _entry(x, x.status, "NOT_PROMOTED", reason)
            if held:
                entry = PolicyEntry(**{**asdict(entry), "owner_decision_ids": tuple(sorted(set(entry.owner_decision_ids) | set(held[1]))),
                                       "excluded_path_patterns": (), "submodule_handling": entry.submodule_handling,
                                       "generated_file_handling": entry.generated_file_handling})
            out.append(entry)
    entries = tuple(out)
    validate_policy(entries)
    return entries


def validate_policy(entries: Iterable[PolicyEntry]) -> None:
    validate_language_targets()
    if sum(CATEGORY_TARGET_PCT.values()) != 100:
        raise PolicyError("category byte targets must sum to 100")
    seen: set[str] = set()
    for e in entries:
        where = f"entry {e.source_id!r}"
        if e.source_id in seen:
            raise PolicyError(f"{where}: duplicate")
        seen.add(e.source_id)
        if e.stage1c_status not in rs.STATUSES:
            raise PolicyError(f"{where}: bad status {e.stage1c_status!r}")
        eligible = (e.tokenizer_fit_eligible, e.model_training_eligible, e.validation_eligible)
        if e.stage1c_status != rs.ACCEPT:
            if any(eligible) or e.repo_cap_bp != 0 or e.license_used_for_eligibility is not None:
                raise PolicyError(f"{where}: only ACCEPT entries may be eligible, capped or carry an eligibility license")
            continue
        if e.status_authority not in ("REGISTRY_CARRIED_OVER", "STAGE1C_PROMOTION"):
            raise PolicyError(f"{where}: ACCEPT without a recognised authority")
        if e.license_used_for_eligibility not in rs.ALLOWED_LICENSE_IDS:
            raise PolicyError(f"{where}: ACCEPT needs an allow-listed license")
        if e.pin_kind in (rs.PIN_COMMIT, rs.PIN_DATASET_REVISION) and not (e.commit and _HEX40.match(e.commit)):
            raise PolicyError(f"{where}: ACCEPT needs a concrete 40-hex commit")
        if e.pin_kind == rs.PIN_NONE:
            raise PolicyError(f"{where}: ACCEPT needs a pin")
        if e.category not in CATEGORY_TARGET_PCT:
            raise PolicyError(f"{where}: category {e.category!r} has no byte target")
        if e.category == rs.CAT_CODE and (e.od08_language not in OD08_LANGUAGE_TARGET_PCT):
            raise PolicyError(f"{where}: code source outside the frozen OD-08 mix")
        if e.category != rs.CAT_CODE and e.od08_language is not None:
            raise PolicyError(f"{where}: only general-code sources carry an OD-08 language")
        if e.repo_cap_bp <= 0:
            raise PolicyError(f"{where}: ACCEPT needs a positive cap")
        if e.status_authority == "STAGE1C_PROMOTION" and not (e.license_text_sha256 and e.evidence_ref):
            raise PolicyError(f"{where}: promoted source lacks Stage 1B evidence")
        if not (e.tokenizer_fit_eligible and e.model_training_eligible and e.validation_eligible):
            raise PolicyError(f"{where}: ACCEPT entries are eligible for fit, training and validation")


def accepted_source_ids(entries: Iterable[PolicyEntry], category: str | None = None) -> tuple[str, ...]:
    """The ACCEPT-only selector: the only way an acquisition stage may choose sources."""
    return tuple(e.source_id for e in entries
                 if e.stage1c_status == rs.ACCEPT and (category is None or e.category == category))


# ---------------------------------------------------------------------------
# Cap audit (CALCULATED)
# ---------------------------------------------------------------------------

def category_ceilings(entries: Sequence[PolicyEntry], cap_bp: int, category: str) -> dict[str, Any]:
    n = len(accepted_source_ids(entries, category))
    return {"accepted_sources": n, "ceiling_pct": min(100.0, n * cap_bp / 100.0)}


def language_ceilings(entries: Sequence[PolicyEntry], cap_bp: int,
                      total_bytes: int = PLANNING_TOTAL_TRAIN_BYTES) -> dict[str, Any]:
    code_bytes = category_target_bytes(rs.CAT_CODE, total_bytes)
    per_repo = code_bytes * cap_bp // 10_000
    out: dict[str, Any] = {}
    for lang, pct in sorted(OD08_LANGUAGE_TARGET_PCT.items()):
        n = sum(1 for e in entries if e.stage1c_status == rs.ACCEPT and e.od08_language == lang)
        target = code_bytes * pct // 100
        ceiling = min(target, n * per_repo)
        out[lang] = {"target_pct_of_code": pct, "accepted_repos": n, "target_bytes": target,
                     "ceiling_bytes": ceiling, "ceiling_pct_of_language_target": round(100 * ceiling / target, 1)}
    total_ceiling = sum(v["ceiling_bytes"] for v in out.values())
    out["_total"] = {"ceiling_bytes": total_ceiling, "ceiling_pct_of_code_category": round(100 * total_ceiling / code_bytes, 1)}
    return out


def cap_audit(entries: Sequence[PolicyEntry]) -> dict[str, Any]:
    audit: dict[str, Any] = {"planning_total_train_bytes": PLANNING_TOTAL_TRAIN_BYTES, "label": "CALCULATED",
                             "note": "Ceilings are the most a category could reach if every counted source supplied its full cap; "
                                     "availability of bytes is UNKNOWN.", "by_cap_bp": {}}
    for cap_bp in CAP_AUDIT_BP:
        audit["by_cap_bp"][str(cap_bp)] = {
            "per_repo_bytes": {c: repo_cap_bytes(cap_bp, c) for c in (rs.CAT_CODE, rs.CAT_CSHARP, rs.CAT_DOCS)},
            "category_ceilings": {c: category_ceilings(entries, cap_bp, c) for c in (rs.CAT_CODE, rs.CAT_CSHARP, rs.CAT_DOCS)},
            "code_language_ceilings": language_ceilings(entries, cap_bp),
        }
    audit["min_cap_bp_to_reach_category_target_if_all_sources_supply_fully"] = {
        c: -(-10_000 // max(1, len(accepted_source_ids(entries, c)))) for c in (rs.CAT_CODE, rs.CAT_CSHARP, rs.CAT_DOCS)}
    return audit


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------

def build_policy(entries: Sequence[PolicyEntry] | None = None) -> dict[str, Any]:
    recs = rs.build_registry()
    ents = tuple(entries) if entries is not None else build_entries(recs)
    by_id = {x.source_id: x for x in recs}
    stage1b_proposed = sorted(x.source_id for x in recs if x.proposed_status == rs.ACCEPT and x.status != rs.ACCEPT)
    source_rows = []
    for e in ents:
        d = asdict(e)
        row = {
            "source_id": e.source_id, "category": e.category, "subcategory": e.subcategory, "od08_language": e.od08_language,
            "stage1c_status": e.stage1c_status, "registry_status": e.registry_status, "status_authority": e.status_authority,
            "owner_decision_ids": list(e.owner_decision_ids),
            "reason": e.reason,
            "eligibility": {"tokenizer_fit": e.tokenizer_fit_eligible, "model_training": e.model_training_eligible,
                            "validation": e.validation_eligible},
            "facts": {"project": e.project, "commit": e.commit, "pin_kind": e.pin_kind,
                      "license_used_for_eligibility": e.license_used_for_eligibility, "license_expression": e.license_expression,
                      "license_text_sha256": e.license_text_sha256, "evidence_ref": e.evidence_ref,
                      "file_level_license_uniformity": by_id[e.source_id].file_level_license_uniformity},
            "proposal": {"repo_cap_bp_of_category_target": e.repo_cap_bp, "excluded_path_patterns": list(e.excluded_path_patterns),
                         "submodule_handling": e.submodule_handling, "generated_file_handling": e.generated_file_handling},
        }
        assert row["facts"]["commit"] == d["commit"]
        source_rows.append(row)
    body: dict[str, Any] = {
        "schema": POLICY_SCHEMA, "not_a_milestone": True, "nb13_assigned": False,
        "acquisition_authorized": False, "tokenizer_training_authorized": False, "model_training_authorized": False,
        "status_authority_note": "stage1c_status is the operative status for acquisition. Promotions are auditor decisions that take "
                                 "effect only when the owner accepts these files; the registry (Stage 1A/1B) is unchanged.",
        "owner_decisions": {
            "category_byte_target_pct": dict(CATEGORY_TARGET_PCT),
            "category_target_meaning": "working BYTE-share targets for the expansion experiment, not final YUG-100M proportions",
            "od08_general_code_language_target_pct": dict(OD08_LANGUAGE_TARGET_PCT),
            "od08_meaning": "byte-share targets within the 24% general-code category; C# is excluded (own category); languages not listed cannot enter",
            "license_allowlist": sorted(rs.ALLOWED_LICENSE_IDS), "license_allowlist_changed_by_stage1c": False,
            "unity_blocked_source_ids": ["gh-unity-csreference", "unity-asset-store-academy", "unity-docs", "unity-learn"],
            "natural_codeswitch_share_pct": 0, "synthetic_codeswitch": "diagnostic only; never fitted or primary-scored",
            "design_principles": list(DESIGN_PRINCIPLES),
        },
        "facts": {
            "registry_fingerprint": rs.registry_fingerprint(recs),
            "stage1b_evidence_fingerprint": rs.STAGE1B_EVIDENCE_FINGERPRINT,
            "evidence_artifact": EVIDENCE_ARTIFACT,
            "stage1b_proposed_accept_sources": stage1b_proposed,
            "stage1b_proposed_accept_count": len(stage1b_proposed),
            "planning_total_train_bytes": PLANNING_TOTAL_TRAIN_BYTES,
            "planning_total_train_bytes_basis": "owner-supplied estimate: ~10M 4096-equivalent train tokens at ~4.12 bytes/token",
        },
        "proposals": {
            "label": "PROPOSAL (awaiting owner approval, OD-12)",
            "stage1c_promotions": [{"source_id": k, "basis": v} for k, v in sorted(STAGE1C_PROMOTIONS.items())],
            "repo_cap_bp_of_category_target": {k: v for k, v in sorted(REPO_CAP_BP.items())},
            "repo_cap_meaning": "absolute byte budget per repository = cap x category target bytes; selection is deterministic (hash-ordered, greedy, never exceeds the budget)",
            "realized_share_gate": "report max realized per-source share at Stage 5; threshold NOT frozen",
            "global_excluded_path_patterns": list(GLOBAL_EXCLUDED_PATTERNS),
            "supersedes": "Stage 1A CR-08 blanket 'Plugins/' exclusion, which would drop UniTask's and UniRx's own source",
            "per_source_excluded_path_patterns": {k: list(v) for k, v in sorted(SOURCE_EXCLUDED_PATTERNS.items())},
            "generated_file_markers": list(GENERATED_FILE_MARKERS),
            "cap_audit": cap_audit(ents),
        },
        "unresolved": {
            "held_stage1b_proposals": [{"source_id": k, "reason": v[0], "owner_decision_ids": list(v[1])}
                                       for k, v in sorted(STAGE1C_HELD_PROPOSALS.items())],
            "not_promoted": [{"source_id": e.source_id, "status": e.stage1c_status, "reason": e.reason,
                              "owner_decision_ids": list(e.owner_decision_ids)}
                             for e in ents if e.stage1c_status != rs.ACCEPT],
            "capacity_shortfalls_at_proposed_cap": {
                "code_languages": language_ceilings(ents, REPO_CAP_BP[rs.CAT_CODE]),
                "note": "C/C++ has no approved source; Python has one. These shortfalls are reported, not backfilled (OD-15).",
            },
            "open_owner_decisions": [d for d in STAGE1C_OPEN_DECISIONS],
        },
        "source_policy": source_rows,
    }
    body["fingerprint"] = _fingerprint(body)
    return body


def _fingerprint(body: dict[str, Any]) -> str:
    payload = {k: v for k, v in body.items() if k != "fingerprint"}
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def policy_json(body: dict[str, Any] | None = None) -> str:
    return json.dumps(body if body is not None else build_policy(), indent=2, sort_keys=True) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the Stage 1C source policy (no network, no acquisition).")
    parser.add_argument("--output", default="artifacts/rep_corpus_10m_v1/01_source_policy.json")
    args = parser.parse_args(argv)
    try:
        body = build_policy()
    except (PolicyError, rs.SourceRegistryError) as exc:
        print(f"POLICY INVALID: {exc}", file=sys.stderr)
        return 2
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(policy_json(body), encoding="utf-8")
    accepted = len(accepted_source_ids(build_entries()))
    print(f"policy written to {out}: {accepted} ACCEPT sources, {body['fingerprint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())