"""Representative-corpus expansion, STAGE 1B: real-fetch license verification.

NOT a milestone (no NB-13). Fetches LICENSE/legal METADATA only: license files,
a README for license statements, notice files, and ``.gitmodules`` presence, all
at a concrete 40-hex commit. It never clones, never downloads source corpora, and
never executes anything it fetched.

SAFETY
  - HTTPS only, explicit host allow-list (github.com, raw.githubusercontent.com).
  - Redirects are refused (a redirect is recorded as a failure, not followed).
  - Per-request timeout and a hard response-size limit.
  - Hashing is SHA-256 over the exact fetched bytes.
  - One failing source never marks any other source verified; a source counts as
    verified only when at least one license file was fetched AND recognised.

DETECTION
  License identification is a deterministic phrase match on the fetched text and is
  reported together with the phrases that matched. It is evidence for a human
  reviewer, not legal advice. Unrecognised text is reported as ``None``.

EVIDENCE ARTIFACT
  ``build_evidence`` returns a dict whose ``fingerprint`` covers everything except the
  retrieval timestamp. Network access is kept out of normal unit tests: tests inject an
  ``opener`` function.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

SCHEMA = "rep-corpus-license-evidence-v1"
ALLOWED_HOSTS = frozenset({"github.com", "raw.githubusercontent.com"})
TIMEOUT_SECONDS = 20.0
MAX_BYTES = 262_144
USER_AGENT = "yug-license-verify/0.1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")

LICENSE_PATHS: tuple[str, ...] = (
    "LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "COPYING", "COPYING.md", "LICENSE-MIT",
    "LICENSE-APACHE", "UNLICENSE", "LICENSE-CODE", "LICENSE-DOCS", "license.txt",
)
README_PATHS: tuple[str, ...] = ("README.md", "README.rst", "README.txt", "README")
NOTICE_PATHS: tuple[str, ...] = ("NOTICE", "NOTICE.md", "THIRD-PARTY-NOTICES.TXT", "ThirdPartyNotices.txt")

# source_id -> (repository, extra license-bearing paths). Candidate list mirrors the registry.
GITHUB_SOURCES: dict[str, tuple[str, tuple[str, ...]]] = {
    "gh-psf-requests": ("psf/requests", ()),
    "gh-expressjs-express": ("expressjs/express", ()),
    "gh-gin-gonic-gin": ("gin-gonic/gin", ()),
    "gh-burntsushi-ripgrep": ("BurntSushi/ripgrep", ()),
    "gh-cysharp-unitask": ("Cysharp/UniTask", ()),
    "gh-mirrornetworking-mirror": ("MirrorNetworking/Mirror", ()),
    "gh-monogame-monogame": ("MonoGame/MonoGame", ()),
    "gh-google-eng-practices": ("google/eng-practices", ()),
    "gh-microsoft-api-guidelines": ("microsoft/api-guidelines", ()),
    "gh-rust-lang-book": ("rust-lang/book", ()),
    "gh-pallets-flask": ("pallets/flask", ()),
    "gh-encode-httpx": ("encode/httpx", ()),
    "gh-lodash-lodash": ("lodash/lodash", ()),
    "gh-spf13-cobra": ("spf13/cobra", ()),
    "gh-tokio-rs-tokio": ("tokio-rs/tokio", ()),
    "gh-serde-rs-serde": ("serde-rs/serde", ()),
    "gh-google-guava": ("google/guava", ()),
    "gh-nlohmann-json": ("nlohmann/json", ("LICENSE.MIT",)),
    "gh-flame-engine-flame": ("flame-engine/flame", ()),
    "gh-dotnet-runtime": ("dotnet/runtime", ("LICENSE.TXT",)),
    "gh-dotnet-roslyn": ("dotnet/roslyn", ("License.txt",)),
    "gh-stride3d-stride": ("stride3d/stride", ()),
    "gh-fna-xna-fna": ("FNA-XNA/FNA", ("licenses/LICENSE", "licenses/lzxdecoder.LICENSE", "licenses/monoxna.LICENSE")),
    "gh-neuecc-unirx": ("neuecc/UniRx", ()),
    "gh-messagepack-csharp": ("MessagePack-CSharp/MessagePack-CSharp", ()),
    "gh-unity-ml-agents": ("Unity-Technologies/ml-agents", ()),
    "gh-rust-lang-reference": ("rust-lang/reference", ()),
    "gh-rust-lang-rust-by-example": ("rust-lang/rust-by-example", ()),
    "gh-rust-lang-rustc-dev-guide": ("rust-lang/rustc-dev-guide", ()),
    "python-docs-cpython-doc": ("python/cpython", ("Doc/license.rst",)),
    "gh-dotnet-docs": ("dotnet/docs", ()),
    "mdn-content": ("mdn/content", ()),
    "gh-kubernetes-website": ("kubernetes/website", ()),
    "docker-docs": ("docker/docs", ()),
}



# Evidence gathered with the interactive web_fetch tool (rendered page text, so no byte-level
# hash exists). Summaries are paraphrased; metadata values are quoted only where they are data.
MANUAL_DATASET_EVIDENCE: tuple[dict[str, Any], ...] = (
    {
        "source_id": "hf-wikipedia-ne-20231101",
        "retrieval_method": "web_fetch of the dataset card and the repository tree page",
        "urls": ["https://huggingface.co/datasets/wikimedia/wikipedia",
                 "https://huggingface.co/datasets/wikimedia/wikipedia/tree/main"],
        "dataset_repo_head_commit_at_retrieval": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        "config": "20231101.ne",
        "config_row_count_on_card": "32.9k (matches the 32,885 reported by the earlier /rows probe)",
        "declared_license_metadata": ["cc-by-sa-3.0", "gfdl"],
        "statement_summary": [
            "Card states original textual content is licensed under GFDL and CC BY-SA 3.0.",
            "Card says some text may be available only under the Creative Commons license (see the Wikimedia Terms of Use).",
            "Card says text by some authors may carry additional licenses or be public domain.",
            "Card points to dumps.wikimedia.org/legal.html for copyright licensing information.",
        ],
        "not_found_in_card": ["any mention of CC BY-SA 4.0"],
        "caveats": [
            "dumps.wikimedia.org/legal.html could not be fetched (cache-only domain), so any newer Wikimedia licensing statement is UNKNOWN.",
            "The existing /rows acquisition path has no revision parameter in this repository's code, so it reads the unpinned main branch; a concrete dataset commit exists but is not what the acquisition uses.",
            "Per-article license metadata is not part of the dataset schema (id, url, title, text).",
        ],
        "verification_result": "PARTIAL: CC-BY-SA-3.0 and GFDL are supported by the card; the card's own caveats and the missing immutable pin keep the owner decision OD-06 open.",
    },
    {
        "source_id": "cp-wikimedia-filtered",
        "retrieval_method": "web_fetch of the dataset card; a third-party listing found by web_search",
        "urls": ["https://huggingface.co/datasets/common-pile/wikimedia_filtered"],
        "declared_license_metadata": [],
        "statement_summary": [
            "No dataset-level license field appears in the card header; licensing is per record (metadata.license).",
            "Sample records carry the exact string 'Creative Commons - Attribution Share-Alike - https://creativecommons.org/licenses/by-sa/4.0/'.",
            "Card says official Wikimedia wikis are released under a CC BY-SA license and lists eight wikis (Wikipedia, Wikinews, Wikibooks, Wikiquote, Wikisource, Wikiversity, Wikivoyage, Wiktionary).",
            "The first sample records are from wikibooks.com and mix namespaces (0, 1, 2, 4, 102, 103).",
            "Card has a 'License Issues' section warning that license laundering or inaccurate metadata can cause wrong license assignment for some documents.",
        ],
        "caveats": [
            "Card says the dumps are from March 2025, while the sample records show an 'added' date of 2024-07-27: provenance statements are internally inconsistent.",
            "The pinned revision 0641bb84... appears as a pinned revision in a third-party dataset listing; its existence on Hugging Face was not verified directly.",
            "Licensing is per record, so only records whose license string maps exactly to CC-BY-SA-4.0 may be treated as ACCEPT (existing code behaviour).",
        ],
        "verification_result": "PARTIAL: per-record licensing and the license-laundering caveat verified; no blanket dataset license; revision existence corroborated by a third party only.",
    },
)


class FetchError(Exception):
    """A bounded fetch failed (host, scheme, redirect, size, HTTP or transport)."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        raise FetchError(f"redirect refused (HTTP {code} -> {newurl})")


_OPENER = urllib.request.build_opener(_NoRedirect)


def _check_url(url: str) -> None:
    m = re.match(r"^(https)://([^/:?#]+)(/|$)", url)
    if not m:
        raise FetchError(f"only https URLs are allowed: {url}")
    if m.group(2).lower() not in ALLOWED_HOSTS:
        raise FetchError(f"host not in allow-list: {m.group(2)}")


def http_fetch(url: str, *, method: str = "GET", data: bytes | None = None, headers: dict[str, str] | None = None,
               max_bytes: int = MAX_BYTES, timeout: float = TIMEOUT_SECONDS) -> bytes:
    """Bounded fetch. Raises :class:`FetchError`; HTTP 404 is reported as ``FetchError('HTTP 404')``."""
    _check_url(url)
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            body = resp.read(max_bytes + 1)
    except FetchError:
        raise
    except urllib.error.HTTPError as exc:
        raise FetchError(f"HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise FetchError(f"transport error: {type(exc).__name__}: {exc}") from exc
    if len(body) > max_bytes:
        raise FetchError(f"response exceeds {max_bytes} bytes")
    return body


def _pkt(text: str) -> bytes:
    raw = text.encode("utf-8")
    return b"%04x" % (len(raw) + 4) + raw


def parse_ls_refs_head(response: bytes) -> str:
    """Extract the HEAD commit SHA from a git protocol-v2 ``ls-refs`` response."""
    pos = 0
    while pos + 4 <= len(response):
        length = int(response[pos:pos + 4], 16)
        if length == 0:
            break
        if length < 4:
            pos += 4
            continue
        line = response[pos + 4:pos + length].decode("utf-8", "replace")
        pos += length
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "HEAD" and _HEX40.match(parts[0]):
            return parts[0]
    raise FetchError("HEAD not found in ls-refs response")


def resolve_head_sha(repo: str, fetch: Callable[..., bytes] = http_fetch) -> str:
    body = (_pkt("command=ls-refs\n") + _pkt("agent=yug-license-verify\n") + b"0001"
            + _pkt("symrefs\n") + _pkt("ref-prefix HEAD\n") + b"0000")
    resp = fetch(f"https://github.com/{repo}.git/git-upload-pack", method="POST", data=body,
                 headers={"Content-Type": "application/x-git-upload-pack-request",
                          "Accept": "application/x-git-upload-pack-result", "Git-Protocol": "version=2"})
    return parse_ls_refs_head(resp)


# ---- license detection (deterministic phrase match) ----

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    # (spdx, all-of phrases, none-of phrases); first match wins, so order matters
    ("CC-BY-SA-4.0", ("attribution-sharealike 4.0 international",), ()),
    ("CC-BY-SA-3.0", ("attribution-sharealike 3.0 unported",), ()),
    ("CC-BY-SA-2.5", ("attribution-sharealike 2.5",), ()),
    ("CC-BY-4.0", ("attribution 4.0 international",), ()),
    ("CC-BY-3.0", ("attribution 3.0 unported",), ()),
    ("CC0-1.0", ("cc0 1.0 universal",), ()),
    ("PSF-2.0", ("python software foundation license version 2",), ()),
    ("MS-PL", ("microsoft public license (ms-pl)",), ()),
    ("Apache-2.0", ("apache license", "version 2.0, january 2004"), ()),
    # short-form notice only (no full license text); reported as such via the caveat below
    ("Apache-2.0", ("licensed under the apache license, version 2.0", "apache.org/licenses/license-2.0"), ()),
    ("Unlicense", ("this is free and unencumbered software released into the public domain",), ()),
    ("BSD-3-Clause", ("redistribution and use in source and binary forms", "neither the name"), ()),
    ("BSD-2-Clause", ("redistribution and use in source and binary forms",), ("neither the name",)),
    ("MIT", ("permission is hereby granted, free of charge, to any person obtaining a copy",
             "the above copyright notice and this permission notice shall be included"), ()),
    ("GPL-family", ("gnu general public license",), ()),
)


def detect_license(text: str) -> tuple[str | None, list[str]]:
    t = _norm(text)
    for spdx, required, forbidden in _RULES:
        if all(p in t for p in required) and not any(p in t for p in forbidden):
            return spdx, list(required)
    return None, []


def all_license_matches(text: str) -> list[str]:
    """Every rule that matches (sorted, unique). A file may contain several license texts or
    mention others; the first-match result of :func:`detect_license` is only the primary."""
    t = _norm(text)
    return sorted({spdx for spdx, req, forb in _RULES if all(p in t for p in req) and not any(p in t for p in forb)})


def title_line(text: str, width: int = 120) -> str:
    for line in text.splitlines():
        if line.strip():
            return re.sub(r"\s+", " ", line).strip()[:width]
    return ""


_STATEMENT = re.compile(r"[^\n]*(?:dual[- ]licen[sc]ed|licen[sc]ed under|either of|at your option|\bunder the terms\b)[^\n]*", re.I)


def statement_excerpts(text: str, limit: int = 3, width: int = 240) -> list[str]:
    out: list[str] = []
    for m in _STATEMENT.finditer(text):
        line = re.sub(r"\s+", " ", m.group(0)).strip()
        if line and line not in out:
            out.append(line[:width])
        if len(out) >= limit:
            break
    return out


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def verify_source(source_id: str, repo: str, commit: str, *, commit_origin: str, extra_paths: Sequence[str] = (),
                  fetch: Callable[..., bytes] = http_fetch, sleep: float = 0.0) -> dict[str, Any]:
    """Collect license evidence for ``repo`` at ``commit``. Never raises for a missing file;
    transport problems are recorded in ``errors``."""
    if not _HEX40.match(commit):
        raise FetchError(f"{source_id}: commit must be 40-hex")
    base = f"https://raw.githubusercontent.com/{repo}/{commit}/"
    rec: dict[str, Any] = {
        "source_id": source_id, "repository": repo, "commit": commit, "commit_origin": commit_origin,
        "license_files": [], "not_found_paths": [], "readme": None, "notice_files": [],
        "gitmodules_present": False, "errors": [], "caveats": [],
    }

    def get(path: str) -> bytes | None:
        if sleep:
            time.sleep(sleep)
        try:
            return fetch(base + path)
        except FetchError as exc:
            if str(exc) == "HTTP 404":
                rec["not_found_paths"].append(path)
            else:
                rec["errors"].append(f"{path}: {exc}")
            return None

    for path in tuple(LICENSE_PATHS) + tuple(extra_paths):
        data = get(path)
        if data is None:
            continue
        text = data.decode("utf-8", "replace")
        spdx, matched = detect_license(text)
        rec["license_files"].append({"path": path, "url": base + path, "sha256": _sha256(data), "bytes": len(data),
                                     "detected": spdx, "matched_phrases": matched,
                                     "also_mentions": [m for m in all_license_matches(text) if m != spdx],
                                     "title_line": title_line(text),
                                     "statement_excerpts": statement_excerpts(text)})
    for path in README_PATHS:
        data = get(path)
        if data is not None:
            rec["readme"] = {"path": path, "url": base + path, "sha256": _sha256(data), "bytes": len(data),
                             "license_statement_excerpts": statement_excerpts(data.decode("utf-8", "replace"))}
            break
    for path in NOTICE_PATHS:
        data = get(path)
        if data is not None:
            rec["notice_files"].append({"path": path, "url": base + path, "sha256": _sha256(data), "bytes": len(data)})
    rec["gitmodules_present"] = get(".gitmodules") is not None

    detected = sorted({f["detected"] for f in rec["license_files"] if f["detected"]})
    rec["detected_licenses"] = detected
    unrecognised = [f["path"] for f in rec["license_files"] if not f["detected"]]
    for f in rec["license_files"]:
        if f["also_mentions"]:
            rec["caveats"].append(f"{f['path']}: also contains/mentions {f['also_mentions']}")
    for f in rec["license_files"]:
        if f["detected"] == "Apache-2.0" and f["bytes"] < 2000:
            rec["caveats"].append(f"{f['path']}: short Apache-2.0 notice, not the full license text")
    if unrecognised:
        rec["caveats"].append(f"license text not recognised by phrase rules: {unrecognised}")
    if len(rec["license_files"]) > 1:
        rec["caveats"].append("multiple license files: scope (OR / per-directory / docs vs code) must be read by a human")
    if rec["notice_files"] or rec["gitmodules_present"]:
        rec["caveats"].append("notice files or submodules present: third-party code with separate terms is likely")
    if not rec["license_files"]:
        rec["caveats"].append("no license file found at the tried paths")
    ok = bool(detected)
    rec["retrieval_status"] = "OK" if ok and not rec["errors"] else ("PARTIAL" if rec["license_files"] else "FAILED")
    rec["repository_license_verified"] = ok
    rec["file_level_license_uniformity"] = "PARTIAL" if (rec["notice_files"] or rec["gitmodules_present"]) else "UNKNOWN"
    return rec


def evidence_fingerprint(body: dict[str, Any]) -> str:
    payload = {k: v for k, v in body.items() if k not in ("fingerprint", "retrieved_at_utc")}
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_evidence(pins: dict[str, str], *, only: Sequence[str] | None = None,
                   fetch: Callable[..., bytes] = http_fetch, sleep: float = 0.0,
                   now: Callable[[], str] | None = None) -> dict[str, Any]:
    """``pins``: source_id -> 40-hex commit already frozen in the registry. Sources absent from
    ``pins`` get their HEAD resolved at retrieval time (recorded as such). A failure in one
    source is recorded for that source only."""
    sources: list[dict[str, Any]] = []
    for source_id in sorted(GITHUB_SOURCES):
        if only and source_id not in only:
            continue
        repo, extra = GITHUB_SOURCES[source_id]
        try:
            if source_id in pins:
                commit, origin = pins[source_id], "pinned_in_registry"
            else:
                commit, origin = resolve_head_sha(repo, fetch), "resolved_head_at_retrieval"
            sources.append(verify_source(source_id, repo, commit, commit_origin=origin, extra_paths=extra,
                                         fetch=fetch, sleep=sleep))
        except FetchError as exc:
            sources.append({"source_id": source_id, "repository": repo, "commit": None, "commit_origin": None,
                            "license_files": [], "not_found_paths": [], "readme": None, "notice_files": [],
                            "gitmodules_present": False, "errors": [str(exc)], "caveats": ["commit not resolved"],
                            "detected_licenses": [], "retrieval_status": "FAILED",
                            "repository_license_verified": False, "file_level_license_uniformity": "UNKNOWN"})
    body: dict[str, Any] = {
        "schema": SCHEMA, "not_a_milestone": True, "nb13_assigned": False,
        "fetcher": {"allowed_hosts": sorted(ALLOWED_HOSTS), "timeout_seconds": TIMEOUT_SECONDS,
                    "max_bytes": MAX_BYTES, "redirects": "refused", "license_paths": list(LICENSE_PATHS),
                    "readme_paths": list(README_PATHS), "notice_paths": list(NOTICE_PATHS)},
        "sources": sources,
        "dataset_sources": list(MANUAL_DATASET_EVIDENCE),
    }
    body["fingerprint"] = evidence_fingerprint(body)
    body["retrieved_at_utc"] = (now or (lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))()
    return body


def main(argv: Sequence[str] | None = None) -> int:
    from training.experiments import rep_corpus_sources as rs

    parser = argparse.ArgumentParser(description="Stage 1B license-evidence fetch (license metadata only).")
    parser.add_argument("--output", default="artifacts/rep_corpus_10m_v1/00_source_license_evidence.json")
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these source ids")
    parser.add_argument("--sleep", type=float, default=0.05, help="delay between requests")
    args = parser.parse_args(argv)
    pins = {x.source_id: x.pin for x in rs.build_registry()
            if x.source_id in GITHUB_SOURCES and x.pin_kind == rs.PIN_COMMIT and x.pin}
    body = build_evidence(pins, only=args.only, sleep=args.sleep)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    verified = sum(1 for s in body["sources"] if s["repository_license_verified"])
    print(f"evidence written to {out}: {verified}/{len(body['sources'])} verified, {body['fingerprint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())