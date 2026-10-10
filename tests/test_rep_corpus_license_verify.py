"""Offline tests for training/experiments/rep_corpus_license_verify.py (no network)."""

from __future__ import annotations

import urllib.request

import pytest

from training.experiments import rep_corpus_license_verify as lv

SHA = "a" * 40
MIT_TEXT = ("MIT License\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software "
            "...\nThe above copyright notice and this permission notice shall be included in all copies.\n")
APACHE_TEXT = "                                 Apache License\n                           Version 2.0, January 2004\n"
BSD3 = ("Redistribution and use in source and binary forms, with or without modification, are permitted ...\n"
        "Neither the name of the copyright holder nor the names of its contributors ...")
BSD2 = "Redistribution and use in source and binary forms, with or without modification ... Redistributions of source code must retain"


def test_detect_known_licenses():
    assert lv.detect_license(MIT_TEXT)[0] == "MIT"
    assert lv.detect_license(APACHE_TEXT)[0] == "Apache-2.0"
    assert lv.detect_license(BSD3)[0] == "BSD-3-Clause"
    assert lv.detect_license(BSD2)[0] == "BSD-2-Clause"
    assert lv.detect_license("Microsoft Public License (Ms-PL)\n...")[0] == "MS-PL"
    assert lv.detect_license("Attribution 4.0 International ...")[0] == "CC-BY-4.0"
    assert lv.detect_license("Attribution-ShareAlike 4.0 International ...")[0] == "CC-BY-SA-4.0"
    assert lv.detect_license("Attribution 3.0 Unported")[0] == "CC-BY-3.0"
    assert lv.detect_license("This is free and unencumbered software released into the public domain.")[0] == "Unlicense"
    assert lv.detect_license("PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2")[0] == "PSF-2.0"
    short = 'Licensed under the Apache License, Version 2.0 (the "License"); see http://www.apache.org/licenses/LICENSE-2.0'
    assert lv.detect_license(short)[0] == "Apache-2.0"
    assert lv.detect_license("All rights reserved. Do not copy.")[0] is None
    assert "MIT" in lv.all_license_matches(MIT_TEXT + BSD2)


def test_statement_excerpts():
    text = "Intro\nLicensed under either of Apache or MIT at your option.\nOther line\n"
    assert lv.statement_excerpts(text) == ["Licensed under either of Apache or MIT at your option."]


def test_url_checks_reject_http_other_hosts_and_userinfo():
    for bad in ("http://raw.githubusercontent.com/x", "https://evil.example.com/x", "ftp://github.com/x",
                "https://github.com.evil.example.com/x"):
        with pytest.raises(lv.FetchError):
            lv.http_fetch(bad)


class _Resp:
    def __init__(self, data):
        self._d = data

    def read(self, n=-1):
        return self._d if n < 0 else self._d[:n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeOpener:
    def __init__(self, data):
        self.data = data

    def open(self, req, timeout=None):
        return _Resp(self.data)


def test_size_limit(monkeypatch):
    monkeypatch.setattr(lv, "_OPENER", _FakeOpener(b"x" * 50))
    assert lv.http_fetch("https://raw.githubusercontent.com/a/b/c", max_bytes=50) == b"x" * 50
    with pytest.raises(lv.FetchError):
        lv.http_fetch("https://raw.githubusercontent.com/a/b/c", max_bytes=49)


def test_redirects_are_refused():
    handler = lv._NoRedirect()
    req = urllib.request.Request("https://github.com/x")
    with pytest.raises(lv.FetchError):
        handler.redirect_request(req, None, 301, "Moved", {}, "https://github.com/y")


def test_parse_ls_refs_head():
    line = f"{SHA} HEAD symref-target:refs/heads/main\n".encode()
    resp = b"%04x" % (len(line) + 4) + line + b"0000"
    assert lv.parse_ls_refs_head(resp) == SHA
    with pytest.raises(lv.FetchError):
        lv.parse_ls_refs_head(b"0000")


def _fake_fetch(files, head=SHA):
    calls = []

    def fetch(url, **kw):
        calls.append(url)
        if url.endswith("git-upload-pack"):
            line = f"{head} HEAD symref-target:refs/heads/main\n".encode()
            return b"%04x" % (len(line) + 4) + line + b"0000"
        for path, data in files.items():
            if url.endswith("/" + path):
                return data
        raise lv.FetchError("HTTP 404")

    fetch.calls = calls
    return fetch


def test_verify_source_hashes_and_flags():
    fetch = _fake_fetch({"LICENSE": MIT_TEXT.encode(), "README.md": b"Licensed under the MIT license.\n", ".gitmodules": b"[submodule]"})
    rec = lv.verify_source("gh-x", "o/r", SHA, commit_origin="pinned_in_registry", fetch=fetch)
    assert rec["repository_license_verified"] and rec["retrieval_status"] == "OK"
    assert rec["license_files"][0]["sha256"] == lv._sha256(MIT_TEXT.encode())
    assert rec["detected_licenses"] == ["MIT"] and rec["gitmodules_present"]
    assert rec["file_level_license_uniformity"] == "PARTIAL"
    assert "LICENSE-APACHE" in rec["not_found_paths"]
    with pytest.raises(lv.FetchError):
        lv.verify_source("gh-x", "o/r", "short", commit_origin="x", fetch=fetch)


def test_unrecognised_or_missing_license_is_not_verified():
    rec = lv.verify_source("gh-x", "o/r", SHA, commit_origin="x", fetch=_fake_fetch({"LICENSE": b"All rights reserved."}))
    assert not rec["repository_license_verified"] and rec["retrieval_status"] == "PARTIAL"
    rec = lv.verify_source("gh-x", "o/r", SHA, commit_origin="x", fetch=_fake_fetch({}))
    assert not rec["repository_license_verified"] and rec["retrieval_status"] == "FAILED"


def test_transport_errors_are_recorded_not_raised():
    def fetch(url, **kw):
        raise lv.FetchError("transport error: boom")

    rec = lv.verify_source("gh-x", "o/r", SHA, commit_origin="x", fetch=fetch)
    assert rec["errors"] and not rec["repository_license_verified"]


def test_failure_isolation_and_pins():
    def fetch(url, **kw):
        if "/BurntSushi/" in url or "BurntSushi" in url:
            raise lv.FetchError("transport error: down")
        if url.endswith("git-upload-pack"):
            line = f"{SHA} HEAD\n".encode()
            return b"%04x" % (len(line) + 4) + line + b"0000"
        if url.endswith("/LICENSE"):
            return MIT_TEXT.encode()
        raise lv.FetchError("HTTP 404")

    body = lv.build_evidence({"gh-psf-requests": "b" * 40},
                             only=["gh-psf-requests", "gh-burntsushi-ripgrep", "gh-pallets-flask"], fetch=fetch,
                             now=lambda: "2000-01-01T00:00:00Z")
    by = {s["source_id"]: s for s in body["sources"]}
    assert by["gh-psf-requests"]["commit"] == "b" * 40 and by["gh-psf-requests"]["commit_origin"] == "pinned_in_registry"
    assert by["gh-pallets-flask"]["commit"] == SHA and by["gh-pallets-flask"]["commit_origin"] == "resolved_head_at_retrieval"
    assert not by["gh-burntsushi-ripgrep"]["repository_license_verified"]
    assert by["gh-pallets-flask"]["repository_license_verified"]
    assert body["not_a_milestone"] is True and body["nb13_assigned"] is False


def test_fingerprint_excludes_timestamp_and_tracks_content():
    kwargs = dict(only=["gh-pallets-flask"], fetch=_fake_fetch({"LICENSE": MIT_TEXT.encode()}))
    a = lv.build_evidence({}, now=lambda: "2000-01-01T00:00:00Z", **kwargs)
    b = lv.build_evidence({}, now=lambda: "2030-01-01T00:00:00Z", **kwargs)
    assert a["fingerprint"] == b["fingerprint"] and a["retrieved_at_utc"] != b["retrieved_at_utc"]
    c = lv.build_evidence({}, now=lambda: "2000-01-01T00:00:00Z", only=["gh-pallets-flask"],
                          fetch=_fake_fetch({"LICENSE": APACHE_TEXT.encode()}))
    assert c["fingerprint"] != a["fingerprint"]
    assert lv.evidence_fingerprint(a) == a["fingerprint"]


def test_module_is_stdlib_only_and_allows_only_github_hosts():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(lv.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert "yug" not in roots and "torch" not in roots
    assert lv.ALLOWED_HOSTS == frozenset({"github.com", "raw.githubusercontent.com"})