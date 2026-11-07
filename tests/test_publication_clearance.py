"""Zero-authority bounded publication candidates and exact human boundaries."""

import hashlib

import pytest

from training.production_spec import SpecificationError, seal
from training.publication_clearance import _classify, decision_batches
from training.source_first_clearance import (
    BOUNDED_RIGHTS,
    BOUNDED_SCOPE,
    validate_candidate_batch,
    validate_rights_record,
)

SHA = "sha256:" + "a" * 64
OTHER = "sha256:" + "b" * 64


def _candidate(path, raw=b'print("hello")\n', *, kind="code", extra=None):
    row = dict(path=path, git_mode="100644")
    source = dict(status="ACCEPT", license_id="MIT")
    observed = dict(status="REVIEW", reason="exact grant needed")
    rows = {path: row}
    history = dict(adequate=True)
    publication = [dict(path="README.md", sha256=SHA)]
    contribution = [dict(path="CONTRIBUTING.md", sha256=OTHER)]
    if extra:
        extra(row, source, observed, rows, history, publication, contribution)
    return _classify(row, observed, source, rows, raw, history, publication, contribution, False)


@pytest.mark.parametrize(
    ("path", "raw", "expected"),
    [
        ("docs/start.md", b"# First party\n", "ROUTINE_CANDIDATE"),
        ("doc/start.rst", b"First party\n===========\n", "ROUTINE_CANDIDATE"),
        ("doc/start.txt", b"First party\n", "REVIEW"),
        ("src/pkg/main.py", b'print("hello")\n', "ROUTINE_CANDIDATE"),
        ("src/vendor/main.py", b'print("hello")\n', "EXCLUDE"),
        ("src/pkg/main.py", b"# generated; do not edit\n", "REVIEW"),
        ("src/pkg/main.py", b"# copied from elsewhere\n", "REVIEW"),
        ("src/pkg/main.py", b"# SPDX-License-Identifier: GPL-3.0\n", "REVIEW"),
        ("src/pkg/main.py", b"\xff\n", "REVIEW"),
        ("docs/start.rst", b".. include:: external.rst\n", "REVIEW"),
        ("tests/test_main.py", b'print("hello")\n', "EXCLUDE"),
        ("src/pkg/main.py", b'print("hello")\n', "ROUTINE_CANDIDATE"),
    ],
)
def test_file_candidate_and_exception_states(path, raw, expected):
    state, _, _ = _candidate(path, raw)
    assert state == expected


def test_nested_license_and_missing_history_fail_closed():
    state, reason, _ = _candidate(
        "src/pkg/main.py", extra=lambda *a: a[3].update({"src/LICENSE": {}})
    )
    assert state == "REVIEW" and "nested" in reason
    state, reason, _ = _candidate("src/pkg/main.py", extra=lambda *a: a[4].update(adequate=False))
    assert state == "REVIEW" and "history" in reason


def test_publication_contribution_and_protected_require_review():
    state, _, _ = _candidate("docs/start.md", extra=lambda *a: a[5].clear())
    assert state == "REVIEW"
    row = dict(path="docs/start.md", git_mode="100644")
    state, reason, _ = _classify(
        row,
        dict(status="REVIEW"),
        dict(status="ACCEPT", license_id="MIT"),
        {row["path"]: row},
        b"# docs",
        dict(adequate=True),
        [dict(path="README", sha256=SHA)],
        [dict(path="CONTRIBUTING", sha256=OTHER)],
        True,
    )
    assert state == "EXCLUDE" and "protected" in reason


def test_batches_are_bounded_stable_and_scope_bound():
    scope = seal(dict(source_id="sample", inventory_identity=SHA, authority="NONE"))
    candidates = [
        seal(
            dict(
                path=f"src/a{i:03}.py", source_id="sample", authority="NONE", training_allowed=False
            )
        )
        for i in range(130)
    ]
    first = decision_batches(list(reversed(candidates)), scope, SHA)
    assert [len(b["paths"]) for b in first] == [64, 64, 2]
    assert first == decision_batches(candidates, scope, SHA)
    assert len({p for b in first for p in b["paths"]}) == 130
    assert all(b["authority"] == "NONE" and b["training_allowed"] is False for b in first)
    changed = decision_batches(candidates, seal(dict(scope, inventory_identity=OTHER)), SHA)
    assert [b["identity"] for b in changed] != [b["identity"] for b in first]
    with pytest.raises(SpecificationError):
        decision_batches(candidates + [candidates[0]], scope, SHA)


def _rights_fixture(path="src/pkg/main.py", raw=b'print("hello")\n', content_type="code"):
    git_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    revision = "1" * 40
    license_ref = dict(path="evidence/license.json", sha256=SHA)
    publication = dict(path="evidence/publication.json", sha256=OTHER)
    contribution = dict(path="evidence/contribution.json", sha256="sha256:" + "c" * 64)
    history_ref = dict(path="evidence/history.json", sha256="sha256:" + "d" * 64)
    candidate_ref = dict(path="evidence/candidate.json", sha256="sha256:" + "e" * 64)
    scope_ref = dict(path="evidence/scope.json", sha256="sha256:" + "f" * 64)
    source_fields = dict(
        source_id="sample",
        locator="https://github.com/x/sample",
        revision=revision,
        license_id="MIT",
    )
    decision = dict(
        source_fields,
        reviewer="owner",
        owner="code",
        subcategory="python",
        attribution="Copyright sample",
        obligations=["retain MIT"],
        source_scope_evidence=[scope_ref],
    )
    file = dict(
        path=path,
        sha256=SHA,
        bytes=len(raw),
        content_type=content_type,
        language="en" if content_type != "code" else "python",
    )
    scope = seal(
        dict(
            schema=BOUNDED_SCOPE,
            decision="APPROVE_BOUNDED_SCOPE",
            reviewer="owner",
            **source_fields,
            inventory_identity=OTHER,
            subtrees=["docs/" if path.startswith("docs/") else "src/"],
            excluded_paths=[],
            root_license_ref=license_ref,
            publication_ref=publication,
            contribution_ref=contribution,
            notice_refs=[],
            protected_source=False,
        )
    )
    history = seal(
        dict(
            source_id="sample",
            revision=revision,
            path=path,
            git_blob=git_blob,
            adequate=True,
            commits=["2" * 40],
        )
    )
    dossier = seal(
        dict(
            policy="bounded-publication-candidates-v1",
            state="ROUTINE_CANDIDATE",
            authority="NONE",
            training_allowed=False,
            grantable=False,
            inventory_identity=OTHER,
            source_scope_candidate_identity=SHA,
            **{k: source_fields[k] for k in ("source_id", "locator", "revision")},
            path=path,
            sha256=SHA,
            bytes=len(raw),
            git_blob=git_blob,
            root_license_evidence=license_ref,
            publication_evidence=[dict(publication, git_blob="3" * 40)],
            contribution_evidence=[dict(contribution, git_blob="4" * 40)],
            notice_copyright_evidence=[],
            file_history_identity=history["identity"],
            file_type="code" if content_type == "code" else "documentation",
        )
    )
    refs = sorted(
        [scope_ref, candidate_ref, history_ref, license_ref, publication, contribution],
        key=lambda x: x["path"],
    )
    rights = seal(
        dict(
            schema=BOUNDED_RIGHTS,
            **source_fields,
            path=path,
            sha256=SHA,
            git_blob=git_blob,
            owner="code",
            subcategory="python",
            reviewer="owner",
            basis="Reviewed bounded first-party publication",
            evidence=refs,
            source_scope_ref=scope_ref,
            candidate_ref=candidate_ref,
            file_history_ref=history_ref,
            project_material_type=("source_code" if content_type == "code" else "documentation"),
            file_type="code" if content_type == "code" else "documentation",
            file_license_conflict=False,
            unresolved_copied_or_adapted=False,
            third_party=False,
            generated=False,
            binary=False,
            media=False,
            translation=False,
            linked_scope_exclusions=[],
            attribution="Copyright sample",
            obligations=["retain MIT"],
        )
    )
    args = dict(
        raw=raw,
        git_object=git_blob,
        source_license_ref=license_ref,
        scope_decision=scope,
        dossier=dossier,
        history=history,
        inventory_identity=OTHER,
    )
    return rights, decision, file, args


@pytest.mark.parametrize(
    ("path", "raw", "content_type"),
    [
        ("src/pkg/main.py", b'print("hello")\n', "code"),
        ("docs/start.rst", b"First party\n===========\n", "prose"),
        ("docs/start.md", b"# First party\n", "markdown"),
    ],
)
def test_bounded_exact_rights_require_human_scope_and_candidate(path, raw, content_type):
    rights, decision, file, args = _rights_fixture(path, raw, content_type)
    assert validate_rights_record(rights, decision, file, **args) == rights["evidence"]
    for field, value in [
        ("scope_decision", None),
        ("dossier", None),
        ("history", None),
        ("inventory_identity", SHA),
    ]:
        bad = dict(args, **{field: value})
        with pytest.raises((SpecificationError, TypeError)):
            validate_rights_record(rights, decision, file, **bad)


@pytest.mark.parametrize(
    ("target", "key", "value"),
    [
        ("rights", "third_party", True),
        ("rights", "generated", True),
        ("rights", "file_license_conflict", True),
        ("scope", "protected_source", True),
        ("scope", "revision", "2" * 40),
        ("dossier", "state", "REVIEW"),
        ("dossier", "authority", "GRANTED"),
        ("history", "adequate", False),
    ],
)
def test_bounded_rights_fail_closed_on_conflict(target, key, value):
    rights, decision, file, args = _rights_fixture()
    if target == "rights":
        rights = seal(dict(rights, **{key: value}))
    else:
        field = "scope_decision" if target == "scope" else target
        args[field] = seal(dict(args[field], **{key: value}))
    with pytest.raises(SpecificationError):
        validate_rights_record(rights, decision, file, **args)


def test_candidate_is_not_a_grant_or_training_decision():
    rights, decision, file, args = _rights_fixture()
    with pytest.raises(SpecificationError):
        validate_rights_record(args["dossier"], decision, file, **args)
    assert args["dossier"]["authority"] == "NONE"
    assert args["dossier"]["training_allowed"] is False


def test_exact_decision_cannot_migrate_file_between_batches():
    scope = seal(dict(source_id="sample", inventory_identity=SHA))
    dossiers = [
        seal(dict(source_scope_candidate_identity=scope["identity"], path=f"src/a{i}.py"))
        for i in range(65)
    ]
    batches = decision_batches(dossiers, scope, SHA)
    ordered = sorted(dossiers, key=lambda d: d["path"])
    decision = dict(files=[dict(path=d["path"]) for d in ordered[:64]])
    source = dict(source_id="sample")
    validate_candidate_batch(batches[0], decision, source, SHA, ordered[:64])
    with pytest.raises(SpecificationError):
        validate_candidate_batch(batches[1], decision, source, SHA, ordered[:64])
    with pytest.raises(SpecificationError):
        validate_candidate_batch(batches[0], decision, source, OTHER, ordered[:64])
    changed = seal(dict(batches[0], paths=batches[0]["paths"][1:] + [batches[1]["paths"][0]]))
    with pytest.raises(SpecificationError):
        validate_candidate_batch(changed, decision, source, SHA, ordered[:64])
