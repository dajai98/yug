"""Fail-closed first-time file-scope authority, independent of cached corpora."""
import copy
import hashlib

import pytest

from training.production_corpus import canonical, identity
from training.production_spec import SpecificationError, seal
from training.source_first_clearance import (
    POLICY, RIGHTS, PROJECT_RIGHTS, _already_registered, build, validate_decision_scope,
    validate_rights_record)


SHA = 'sha256:' + 'a' * 64
OTHER = 'sha256:' + 'b' * 64
REV = '1' * 40
SOURCE = 'gh-example-new'
LOCATOR = 'https://github.com/example/new'


def sample():
    source = dict(source_id=SOURCE, locator=LOCATOR, revision=REV,
                  license_id='MIT', category='software_docs',
                  subcategory='curated_subset', deny_paths=['**/vendor/**'])
    row = dict(source_id=SOURCE, locator=LOCATOR, revision=REV,
               path='docs/guide.md', sha256=SHA, bytes=100)
    f = dict(path=row['path'], sha256=SHA, bytes=100,
             content_type='markdown', language='en',
             rights_ref=dict(path='reviews/guide.json', sha256=OTHER))
    decision = seal(dict(schema=POLICY, decision='APPROVE_EXACT_FILES',
        reviewer='reviewer A', decided_at='2026-10-31', source_id=SOURCE,
        locator=LOCATOR, revision=REV, owner='software_docs',
        subcategory='curated_subset', license_id='MIT', split='train',
        cap_tokens=1000, attribution='Example copyright and MIT license',
        obligations=['preserve MIT attribution'], source_scope_evidence=[
            dict(path='evidence/contributors.json', sha256=OTHER),
            dict(path='evidence/license.json', sha256=SHA)], files=[f]))
    rights = seal(dict(schema=RIGHTS, source_id=SOURCE, locator=LOCATOR,
        revision=REV, path=f['path'], sha256=SHA, owner='software_docs',
        subcategory='curated_subset', license_id='MIT', first_party=True,
        third_party=False, generated=False, reviewer='reviewer A',
        basis='Exact human review of authorship and retained notices',
        evidence=[dict(path='evidence/contributors.json', sha256=OTHER)]))
    return source, {row['path']: row}, decision, rights


def test_first_time_scope_and_rights_are_exact_and_deterministic():
    source, rows, decision, rights = sample()
    assert validate_decision_scope(decision, source, rows) == decision['files']
    assert validate_rights_record(rights, decision, decision['files'][0]) == rights['evidence']
    assert canonical(decision) == canonical(copy.deepcopy(decision))
    assert identity(decision) == identity(copy.deepcopy(decision))


@pytest.mark.parametrize('field,value', [
    ('source_id', 'gh-other'), ('locator', 'https://github.com/other/repo'),
    ('revision', '2' * 40), ('owner', 'code'), ('subcategory', 'python'),
    ('license_id', 'Apache-2.0'), ('attribution', ''), ('reviewer', ''),
    ('split', 'unassigned'), ('cap_tokens', 0),
])
def test_source_revision_owner_and_obligation_substitution_fails(field, value):
    source, rows, decision, _ = sample()
    decision = seal(dict(decision, **{field: value}))
    with pytest.raises(SpecificationError):
        validate_decision_scope(decision, source, rows)


@pytest.mark.parametrize('field,value', [
    ('path', 'docs/other.md'), ('sha256', OTHER), ('bytes', 101),
    ('content_type', 'binary'), ('language', ''), ('rights_ref', None),
])
def test_file_substitution_or_missing_file_rights_fails(field, value):
    source, rows, decision, _ = sample()
    decision['files'][0][field] = value
    decision = seal(decision)
    with pytest.raises(SpecificationError):
        validate_decision_scope(decision, source, rows)


def test_vendor_path_and_supersession_fields_cannot_enter_first_time_route():
    source, rows, decision, _ = sample()
    row = rows.pop('docs/guide.md')
    row['path'] = 'docs/vendor/guide.md'
    rows[row['path']] = row
    decision['files'][0]['path'] = row['path']
    decision = seal(decision)
    with pytest.raises(SpecificationError):
        validate_decision_scope(decision, source, rows)
    source, rows, decision, _ = sample()
    decision['prior_clearance_identity'] = OTHER
    decision = seal(decision)
    with pytest.raises(SpecificationError):
        validate_decision_scope(decision, source, rows)


@pytest.mark.parametrize('field,value', [
    ('source_id', 'gh-other'), ('locator', 'https://github.com/other/repo'),
    ('revision', '2' * 40),
    ('path', 'docs/other.md'), ('sha256', OTHER),
    ('owner', 'code'), ('subcategory', 'python'),
    ('license_id', 'Apache-2.0'), ('reviewer', 'other'),
    ('first_party', False), ('third_party', True),
    ('generated', True), ('evidence', []), ('basis', ''),
])
def test_file_rights_ambiguity_and_third_party_fail(field, value):
    _, _, decision, rights = sample()
    rights = seal(dict(rights, **{field: value}))
    with pytest.raises(SpecificationError):
        validate_rights_record(rights, decision, decision['files'][0])


def test_root_license_alone_and_duplicate_files_fail():
    source, rows, decision, _ = sample()
    decision['source_scope_evidence'] = decision['source_scope_evidence'][-1:]
    with pytest.raises(SpecificationError):
        validate_decision_scope(seal(decision), source, rows)
    source, rows, decision, _ = sample()
    decision['files'].append(copy.deepcopy(decision['files'][0]))
    with pytest.raises(SpecificationError):
        validate_decision_scope(seal(decision), source, rows)


def test_missing_external_pin_cannot_create_grant(tmp_path):
    output = tmp_path/'artifacts'/'grant'
    with pytest.raises(SpecificationError):
        build(tmp_path/'decision.json', output, expected_decision_sha256=None,
              cache=tmp_path/'cache', catalog=tmp_path/'catalog.json',
              review=tmp_path/'review', references=tmp_path/'references.jsonl',
              root=tmp_path)
    assert not output.exists()


def test_legacy_null_locator_does_not_disable_source_alias_checks():
    source, _, _, _ = sample()
    registry = {'sources': [{'source_id': 'gh-legacy', 'locator': None}]}
    assert not _already_registered(source, registry)
    registry['sources'].append({'source_id': source['source_id'], 'locator': None})
    assert _already_registered(source, registry)
    registry['sources'][-1] = {'source_id': 'gh-other', 'locator': source['locator'] + '/'}
    assert _already_registered(source, registry)


def project_sample():
    source, _, decision, _ = sample()
    raw = (b'# Guide\n\nFirst-party documentation.\n'
           b'![Image](media/scene.jpg)\n'
           b'[!INCLUDE [News](../includes/news.md)]\n')
    sha = 'sha256:' + hashlib.sha256(raw).hexdigest()
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    path = 'docs/guide.md'
    row = dict(source_id=SOURCE, locator=LOCATOR, revision=REV,
               path=path, sha256=sha, bytes=len(raw))
    decision['files'][0].update(sha256=sha, bytes=len(raw))
    license_ref = dict(path='evidence/license.json', sha256=SHA)
    publication = dict(path='evidence/publication.json', sha256=OTHER)
    contribution = dict(path='evidence/contribution.json', sha256=OTHER)
    history = dict(path='evidence/history.json', sha256=OTHER)
    decision['source_scope_evidence'] = sorted(
        [license_ref, publication, contribution, history], key=lambda x: x['path'])
    decision = seal(decision)
    rights = seal(dict(schema=PROJECT_RIGHTS, source_id=SOURCE, locator=LOCATOR,
        revision=REV, path=path, sha256=sha, git_blob=blob,
        owner='software_docs', subcategory='curated_subset', license_id='MIT',
        reviewer='reviewer A', basis='Exact project publication and file review',
        evidence=decision['source_scope_evidence'],
        project_publication_ref=publication,
        contribution_policy_ref=contribution, file_history_ref=history,
        project_material_type='documentation', file_license_conflict=False,
        unresolved_copied_or_adapted=False, third_party=False, generated=False,
        binary=False, media=False, translation=False,
        linked_scope_exclusions=['docs/media/scene.jpg', 'includes/news.md'],
        attribution=decision['attribution'], obligations=decision['obligations']))
    return source, {path: row}, decision, rights, raw, blob, license_ref


def project_check(source, rows, decision, rights, raw, blob, license_ref):
    approved = validate_decision_scope(decision, source, rows)[0]
    return validate_rights_record(rights, decision, approved, raw=raw,
                                  git_object=blob, source_license_ref=license_ref)


def project_with_links(content, exclusions):
    args = list(project_sample())
    raw = b'# Guide\n\nFirst-party documentation.\n' + content
    sha = 'sha256:' + hashlib.sha256(raw).hexdigest()
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    args[1]['docs/guide.md'].update(sha256=sha, bytes=len(raw))
    args[2]['files'][0].update(sha256=sha, bytes=len(raw))
    args[2] = seal(args[2])
    args[3] = seal(dict(args[3], sha256=sha, git_blob=blob,
                        linked_scope_exclusions=exclusions))
    args[4], args[5] = raw, blob
    return args


@pytest.mark.parametrize('content,exclusion', [
    (b'[Guide](local/file.md)\n', 'docs/local/file.md'),
    (b'![Image](image.png)\n', 'docs/image.png'),
    (b'[Guide][ref]\n\n[ref]: docs/file.md\n', 'docs/docs/file.md'),
    (b'![Image][img]\n\n[img]: image.png\n', 'docs/image.png'),
    (b'<img src="image.png">\n', 'docs/image.png'),
    (b'<a href="local.md">Guide</a>\n', 'docs/local.md'),
])
def test_local_markdown_and_html_references_require_exact_exclusions(content, exclusion):
    args = project_with_links(content, [exclusion])
    assert project_check(*args) == args[3]['evidence']
    assert [f['path'] for f in validate_decision_scope(args[2], args[0], args[1])] == [
        'docs/guide.md']
    args[3] = seal(dict(args[3], linked_scope_exclusions=[]))
    with pytest.raises(SpecificationError, match='scope exclusion'):
        project_check(*args)


def test_external_and_fragment_references_do_not_add_local_files():
    content = (b'[Web](https://example.com/a) [Section](#details)\n'
               b'[Remote][ref]\n\n[ref]: https://example.com/b\n'
               b'<a href="https://example.com/c">Web</a>\n'
               b'<img src="data:image/png;base64,AAAA">\n')
    args = project_with_links(content, [])
    assert project_check(*args) == args[3]['evidence']
    assert args[3]['linked_scope_exclusions'] == []


@pytest.mark.parametrize('content', [
    b'[Link](../../outside.md)\n',
    b'[Link][ref]\n\n[ref]: ../../outside.md\n',
    b'<img src="../../outside.md">\n',
    b'<a href="C:/outside.md">Outside</a>\n',
    b'<img src="file:///tmp/image.png">\n',
    b'[Link][missing]\n',
    b'<img srcset="image.png 1x">\n',
    b'[Link](local(file).md)\n',
])
def test_unresolvable_or_escaping_local_references_fail_closed(content):
    args = project_with_links(content, [])
    with pytest.raises(SpecificationError):
        project_check(*args)


def test_link_exclusions_are_deterministic_and_never_widen_file_scope():
    content = (b'[Guide](image.png) ![Image][img]\n'
               b'[img]: image.png\n<img src="image.png">\n')
    args = project_with_links(content, ['docs/image.png'])
    original = canonical(args[3])
    assert project_check(*args) == project_check(*args)
    assert canonical(args[3]) == original
    assert [f['path'] for f in args[2]['files']] == ['docs/guide.md']
    assert 'docs/image.png' not in [f['path'] for f in args[2]['files']]


def test_project_publication_profile_can_qualify_without_individual_clas():
    args = project_sample()
    assert project_check(*args) == args[3]['evidence']
    assert canonical(args[3]) == canonical(copy.deepcopy(args[3]))


@pytest.mark.parametrize('missing', [
    'project_publication_ref', 'contribution_policy_ref', 'file_history_ref'])
def test_root_or_spdx_alone_cannot_qualify(missing):
    args = list(project_sample())
    rights = dict(args[3]); rights.pop(missing)
    args[3] = seal(rights)
    with pytest.raises(SpecificationError):
        project_check(*args)


def test_matching_spdx_without_project_publication_still_fails():
    args = list(project_sample())
    raw = b'SPDX-License-Identifier: MIT\n' + args[4]
    sha = 'sha256:' + hashlib.sha256(raw).hexdigest()
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    args[1]['docs/guide.md'].update(sha256=sha, bytes=len(raw))
    args[2]['files'][0].update(sha256=sha, bytes=len(raw))
    args[2] = seal(args[2])
    rights = dict(args[3], sha256=sha, git_blob=blob)
    rights['evidence'] = [r for r in rights['evidence']
                          if r != rights['project_publication_ref']]
    args[3] = seal(rights)
    args[4], args[5] = raw, blob
    with pytest.raises(SpecificationError, match='Root license alone'):
        project_check(*args)


def test_project_profile_does_not_mutate_legacy_rights_or_decision():
    _, _, legacy_decision, legacy_rights = sample()
    prior_decision = canonical(legacy_decision)
    prior_rights = canonical(legacy_rights)
    args = project_sample()
    assert project_check(*args) == args[3]['evidence']
    assert canonical(legacy_decision) == prior_decision
    assert canonical(legacy_rights) == prior_rights
    assert validate_rights_record(legacy_rights, legacy_decision,
                                  legacy_decision['files'][0]) == legacy_rights['evidence']


@pytest.mark.parametrize('field,value', [
    ('source_id', 'gh-other'), ('revision', '2' * 40),
    ('path', 'docs/other.md'), ('sha256', OTHER),
    ('git_blob', 'f' * 40), ('owner', 'graphics'),
    ('subcategory', 'other'), ('license_id', 'Apache-2.0'),
    ('attribution', ''), ('obligations', []), ('reviewer', 'other'),
    ('file_license_conflict', True), ('unresolved_copied_or_adapted', True),
    ('third_party', True), ('generated', True), ('media', True),
    ('translation', True), ('linked_scope_exclusions', []),
])
def test_project_file_substitutions_and_ambiguity_fail(field, value):
    args = list(project_sample())
    args[3] = seal(dict(args[3], **{field: value}))
    with pytest.raises(SpecificationError):
        project_check(*args)


@pytest.mark.parametrize('text', [
    b'# SPDX-License-Identifier: GPL-3.0\n',
    b'# SPDX-License-Identifier: MIT OR GPL-3.0\n',
    b'# Adapted from another project\n',
    b'# Copied from third-party docs\n',
])
def test_conflicting_file_license_or_adapted_content_fails(text):
    args = list(project_sample())
    args[4] = args[4] + text
    with pytest.raises(SpecificationError):
        project_check(*args)


def test_human_decision_and_external_pin_are_required_for_project_route(tmp_path):
    args = list(project_sample())
    args[2] = seal(dict(args[2], decision='REVIEW'))
    with pytest.raises(SpecificationError):
        project_check(*args)
    with pytest.raises(SpecificationError):
        build(tmp_path/'decision.json', tmp_path/'artifacts'/'grant',
              expected_decision_sha256=None, cache=tmp_path/'cache',
              catalog=tmp_path/'catalog.json', review=tmp_path/'review',
              references=tmp_path/'references.jsonl', root=tmp_path)
    assert not (tmp_path/'artifacts'/'grant').exists()
