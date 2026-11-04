"""Materialize an externally approved, exact-file first source clearance.

This is distinct from revision supersession. It never chooses files or infers
rights from a repository license, an SPDX label, or a review-cache entry.
The decision digest must be supplied from outside the mutable output. Each
file additionally needs a pinned, human-reviewed rights record under one of the
two bounded first-time evidence profiles.
No acquisition, tokenization, accounting, or training occurs here.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import posixpath
import re
from html import unescape
from pathlib import Path
from urllib.parse import urlsplit

from training.category_ownership import decision as ownership_decision
from training.clearance_evidence import verify_bundle
from training.production_acquisition import item_key, read_plan, validate_clearance
from training.production_corpus import atomic_json, canonical, file_digest
from training.production_spec import (local_path, need, path_allowed, seal,
                                      validate, verify_seal)
from training.source_admission import cached_sources, holdout_groups
from training.source_approval import payload
from training.source_prepare import ROOT, LICENSES, screening
from training.license_evidence import resolve_license

POLICY = 'reviewed-first-source-clearance-v1'
RIGHTS = 'reviewed-first-file-rights-v1'
PROJECT_RIGHTS = 'project-authorized-permissive-v1'
SHA = re.compile(r'sha256:[0-9a-f]{64}\Z')
REV = re.compile(r'[0-9a-f]{40}\Z')
LOCAL_LINK = re.compile(r'!?\[[^\]]*\]\(([^)]+)\)')
REFERENCE_DEFINITION = re.compile(r'^[ \t]{0,3}\[([^\]\r\n]+)\]:[ \t]*(.*)$', re.M)
REFERENCE_USE = re.compile(r'!?\[([^\]\r\n]*)\]\[([^\]\r\n]*)\]')
HTML_TAG = re.compile(r'<[A-Za-z][^>]*>', re.S)
HTML_RESOURCE = re.compile(
    r'\b(?:src|href)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+))', re.I)
PROVENANCE_SIGNAL = re.compile(
    r'\b(copied|adapted|derived|borrowed|vendored|third.party)\b', re.I)
UNSUPPORTED_PATH_PARTS = frozenset((
    'media', 'assets', 'translation', 'translations', 'vendor', 'vendored',
    'third_party', 'thirdparty', 'generated', 'dist', 'build', 'node_modules'))


def _unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate approval/evidence field')
        result[key] = value
    return result


def _read(path, *, maximum=1_048_576, canonical_required=True):
    path = Path(path)
    need(path.is_file() and 0 < path.stat().st_size <= maximum,
         'Missing or oversized approval/evidence')
    data = path.read_bytes()
    try:
        value = json.loads(data, object_pairs_hook=_unique)
    except (ValueError, UnicodeError) as exc:
        raise ValueError('Malformed approval/evidence JSON') from exc
    need(type(value) is dict and
         (not canonical_required or canonical(value) == data),
         'Noncanonical approval/evidence JSON')
    verify_seal(value)
    return value


def _ref(root, ref, output):
    need(type(ref) is dict and set(ref) == {'path', 'sha256'} and
         type(ref['sha256']) is str and SHA.fullmatch(ref['sha256']),
         'Malformed independent evidence reference')
    path = local_path(root, ref['path'])
    need(not path.is_relative_to(output) and path.is_file() and
         path.stat().st_size <= 1_048_576 and file_digest(path) == ref['sha256'],
         'Independent evidence missing, changed, or output-local')
    return path


def _already_registered(source, registry):
    """Treat absent legacy locators as unknown, never as an alias."""
    return any(s['source_id'] == source['source_id'] or
               (s.get('locator') or '').casefold().rstrip('/') ==
               source['locator'].casefold().rstrip('/')
               for s in registry['sources'])


def validate_decision_scope(decision, source, rows):
    """Pure exact-scope check; rights records and payloads are checked by build."""
    need(type(decision) is dict and set(decision) == {
        'schema', 'decision', 'reviewer', 'decided_at', 'source_id', 'locator',
        'revision', 'owner', 'subcategory', 'license_id', 'split', 'cap_tokens',
        'attribution', 'obligations', 'source_scope_evidence', 'files', 'identity'},
        'Incomplete first-time decision')
    verify_seal(decision)
    need(decision['schema'] == POLICY and decision['decision'] == 'APPROVE_EXACT_FILES'
         and type(decision['reviewer']) is str and bool(decision['reviewer'].strip())
         and type(decision['decided_at']) is str and bool(decision['decided_at'].strip()),
         'Missing external first-time governance decision')
    need(all(decision[key] == source[key] for key in
             ('source_id', 'locator', 'revision', 'license_id')) and
         decision['owner'] == source['category'] and
         decision['subcategory'] == source['subcategory'] and
         REV.fullmatch(decision['revision']) and source['license_id'] in LICENSES,
         'First-time decision source/revision/license/owner mismatch')
    need(decision['split'] in ('train', 'validation') and
         type(decision['cap_tokens']) is int and 0 < decision['cap_tokens'] <= 1_000_000,
         'Invalid split or bounded source cap')
    need(type(decision['attribution']) is str and bool(decision['attribution'].strip())
         and type(decision['obligations']) is list and bool(decision['obligations'])
         and all(type(x) is str and bool(x.strip()) for x in decision['obligations'])
         and decision['obligations'] == sorted(set(decision['obligations'])),
         'Missing attribution/NOTICE obligations')
    need(type(decision['source_scope_evidence']) is list and
         len(decision['source_scope_evidence']) >= 2 and
         all(type(x) is dict and set(x) == {'path', 'sha256'} and
             type(x['sha256']) is str and SHA.fullmatch(x['sha256'])
             for x in decision['source_scope_evidence']) and
         decision['source_scope_evidence'] == sorted(
             decision['source_scope_evidence'], key=lambda x: x['path']) and
         len({x['path'] for x in decision['source_scope_evidence']}) ==
         len(decision['source_scope_evidence']),
         'Root license alone cannot establish bounded source scope')
    files = decision['files']
    need(type(files) is list and 0 < len(files) <= 64 and
         all(type(f) is dict and set(f) == {
             'path', 'sha256', 'bytes', 'content_type', 'language', 'rights_ref'}
             for f in files), 'Incomplete or unbounded exact-file decision')
    need(files == sorted(files, key=lambda f: f['path']) and
         len({f['path'].casefold() for f in files}) == len(files),
         'Duplicate, unsorted, or ambiguous first-time file set')
    for f in files:
        row = rows.get(f['path'])
        need(row is not None and row['source_id'] == source['source_id'] and
             row['locator'] == source['locator'] and
             row['revision'] == source['revision'] and
             row['sha256'] == f['sha256'] and row['bytes'] == f['bytes'] and
             type(f['bytes']) is int and 0 < f['bytes'] <= 1_048_576 and
             type(f['sha256']) is str and SHA.fullmatch(f['sha256']) and
             f['content_type'] in ('code', 'markdown', 'prose') and
             type(f['language']) is str and bool(f['language']) and
             type(f['rights_ref']) is dict and
             set(f['rights_ref']) == {'path', 'sha256'} and
             type(f['rights_ref']['sha256']) is str and
             SHA.fullmatch(f['rights_ref']['sha256']) and
             path_allowed(f['path'], [f['path']], source['deny_paths']),
             'First-time file source/path/content/type mismatch')
    return files


def _local_links(raw, path):
    """List linked local files; links never become part of an exact-file grant."""
    text = raw.decode('utf-8-sig')
    links = set()

    def add(target):
        target = unescape(target.strip())
        need('(' not in target and ')' not in target,
             'Unsupported document link syntax')
        need(not re.match(r'^[A-Za-z]:', target),
             'Absolute document link needs separate review')
        if not target:
            return
        scheme = urlsplit(target).scheme.casefold()
        if scheme:
            need(scheme in ('http', 'https', 'ftp', 'ftps', 'mailto', 'tel', 'data'),
                 'Unsupported document URI scheme')
            return
        if target.startswith('//'):
            return
        need('\\' not in target and '%' not in target and
             not any(ord(char) < 32 for char in target),
             'Unsupported local document link')
        target = target.split('#', 1)[0].split('?', 1)[0]
        if not target:
            return
        need(not target.startswith('/'), 'Absolute document link needs separate review')
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(path), target))
        need(resolved != '..' and not resolved.startswith('../'),
             'Document link escapes repository')
        if resolved != path:
            links.add(resolved)

    def markdown_target(body):
        body = body.strip()
        if body.startswith('<'):
            end = body.find('>')
            need(end > 1, 'Unsupported document link syntax')
            target, suffix = body[1:end], body[end + 1:].strip()
        else:
            parts = body.split(None, 1)
            need(bool(parts), 'Unsupported document link syntax')
            target, suffix = parts[0], parts[1].strip() if len(parts) == 2 else ''
        need(not suffix or re.fullmatch(r'"[^"\r\n]*"|\'[^\'\r\n]*\'|\([^()\r\n]*\)',
                                         suffix), 'Unsupported document link title')
        return target

    inline = list(LOCAL_LINK.finditer(text))
    need(text.count('](') == len(inline), 'Unsupported inline document link')
    for match in inline:
        add(markdown_target(match.group(1)))

    definitions = {}
    for match in REFERENCE_DEFINITION.finditer(text):
        label = ' '.join(match.group(1).split()).casefold()
        need(label not in definitions, 'Duplicate document reference label')
        definitions[label] = markdown_target(match.group(2))
        add(definitions[label])
    for match in REFERENCE_USE.finditer(text):
        label = ' '.join((match.group(2) or match.group(1)).split()).casefold()
        need(label in definitions, 'Unresolved document reference link')
    for line in text.splitlines():
        if '[!INCLUDE' in line.upper():
            need(LOCAL_LINK.search(line) or REFERENCE_USE.search(line),
                 'Unsupported document include syntax')

    need(not re.search(r'\bsrcset\s*=', text, re.I),
         'Unsupported HTML resource set')
    tags = list(HTML_TAG.finditer(text))
    candidates = re.finditer(r'<[A-Za-z][^>]*?\b(?:src|href)\s*=', text,
                             re.I | re.S)
    need(all(any(tag.start() <= candidate.start() < tag.end() for tag in tags)
             for candidate in candidates), 'Malformed HTML resource reference')
    for tag in tags:
        value = tag.group()
        matches = list(HTML_RESOURCE.finditer(value))
        need(len(matches) == len(re.findall(r'\b(?:src|href)\s*=', value, re.I)),
             'Unsupported HTML resource reference')
        for match in matches:
            add(next(part for part in match.groups() if part is not None))
    return sorted(links)


def validate_rights_record(rights, decision, approved_file, *,
                           raw=None, git_object=None, source_license_ref=None):
    """Require a pinned file review under exactly one supported rights profile."""
    if type(rights) is dict and rights.get('schema') == PROJECT_RIGHTS:
        need(set(rights) == {
            'schema', 'source_id', 'locator', 'revision', 'path', 'sha256',
            'git_blob', 'owner', 'subcategory', 'license_id', 'reviewer',
            'basis', 'evidence', 'project_publication_ref',
            'contribution_policy_ref', 'file_history_ref',
            'project_material_type', 'file_license_conflict',
            'unresolved_copied_or_adapted', 'third_party', 'generated',
            'binary', 'media', 'translation', 'linked_scope_exclusions',
            'attribution', 'obligations', 'identity'},
            'Incomplete project-authorized permissive evidence')
        verify_seal(rights)
        need(decision.get('decision') == 'APPROVE_EXACT_FILES' and
             type(decision.get('reviewer')) is str and
             bool(decision['reviewer'].strip()) and
             decision['license_id'] in ('MIT', 'Apache-2.0',
                                        'BSD-2-Clause', 'BSD-3-Clause') and
             all(rights[k] == decision[k] for k in
                 ('source_id', 'locator', 'revision', 'license_id', 'reviewer')) and
             rights['path'] == approved_file['path'] and
             rights['sha256'] == approved_file['sha256'] and
             rights['owner'] == decision['owner'] and
             rights['subcategory'] == decision['subcategory'] and
             rights['attribution'] == decision['attribution'] and
             rights['obligations'] == decision['obligations'] and
             rights['project_material_type'] == 'documentation' and
             approved_file['content_type'] == 'markdown' and
             approved_file['language'] == 'en' and
             approved_file['path'].lower().endswith('.md') and
             not any(part.casefold() in UNSUPPORTED_PATH_PARTS
                     for part in approved_file['path'].split('/')) and
             rights['file_license_conflict'] is False and
             rights['unresolved_copied_or_adapted'] is False and
             rights['third_party'] is False and rights['generated'] is False and
             rights['binary'] is False and rights['media'] is False and
             rights['translation'] is False and
             type(rights['basis']) is str and bool(rights['basis'].strip()),
             'Project publication/rights scope is unresolved')
        need(type(raw) is bytes and type(git_object) is str and
             re.fullmatch(r'[0-9a-f]{40}', git_object) and
             rights['git_blob'] == git_object and
             hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
             == git_object,
             'Project-authorized exact file bytes/Git blob differ')
        text = raw.decode('utf-8-sig')
        file_labels = [line.split(':', 1)[1].strip() for line in
                       re.findall(r'SPDX-License-Identifier:[^\r\n]*',
                                  text, re.I)]
        need(not file_labels or file_labels == [decision['license_id']],
             'Conflicting exact-file license')
        need(not PROVENANCE_SIGNAL.search(text),
             'Copied/adapted/third-party signal requires separate review')
        need(type(rights['linked_scope_exclusions']) is list and
             rights['linked_scope_exclusions'] == _local_links(raw, approved_file['path']),
             'Linked/include material lacks explicit scope exclusion')
        refs = rights['evidence']
        need(type(refs) is list and len(refs) >= 4 and
             all(type(ref) is dict and set(ref) == {'path', 'sha256'} and
                 type(ref['sha256']) is str and SHA.fullmatch(ref['sha256'])
                 for ref in refs) and
             refs == sorted(refs, key=lambda ref: ref['path']) and
             len({ref['path'] for ref in refs}) == len(refs) and
             type(source_license_ref) is dict and
             any(ref['path'] == source_license_ref.get('path') and
                 ref['sha256'] == source_license_ref.get('sha256') for ref in refs),
             'Root license alone cannot establish project-authorized scope')
        extra = [rights[k] for k in ('project_publication_ref',
                                    'contribution_policy_ref', 'file_history_ref')]
        need(all(ref in refs for ref in extra) and
             len({ref['path'] for ref in extra}) == 3 and
             all(ref['path'] != source_license_ref['path'] for ref in extra) and
             all(ref in refs for ref in decision['source_scope_evidence']),
             'Independent project publication/contribution/history evidence missing')
        return refs
    need(type(rights) is dict and set(rights) == {
        'schema', 'source_id', 'locator', 'revision', 'path', 'sha256',
        'owner', 'subcategory', 'license_id', 'first_party',
        'third_party', 'generated', 'reviewer', 'basis', 'evidence', 'identity'},
        'Incomplete per-file rights review')
    verify_seal(rights)
    need(rights['schema'] == RIGHTS and
         all(rights[k] == decision[k] for k in
             ('source_id', 'locator', 'revision', 'license_id', 'reviewer')) and
         rights['path'] == approved_file['path'] and
         rights['sha256'] == approved_file['sha256'] and
         rights['owner'] == decision['owner'] and
         rights['subcategory'] == decision['subcategory'] and
         rights['first_party'] is True and rights['third_party'] is False and
         rights['generated'] is False and
         type(rights['basis']) is str and bool(rights['basis'].strip()) and
         type(rights['evidence']) is list and bool(rights['evidence']),
         'Incomplete or contradictory per-file rights review')
    return rights['evidence']


def build(decision_path, output, *, expected_decision_sha256, cache,
          catalog, review, references, root=ROOT,
          base_registry=None, base_spec=None):
    root = Path(root).resolve()
    resolve = lambda p: Path(p).resolve() if Path(p).is_absolute() else local_path(root, str(p))
    decision_path, output, cache, catalog, review, references = map(
        resolve, (decision_path, output, cache, catalog, review, references))
    base_registry = resolve(base_registry or 'training/specifications/source_registry_v1.json')
    base_spec = resolve(base_spec or 'training/specifications/production_corpus_v1.json')
    inputs = [decision_path, catalog, references, base_registry, base_spec,
              review/'request.json', review/'complete.json', review/'evidence.jsonl',
              cache/'status.json']
    need(output.is_relative_to(root/'artifacts') and not output.is_relative_to(cache)
         and all(not p.is_relative_to(output) for p in inputs),
         'First-time output overlaps inputs or escapes artifacts')
    need(type(expected_decision_sha256) is str and
         SHA.fullmatch(expected_decision_sha256) and
         file_digest(decision_path) == expected_decision_sha256,
         'Externally pinned first-time decision missing or changed')
    decision = _read(decision_path, maximum=65536)
    registry, spec, catalog_value = (_read(p, maximum=16_777_216,
                                           canonical_required=False)
                                     for p in (base_registry, base_spec, catalog))
    validate(spec, registry, root)
    need(catalog_value['schema'] == registry['schema'], 'Wrong discovery catalog')
    sources = [s for s in catalog_value['sources']
               if s['source_id'] == decision['source_id']]
    need(len(sources) == 1 and sources[0]['status'] == 'ACCEPT',
         'Missing or unapproved discovery source')
    source = sources[0]
    need(not _already_registered(source, registry),
         'Existing source/alias requires its existing approval route')
    bundle = verify_bundle(review, root)
    need(bundle['complete'] and bundle['training_allowed'] is False and
         bundle['counts']['EVIDENCED'] == 0,
         'Only complete review-only evidence is accepted')
    review_request = _read(review/'request.json', canonical_required=False)
    for p in (catalog, references, base_registry, base_spec, cache/'status.json'):
        need(review_request['inputs'].get(p.relative_to(root).as_posix()) == file_digest(p),
             'Review evidence does not bind current source/cache/policy')
    caches = [x for x in cached_sources(cache, catalog_value, spec, root)
              if x[1]['source_id'] == source['source_id']]
    need(len(caches) == 1, 'Missing/ambiguous pinned source cache')
    directory, _, inventory, done = caches[0]
    need(review_request['inputs'].get(inventory.relative_to(root).as_posix()) ==
         file_digest(inventory), 'Review evidence inventory changed')
    inventory_rows = list(read_plan(inventory))
    rows = {r['path']: r for r in inventory_rows}
    need(len(rows) == len(inventory_rows) == sum(done['counts'].values()),
         'Duplicate or incomplete pinned inventory')
    approved = validate_decision_scope(decision, source, rows)
    heldout = holdout_groups(references, registry) | holdout_groups(references, catalog_value)
    need(source['split_group'] not in heldout, 'Retained protected source group')
    review_rows = {r['path']: r for r in read_plan(review/'evidence.jsonl')
                   if r['source_id'] == source['source_id']}
    need(len(review_rows) == len(rows), 'Missing or duplicate review decisions')
    evidence = []
    for ref in decision['source_scope_evidence']:
        _ref(root, ref, output)
        need(ref not in evidence, 'Duplicate source scope evidence')
        evidence.append(ref)
    license_ref = source['license_evidence']
    need(type(license_ref) is dict and
         any(r['path'] == license_ref['path'] and
             r['sha256'] == license_ref['sha256'] for r in evidence),
         'Pinned source license evidence omitted from reviewed scope')
    lic = _read(local_path(root, license_ref['path']))
    need(lic['url'] == 'https://api.github.com/repos/'+source['name']+
         '/license?ref='+source['revision'] and
         lic['body']['path'] == license_ref['upstream_path'] and
         lic['body']['license']['spdx_id'] == source['license_id'],
         'Source license metadata does not bind exact revision')
    need(license_ref['upstream_path'] in rows and
         rows[license_ref['upstream_path']]['sha256'] == license_ref['text_sha256'],
         'Pinned root license differs from verified cache')
    license_bytes = payload(directory, rows[license_ref['upstream_path']])
    need(license_bytes == base64.b64decode(
         ''.join(lic['body']['content'].split()), validate=True) and
         resolve_license(source['license_id'], license_bytes) == source['license_id'],
         'Root license bytes/terms differ from pinned metadata')
    output_files = []; prepared = []
    for f in approved:
        row = rows[f['path']]
        observed = review_rows[f['path']]
        need(observed['status'] == 'REVIEW' and
             all(observed[k] == row[k] for k in
                 ('source_id', 'revision', 'path', 'sha256', 'bytes')) and
             observed['cache_inventory_sha256'] == done['inventory_sha256'],
             'File is not exact reviewed, unexcluded material')
        raw = payload(directory, row)
        screened, _ = screening(f['path'], row['git_mode'], row['bytes'], raw,
                                source, spec)
        need(screened != 'EXCLUDED', 'Excluded/third-party/generated file')
        ref = f['rights_ref']
        rights = _read(_ref(root, ref, output), maximum=16384)
        for support in validate_rights_record(
                rights, decision, f, raw=raw, git_object=row['git_object'],
                source_license_ref=source['license_evidence']):
            _ref(root, support, output)
        evidence.append(ref)
        own = seal(dict(source_id=source['source_id'], revision=source['revision'],
                        path=f['path'], sha256=f['sha256'],
                        owner=decision['owner'], subcategory=decision['subcategory'],
                        decision_type='manual_review', rationale_ref=decision['identity'],
                        reviewer=decision['reviewer']))
        own_path = output/'ownership'/(hashlib.sha256(f['path'].encode()).hexdigest()+'.json')
        prepared.append((f, row, own_path, own))
    evidence.extend([dict(path=p.relative_to(root).as_posix(), sha256=file_digest(p))
                     for p in (decision_path, inventory, review/'complete.json')])
    need(len({x['path'] for x in evidence}) == len(evidence),
         'Duplicate or conflicting source obligation evidence')
    output.mkdir(parents=True, exist_ok=True)
    (output/'ownership').mkdir(exist_ok=True)

    def publish(path, value):
        encoded = canonical(value)
        if path.exists():
            need(path.read_bytes() == encoded, 'First-time replay differs: '+str(path))
        else:
            atomic_json(path, value)

    for _, _, path, own in prepared:
        publish(path, own)
    for f, row, path, _ in prepared:
        output_files.append(dict(path=f['path'], sha256=f['sha256'],
            bytes=f['bytes'], owner=decision['owner'],
            subcategory=decision['subcategory'], content_type=f['content_type'],
            license_id=decision['license_id'], tags=[],
            ownership_decision=ownership_decision(
                source_id=source['source_id'], revision=source['revision'],
                path=f['path'], sha256=f['sha256'], owner=decision['owner'],
                subcategory=decision['subcategory'], decision_type='manual_review',
                rationale_ref=decision['identity'], evidence_ref=dict(
                    path=path.relative_to(root).as_posix(), sha256=file_digest(path)))))
    manifest = seal(dict(source_id=source['source_id'],
                         revision=source['revision'], files=output_files))
    publish(output/'files.json', manifest)
    clearance = seal(dict(source_id=source['source_id'],
        revision=source['revision'], file_manifest_identity=manifest['identity'],
        decision='ACCEPT', evidence=evidence, attribution=decision['attribution'],
        obligations=decision['obligations'],
        governance_decision_identity=decision['identity']))
    publish(output/'clearance.json', clearance)
    cap = min(decision['cap_tokens'], sum(f['bytes']+1 for f in approved))
    s = copy.deepcopy(source)
    s.update(target_cap_tokens=cap, category_caps_tokens={decision['owner']:cap},
             cap_basis='Externally reviewed first-time exact-file ceiling',
             split=decision['split'], allow_paths=[f['path'] for f in approved],
             origin_verified=True, training_allowed_after_clearance=True,
             file_manifest=(output/'files.json').relative_to(root).as_posix(),
             clearance_record=(output/'clearance.json').relative_to(root).as_posix())
    new_registry = copy.deepcopy(registry)
    new_registry['sources'].append(s)
    new_registry = seal(new_registry)
    new_spec = copy.deepcopy(spec)
    for kind in ('repository_group', 'domain_group'):
        need(s[kind] not in new_spec['group_caps'][kind],
             'Existing concentration cap cannot be raised')
        new_spec['group_caps'][kind][s[kind]] = cap
    new_spec = seal(dict(new_spec, registry_identity=new_registry['identity']))
    validate(new_spec, new_registry, root)
    records = []
    for (f, row, _, _), approved_file in zip(prepared, output_files):
        record = seal(dict(source_id=source['source_id'],
            revision=source['revision'], **approved_file, clearance_status='ACCEPT',
            split=s['split'], split_group=s['split_group'],
            split_unit_kind='repository', allow_rule=f['path'],
            deny_rules_checked=s['deny_paths'], language=f['language'],
            license_evidence=evidence, attribution=decision['attribution'],
            source_clearance_identity=clearance['identity'], quality_review='ACCEPT',
            input_path=(directory/row['local_blob']).relative_to(root).as_posix()))
        validate_clearance(record, s, new_spec, root, approved_file)
        records.append(record)
    publish(output/'registry.json', new_registry)
    publish(output/'spec.json', new_spec)
    plan = b''.join(canonical(r)+b'\n' for r in sorted(records, key=item_key))
    if (output/'plan.jsonl').exists():
        need((output/'plan.jsonl').read_bytes() == plan, 'First-time plan replay differs')
    else:
        (output/'plan.jsonl').write_bytes(plan)
    artifacts = [output/name for name in
                 ('files.json', 'clearance.json', 'registry.json', 'spec.json', 'plan.jsonl')]
    artifacts += [x[2] for x in prepared]
    result = seal(dict(policy=POLICY, decision_identity=decision['identity'],
        source_id=source['source_id'], revision=source['revision'],
        files=len(approved), production_tokens=0, training_authorized=False,
        artifacts={p.relative_to(output).as_posix(): file_digest(p) for p in artifacts}))
    publish(output/'complete.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision', required=True, type=Path)
    parser.add_argument('--approved-decision-sha256', required=True)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--catalog', required=True, type=Path)
    parser.add_argument('--review', required=True, type=Path)
    parser.add_argument('--references', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = build(args.decision, args.output,
                       expected_decision_sha256=args.approved_decision_sha256,
                       cache=args.cache, catalog=args.catalog, review=args.review,
                       references=args.references)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(f'First-time clearance stopped: {type(exc).__name__}: {exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
