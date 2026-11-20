"""Review-only comparison of two complete, pinned source preparation caches.

This module creates technical evidence, never clearance, admission or authority.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit

from training.production_acquisition import MAX_ITEM_BYTES
from training.production_corpus import atomic_json, canonical, file_digest
from training.production_spec import (iter_manifest_files, need, path_allowed, seal,
                                      validate, verify_seal)
from training.source_admission import holdout_groups
from training.source_prepare import POLICY as CACHE_POLICY, ROOT, git, screening

POLICY = 'pinned-revision-inventory-comparison-v1'
SHA = re.compile(r'sha256:[0-9a-f]{64}\Z')
REVISION = re.compile(r'[0-9a-f]{40}\Z')
OID = re.compile(r'[0-9a-f]{40}\Z')
WINDOWS_RESERVED = {'con', 'prn', 'aux', 'nul'} | {
    f'{prefix}{n}' for prefix in ('com', 'lpt') for n in range(1, 10)}


def _read(path):
    value = json.loads(path.read_bytes(), object_pairs_hook=_unique)
    verify_seal(value)
    return value


def _unique(pairs):
    value = {}
    for key, item in pairs:
        need(key not in value, 'Duplicate JSON field in revision evidence')
        value[key] = item
    return value


def _safe_path(path):
    need(type(path) is str and path_allowed(path, ['**'], []) and
         unicodedata.normalize('NFC', path) == path and
         not any(ord(c) < 32 or ord(c) == 127 or 0xD800 <= ord(c) <= 0xDFFF
                 for c in path) and
         all(part.rstrip(' .') == part and
             part.split('.')[0].casefold() not in WINDOWS_RESERVED
             for part in path.split('/')),
         'Unsafe or ambiguous Git path')


def _tree(repo, revision):
    raw = git(repo, 'ls-tree', '-r', '-z', '--long', revision).stdout
    if not raw:
        return
    need(raw.endswith(b'\0'), 'Incomplete Git tree')
    for entry in raw[:-1].split(b'\0'):
        meta, encoded = entry.split(b'\t', 1)
        mode, kind, oid, size = meta.split()
        path = encoded.decode('utf-8', 'strict')
        _safe_path(path)
        need(mode in (b'100644', b'100755') and kind == b'blob' and
             OID.fullmatch(oid.decode()) and size.isdigit(),
             'Unsupported Git entry or file mode')
        yield mode.decode(), oid.decode(), int(size), path


def _inventory(directory, source, spec, expected_hash):
    """Reconcile every cached row with the complete pinned Git tree and raw blobs."""
    need(directory.is_dir() and directory.name == source['source_id'],
         'Missing or misbound source cache')
    request = _read(directory / 'request.json')
    done = _read(directory / 'complete.json')
    need(set(done) == {'request_identity', 'revision', 'inventory_sha256',
                       'counts', 'training_allowed', 'identity'} and
         type(done['counts']) is dict and
         all(type(n) is int and n > 0 for n in done['counts'].values()),
         'Malformed cache completion receipt')
    need(request.get('policy') == CACHE_POLICY and request.get('source') == source and
         request.get('filters') == spec['filters'] and
         request.get('max_bytes') == MAX_ITEM_BYTES and
         set(request) <= {'policy', 'source', 'filters', 'max_bytes',
                          'review_context', 'identity'} and
         (not request.get('review_context') or
          request['review_context'].get('training_allowed') is False),
         'Cache request differs from pinned source or screening policy')
    inv = directory / 'inventory.jsonl'
    actual_hash = file_digest(inv)
    need(actual_hash == expected_hash == done.get('inventory_sha256') and
         done.get('request_identity') == request['identity'] and
         done.get('revision') == source['revision'] and
         done.get('training_allowed') is False,
         'Missing, stale or contradictory cache completion')
    repo = directory / 'git'
    config = json.loads((directory / 'git_config.json').read_bytes(),
                        object_pairs_hook=_unique)
    need(file_digest(repo / 'config') == config.get('sha256'),
         'Git cache configuration changed')
    revision = source['revision']
    need(git(repo, 'rev-parse', revision + '^{commit}').stdout.decode().strip() == revision and
         git(repo, 'rev-parse', 'HEAD').stdout.decode().strip() == revision,
         'Cache does not contain the exact pinned commit')
    git(repo, 'fsck', '--full', '--strict', '--no-reflogs')
    inventory_bytes = inv.read_bytes()
    need(not inventory_bytes or inventory_bytes.endswith(b'\n'),
         'Truncated Git inventory JSONL')
    lines = inventory_bytes.splitlines()
    tree = list(_tree(repo, revision))
    need(len(lines) == len(tree) and
         len(lines) == sum(done.get('counts', {}).values()),
         'Incomplete Git inventory or receipt counts')
    rows = {}
    folded = set()
    counts = Counter()
    for line, (mode, oid, size, path) in zip(lines, tree):
        need(bool(line), 'Blank Git inventory row')
        row = json.loads(line, object_pairs_hook=_unique)
        need(line == canonical(row), 'Noncanonical Git inventory row')
        _safe_path(row.get('path'))
        need(row['path'] == path and path.casefold() not in folded,
             'Duplicate, reordered or case-folded Git path')
        folded.add(path.casefold())
        blob_ref = 'files/' + oid + '.blob'
        blob_path = directory / blob_ref
        need(blob_path.is_file() and not blob_path.is_symlink() and
             blob_path.resolve().is_relative_to(directory) and
             blob_path.stat().st_size == size and
             row.get('local_blob') == blob_ref,
             'Missing or misbound cached blob')
        git_sha = hashlib.sha1(b'blob ' + str(size).encode() + b'\0')
        raw_sha = hashlib.sha256()
        small = bytearray()
        with blob_path.open('rb') as stream:
            for block in iter(lambda: stream.read(65536), b''):
                git_sha.update(block)
                raw_sha.update(block)
                if size <= MAX_ITEM_BYTES:
                    small.extend(block)
        data = bytes(small)
        need(git_sha.hexdigest() == oid,
             'Cached blob differs from Git object')
        sha = 'sha256:' + raw_sha.hexdigest()
        status, reason = screening(path, mode, size,
                                   data, source, spec)
        detected = ('oversized' if size > MAX_ITEM_BYTES else
                    'binary' if b'\0' in data else 'text')
        if detected == 'text':
            try:
                data.decode('utf-8-sig')
            except UnicodeError:
                detected = 'non-UTF8/binary'
        need(row.get('source_id') == source['source_id'] and
             row.get('locator') == source['locator'] and
             row.get('revision') == revision and row.get('git_mode') == mode and
             row.get('git_object') == oid and row.get('bytes') == size and
             row.get('sha256') == sha and row.get('screening') == status and
             row.get('reason') == reason and row.get('training_allowed') is False and
             row.get('category') == source.get('category') and
             row.get('subcategory') == source.get('subcategory') and
             row.get('source_license_id') == source.get('license_id') and
             row.get('source_license_evidence') == source.get('license_evidence') and
             row.get('extension') == Path(path).suffix.lower() and
             row.get('detected_type') == detected,
             'Cached inventory row contradicts pinned Git tree or source policy')
        counts[status] += 1
        rows[path] = dict(path=path, revision=revision, git_object=oid,
                          sha256=sha, bytes=size, git_mode=mode,
                          screening=status, reason=reason)
    need(dict(counts) == done['counts'], 'Cache screening counts changed')
    return rows, request, done


def build(old_cache, new_cache, registry_path, spec_path, manifest_path,
          references_path, output, *, source_id, locator, old_revision,
          new_revision, expected_registry_sha256, expected_spec_sha256,
          expected_manifest_sha256, expected_references_sha256,
          expected_old_inventory_sha256, expected_new_inventory_sha256,
          expected_old_receipt_sha256, expected_new_receipt_sha256,
          root=ROOT):
    """Publish an immutable technical comparison, without changing any input."""
    root = Path(root).resolve()
    paths = [Path(p).resolve() for p in (old_cache, new_cache, registry_path,
             spec_path, manifest_path, references_path, output)]
    old_cache, new_cache, registry_path, spec_path, manifest_path, references_path, output = paths
    need(all(p.is_relative_to(root) for p in paths) and
         all(not p.is_relative_to(output) and not output.is_relative_to(p)
             for p in paths[:-1]) and old_cache != new_cache,
         'Comparison output overlaps or escapes immutable inputs')
    need(type(source_id) is str and re.fullmatch(r'[a-z0-9][a-z0-9-]*', source_id) and
         REVISION.fullmatch(old_revision or '') and REVISION.fullmatch(new_revision or '') and
         old_revision != new_revision,
         'Source and two distinct immutable revisions required')
    u = urlsplit(locator or '')
    need(u.scheme == 'https' and u.netloc == 'github.com' and not u.query and
         not u.fragment and re.fullmatch(r'/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', u.path) and
         locator == locator.rstrip('/'), 'Canonical GitHub locator required')
    pins = [(registry_path, expected_registry_sha256),
            (spec_path, expected_spec_sha256),
            (manifest_path, expected_manifest_sha256),
            (references_path, expected_references_sha256),
            (old_cache / 'complete.json', expected_old_receipt_sha256),
            (new_cache / 'complete.json', expected_new_receipt_sha256)]
    need(all(SHA.fullmatch(sha or '') and path.is_file() and
             file_digest(path) == sha for path, sha in pins),
         'Trusted input byte pin missing or stale')
    need(all(SHA.fullmatch(sha or '') for sha in
             (expected_old_inventory_sha256, expected_new_inventory_sha256)),
         'Trusted inventory byte pins required')
    registry = _read(registry_path)
    spec = _read(spec_path)
    validate(spec, registry, root)
    matches = [s for s in registry['sources'] if s['source_id'] == source_id or
               (s.get('locator') or '').casefold().rstrip('/') == locator.casefold()]
    need(len(matches) == 1 and matches[0]['source_id'] == source_id and
         matches[0]['locator'] == locator and matches[0]['revision'] == old_revision and
         matches[0]['status'] == 'ACCEPT' and matches[0].get('file_manifest') and
         matches[0].get('clearance_record'),
         'Ambiguous, uncleared or mismatched original source')
    source = matches[0]
    need(source['split_group'] not in holdout_groups(references_path, registry),
         'Protected source group conflict')
    manifest = _read(manifest_path)
    need((root / source['file_manifest']).resolve() == manifest_path and
         manifest.get('source_id') == source_id and
         manifest.get('revision') == old_revision,
         'Old approved manifest is not the registered source manifest')
    old_cache_request = _read(old_cache / 'request.json')
    old_cache_source = old_cache_request.get('source')
    stable = ('source_id', 'locator', 'revision', 'license_id',
              'license_evidence', 'repository_group', 'domain_group',
              'split_group', 'category', 'subcategory', 'deny_paths')
    need(type(old_cache_source) is dict and
         all(old_cache_source.get(field) == source.get(field) for field in stable),
         'Historical cache differs from registered source identity or original policy')
    new_source = dict(source, revision=new_revision)
    old_rows, old_request, old_done = _inventory(old_cache, old_cache_source, spec,
                                                 expected_old_inventory_sha256)
    new_rows, new_request, new_done = _inventory(new_cache, new_source, spec,
                                                 expected_new_inventory_sha256)
    approved = {}
    for f in iter_manifest_files(root, manifest):
        path = f['path']
        _safe_path(path)
        need(path not in approved and path in old_rows and
             f['sha256'] == old_rows[path]['sha256'] and
             f['bytes'] == old_rows[path]['bytes'],
             'Old approval absent from complete R1 inventory or changed')
        approved[path] = f
    need(approved, 'Empty historical approval scope')
    classes = {}
    for path in sorted(old_rows.keys() | new_rows.keys()):
        old, new = old_rows.get(path), new_rows.get(path)
        kind = ('REMOVED' if new is None else 'ADDED' if old is None else
                'UNCHANGED' if old['sha256'] == new['sha256'] else 'CHANGED')
        classes[path] = kind
    comparison = seal(dict(policy=POLICY, source_id=source_id, locator=locator,
        old_revision=old_revision, new_revision=new_revision,
        input_sha256={p.relative_to(root).as_posix(): sha for p, sha in pins},
        old_inventory_sha256=expected_old_inventory_sha256,
        new_inventory_sha256=expected_new_inventory_sha256,
        old_request_identity=old_request['identity'], new_request_identity=new_request['identity'],
        old_receipt_identity=old_done['identity'], new_receipt_identity=new_done['identity'],
        old_manifest_identity=manifest['identity'], old_registry_identity=registry['identity'],
        old_spec_identity=spec['identity'],
        files=[dict(path=path, classification=kind, old=old_rows.get(path),
                    new=new_rows.get(path)) for path, kind in classes.items()],
        counts=dict(sorted(Counter(classes.values()).items())),
        approval_scope=dict(previously_approved_unchanged=sorted(
            p for p in approved if classes[p] == 'UNCHANGED'),
            previously_approved_changed=sorted(p for p in approved if classes[p] == 'CHANGED'),
            previously_approved_removed=sorted(p for p in approved if classes[p] == 'REMOVED'),
            newly_added=sorted(p for p, kind in classes.items() if kind == 'ADDED'),
            previously_unapproved_in_r1=sorted(old_rows.keys() - approved.keys()),
            ineligible_in_r2=sorted(p for p, row in new_rows.items()
                                    if row['screening'] == 'EXCLUDED')),
        review_only=True, training_authorized=False, production_tokens=0))
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'comparison.json'
    complete_path = output / 'complete.json'
    def publish(path, value):
        encoded = canonical(value)
        if path.exists():
            need(path.read_bytes() == encoded, 'Conflicting comparison replay artifact')
        else:
            atomic_json(path, value)
    publish(report_path, comparison)
    complete = seal(dict(policy=POLICY, comparison_identity=comparison['identity'],
                         comparison_sha256=file_digest(report_path), review_only=True,
                         training_authorized=False, production_tokens=0))
    publish(complete_path, complete)
    return comparison
