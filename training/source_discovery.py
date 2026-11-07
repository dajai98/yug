"""Bounded GitHub metadata/license discovery. Never fetches corpus payloads.

License acceptance is for source discovery/cache review only, not file clearance.
Resume reuses sealed responses, including the first resolved immutable commit.
Only explicit --fetch permits HTTP; offline replay is the default.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from training.conditional_license import CONDITIONAL, pending
from training.license_evidence import resolve_license
from training.production_corpus import atomic_json, canonical, file_digest
from training.production_spec import need, seal, verify_seal
from training.source_prepare import LICENSES, ROOT
from training.token_shards import _writer_lock

VERSION = 'bounded-source-discovery-v1'
MAX_RESPONSE = 524288
CANONICAL_LICENSES = frozenset(('LICENSE', 'LICENSE.TXT', 'LICENSE.MD',
                                'COPYING', 'COPYING.TXT', 'COPYING.MD'))
CANONICAL_COMPANIONS = frozenset(('NOTICE', 'COPYRIGHT'))
PILOT = [dict(locator='https://github.com/'+repo, branch='main', category='code', subcategory='python')
         for repo in ('python-attrs/attrs', 'pallets/click', 'pytest-dev/pytest')]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('Metadata redirects are not allowed')


def fetch_json(url):
    pattern = (r'https://api\.github\.com/repos/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/'
               r'(git/ref/heads/[A-Za-z0-9_.-]+|license\?ref=[0-9a-f]{40}|'
               r'git/(?:commits|trees|blobs)/[0-9a-f]{40})')
    need(re.fullmatch(pattern, url), 'Non-metadata URL rejected')
    request = urllib.request.Request(url, headers={'Accept': 'application/vnd.github+json',
                                                  'User-Agent': 'YUG-source-discovery-v1'})
    with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
        data = response.read(MAX_RESPONSE+1)
    need(len(data) <= MAX_RESPONSE, 'Metadata response exceeds bound')
    return json.loads(data)


def response(directory, name, url, fetcher):
    path = directory/(name+'.json')
    if path.exists():
        retained = json.loads(path.read_bytes()); verify_seal(retained)
        need(retained['url'] == url, 'Retained response URL mismatch')
        return retained['body']
    need(fetcher is not None, 'Metadata missing; explicit --fetch required')
    body = fetcher(url)
    need(len(canonical(body)) <= MAX_RESPONSE, 'Metadata response exceeds bound')
    if '/git/ref/heads/' in url:
        need(isinstance(body, dict) and set(body) <= {'ref', 'node_id', 'url', 'object'},
             'Unexpected non-ref metadata fields')
    atomic_json(path, seal(dict(url=url, body=body)))
    return body


def _canonical_fallback(directory, root, api, locator, revision, primary, fetcher):
    """Retain only root license/notice blobs from the pinned Git tree.

    This is source-level review evidence. It cannot grant owned file scope or use.
    """
    sid = 'gh-' + locator.removeprefix('https://github.com/').lower().replace('/', '-')
    commit = response(directory, 'pinned_commit', api+'/git/commits/'+revision, fetcher)
    tree_sha = (commit.get('tree') or {}).get('sha')
    need(commit.get('sha') == revision and isinstance(tree_sha, str) and
         re.fullmatch('[0-9a-f]{40}', tree_sha), 'Pinned commit/tree mismatch')
    tree = response(directory, 'root_tree', api+'/git/trees/'+tree_sha, fetcher)
    need(tree.get('sha') == tree_sha and tree.get('truncated') is False and
         isinstance(tree.get('tree'), list), 'Incomplete or mismatched pinned root tree')
    entries = {}
    for row in tree['tree']:
        path = row.get('path')
        if (not isinstance(path, str) or
                path.upper() not in CANONICAL_LICENSES | CANONICAL_COMPANIONS):
            continue
        need(path not in entries and row.get('type') == 'blob' and
             row.get('mode') in ('100644', '100755') and
             isinstance(row.get('sha'), str) and re.fullmatch('[0-9a-f]{40}', row['sha']),
             'Ambiguous canonical root entry')
        entries[path] = row['sha']
    need(len(entries) <= len(CANONICAL_LICENSES | CANONICAL_COMPANIONS),
         'Too many canonical root entries')
    files = []
    for path, blob_sha in sorted(entries.items()):
        blob = response(directory, 'canonical_'+path.lower().replace('.', '_'),
                        api+'/git/blobs/'+blob_sha, fetcher)
        need(blob.get('sha') == blob_sha and blob.get('encoding') == 'base64' and
             type(blob.get('size')) is int and 0 < blob['size'] <= 131072,
             'Invalid canonical Git blob response')
        data = base64.b64decode(''.join(blob['content'].split()), validate=True)
        need(len(data) == blob['size'] and
             hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest() == blob_sha,
             'Canonical Git blob mismatch')
        text = data.decode('utf-8')
        files.append(dict(source_id=sid, locator=locator, revision=revision,
                          path=path, git_blob=blob_sha,
                          sha256='sha256:'+hashlib.sha256(data).hexdigest(),
                          text=text, role=('LICENSE' if path.upper() in CANONICAL_LICENSES
                                           else 'COMPANION'),
                          resolver_method='pinned-root-git-blob-v1'))
    primary_row = next((row for row in files if row['path'] == primary['path']), None)
    if primary_row is not None:
        need(primary_row['git_blob'] == primary['sha'] and
             primary_row['sha256'] == 'sha256:'+hashlib.sha256(primary['data']).hexdigest(),
             'API license and pinned root blob disagree')
    matches = [resolve_license('NOASSERTION', row['text'].encode('utf-8'))
               for row in files if row['role'] == 'LICENSE']
    resolved = (matches[0] if primary_row is not None and matches and
                all(value == matches[0] for value in matches) and
                matches[0] in LICENSES else None)
    record = seal(dict(policy='pinned-canonical-license-evidence-v1', source_id=sid,
                       locator=locator, revision=revision, tree_sha=tree_sha,
                       files=files, resolved_license=resolved, authority='NONE'))
    path = directory/'canonical_evidence.json'
    if path.exists():
        need(json.loads(path.read_bytes()) == record, 'Canonical evidence changed')
    else:
        atomic_json(path, record)
    reference = dict(path=path.relative_to(root).as_posix(), sha256=file_digest(path),
                     identity=record['identity'])
    return resolved, reference, any(row['role'] == 'COMPANION' for row in files)


def discover(seed, directory, root, spec, fetcher):
    locator = seed['locator']; repo = locator.removeprefix('https://github.com/')
    api = 'https://api.github.com/repos/'+repo
    commit = response(directory, 'commit', api+'/git/ref/heads/'+seed['branch'], fetcher)
    need(commit.get('ref') == 'refs/heads/'+seed['branch'] and
         (commit.get('object') or {}).get('type') == 'commit', 'Invalid Git ref metadata')
    revision = commit['object'].get('sha')
    need(isinstance(revision, str) and re.fullmatch('[0-9a-f]{40}', revision), 'Invalid pinned commit')
    license = response(directory, 'license', api+'/license?ref='+revision, fetcher)
    license_id = (license.get('license') or {}).get('spdx_id')
    base = dict(locator=locator, revision=revision, license_id=license_id,
                category=seed['category'], subcategory=seed['subcategory'])
    if license_id and license_id != 'NOASSERTION' and license_id not in LICENSES | CONDITIONAL.keys():
        return dict(base, status='REJECTED', reason='license outside existing allowlist'), None
    need(license.get('encoding') == 'base64' and license.get('type') == 'file', 'Invalid license payload')
    data = base64.b64decode(''.join(license['content'].split()), validate=True)
    need(0 < len(data) <= 131072 and license.get('size') == len(data), 'Invalid license size')
    data.decode('utf-8')
    need(hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest() == license.get('sha'), 'License Git blob hash mismatch')
    path = license.get('path')
    need(isinstance(path, str) and re.fullmatch(r'[A-Za-z0-9_.-]+', path), 'Non-root/ambiguous license scope')
    need(license.get('html_url') == locator+'/blob/'+revision+'/'+path, 'License evidence is not revision-bound')
    conditional = license_id in CONDITIONAL
    canonical_evidence = None
    companion_review = False
    if license_id in (None, '', 'NOASSERTION'):
        try:
            license_id, canonical_evidence, companion_review = _canonical_fallback(
                directory, root, api, locator, revision,
                dict(path=path, sha=license['sha'], data=data), fetcher)
        except urllib.error.URLError:
            return dict(base, status='REVIEW', reason='pinned canonical evidence unavailable'), None
    elif not conditional:
        license_id = resolve_license(license_id, data)
    base['license_id'] = license_id
    if license_id not in LICENSES | CONDITIONAL.keys():
        return dict(base, status='REVIEW',
                    reason='pinned canonical license text unresolved or conflicting',
                    canonical_evidence=canonical_evidence), None
    if companion_review:
        return dict(base, status='REVIEW',
                    reason='pinned NOTICE/COPYRIGHT companion requires obligation review',
                    canonical_evidence=canonical_evidence), None
    sid = 'gh-'+repo.lower().replace('/', '-')
    evidence = dict(path=(directory/'license.json').relative_to(root).as_posix(),
                    sha256=file_digest(directory/'license.json'), upstream_path=path,
                    text_sha256='sha256:'+hashlib.sha256(data).hexdigest(),
                    url=license['html_url'], scope='repository license evidence; no per-file clearance')
    if canonical_evidence is not None:
        evidence['canonical_evidence'] = canonical_evidence
    source = dict(source_id=sid, name=repo, **base, status='ACCEPT',
                  status_authority=VERSION, status_reason='Metadata/license accepted for review-cache preparation only',
                  license_evidence=evidence, license_text_sha256=evidence['text_sha256'],
                  allowed_categories=[seed['category']], target_cap_tokens=0,
                  category_caps_tokens={seed['category']: 0}, cap_basis='No production allocation authorized',
                  repository_group=sid, domain_group=sid, split_group=sid, split='unassigned',
                  allow_paths=[], deny_paths=spec['filters']['deny_paths'],
                  file_manifest=None, clearance_record=None, origin_verified=False,
                  provenance=dict(discovery_version=VERSION, metadata_only=True),
                  training_allowed_after_clearance=False)
    if conditional:
        source.update(status='CONDITIONAL',status_reason='Conditional metadata claim only; no file approval',conditional_license=pending(license_id))
        return dict(base,source_id=sid,status='REVIEW',reason='CONDITIONAL; source-specific rights and obligation approval required',license_evidence=evidence),source
    return dict(base, source_id=sid, status='ACCEPTED', reason='Pinned source-level license evidence only',
                license_evidence=evidence), source


def build(registry, spec, output, *, seeds=None, root=ROOT, resume=False, fetcher=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    need(output.is_relative_to(root), 'Output must remain under repository root')
    for p in (registry, spec):
        need(not Path(p).resolve().is_relative_to(output), 'Output overlaps immutable inputs')
    catalog = json.loads(Path(registry).read_bytes()); verify_seal(catalog)
    policy = json.loads(Path(spec).read_bytes()); verify_seal(policy)
    seeds = sorted(PILOT if seeds is None else seeds, key=lambda s: s['locator'])
    need(0 < len(seeds) <= 10, 'Discovery requires 1..10 explicit seed repositories')
    categories = {c['id']: c['subcategories'] for c in policy['categories']}
    seen = set()
    for seed in seeds:
        url = seed['locator']
        need(re.fullmatch(r'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', url), 'Invalid discovery locator')
        need(re.fullmatch(r'[A-Za-z0-9_.-]+', seed.get('branch', '')), 'Explicit simple branch required')
        need(url.lower() not in seen, 'Duplicate seed locator'); seen.add(url.lower())
        need(seed['subcategory'] in categories.get(seed['category'], []), 'Invalid category route')
    request = seal(dict(version=VERSION, registry_sha256=file_digest(registry), spec_sha256=file_digest(spec),
                        seeds=seeds, license_allowlist=sorted(LICENSES)))
    output.mkdir(parents=True, exist_ok=True)
    if (output/'request.json').exists():
        need(resume, 'Existing discovery requires --resume')
        need(json.loads((output/'request.json').read_bytes()) == request, 'Discovery inputs changed')
    else:
        need(not any(output.iterdir()), 'Unknown discovery output')
        atomic_json(output/'request.json', request)
    prior = None
    if (output/'complete.json').exists():
        prior = json.loads((output/'complete.json').read_bytes()); verify_seal(prior)
        need(prior['request_identity'] == request['identity'], 'Receipt mismatch')
        for name, digest in prior['artifacts'].items():
            p = (output/name).resolve()
            need(p.is_relative_to(output) and file_digest(p) == digest, 'Discovery artifact corrupt')
    known = {str(s.get('locator')).lower().rstrip('/') for s in catalog['sources']}
    ids = {s['source_id'] for s in catalog['sources']}
    decisions = []; sources = []
    for index, seed in enumerate(seeds):
        directory = output/f'source-{index:02d}'; directory.mkdir(exist_ok=True)
        sid = 'gh-'+seed['locator'].removeprefix('https://github.com/').lower().replace('/', '-')
        if seed['locator'].lower() in known or sid in ids:
            decision, source = dict(seed, status='REJECTED', reason='already registered source'), None
        else:
            decision, source = discover(seed, directory, root, policy, None if prior else fetcher)
        decisions.append(decision)
        if source is not None:
            sources.append(source); ids.add(source['source_id'])
    results = seal(dict(version=VERSION, sources=decisions, training_allowed=False))
    candidate_catalog = seal(dict(catalog, sources=sources))
    if prior:
        need(json.loads((output/'decisions.json').read_bytes()) == results and
             json.loads((output/'catalog.json').read_bytes()) == candidate_catalog, 'Discovery replay mismatch')
        return prior
    atomic_json(output/'decisions.json', results); atomic_json(output/'catalog.json', candidate_catalog)
    counts = Counter(d['status'] for d in decisions)
    artifacts = {p.relative_to(output).as_posix(): file_digest(p) for p in sorted(output.rglob('*.json'))
                 if p.name != 'complete.json'}
    result = seal(dict(version=VERSION, request_identity=request['identity'], candidate_count=len(seeds),
                       counts={k: counts[k] for k in ('ACCEPTED', 'REJECTED', 'REVIEW')},
                       artifacts=artifacts, training_allowed=False))
    atomic_json(output/'complete.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=ROOT/'training/specifications/source_registry_v1.json')
    parser.add_argument('--spec', type=Path, default=ROOT/'training/specifications/production_corpus_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=Path, help='JSON array of at most ten locator/branch/category/subcategory records')
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    try:
        need(args.output.resolve().is_relative_to(ROOT/'artifacts'), 'Output must stay under artifacts')
        seeds = json.loads(args.seeds.read_bytes()) if args.seeds else None
        for path in (args.registry, args.spec, args.seeds):
            if path: need(not path.resolve().is_relative_to(args.output.resolve()), 'Output overlaps inputs')
        with _writer_lock(args.output.with_name(args.output.name+'.discovery')):
            result = build(args.registry, args.spec, args.output, seeds=seeds, resume=args.resume,
                           fetcher=fetch_json if args.fetch else None)
        print(json.dumps(result, sort_keys=True)); return 0
    except Exception as exc:
        print(f'Discovery stopped: {type(exc).__name__}: {exc}'); return 2


if __name__ == '__main__': raise SystemExit(main())
