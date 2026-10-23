"""Bounded GitHub metadata/license discovery. Never fetches corpus payloads.

License acceptance is for source discovery/cache review only, not file clearance.
Resume reuses sealed responses, including the first resolved immutable commit.
Only explicit --fetch permits HTTP; offline replay is the default.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import urllib.request

from training.production_corpus import atomic_json, canonical, file_digest
from training.production_spec import seal, verify_seal, need
from training.source_prepare import ROOT, LICENSES
from training.token_shards import _writer_lock
from training.license_evidence import resolve_license
from training.conditional_license import CONDITIONAL, pending

VERSION = 'bounded-source-discovery-v1'
MAX_RESPONSE = 524288
PILOT = [dict(locator='https://github.com/'+repo, branch='main', category='code', subcategory='python')
         for repo in ('python-attrs/attrs', 'pallets/click', 'pytest-dev/pytest')]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('Metadata redirects are not allowed')


def fetch_json(url):
    need(re.fullmatch(r'https://api\.github\.com/repos/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/(git/ref/heads/[A-Za-z0-9_.-]+|license\?ref=[0-9a-f]{40})', url), 'Non-metadata URL rejected')
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
    license_id = license_id if conditional else resolve_license(license_id, data)
    base['license_id'] = license_id
    if license_id not in LICENSES | CONDITIONAL.keys():
        return dict(base, status='REVIEW', reason='retained license text unresolved; no filename or cross-file inference'), None
    sid = 'gh-'+repo.lower().replace('/', '-')
    evidence = dict(path=(directory/'license.json').relative_to(root).as_posix(),
                    sha256=file_digest(directory/'license.json'), upstream_path=path,
                    text_sha256='sha256:'+hashlib.sha256(data).hexdigest(),
                    url=license['html_url'], scope='repository license evidence; no per-file clearance')
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
