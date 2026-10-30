"""Offline, exact-file approval materialization; no repository-wide inheritance.

Supported evidence profile: Python src packages explicitly listed by Hatch,
PEP 621 repository/license/author declarations, matching pinned root license,
and an exact per-file SPDX header. An alternative MIT-only profile verifies full
root terms, canonical project metadata and matching per-file attribution/SPDX
without requiring explicit Hatch package declarations. Ambiguity stays REVIEW.
This produces grants and a dry-run plan, never production tokens or training.
"""
from __future__ import annotations
import argparse
import ast
import base64
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 remains supported, with fail-closed review.
    tomllib = None

from training.clearance_evidence import verify_bundle
from training.corpus_prep import check_quality, QualityConfig
from training.experiments.representative_mini_corpus import normalize_structured_text
from training.production_acquisition import read_plan, item_key, validate_clearance, ProductionBuild
from training.production_corpus import atomic_json, atomic_replace, canonical, file_digest
from training.production_spec import seal, verify_seal, validate, need, local_path, path_allowed
from training.source_admission import cached_sources, holdout_groups
from training.source_prepare import ROOT, LICENSES, screening
from training.token_shards import _writer_lock
from training.license_evidence import resolve_license, package_license_matches

POLICY = 'package-or-mit-attributed-file-grants-v3'

MIT_TERMS = '''Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.'''


def mit_attribution(text):
    """Parse presentation-only MIT preamble; never strip or alter legal terms."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    notices = []
    headings = 0
    while lines:
        if lines[0] in ('MIT License', 'The MIT License'):
            headings += 1
        elif lines[0].startswith('Copyright (c) ') and lines[0][14:].strip():
            notices.append(lines[0])
        else:
            break
        lines.pop(0)
    if not notices or headings > 1 or ' '.join(' '.join(lines).split()) != ' '.join(MIT_TERMS.split()):
        return None
    return '; '.join(notices)


def mit_file_scope(source, rows, directory, root):
    """Alternative evidence only; selecting src/name does not license its files."""
    if source.get('license_id') != 'MIT' or tomllib is None:
        return None, 'MIT file profile unavailable'
    e = source.get('license_evidence', {})
    if e.get('upstream_path') != 'LICENSE' or not all(p in rows for p in ('LICENSE', 'pyproject.toml')):
        return None, 'MIT root/project evidence missing'
    if any('/' not in p and p != 'LICENSE' and re.search(r'license|copying|notice|copyright',p,re.I) for p in rows):
        return None, 'additional root rights scope requires review'
    for p in ('LICENSE', 'pyproject.toml'):
        if rows[p]['git_mode'] not in ('100644', '100755') or not rows[p]['bytes'] or rows[p]['bytes'] > 1048576:
            return None, 'MIT evidence nonregular or oversized'
    ep = local_path(root, e['path'])
    need(file_digest(ep) == e['sha256'], 'MIT license evidence corrupt')
    response = json.loads(ep.read_bytes()); verify_seal(response)
    need(source['locator'] == 'https://github.com/'+source['name'], 'MIT canonical source mismatch')
    need(response['url'] == 'https://api.github.com/repos/'+source['name']+'/license?ref='+source['revision'], 'MIT revision mismatch')
    body = response['body']
    raw = payload(directory, rows['LICENSE'])
    need(body['path'] == 'LICENSE' and body['license']['spdx_id'] == 'MIT'
         and raw == base64.b64decode(''.join(body['content'].split()), validate=True)
         and file_digest(directory/rows['LICENSE']['local_blob']) == e['text_sha256'], 'MIT root binding mismatch')
    attribution = mit_attribution(raw.decode('utf-8'))
    if attribution is None:
        return None, 'MIT root copyright/terms ambiguous or modified'
    try:
        meta = tomllib.loads(payload(directory, rows['pyproject.toml']).decode('utf-8'))
        project = meta['project']; name = project['name']
        authors = [a['name'] for a in project['authors']]
    except (ValueError, KeyError, TypeError): return None, 'MIT project provenance ambiguous'
    if project.get('license') not in ('MIT', {'text':'MIT'}, {'file':'LICENSE'}) or project.get('license-files') not in (None, ['LICENSE']):
        return None, 'MIT project license conflict'
    if source['locator'] not in project.get('urls', {}).values() or not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
        return None, 'MIT canonical project/path ambiguous'
    owners = [a for a in authors if isinstance(a, str) and a and re.search(r'(?<!\w)'+re.escape(a)+r'(?!\w)', attribution)]
    if not owners: return None, 'MIT attribution/project mismatch'
    return dict(profile='mit-attributed-file-v1', packages=['src/'+name], owners=owners,
                attribution=attribution, evidence=[dict(path=e['path'], sha256=e['sha256'])]+[
                    dict(path=(directory/rows[p]['local_blob']).relative_to(root).as_posix(), sha256=rows[p]['sha256'])
                    for p in ('LICENSE','pyproject.toml')]), None


def mit_file_reason(row, data, source, scope, all_paths):
    text = data.decode('utf-8-sig')
    lines = text.splitlines(keepends=True)
    header = []
    while lines and (lines[0].startswith('#') or not lines[0].strip()): header.append(lines.pop(0))
    copyrights = [l.rstrip('\r\n') for l in header if l.startswith('# SPDX-FileCopyrightText: ')]
    if not copyrights or any(not any(re.fullmatch(r'# SPDX-FileCopyrightText: \d{4}(?:-\d{4})? '+re.escape(owner)+r'(?: <[^<>\s]+@[^<>\s]+>)?', l) for owner in scope['owners']) for l in copyrights):
        return 'missing/conflicting file attribution'
    tags = re.findall(r'^# SPDX-License-Identifier: (.+)$', ''.join(header), re.M)
    if tags != ['MIT']: return 'missing/conflicting file MIT header'
    if re.search(r'generated|do not edit|auto.?generated', text, re.I): return 'generated material requires review'
    # Reuse the legacy exception/quality checks on a screening copy only.
    # Raw bytes, payload hashes and production content are never rewritten.
    rest = ''.join(l for l in header if not l.startswith(('# SPDX-FileCopyrightText:', '# SPDX-License-Identifier:'))) + ''.join(lines)
    return file_reason(row, ('# SPDX-License-Identifier: MIT\n'+rest).encode(), source, scope, all_paths)


def payload(directory, row):
    p = local_path(directory, row['local_blob'])
    need(p.is_relative_to(directory/'files') and p.is_file(), 'Invalid cache blob')
    need(p.stat().st_size == row['bytes'] and file_digest(p) == row['sha256'], 'Cache payload corrupt')
    digest = hashlib.sha1(f"blob {row['bytes']}\0".encode())
    with p.open('rb') as stream:
        while chunk := stream.read(65536): digest.update(chunk)
    need(digest.hexdigest() == row['git_object'], 'Git blob mismatch')
    return p.read_bytes() if row['bytes'] <= 1_048_576 else b''


def package_scope(source, rows, directory, root):
    """No build hooks/imports executed. Failure to prove this profile is review."""
    if tomllib is None:
        return None, 'TOML evidence parser unavailable; Python 3.11+ required for this profile'
    e = source.get('license_evidence')
    if not isinstance(e, dict) or source.get('license_id') not in LICENSES:
        return None, 'unsupported source license evidence'
    evidence_path = local_path(root, e['path'])
    need(file_digest(evidence_path) == e['sha256'], 'License evidence corrupt')
    response = json.loads(evidence_path.read_bytes()); verify_seal(response)
    license = response['body']; sid = source['license_id']
    need(response['url'] == 'https://api.github.com/repos/'+source['name']+'/license?ref='+source['revision'], 'License revision mismatch')
    retained_bytes = base64.b64decode(''.join(license['content'].split()), validate=True)
    need(resolve_license(license['license']['spdx_id'], retained_bytes) == sid and license['path'] == e['upstream_path'], 'License binding mismatch')
    root_license = rows.get(e['upstream_path']); project = rows.get('pyproject.toml')
    if root_license is None or project is None: return None, 'package/root license evidence missing'
    if '/' in root_license['path'] or root_license['git_mode'] not in ('100644','100755') or project['git_mode'] not in ('100644','100755'):
        return None, 'nonregular/ambiguous package evidence'
    if any(r['bytes'] is None or r['bytes'] > 1_048_576 for r in (root_license, project)):
        return None, 'package evidence exceeds document bound'
    license_bytes = payload(directory, root_license)
    need(license_bytes == base64.b64decode(''.join(license['content'].split()), validate=True)
         and root_license['sha256'] == e['text_sha256'], 'Pinned license differs from cache')
    try:
        metadata = tomllib.loads(payload(directory, project).decode('utf-8'))
        p = metadata['project']; packages = metadata['tool']['hatch']['build']['targets']['wheel']['packages']
    except (ValueError, KeyError, TypeError): return None, 'unsupported/ambiguous package declarations'
    if not package_license_matches(p, sid, root_license['path']):
        return None, 'package license does not explicitly match root evidence'
    if source['locator'] not in p.get('urls', {}).values(): return None, 'package does not declare canonical repository'
    authors = [a.get('name', '') for a in p.get('authors', [])]
    text = license_bytes.decode('utf-8')
    copyright_lines = [line for line in text.splitlines() if 'copyright' in line.casefold() and '(c)' in line.casefold()]
    if not authors or not any(name and name in line for name in authors for line in copyright_lines):
        return None, 'package authors not bound to license copyright'
    if not isinstance(packages, list) or not packages or any(not isinstance(p, str) or not re.fullmatch(r'src/[A-Za-z_][A-Za-z0-9_]*', p) for p in packages):
        return None, 'unbounded/unsupported package scope'
    return dict(packages=sorted(set(packages)), attribution='; '.join(copyright_lines),
                evidence=[dict(path=e['path'], sha256=e['sha256']),
                          dict(path=(directory/project['local_blob']).relative_to(root).as_posix(), sha256=project['sha256']),
                          dict(path=(directory/root_license['local_blob']).relative_to(root).as_posix(), sha256=root_license['sha256'])]), None


def file_reason(row, data, source, scope, all_paths):
    path = row['path']
    if not path.endswith('.py') or not any(path.startswith(p+'/') for p in scope['packages']):
        return 'outside explicit Python package file scope'
    if any(c.casefold() in ('tests','test','vendor','vendored','third_party','thirdparty','fixtures','generated') for c in PurePosixPath(path).parts):
        return 'third-party/test/generated scope not supported'
    if any(x in path for x in '*?[]'): return 'path cannot form an exact allow rule'
    parents = PurePosixPath(path).parents
    for name in all_paths:
        q = PurePosixPath(name)
        if q.parent != PurePosixPath('.') and q.parent in parents and any(t in q.name.casefold() for t in ('license','copying','notice')):
            return 'nested license/notice ambiguity'
    text = data.decode('utf-8-sig')
    headers = re.findall(r'^# SPDX-License-Identifier: (.+)$', text, re.M)
    if not text.startswith('# SPDX-License-Identifier: '+source['license_id']+'\n') or headers != [source['license_id']]:
        return 'missing/conflicting exact file SPDX header'
    if re.search(r'copied|adapted from|derived from|borrowed|third.party|copyright|all rights reserved|licensed under|SPDX-FileCopyrightText', text, re.I):
        return 'additional provenance/license signals require review'
    try: ast.parse(text)
    except (SyntaxError, ValueError): return 'unsupported Python syntax'
    if check_quality(normalize_structured_text(text), QualityConfig()) is not None:
        return 'existing quality screen did not pass'
    return None


def build(cache, catalog, evidence, registry, spec, plan, references, output, *, root=ROOT,
          cap=100000, validation_groups=1, resume=False):
    root=Path(root).resolve(); output=Path(output).resolve(); cache=Path(cache).resolve()
    need(output.is_relative_to(root) and not output.is_relative_to(cache) and not cache.is_relative_to(output), 'Output overlaps cache/outside root')
    inputs = [catalog, registry, spec, plan, references, Path(evidence)/'request.json', Path(evidence)/'complete.json', Path(evidence)/'evidence.jsonl']
    need(all(not Path(p).resolve().is_relative_to(output) for p in inputs), 'Output overlaps immutable inputs')
    need(type(cap) is int and 0 < cap <= 1000000, 'Cap must be 1..1000000 tokens per source')
    base=json.loads(Path(catalog).read_bytes()); verify_seal(base)
    reg=json.loads(Path(registry).read_bytes()); policy=json.loads(Path(spec).read_bytes()); validate(policy,reg,root)
    bundle=verify_bundle(evidence, root); need(bundle['complete'], 'Evidence incomplete')
    er=json.loads((Path(evidence)/'request.json').read_bytes())
    for p in (catalog, registry, spec, plan, references):
        need(er['inputs'].get(Path(p).resolve().relative_to(root).as_posix()) == file_digest(p), 'Evidence inputs differ')
    caches=list(cached_sources(cache,base,policy,root)); need(caches, 'No caches')
    groups=sorted({s['split_group'] for _,s,_,_ in caches})
    need(type(validation_groups) is int and 0 <= validation_groups <= len(groups), 'Invalid validation group count')
    ranked=sorted(groups,key=lambda g: hashlib.sha256(f"{policy['split']['seed']}:{g}".encode()).hexdigest())
    assignments={g:('validation' if g in ranked[:validation_groups] else 'train') for g in groups}
    heldout=holdout_groups(references,base)|holdout_groups(references,reg)
    request=seal(dict(policy=POLICY,inputs={Path(p).resolve().relative_to(root).as_posix():file_digest(p) for p in inputs},
                      cap=cap,validation_groups=validation_groups,assignments=assignments))
    output.mkdir(parents=True,exist_ok=True)
    if (output/'request.json').exists():
        need(resume and json.loads((output/'request.json').read_bytes())==request,'Resume request changed')
    else:
        need(not any(output.iterdir()),'Unknown output'); atomic_json(output/'request.json',request)
    prior=None
    if (output/'complete.json').exists():
        prior=json.loads((output/'complete.json').read_bytes()); verify_seal(prior)
        need(prior['request_identity']==request['identity'],'Receipt differs')
        for name,digest in prior['artifacts'].items(): need(file_digest(local_path(output,name))==digest,'Approval artifact corrupt')
    def publish(path, value):
        if prior is not None:
            need(Path(path).read_bytes()==canonical(value),'Approval replay differs')
        else:
            atomic_json(path,value)
    decisions={(r['source_id'],r['path']):r for r in read_plan(Path(evidence)/'evidence.jsonl')}
    records=list(read_plan(plan)); original=len(records); results=[]; source_caps={}
    known={s['source_id'] for s in reg['sources']}
    known_origins={str(s.get('locator')).casefold().rstrip('/') for s in reg['sources']}
    for directory,source,inventory,done in caches:
        need(source['source_id'] not in known and source['locator'].casefold().rstrip('/') not in known_origins,'Existing source/alias requires existing approval workflow')
        need(er['inputs'].get(inventory.relative_to(root).as_posix())==file_digest(inventory),'Unbound cache inventory')
        rows={r['path']:r for r in read_plan(inventory)}
        need(len(rows)==sum(done['counts'].values()),'Duplicate/incomplete inventory')
        scope,scope_reason=package_scope(source,rows,directory,root)
        alternative,alternative_reason=mit_file_scope(source,rows,directory,root) if source.get('license_id')=='MIT' else (None,None)
        if scope is None:
            scope,scope_reason=alternative,alternative_reason or scope_reason
        selected=[]
        for path,row in sorted(rows.items()):
            e=decisions.pop((source['source_id'],path)); verify_seal(e)
            need(all(e[k]==row[k] for k in ('source_id','revision','path','sha256','bytes')),'Evidence row mismatch')
            need(row['source_id']==source['source_id'] and row['revision']==source['revision'] and row['locator']==source['locator'],'Row origin mismatch')
            data=payload(directory,row) if row['local_blob'] else b''
            status,reason='REVIEW',scope_reason or 'not eligible review evidence'
            screened,why=screening(path,row['git_mode'],row['bytes'] or 0,data,source,policy)
            if e['status']=='EXCLUDED' or screened=='EXCLUDED': status,reason='EXCLUDED',e['reason'] if e['status']=='EXCLUDED' else why
            elif source['split_group'] in heldout: status,reason='EXCLUDED','retained holdout source group'
            elif e['status']=='REVIEW' and scope:
                checker=mit_file_reason if scope.get('profile')=='mit-attributed-file-v1' else file_reason
                reason=checker(row,data,source,scope,rows)
                if reason is not None and alternative:
                    reason=mit_file_reason(row,data,source,alternative,rows)
                if reason is None: status,reason='APPROVED','explicit file SPDX + pinned license + ownership evidence'; selected.append(row)
            results.append(dict(source_id=source['source_id'],path=path,revision=row['revision'],sha256=row['sha256'],status=status,reason=reason))
        if not selected: continue
        sid=source['source_id']; s=copy.deepcopy(source)
        bounded=min(cap,sum(r['bytes']+1 for r in selected))
        s.update(target_cap_tokens=bounded,category_caps_tokens={s['category']:bounded},cap_basis='Bounded materialization ceiling; not a yield estimate',
                 split=assignments[s['split_group']],allow_paths=sorted(r['path'] for r in selected),
                 training_allowed_after_clearance=True,origin_verified=True,
                 file_manifest=(output/(sid+'-files.json')).relative_to(root).as_posix(),
                 clearance_record=(output/(sid+'-clearance.json')).relative_to(root).as_posix())
        from training.category_ownership import decision as ownership_decision
        files=[]
        for r in selected:
            f=dict(path=r['path'],sha256=r['sha256'],bytes=r['bytes'],owner=s['category'],
                   subcategory=s['subcategory'],content_type='code',license_id=s['license_id'])
            f['ownership_decision']=ownership_decision(source_id=sid,revision=s['revision'],
                path=r['path'],sha256=r['sha256'],owner=f['owner'],subcategory=f['subcategory'],
                decision_type='structural',rationale_ref='approved-source-and-extension-v1',
                normalized_sha256='sha256:'+hashlib.sha256(normalize_structured_text(
                    payload(directory,r).decode('utf-8-sig')).encode('utf-8')).hexdigest())
            files.append(f)
        manifest=seal(dict(source_id=sid,revision=s['revision'],files=files)); publish(root/s['file_manifest'],manifest)
        evidence_refs=scope['evidence']+[dict(path=inventory.relative_to(root).as_posix(),sha256=done['inventory_sha256']),dict(path=(output/'request.json').relative_to(root).as_posix(),sha256=file_digest(output/'request.json'))]
        attribution=scope['attribution']+'; '+s['license_id']+'; preserve license and notices.'
        clearance=seal(dict(source_id=sid,revision=s['revision'],file_manifest_identity=manifest['identity'],decision='ACCEPT',evidence=evidence_refs,attribution=attribution))
        publish(root/s['clearance_record'],clearance)
        for row,f in zip(selected,files):
            record=seal(dict(source_id=sid,revision=s['revision'],**f,clearance_status='ACCEPT',split=s['split'],
                             split_group=s['split_group'],split_unit_kind='repository',allow_rule=row['path'],deny_rules_checked=s['deny_paths'],
                             language='python',license_evidence=evidence_refs,attribution=attribution,
                             source_clearance_identity=clearance['identity'],quality_review='ACCEPT',
                             input_path=(directory/row['local_blob']).relative_to(root).as_posix()))
            validate_clearance(record,s,policy,root,f); records.append(record)
        reg['sources'].append(s); known.add(sid); known_origins.add(s['locator'].casefold().rstrip('/')); source_caps[sid]=bounded
        for kind in ('repository_group','domain_group'):
            need(s[kind] not in policy['group_caps'][kind],'Existing concentration group cannot be raised')
            policy['group_caps'][kind][s[kind]]=bounded
    need(not decisions,'Extra evidence files')
    reg=seal(reg); policy=seal(dict(policy,registry_identity=reg['identity'])); validate(policy,reg,root)
    publish(output/'registry.json',reg); publish(output/'spec.json',policy)
    publish(output/'decisions.json',seal(dict(policy=POLICY,files=results)))
    plan_bytes=b''.join(canonical(r)+b'\n' for r in sorted(records,key=item_key))
    if prior is not None:
        need((output/'plan.jsonl').read_bytes()==plan_bytes,'Approval plan replay differs')
    else:
        (output/'plan.part').write_bytes(plan_bytes)
        atomic_replace(output/'plan.part',output/'plan.jsonl')
    if prior is None:
        with ProductionBuild(output/'dry_run',policy,reg,repository_root=root,reference_plan=references,
                             diagnostic=True,
                             expected_plan_sha256=file_digest(output/'plan.jsonl')) as b:
            preflight=b.dry_run(output/'plan.jsonl')
        # Database page allocation is diagnostic, not deterministic evidence.
        preflight.pop('storage_bytes',None)
        preflight.pop('identity',None)
        atomic_json(output/'dry_run.json',seal(preflight))
    counts=Counter(r['status'] for r in results)
    artifact_names=['request.json','registry.json','spec.json','plan.jsonl','decisions.json','dry_run.json']
    for sid in sorted(source_caps): artifact_names.extend((sid+'-files.json',sid+'-clearance.json'))
    result=seal(dict(request_identity=request['identity'],counts={k:counts[k] for k in ('APPROVED','EXCLUDED','REVIEW')},
                     new_plan_files=len(records)-original,carried_forward_files=original,source_caps=source_caps,assignments=assignments,
                     accepted_tokens_added=0,artifacts={name:file_digest(output/name) for name in artifact_names}))
    if prior: need(result==prior,'Approval replay differs')
    else: atomic_json(output/'complete.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    base=ROOT/'artifacts/discovered_sources_pipeline_v1'; approved=ROOT/'artifacts/production_batch_1_v1/admission'
    for key,default in dict(cache=base/'cache',catalog=ROOT/'artifacts/source_discovery_v1/catalog.json',evidence=base/'evidence',
                            registry=approved/'registry.json',spec=approved/'spec.json',plan=approved/'plan.jsonl',
                            references=ROOT/'artifacts/acquisition_readiness_gate_v1/heldout_references.jsonl').items():
        p.add_argument('--'+key,type=Path,default=default)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--cap',type=int,default=100000)
    p.add_argument('--validation-groups',type=int,default=1); p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    try:
        need(args.output.resolve().is_relative_to(ROOT/'artifacts'),'Output must stay under artifacts')
        need(not args.output.resolve().is_relative_to(args.cache.resolve()),'Output overlaps cache')
        with _writer_lock(args.output.with_name(args.output.name+'.approval')): result=build(**vars(args))
        print(json.dumps(result,sort_keys=True)); return 0
    except Exception as exc: print(f'Approval stopped: {type(exc).__name__}: {exc}'); return 2


if __name__=='__main__': raise SystemExit(main())
