"""Versioned, file-level evidence for one production budget owner.

The sealed file manifest is an approved input, not a decision inferred from a
locally mutable production item. The external plan digest pins its copy.
"""
from __future__ import annotations

import json
import re
import ast
from pathlib import Path

from training.production_corpus import canonical, file_digest, identity
from training.production_spec import QUOTAS, need, local_path, seal, verify_seal


POLICY = dict(schema='yug-category-ownership-v1',
              decision_types=['structural', 'source_evidence', 'manual_review'],
              structural_extensions={'python': '.py'})
POLICY_IDENTITY = identity(POLICY)
OWNER_ALIASES = frozenset(('category', 'categories', 'owners', 'owner_id',
    'budget_owner', 'budget_category', 'accounting_owner', 'accounting_category',
    'allocation_owner', 'allocation_category', 'category_owner', 'owner_category',
    'top_level_category'))
TAG_PATTERN = re.compile(r'[a-z][a-z0-9_]{0,63}\Z')
APPROVAL_SCHEMA = 'yug-ownership-plan-approval-v1'
APPROVAL_SCOPE = 'production_admission'
SHA256_PATTERN = re.compile(r'sha256:[0-9a-f]{64}\Z')


def approval(*, plan_sha256, approver, approved_at):
    """Represent an external approval decision; its digest must be pinned separately."""
    need(type(plan_sha256) is str and SHA256_PATTERN.fullmatch(plan_sha256),
         'Invalid approved plan identity')
    need(type(approver) is str and bool(approver.strip()) and
         type(approved_at) is str and bool(approved_at.strip()),
         'Incomplete ownership approval authority')
    return seal(dict(schema=APPROVAL_SCHEMA, plan_sha256=plan_sha256,
        ownership_policy_identity=POLICY_IDENTITY, approver=approver,
        scope=APPROVAL_SCOPE, approved_at=approved_at))


def verify_approval(path, expected_sha256, plan_sha256, build_directory):
    need(type(expected_sha256) is str and SHA256_PATTERN.fullmatch(expected_sha256),
         'Missing external ownership approval identity')
    need(type(plan_sha256) is str and SHA256_PATTERN.fullmatch(plan_sha256),
         'Missing external approved plan identity')
    need(path is not None, 'Missing external ownership approval artifact')
    artifact=Path(path).resolve()
    need(not artifact.is_relative_to(Path(build_directory).resolve()),
         'Ownership approval must be external to mutable build state')
    need(artifact.is_file() and artifact.stat().st_size <= 16384 and
         file_digest(artifact)==expected_sha256,
         'External ownership approval artifact missing or changed')
    try:
        value=json.loads(artifact.read_bytes(), object_pairs_hook=_unique_pairs)
    except (ValueError, UnicodeError) as exc:
        raise ValueError('Malformed external ownership approval artifact') from exc
    need(type(value) is dict and set(value)=={
        'schema','plan_sha256','ownership_policy_identity','approver',
        'scope','approved_at','identity'}, 'Malformed ownership approval fields')
    verify_seal(value)
    need(value['schema']==APPROVAL_SCHEMA and value['scope']==APPROVAL_SCOPE and
         value['ownership_policy_identity']==POLICY_IDENTITY and
         value['plan_sha256']==plan_sha256 and
         type(value['approver']) is str and bool(value['approver'].strip()) and
         type(value['approved_at']) is str and bool(value['approved_at'].strip()),
         'Ownership approval does not authorize this plan and policy')
    return value['identity']


def _unique_pairs(pairs):
    value={}
    for key,item in pairs:
        need(key not in value, 'Duplicate ownership approval field')
        value[key]=item
    return value


def verify_structure(raw, record):
    """Check the actual bounded payload for structural decisions, not metadata."""
    if record['ownership_decision']['decision_type']!='structural':
        return
    need(record['subcategory']=='python' and record['language']=='python',
         'Unsupported structural language; source/manual evidence required')
    need(type(raw) is bytes and len(raw)<=1024*1024,
         'Unbounded structural payload')
    try:
        text=raw.decode('utf-8-sig')
        tree=ast.parse(text)
    except (UnicodeError, SyntaxError, ValueError, RecursionError, MemoryError):
        need(False, 'Invalid structural Python source')
    need(any(isinstance(node,(ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
         ast.Import, ast.ImportFrom, ast.Assign, ast.AugAssign,
         ast.For, ast.While, ast.If, ast.With, ast.Try)) or
         (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call))
         for node in tree.body), 'Structural Python source contains no code')


def canonical_tags(value):
    need(type(value) is list and len(value) <= 16 and
         all(type(tag) is str and TAG_PATTERN.fullmatch(tag) for tag in value) and
         value == sorted(set(value)), 'Invalid descriptive tags')
    return value


def decision(*, source_id, revision, path, sha256, owner, subcategory,
             decision_type, rationale_ref, tags=None, normalized_sha256=None,
             evidence_ref=None):
    """Construct a decision from approved file evidence, never from a plan row."""
    data=dict(policy_identity=POLICY_IDENTITY,source_id=source_id,revision=revision,
              path=path,sha256=sha256,owner=owner,subcategory=subcategory,
              decision_type=decision_type,rationale_ref=rationale_ref,
              tags=canonical_tags([] if tags is None else tags),
              normalized_sha256=normalized_sha256,evidence_ref=evidence_ref)
    return seal(data)


def verify(record, approved_file, source, root):
    need(not (OWNER_ALIASES & record.keys()) and not (OWNER_ALIASES & approved_file.keys()),
         'Ambiguous budget-owner alias')
    need(type(record.get('owner')) is str and record['owner'] in QUOTAS,
         'Missing or unsupported single budget owner')
    need(type(record.get('subcategory')) is str and bool(record['subcategory']),
         'Missing item subcategory')
    need(record.get('subcategory')==approved_file.get('subcategory'),
         'Item subcategory differs from approved file evidence')
    tags=canonical_tags(record.get('tags',[]))
    need(tags==canonical_tags(approved_file.get('tags',[])),
         'Descriptive tags differ from approved file evidence')
    approved=approved_file.get('ownership_decision')
    need(type(approved) is dict, 'Missing approved file ownership decision')
    need(not (OWNER_ALIASES & approved.keys()), 'Ambiguous ownership decision alias')
    verify_seal(approved)
    need(record.get('ownership_decision')==approved,
         'Item ownership decision differs from approved file evidence')
    need(approved.get('policy_identity')==POLICY_IDENTITY,
         'Unsupported ownership decision policy')
    for key in ('source_id','revision','path','sha256','owner','subcategory'):
        expected=source[key] if key in ('source_id','revision') else record[key]
        need(approved.get(key)==expected, 'Ownership decision '+key+' mismatch')
    need(approved.get('tags')==tags and
         approved.get('decision_type') in POLICY['decision_types'] and
         type(approved.get('rationale_ref')) is str and bool(approved['rationale_ref']),
         'Incomplete ownership decision')
    nsha=approved.get('normalized_sha256')
    need(nsha is None or (type(nsha) is str and re.fullmatch(r'sha256:[0-9a-f]{64}',nsha)),
         'Invalid normalized ownership identity')
    if approved['decision_type']=='structural':
        extension=POLICY['structural_extensions'].get(record['subcategory'])
        need(extension is not None and record['path'].lower().endswith(extension) and
             record['owner']=='code' and record['content_type']=='code' and
             source['category']=='code' and source['subcategory']==record['subcategory'] and
             approved['rationale_ref']=='approved-source-and-extension-v1' and
             approved.get('evidence_ref') is None,
             'Structural ownership evidence insufficient')
    else:
        ref=approved.get('evidence_ref')
        need(type(ref) is dict and set(ref)=={'path','sha256'},
             'Missing source/manual ownership evidence reference')
        path=local_path(root,ref['path'])
        need(path.stat().st_size<=16384 and file_digest(path)==ref['sha256'],
             'Ownership evidence corrupt')
        evidence=json.loads(path.read_bytes());verify_seal(evidence)
        need(not (OWNER_ALIASES & evidence.keys()), 'Ambiguous ownership evidence alias')
        for key in ('source_id','revision','path','sha256','owner','subcategory'):
            need(evidence.get(key)==approved[key], 'Ownership evidence '+key+' mismatch')
        need(evidence.get('decision_type')==approved['decision_type'] and
             evidence.get('rationale_ref')==approved['rationale_ref'] and
             type(evidence.get('reviewer')) is str and bool(evidence['reviewer']),
             'Ownership evidence incomplete')
    return approved['identity']
