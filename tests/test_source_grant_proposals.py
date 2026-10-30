"""Proposal-only bridge checks. Synthetic caches cannot grant production use."""
import json
import shutil

import pytest

from tests.test_source_admission import inputs
from tests.test_clearance_evidence import review_only_inputs
from training import clearance_evidence
from training.production_acquisition import read_plan
from training.production_corpus import canonical, file_digest
from training.production_spec import SpecificationError, seal
from training.source_grant_proposals import compile_proposals, _file_scope


@pytest.fixture
def proposal_inputs(inputs):
    args, _ = review_only_inputs(inputs)
    root = inputs['root']
    target = root/'training/specifications'
    target.mkdir(parents=True)
    for name, key in [('source_registry_v1.json', 'root_registry'),
                      ('production_corpus_v1.json', 'root_spec')]:
        dest = target/name
        shutil.copyfile(args[key], dest)
        args[key] = dest
    clearance_evidence.build(**args)
    return dict(root=root, cache=args['cache'], catalog=args['cache_registry'],
                evidence=args['output'], references=args['references'],
                output=root/'artifacts/proposals')


def test_review_remains_ungranted_and_excluded_remains_excluded(proposal_inputs):
    result = compile_proposals(**proposal_inputs)
    rows = list(read_plan(proposal_inputs['output']/'proposals.jsonl'))
    assert result['total'] == 3
    assert result['reviewed'] == 2 and result['excluded'] == 1
    assert result['legitimately_grantable'] == 0
    assert all(row['grantable'] is False and row['training_allowed'] is False for row in rows)
    assert any(row['clearance_status'] == 'EXCLUDED' for row in rows)
    assert all(row['request_identity'] == result['request_identity'] for row in rows)


def test_proposal_replay_is_deterministic_and_cache_immutable(proposal_inputs):
    cache = proposal_inputs['cache']
    before = {p: file_digest(p) for p in cache.rglob('*') if p.is_file()}
    first = compile_proposals(**proposal_inputs)
    assert first == compile_proposals(**proposal_inputs)
    assert before == {p: file_digest(p) for p in cache.rglob('*') if p.is_file()}


@pytest.mark.parametrize('damage', ['revision', 'source', 'path', 'content'])
def test_changed_cached_identity_fails_closed(proposal_inputs, damage):
    directory = next(p for p in proposal_inputs['cache'].iterdir() if p.is_dir())
    inventory = directory/'inventory.jsonl'
    rows = list(read_plan(inventory))
    if damage == 'revision':
        rows[0]['revision'] = 'a'*40
    elif damage == 'source':
        rows[0]['source_id'] = 'other-source'
    elif damage == 'path':
        rows[0]['path'] = 'src/other.py'
    else:
        blob = directory/rows[0]['local_blob']
        blob.write_bytes(b'changed content')
        rows[0]['bytes'] = blob.stat().st_size
        rows[0]['sha256'] = file_digest(blob)
    inventory.write_bytes(b''.join(canonical(row)+b'\n' for row in rows))
    done_path = directory/'complete.json'
    done = json.loads(done_path.read_bytes())
    done['inventory_sha256'] = file_digest(inventory)
    done_path.write_bytes(canonical(seal(done)))
    with pytest.raises((SpecificationError, ValueError)):
        compile_proposals(**proposal_inputs)
    assert not proposal_inputs['output'].exists()


def test_catalog_or_reference_substitution_fails_closed(proposal_inputs):
    proposal_inputs['references'].write_bytes(b'changed references\n')
    with pytest.raises((SpecificationError, ValueError)):
        compile_proposals(**proposal_inputs)


def test_conditional_source_cannot_become_file_scope_candidate(tmp_path):
    scope, reason = _file_scope({'status': 'CONDITIONAL', 'license_id': 'CC-BY-SA-4.0'},
                                {}, tmp_path, tmp_path)
    assert scope is None and 'governance' in reason


def test_existing_grant_inputs_unchanged(proposal_inputs):
    root = proposal_inputs['root']
    historical = [root/name for name in ('historical_spec.json', 'reg.json', 'plan.jsonl')]
    existing = {p: file_digest(p) for p in historical if p.exists()}
    compile_proposals(**proposal_inputs)
    assert existing == {p: file_digest(p) for p in existing}


def test_replay_rejects_corrupt_proposal_output(proposal_inputs):
    compile_proposals(**proposal_inputs)
    (proposal_inputs['output']/'proposals.jsonl').write_bytes(b'corrupt\n')
    with pytest.raises(SpecificationError):
        compile_proposals(**proposal_inputs)
