"""The supplied Phase 3 intake is review data, never production authority."""
import json
import socket
from collections import defaultdict
from pathlib import Path

from training.production_corpus import canonical
from training.production_spec import verify_seal
from training.source_expansion import build
from training.source_prepare import classify


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'training/specifications/phase3_source_candidate_registry_v1.json'
ACTIVE = ROOT / 'training/specifications/source_registry_v1.json'
SPEC = ROOT / 'training/specifications/production_corpus_v1.json'


def test_intake_is_sealed_and_cannot_change_active_approval():
    catalog = json.loads(CATALOG.read_bytes())
    active = json.loads(ACTIVE.read_bytes())
    spec = json.loads(SPEC.read_bytes())
    for value in (catalog, active, spec):
        verify_seal(value)
    assert CATALOG.read_bytes() == canonical(catalog)
    assert catalog['catalog_role'] == 'phase3-review-intake-only'
    assert catalog['external_candidate_input_sha256'] == (
        'sha256:15f74a8b4c3c3efdf91976c57351cb755f409f70ffa0255defb7220718854966'
    )
    assert spec['registry_identity'] == active['identity'] == catalog['active_registry_identity']
    assert catalog['identity'] != active['identity']
    assert len(catalog['sources']) == 41
    assert len(catalog['existing_registry_matches']) == 6
    assert {r['source_id'] for r in catalog['sources']}.isdisjoint(
        {r['source_id'] for r in active['sources']}
    )
    assert {r['source_id'] for r in catalog['existing_registry_matches']} <= {
        r['source_id'] for r in active['sources']
    }
    submitted = [r['provenance']['submitted_candidate'] for r in catalog['sources']]
    submitted += [r['submitted_candidate'] for r in catalog['existing_registry_matches']]
    assert len({row['candidate_id'] for row in submitted}) == 47
    assert all(row['revision'] is None and row['auto_approve'] is False
               for row in submitted)


def test_every_new_source_remains_unapproved_and_unpreparable():
    catalog = json.loads(CATALOG.read_bytes())
    spec = json.loads(SPEC.read_bytes())
    owners = {category['id'] for category in spec['categories']}
    groups = defaultdict(set)
    for source in catalog['sources']:
        assert source['status'] == 'REVIEW'
        assert source['category'] in owners
        assert source['revision'] is None
        assert source['license_id'] is None
        assert source['license_evidence'] is None
        assert source['file_manifest'] is None
        assert source['clearance_record'] is None
        assert source['allow_paths'] == []
        assert source['split'] == 'unassigned'
        assert source['target_cap_tokens'] == 0
        assert source['category_caps_tokens'] == {source['category']: 0}
        assert source['origin_verified'] is False
        assert source['training_allowed_after_clearance'] is False
        assert classify(source) != 'CACHE_READY'
        assert source['provenance']['input_sha256'] == catalog['external_candidate_input_sha256']
        groups[source['locator'].casefold().rstrip('/')].add(source['split_group'])
    assert all(len(values) == 1 for values in groups.values())


def test_existing_expansion_rejects_every_review_row_without_network(tmp_path, monkeypatch):
    def no_network(*_args, **_kwargs):
        raise AssertionError('Candidate intake must remain offline')

    monkeypatch.setattr(socket, 'socket', no_network)
    output = tmp_path / 'expansion'
    first = build(CATALOG, SPEC, output)
    assert first['candidate_sources'] == 0
    assert first['rejected_sources'] == 41
    assert first == build(CATALOG, SPEC, output, resume=True)
    decisions = json.loads((output / 'decisions.json').read_bytes())['sources']
    assert all(row['status'] == 'REJECTED' and row['training_allowed'] is False
               for row in decisions)
