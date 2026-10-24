"""Use-specific authorization, not a rights inference or a complete safety gate.

Receipts are supplied by the controlled owner/reviewer workflow. Seals establish
integrity, not reviewer authentication. No function here issues an approval.
Legacy preparation and generic shard replay are NOT training authorization.
"""
import json
from pathlib import Path

from training.production_corpus import file_digest, identity
from training.production_spec import need, verify_seal, local_path
from training.conditional_license import validate_approval

SCHEMA = 'yug-corpus-decision-v1'
POLICY = 'explicit-intended-use-v1'
BUNDLE = 'yug-corpus-decisions-v1'
USES = ('discovery', 'acquisition', 'inspection', 'production_preparation',
        'evaluation', 'commercial_training')
MAX_BUNDLE_BYTES = 16 * 1024 * 1024


def binding(record, source, spec):
    """Bind the immutable clearance too: paths, split and obligations cannot drift."""
    return dict(clearance_identity=record['identity'], source_identity=identity(source),
                spec_identity=spec['identity'], source_id=record['source_id'],
                revision=record['revision'], path=record['path'], sha256=record['sha256'])


def verify_receipt(receipt, record, source, spec, root, intended_use):
    need(intended_use in USES, 'Unsupported intended use')
    need(isinstance(receipt, dict), 'Missing decision receipt')
    verify_seal(receipt)
    need(receipt.get('schema') == SCHEMA and receipt.get('policy') == POLICY,
         'Unsupported decision schema/policy')
    verify_seal(record)
    need(receipt.get('binding') == binding(record, source, spec), 'Decision item/source/evidence binding mismatch')
    need(record.get('clearance_status') == 'ACCEPT', 'Clearance is not ACCEPT')
    need(receipt.get('decision') == 'ACCEPT', 'Decision is not ACCEPT')
    uses = receipt.get('intended_uses')
    need(isinstance(uses, list) and uses and len(uses) == len(set(uses))
         and all(u in USES for u in uses) and intended_use in uses,
         'Intended use not explicitly authorized')
    authority = receipt.get('authority')
    need(isinstance(authority, dict) and authority.get('kind') == 'human'
         and isinstance(authority.get('id'), str) and authority['id'].strip()
         and isinstance(receipt.get('decided_at'), str) and receipt['decided_at'].strip(),
         'Missing explicit decision authority/date')
    need(receipt.get('human_approval_required') is True and receipt.get('human_approved') is True,
         'Explicit human approval required')
    for field in ('reason', 'obligations', 'restrictions'):
        need(isinstance(receipt.get(field), str) and receipt[field].strip(), 'Missing decision '+field)
    evidence = receipt.get('evidence')
    need(isinstance(evidence, list) and evidence, 'Missing decision evidence')
    roles = set()
    for e in evidence:
        need(isinstance(e, dict) and e.get('role') in ('approval', 'rights_scope', 'provenance'),
             'Unsupported decision evidence role')
        need(file_digest(local_path(root, e.get('path'))) == e.get('sha256'), 'Decision evidence hash mismatch')
        roles.add(e['role'])
    need(roles == {'approval', 'rights_scope', 'provenance'}, 'Incomplete decision evidence')
    for e in record['license_evidence']:
        need(file_digest(local_path(root, e['path'])) == e['sha256'], 'License evidence hash mismatch')
    approval = validate_approval(record, source, root)
    if approval is not None and intended_use == 'commercial_training':
        need(approval['intended_use']['commercial_training']['decision'] == 'APPROVED',
             'Conditional commercial training not approved')
    return receipt['identity']


def load_receipts(path):
    need(path is not None, 'Explicit decision receipts required for this use')
    path = Path(path)
    need(path.stat().st_size <= MAX_BUNDLE_BYTES, 'Decision bundle exceeds bounded metadata limit')
    bundle = json.loads(path.read_bytes())
    verify_seal(bundle)
    need(bundle.get('schema') == BUNDLE and isinstance(bundle.get('receipts'), list),
         'Unsupported decision bundle')
    receipts = {}
    for r in bundle['receipts']:
        need(isinstance(r, dict) and isinstance(r.get('binding'), dict), 'Invalid decision row')
        key = r['binding'].get('clearance_identity')
        need(isinstance(key, str) and key not in receipts, 'Duplicate/missing decision item')
        receipts[key] = r
    return bundle['identity'], receipts
