"""Explicit local approval contracts; never infers or issues rights grants."""
import json
from training.production_spec import need, verify_seal, local_path
from training.production_corpus import file_digest

VERSION='source-specific-conditional-license-v1'
CONDITIONAL={
    'CC-BY-SA-3.0': 'attribution, modification notices and applicable ShareAlike obligations',
    'CC-BY-SA-4.0': 'attribution, modification notices, database rights and applicable ShareAlike obligations',
    'MPL-2.0': 'notices and applicable covered-file source/redistribution obligations',
    'Apache-2.0 WITH LLVM-exception': 'Apache license/NOTICE/change notices and exact LLVM exception scope',
}
USES=('production_preparation','commercial_training','dataset_distribution','model_distribution','hosted_inference')


def pending(license_id):
    return dict(policy=VERSION,tier='CONDITIONAL',license_id=license_id,
        obligations_required=CONDITIONAL[license_id],approval_required=True,
        scope_verified=False,training_allowed=False)


def validate_approval(record, source, root):
    """Called in the same trust boundary as existing explicit per-file grants.

    A hash/seal is integrity, not proof of reviewer identity or legal authority.
    Only the project owner's controlled approval workflow may supply this record.
    """
    licenses={record.get('license_id'),source.get('license_id')}
    if not licenses.intersection(CONDITIONAL):return None
    need(record.get('license_id')==source.get('license_id'),'Conditional mixed scope requires separate source')
    reference=source.get('conditional_approval')
    need(isinstance(reference,dict),'Missing source-specific conditional approval')
    path=local_path(root,reference.get('path'))
    need(file_digest(path)==reference.get('sha256'),'Conditional approval hash mismatch')
    approval=json.loads(path.read_bytes());verify_seal(approval)
    need(approval.get('policy')==VERSION and approval.get('decision')=='APPROVED','Conditional approval not granted')
    for field in ('source_id','locator','revision','license_id'):
        need(approval.get(field)==source.get(field),'Conditional approval '+field+' mismatch')
    need(isinstance(approval.get('approved_by'),str) and approval['approved_by'].strip()
         and isinstance(approval.get('approved_at'),str) and approval['approved_at'].strip(),'Missing approval authority/date')
    uses=approval.get('intended_use')
    need(isinstance(uses,dict) and set(uses)==set(USES),'Incomplete intended-use assessment')
    for use,value in uses.items():
        need(isinstance(value,dict) and value.get('decision') in ('APPROVED','FORBIDDEN')
             and isinstance(value.get('rationale'),str) and value['rationale'].strip(),'Incomplete use decision')
    need(uses['production_preparation']['decision']=='APPROVED','Preparation not approved')
    obligations=approval.get('obligations')
    required={'attribution','notices','modifications','reciprocity','release_controls'}
    need(isinstance(obligations,dict) and set(obligations)==required,'Incomplete conditional obligations')
    for value in obligations.values():
        need(isinstance(value,str) and value.strip(),'Empty conditional obligation')
    evidence=approval.get('evidence')
    need(isinstance(evidence,list) and evidence,'Missing conditional evidence')
    roles=set()
    for item in evidence:
        need(isinstance(item,dict) and item.get('role') in ('rights_scope','provenance','obligations','approval'),'Unknown evidence role')
        need(file_digest(local_path(root,item.get('path')))==item.get('sha256'),'Conditional evidence corrupt')
        roles.add(item['role'])
    need(roles=={'rights_scope','provenance','obligations','approval'},'Incomplete rights/provenance/obligation evidence')
    scope=approval.get('files')
    need(isinstance(scope,list) and scope,'No approved file scope')
    need(len({f.get('path') for f in scope})==len(scope),'Duplicate conditional file scope')
    matches=[f for f in scope if f.get('path')==record['path']]
    need(len(matches)==1 and matches[0].get('sha256')==record['sha256'],'Record outside exact conditional scope')
    need(record.get('conditional_approval_identity')==approval['identity'],'Item not bound to conditional approval')
    return approval
