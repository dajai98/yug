"""Synthetic approvals only; these fixtures authorize no real corpus."""
import json

import pytest

from training.corpus_gate import SCHEMA, POLICY, BUNDLE, binding, verify_receipt
from training.production_spec import seal, SpecificationError
from training.production_corpus import canonical, file_digest
from training.production_acquisition import ProductionBuild
from tests.test_production_acquisition import Tokenizer, prior_inventory, approval_kwargs
from tests.test_production_cross_batch import setup
from tests.test_conditional_license import approved


def receipt(r,s,spec,root,uses=None):
    e=root/'review-decision.txt'
    if not e.exists():e.write_text('Synthetic owner approval, provenance and rights scope; test only.')
    return seal(dict(schema=SCHEMA,policy=POLICY,binding=binding(r,s,spec),decision='ACCEPT',
        authority=dict(kind='human',id='fixture reviewer'),decided_at='2026-10-23',
        human_approval_required=True,human_approved=True,intended_uses=uses or ['commercial_training'],
        reason='Fixture decision',obligations='Retain test attribution',restrictions='Test only; no real training',
        evidence=[dict(role=role,path=e.name,sha256=file_digest(e)) for role in ('approval','rights_scope','provenance')]))


@pytest.fixture
def case(tmp_path):
    data=setup(tmp_path)
    spec,reg,refs,plans=data
    r=json.loads(plans[0].read_bytes())
    return tmp_path,data,r,reg['sources'][0],receipt(r,reg['sources'][0],spec,tmp_path)


def test_explicit_authorization_deterministic(case):
    root,data,r,s,d=case
    assert verify_receipt(d,r,s,data[0],root,'commercial_training')==d['identity']
    assert canonical(receipt(r,s,data[0],root))==canonical(d)


@pytest.mark.parametrize('mutation', ['missing','corrupt','schema','policy','review','exclude','reject',
    'preparation','identity','source','evidence','authority','approval','obligations','role','license'])
def test_no_upgrade(case,mutation):
    root,data,r,s,d=case
    if mutation=='missing':d=None
    elif mutation=='corrupt':d['reason']='tampered'
    elif mutation=='license':(root/'license.txt').write_text('Changed license')
    elif mutation=='evidence':(root/'review-decision.txt').write_text('Changed evidence')
    else:
        if mutation in ('schema','policy'):d[mutation]='unknown'
        elif mutation in ('review','exclude','reject'):d['decision']=mutation.upper()
        elif mutation=='preparation':d['intended_uses']=['production_preparation']
        elif mutation=='identity':d['binding']['sha256']='sha256:'+'0'*64
        elif mutation=='source':s=dict(s,revision='f'*40)
        elif mutation=='authority':d['authority']['id']=''
        elif mutation=='approval':d['human_approved']=False
        elif mutation=='obligations':d['obligations']=''
        elif mutation=='role':d['evidence']=d['evidence'][:1]
        d=seal(d)
    with pytest.raises(SpecificationError):verify_receipt(d,r,s,data[0],root,'commercial_training')


@pytest.mark.parametrize('allowed',[False,True])
def test_conditional_training_needs_matching_use(approved,allowed):
    root,s,r,a,publish=approved
    if allowed:a['intended_use']['commercial_training']['decision']='APPROVED'
    publish(a);r=seal(dict(r,clearance_status='ACCEPT',license_evidence=a['evidence']))
    spec=seal(dict(test=True));d=receipt(r,s,spec,root)
    if allowed:assert verify_receipt(d,r,s,spec,root,'commercial_training')==d['identity']
    else:
        with pytest.raises(SpecificationError,match='Conditional commercial'):verify_receipt(d,r,s,spec,root,'commercial_training')
    s.pop('conditional_approval')
    d=receipt(r,s,spec,root)
    with pytest.raises(SpecificationError,match='Missing source-specific'):verify_receipt(d,r,s,spec,root,'commercial_training')


def bundle(root,data,uses=None):
    spec,reg,refs,plans=data
    rows=[json.loads(p.read_bytes()) for p in plans]
    docs=[receipt(r,s,spec,root,uses) for r,s in zip(rows,reg['sources'])]
    p=root/'decisions.json';p.write_bytes(canonical(seal(dict(schema=BUNDLE,receipts=docs))))
    return p


def builder(root,data):
    return ProductionBuild(root/'build',data[0],data[1],repository_root=root,
                           reference_plan=data[2],prior_production=prior_inventory(root),
                           expected_plan_sha256=file_digest(data[3][0]),
                           **approval_kwargs(root,file_digest(data[3][0])))


def test_runner_missing_receipt_prevents_fetch_and_accounting(case):
    root,data,r,s,d=case
    with builder(root,data) as b:
        with pytest.raises(SpecificationError,match='receipts required'):
            b.run(data[3][0],Tokenizer(),intended_use='commercial_training',fetch=lambda _:pytest.fail('fetch'))
        assert b.db.execute('SELECT count(*) FROM totals').fetchone()[0]==0


def test_authorized_run_export_replay_and_revocation(case):
    root,data,r,s,d=case;p=bundle(root,data)
    with builder(root,data) as b:
        b.run(data[3][0],Tokenizer(),intended_use='commercial_training',decision_receipts=p)
        shards=b.export_shards(root/'build/authorized',intended_use='commercial_training',decision_receipts=p)
        class NoEncode(Tokenizer):
            def encode(self,*args):pytest.fail('retokenization')
        b.run(data[3][0],NoEncode(),intended_use='commercial_training',decision_receipts=p,fetch=lambda _:pytest.fail('fetch'))
        assert shards==b.export_shards(root/'build/authorized',intended_use='commercial_training',decision_receipts=p)
        stream=b.replay_for_use(root/'build/authorized',split='train',intended_use='commercial_training',decision_receipts=p)
        assert list(stream)
        with pytest.raises(SpecificationError,match='required on resume'):b.run(data[3][0],NoEncode())
        (root/'review-decision.txt').write_text('revoked/mutated')
        with pytest.raises(SpecificationError,match='evidence hash'):b.replay_for_use(root/'build/authorized',split='train',intended_use='commercial_training',decision_receipts=p)


def test_legacy_preparation_not_training_and_separate_export(case):
    root,data,r,s,d=case
    with builder(root,data) as b:
        b.run(data[3][0],Tokenizer())
        legacy=b.export_shards(root/'build/legacy')
        original=file_digest(root/'build/legacy/prepared.json')
        with pytest.raises(SpecificationError):b.export_shards(root/'build/missing',intended_use='commercial_training')
        b.run(data[3][0],Tokenizer())  # Existing FAILED export state requires explicit replay recovery.
        p=bundle(root,data)
        b.expected_governance_sha256=approval_kwargs(root,file_digest(data[3][0]))['expected_governance_sha256']
        with pytest.raises(SpecificationError,match='not authorized'):b.replay_for_use(root/'build/legacy',split='train',intended_use='commercial_training',decision_receipts=p)
        b.export_shards(root/'build/authorized',intended_use='commercial_training',decision_receipts=p)
        assert file_digest(root/'build/legacy/prepared.json')==original
        assert b.export_shards(root/'build/legacy')==legacy


def test_modified_payload_blocks_export(case):
    root,data,r,s,d=case;p=bundle(root,data)
    with builder(root,data) as b:
        b.run(data[3][0],Tokenizer())
        next((root/'build/raw').iterdir()).write_text('tampered')
        with pytest.raises(SpecificationError,match='payload corrupt'):
            b.export_shards(intended_use='commercial_training',decision_receipts=p)


def test_validation_requires_own_use(tmp_path):
    data=setup(tmp_path,splits=['validation','train']);p=bundle(tmp_path,data)
    with builder(tmp_path,data) as b:
        with pytest.raises(SpecificationError,match='not explicitly authorized'):
            b.run(data[3][0],Tokenizer(),intended_use='commercial_training',decision_receipts=p)
        p=bundle(tmp_path,data,['commercial_training','evaluation'])
        b.expected_governance_sha256=approval_kwargs(tmp_path,file_digest(data[3][0]))['expected_governance_sha256']
        b.run(data[3][0],Tokenizer(),intended_use='commercial_training',decision_receipts=p)


@pytest.mark.parametrize('damage',['duplicate','version','seal','missing_row'])
def test_invalid_bundle_no_fallback(case,damage):
    root,data,r,s,d=case;p=bundle(root,data);value=json.loads(p.read_bytes())
    if damage=='duplicate':value['receipts'].append(value['receipts'][0])
    elif damage=='version':value['schema']='unknown'
    elif damage=='missing_row':value['receipts']=[]
    else:value['identity']='bad'
    p.write_bytes(canonical(value if damage=='seal' else seal(value)))
    with builder(root,data) as b:
        with pytest.raises(SpecificationError):b.run(data[3][0],Tokenizer(),intended_use='commercial_training',decision_receipts=p)


def test_no_relabel_of_legacy_export(case):
    root,data,r,s,d=case;p=bundle(root,data)
    with builder(root,data) as b:
        b.run(data[3][0],Tokenizer());b.export_shards()
        before=file_digest(root/'build/prepared.json')
        with pytest.raises(SpecificationError,match='new destination'):
            b.export_shards(intended_use='commercial_training',decision_receipts=p)
        assert file_digest(root/'build/prepared.json')==before


def test_diagnostic_cannot_authorize(case):
    root,data,r,s,d=case;p=bundle(root,data)
    with ProductionBuild(root/'diagnostic',data[0],data[1],repository_root=root,diagnostic=True) as b:
        with pytest.raises(SpecificationError,match='Diagnostic'):
            b.run(data[3][0],Tokenizer(),intended_use='commercial_training',decision_receipts=p)


def test_bound_bundle_cannot_change(case):
    root,data,r,s,d=case;p=bundle(root,data)
    with builder(root,data) as b:
        b.run(data[3][0],Tokenizer(),intended_use='commercial_training',decision_receipts=p)
        doc=json.loads(p.read_bytes());doc['note']='different version';p.write_bytes(canonical(seal(doc)))
        with pytest.raises(SpecificationError,match='changed on resume'):
            b.run(data[3][0],Tokenizer(),intended_use='commercial_training',decision_receipts=p)


@pytest.mark.parametrize('dry_run',[False,True])
def test_cli_runner_does_not_drop_use(case,monkeypatch,dry_run):
    from types import SimpleNamespace
    from training import production_build as runner
    root,data,r,s,d=case
    for name,value in [('spec',data[0]),('registry',data[1])]:
        (root/(name+'.json')).write_bytes(canonical(value))
    monkeypatch.setattr(runner.ByteLevelBpeTokenizer,'load',lambda _:Tokenizer())
    monkeypatch.setattr(runner.Fetcher,'__call__',lambda *args:pytest.fail('network'))
    authority=approval_kwargs(root,file_digest(data[3][0]))
    args=SimpleNamespace(root=root,directory=root/'cli',spec=root/'spec.json',registry=root/'registry.json',
        target_tokens=100000,chunk_items=1,reserve_mib=0,request_interval=0,resume=False,
        plan=data[3][0],tokenizer='unused',references=data[2],prior_production=prior_inventory(root),
        dry_run=dry_run,intended_use='commercial_training',decision_receipts=None,
        approved_plan_sha256=file_digest(data[3][0]),
        ownership_approval=authority['ownership_approval'],
        approved_ownership_approval_sha256=authority['expected_ownership_approval_sha256'],
        governance_log=authority['governance_log'],
        approved_governance_sha256=authority['expected_governance_sha256'])
    with pytest.raises(SpecificationError,match='receipts required'):runner.execute(args)


@pytest.mark.parametrize('reseal_local',[False,True])
def test_replay_rejects_legacy_authorization_promotion(case,reseal_local):
    """Exact exploit: legacy 619-token export plus valid receipts, changed local header."""
    from training.production_corpus import identity, PipelineError
    from training.token_shards import TokenShardStream
    root,data,r,s,d=case;p=bundle(root,data)
    with builder(root,data) as b:
        b.run(data[3][0],Tokenizer());b.export_shards()
        path=root/'build/prepared.json';original=json.loads(path.read_bytes())
        shards=root/'build/shards';before=file_digest(shards/'manifest.json')
        generic=lambda:TokenShardStream(shards,split='train',tokenizer_identity=Tokenizer.identity,
                                       prepared_corpus_identity=original['prepared_corpus_identity'])
        assert len(list(generic()))==619
        changed=dict(original,use_authorization=b.authorize_use('commercial_training',p,accepted_only=True))
        if reseal_local:
            changed.pop('prepared_corpus_identity')
            changed['prepared_corpus_identity']=identity(changed)
        path.write_bytes(canonical(changed))
        with pytest.raises((SpecificationError,PipelineError),match='identity mismatch'):
            b.replay_for_use(root/'build',split='train',intended_use='commercial_training',decision_receipts=p)
        assert file_digest(shards/'manifest.json')==before
        assert len(list(generic()))==619  # Integrity decoding confers no authorization.


@pytest.mark.parametrize('damage',['authorization','identity','content','embedded'])
def test_authorized_replay_manifest_tampering(case,damage):
    from training.production_corpus import identity, PipelineError
    root,data,r,s,d=case;p=bundle(root,data)
    with builder(root,data) as b:
        b.run(data[3][0],Tokenizer());b.export_shards(intended_use='commercial_training',decision_receipts=p)
        replay=lambda:b.replay_for_use(root/'build',split='train',intended_use='commercial_training',decision_receipts=p)
        assert list(replay())==list(replay())
        path=root/'build/prepared.json'
        if damage=='embedded':
            path=root/'build/shards/manifest.json';m=json.loads(path.read_bytes())
            m['prepared_manifest']['use_authorization']['intended_use']='evaluation'
            m.pop('manifest_identity');m['manifest_identity']=identity(m)
        else:
            m=json.loads(path.read_bytes())
            if damage=='authorization':m['use_authorization']['intended_use']='evaluation'
            elif damage=='identity':m['prepared_corpus_identity']='sha256:'+'0'*64
            else:
                m['note']='Locally resealed, but not the exported manifest'
                m.pop('prepared_corpus_identity');m['prepared_corpus_identity']=identity(m)
        path.write_bytes(canonical(m))
        with pytest.raises((SpecificationError,PipelineError)):
            replay()
