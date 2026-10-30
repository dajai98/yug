"""Approved inventory runner recovery and target-prefix contracts."""
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import pytest
from tests.test_production_acquisition import fixture_plan,Tokenizer,ROOT,prior_inventory,approval_kwargs
from training.production_acquisition import ProductionBuild,item_key
from training.production_build import compile_inventory,Fetcher,disk_preflight
from training.production_corpus import canonical,file_digest
from training.production_spec import SpecificationError,seal,verify_seal


def test_compiler_replay_and_corruption(tmp_path):
    spec,reg,plan,records=fixture_plan(tmp_path)
    output,receipt=compile_inventory(spec,reg,tmp_path,tmp_path/'runner',plan)
    assert receipt['scheduled']==2 and file_digest(output)==file_digest(plan)
    assert compile_inventory(spec,reg,tmp_path,tmp_path/'runner',plan)[1]==receipt
    output.write_bytes(b'corrupt')
    with pytest.raises(SpecificationError,match='corrupt'):compile_inventory(spec,reg,tmp_path,tmp_path/'runner',plan)


@pytest.mark.parametrize('reason',['unknown_license','unknown_source','unsupported_transport'])
def test_unresolved_skipped_safe_continues(tmp_path,reason):
    spec,reg,plan,records=fixture_plan(tmp_path)
    if reason=='unknown_license':records[0]['license_id']='UNKNOWN'
    if reason=='unknown_source':records[0]['source_id']='not-registered'
    if reason=='unsupported_transport':records[0].pop('input_path')
    plan.write_bytes(b''.join(canonical(r)+b'\n' for r in records))
    output,receipt=compile_inventory(spec,reg,tmp_path,tmp_path/'runner',plan)
    assert receipt['scheduled']==1 and receipt['unresolved']==1
    assert json.loads(output.read_bytes())['source_id']==records[1]['source_id']


def test_target_prefix_resume_accounting_and_exports(tmp_path):
    spec,reg,plan,records=fixture_plan(tmp_path)
    tok=Tokenizer()
    with ProductionBuild(tmp_path/'out',spec,reg,repository_root=tmp_path,diagnostic=True) as b:
        first=min(records,key=item_key);limit=len((tmp_path/first['input_path']).read_bytes())
        progress=b.run(plan,tok,target_tokens=limit)
        assert progress['accepted_clean_train_tokens']==limit
        assert sum(progress['category_tokens'].values())==limit and len(progress['category_tokens'])==16
        prefix=b.export_shards(tmp_path/'out/exports/first')
        calls=tok.calls
        b.run(plan,tok,target_tokens=limit)
        assert tok.calls==calls
        assert b.export_shards(tmp_path/'out/exports/first')==prefix
        progress=b.run(plan,tok,target_tokens=100000)
        assert progress['state']=='COMPLETE' and tok.calls==2
        full=b.export_shards(tmp_path/'out/exports/full')
        assert full['total_token_count']>prefix['total_token_count']
        b.run(plan,tok,target_tokens=100000)
        assert tok.calls==2


def test_boundary_does_not_discard_cached_document(tmp_path):
    spec,reg,plan,records=fixture_plan(tmp_path)
    tok=Tokenizer()
    with ProductionBuild(tmp_path/'out',spec,reg,repository_root=tmp_path,diagnostic=True) as b:
        assert b.run(plan,tok,target_tokens=1)['state']=='TARGET_BOUNDARY'
        assert tok.calls==1
        assert b.run(plan,tok,target_tokens=1)['accepted_clean_train_tokens']==0
        assert tok.calls==1
        assert b.run(plan,tok,target_tokens=100000)['state']=='COMPLETE'
        assert tok.calls==2


@pytest.mark.parametrize('stage',['_prepare','_cached_tokens','shards'])
def test_interrupted_stage_recovery(tmp_path,monkeypatch,stage):
    spec,reg,plan,records=fixture_plan(tmp_path)
    tok=Tokenizer()
    with ProductionBuild(tmp_path/'out',spec,reg,repository_root=tmp_path,diagnostic=True) as b:
        if stage=='shards':
            b.run(plan,tok)
            import training.production_acquisition as module
            original=module.build_token_shards
            def crash(*args,**kwargs):
                original(*args,**kwargs)
                raise KeyboardInterrupt()
            monkeypatch.setattr(module,'build_token_shards',crash)
            with pytest.raises(KeyboardInterrupt):b.export_shards()
            monkeypatch.setattr(module,'build_token_shards',original)
        else:
            original=getattr(b,stage)
            def crash(*args,**kwargs):
                original(*args,**kwargs)
                raise KeyboardInterrupt()
            monkeypatch.setattr(b,stage,crash)
            with pytest.raises(KeyboardInterrupt):b.run(plan,tok)
    with ProductionBuild(tmp_path/'out',spec,reg,repository_root=tmp_path,diagnostic=True,
                         expected_plan_sha256=file_digest(plan)) as b:
        assert b.run(plan,tok)['state']=='COMPLETE'
        assert b.export_shards()['total_token_count']>0
        assert tok.calls==2


def test_fresh_process_crash_resume(tmp_path):
    spec,reg,plan,records=fixture_plan(tmp_path)
    (tmp_path/'config.json').write_text(json.dumps([spec,reg]))
    script='''import json,sys,os
from pathlib import Path
from training.production_acquisition import ProductionBuild
from training.production_corpus import file_digest
from training.production_build import Fetcher
from yug.tokenizer.bpe import ByteLevelBpeTokenizer
root=Path(sys.argv[1]);spec,reg=json.loads((root/'config.json').read_bytes())
tok=ByteLevelBpeTokenizer.load(sys.argv[2])
with ProductionBuild(root/'out',spec,reg,repository_root=root,diagnostic=True,
                     expected_plan_sha256=file_digest(root/'plan.jsonl')) as b:
 if sys.argv[3]=='crash':
  original=b._prepare
  def crash(*args):
   original(*args)
   os._exit(17)
  b._prepare=crash
 result=b.run(root/'plan.jsonl',tok,fetch=Fetcher(root,reg))
 b.export_shards()
 print(result['accepted_clean_train_tokens'])
'''
    args=[sys.executable,'-B','-c',script,str(tmp_path),str(ROOT/'training/resources/phase1/bpe_fitter_optimization_gate_v1/comparison/run-4096-0/tokenizer.json')]
    assert subprocess.run(args+['crash'],cwd=ROOT,stdin=subprocess.DEVNULL,capture_output=True).returncode==17
    first=subprocess.run(args+['resume'],cwd=ROOT,stdin=subprocess.DEVNULL,capture_output=True,text=True)
    assert first.returncode==0,first.stderr
    second=subprocess.run(args+['resume'],cwd=ROOT,stdin=subprocess.DEVNULL,capture_output=True,text=True)
    assert second.returncode==0 and second.stdout==first.stdout


def test_fetcher_cache_and_disk_guard(tmp_path,monkeypatch):
    spec,reg,plan,records=fixture_plan(tmp_path)
    fetch=Fetcher(tmp_path,reg)
    assert b''.join(fetch(records[0]))==(tmp_path/records[0]['input_path']).read_bytes()
    assert fetch.network_bytes==0
    (tmp_path/records[0]['input_path']).write_bytes(b'corrupt')
    with pytest.raises(SpecificationError,match='corrupt'):list(fetch(records[0]))
    with pytest.raises(SpecificationError,match='disk'):disk_preflight(tmp_path,10**30)



def test_registry_manifest_compilation_without_handwritten_plan(tmp_path):
    spec,reg,plan,records=fixture_plan(tmp_path)
    for i,r in enumerate(records):
        path=tmp_path/f'files{i}.json';m=json.loads(path.read_bytes())
        m['files'][0].update(subcategory=r['subcategory'],input_path=r['input_path'],language='en')
        m=seal(m);path.write_bytes(canonical(m))
        path=tmp_path/f'clear{i}.json';c=json.loads(path.read_bytes());c['file_manifest_identity']=m['identity'];path.write_bytes(canonical(seal(c)))
    output,receipt=compile_inventory(spec,reg,tmp_path,tmp_path/'runner')
    assert receipt['scheduled']==2 and receipt['unresolved']==0
    with ProductionBuild(tmp_path/'out',spec,reg,repository_root=tmp_path,diagnostic=True) as b:
        assert b.run(output,Tokenizer())['stages']=={'ACCEPTED':2}


def test_cli_dry_run_resume_complete_and_no_repeat(tmp_path):
    spec,reg,plan,records=fixture_plan(tmp_path)
    (tmp_path/'spec.json').write_bytes(canonical(spec));(tmp_path/'registry.json').write_bytes(canonical(reg))
    refs=tmp_path/'heldout.jsonl'
    from tests.test_production_acquisition import fixture_reference
    refs.write_bytes(canonical(fixture_reference('Independent historic observations of planetary spectroscopy and stellar spectra.',
        'unrelated-heldout', {'source':'fixture'}))+b'\n')
    args=[sys.executable,'-B','-m','training.production_build','--root',str(tmp_path),'--spec',str(tmp_path/'spec.json'),'--registry',str(tmp_path/'registry.json'),'--plan',str(plan),'--directory',str(tmp_path/'out'),'--references',str(refs),'--tokenizer',str(ROOT/'training/resources/phase1/bpe_fitter_optimization_gate_v1/comparison/run-4096-0/tokenizer.json'),'--target-tokens','100000','--reserve-mib','0','--chunk-items','1']
    args+=['--prior-production',str(prior_inventory(tmp_path)),
           '--approved-plan-sha256',file_digest(plan)]
    approval=approval_kwargs(tmp_path,file_digest(plan))
    args+=['--ownership-approval',str(approval['ownership_approval']),
               '--approved-ownership-approval-sha256',approval['expected_ownership_approval_sha256'],
               '--governance-log',str(approval['governance_log']),
               '--approved-governance-sha256',approval['expected_governance_sha256'],
               '--corpus-ledger',str(approval['corpus_ledger']),
               '--approved-corpus-head',approval['expected_corpus_head']]
    dry=subprocess.run(args+['--dry-run'],cwd=ROOT,stdin=subprocess.DEVNULL,capture_output=True,text=True)
    assert dry.returncode==0,dry.stdout+dry.stderr
    assert not list((tmp_path/'out/raw').glob('*.txt'))
    first=subprocess.run(args+['--resume'],cwd=ROOT,stdin=subprocess.DEVNULL,capture_output=True,text=True)
    assert first.returncode==3,first.stdout+first.stderr  # inventory exhausted, not a fabricated 100K pass
    before={str(p):p.stat().st_mtime_ns for folder in ('raw','normalized','tokens') for p in (tmp_path/'out'/folder).glob('*')}
    progress=json.loads((tmp_path/'out/runner/progress.json').read_bytes())
    verify_seal(progress)
    assert progress['inventory_exhausted'] and progress['network_bytes_this_process']==0
    stale=subprocess.run(args+['--resume'],cwd=ROOT,stdin=subprocess.DEVNULL,capture_output=True,text=True)
    assert stale.returncode==2 and 'Stale or untrusted corpus head' in stale.stdout
    with sqlite3.connect(approval['corpus_ledger']) as ledger:
        current_head=ledger.execute("SELECT value FROM meta WHERE key='head'").fetchone()[0]
    args[args.index('--approved-corpus-head')+1]=current_head
    second=subprocess.run(args+['--resume'],cwd=ROOT,stdin=subprocess.DEVNULL,capture_output=True,text=True)
    assert second.returncode==3,second.stdout+second.stderr
    after={str(p):p.stat().st_mtime_ns for folder in ('raw','normalized','tokens') for p in (tmp_path/'out'/folder).glob('*')}
    assert before==after
    assert json.loads((tmp_path/'out/runner/progress.json').read_bytes())['shard_manifest_identity']==progress['shard_manifest_identity']
