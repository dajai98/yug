"""Optional AMP/CUDA checkpoint state, with real CPU GradScaler coverage."""
import json
import random
import pytest
import torch
from training.checkpoint import save_checkpoint,load_checkpoint
from training.config import TrainingConfig
from training.errors import CheckpointError
from training.loss import causal_lm_loss
from yug.native.config import ModelConfig
from yug.native.model import YugTransformer

def make(device='cpu'):
    cfg=ModelConfig(vocab_size=32,tokenizer_identity='sha256:'+'0'*64,hidden_size=32,num_layers=1,num_heads=4,ffn_size=64,context_length=16)
    m=YugTransformer(cfg).to(device)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,betas=(.9,.95),foreach=False)
    tc=TrainingConfig(seed=123,learning_rate=1e-3,batch_size=1,sequence_length=8,max_steps=4,eval_interval=4,checkpoint_interval=2,device=device)
    return cfg,m,opt,tc

def save(path,parts,scaler=None,step=2):
    cfg,m,opt,tc=parts
    save_checkpoint(path,model=m,optimizer=opt,model_config=cfg,training_config=tc,step=step,tokenizer_identity=cfg.tokenizer_identity,scaler=scaler)

def update(parts,scaler,amp=False):
    cfg,m,opt,tc=parts
    ids=torch.randint(0,32,(1,9),device=tc.device)
    opt.zero_grad()
    with torch.autocast(device_type=tc.device,dtype=torch.float16,enabled=amp):
        loss=causal_lm_loss(m(ids[:,:-1]),ids[:,1:])
    scaler.scale(loss).backward();scaler.unscale_(opt)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
    scaler.step(opt);scaler.update()
    assert all(torch.isfinite(p).all() for p in m.parameters())
    return float(loss.detach())

def equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b

def test_real_cpu_scaler_roundtrip_and_exact_continuation(tmp_path):
    torch.manual_seed(123);a=make();s=torch.amp.GradScaler('cpu',init_scale=128,growth_interval=2)
    update(a,s);update(a,s);save(tmp_path,a,s)
    expected_loss=update(a,s)
    b=make();t=torch.amp.GradScaler('cpu',init_scale=16,growth_interval=99)
    metadata=load_checkpoint(tmp_path,model=b[1],optimizer=b[2],scaler=t)
    assert metadata['step']==2
    assert update(b,t)==expected_loss
    equal(a[1].state_dict(),b[1].state_dict());equal(a[2].state_dict(),b[2].state_dict());equal(s.state_dict(),t.state_dict())

@pytest.mark.parametrize('disabled',[False,True])
def test_missing_or_empty_scaler_leaves_supplied_state_unchanged(tmp_path,disabled):
    a=make();s=torch.amp.GradScaler('cpu',enabled=False) if disabled else None
    save(tmp_path,a,s)
    target=torch.amp.GradScaler('cpu',init_scale=17,growth_interval=19);before=target.state_dict()
    load_checkpoint(tmp_path,model=a[1],optimizer=a[2],scaler=target)
    assert target.state_dict()==before

def test_amp_checkpoint_loads_without_scaler_for_fp32(tmp_path):
    a=make();s=torch.amp.GradScaler('cpu');update(a,s);save(tmp_path,a,s)
    b=make();meta=load_checkpoint(tmp_path,model=b[1],optimizer=b[2])
    assert meta['step']==2;equal(a[1].state_dict(),b[1].state_dict());equal(a[2].state_dict(),b[2].state_dict())

def test_restore_rng_false_still_restores_scaler(tmp_path):
    a=make();s=torch.amp.GradScaler('cpu',init_scale=128);update(a,s);save(tmp_path,a,s)
    target=torch.amp.GradScaler('cpu',init_scale=16);torch.manual_seed(777);random.seed(777)
    cpu=torch.get_rng_state().clone();py=random.getstate()
    load_checkpoint(tmp_path,model=a[1],optimizer=a[2],scaler=target,restore_rng_state=False)
    equal(target.state_dict(),s.state_dict());assert torch.equal(cpu,torch.get_rng_state());assert py==random.getstate()

def test_cpu_save_does_not_initialize_cuda(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    def forbidden():raise AssertionError('CPU save must not read/initialize CUDA RNG')
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',forbidden)
    save(tmp_path,make())
    assert 'cuda_rng_states_hex' not in json.loads((tmp_path/'metadata.json').read_text())

def test_cuda_rng_metadata_protocol_without_gpu(tmp_path,monkeypatch):
    states=[torch.tensor([1,2,3],dtype=torch.uint8),torch.tensor([7,8],dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:states)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    restored=[];monkeypatch.setattr(torch.cuda,'set_rng_state_all',lambda x:restored.extend(x))
    a=make();save(tmp_path,a);load_checkpoint(tmp_path,model=a[1],optimizer=a[2]);equal(restored,states)

def test_cuda_metadata_requires_matching_device_count_only_when_restoring(tmp_path,monkeypatch):
    a=make();save(tmp_path,a);path=tmp_path/'metadata.json';meta=json.loads(path.read_text());meta['cuda_rng_states_hex']=['010203'];path.write_text(json.dumps(meta))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(CheckpointError,match='device count'):load_checkpoint(tmp_path,model=a[1],optimizer=a[2])
    load_checkpoint(tmp_path,model=a[1],optimizer=a[2],restore_rng_state=False)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required for real FP16 GradScaler resume')
def test_cuda_fp16_exact_resume_with_scaler_and_cuda_rng(tmp_path,monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    old=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True);torch.manual_seed(123);torch.cuda.manual_seed_all(123)
        a=make('cuda');s=torch.amp.GradScaler('cuda',init_scale=128,growth_interval=2)
        update(a,s,True);update(a,s,True);save(tmp_path,a,s)
        expected=[update(a,s,True),update(a,s,True)]
        b=make('cuda');t=torch.amp.GradScaler('cuda',init_scale=16,growth_interval=99)
        torch.rand(37,device='cuda')
        assert load_checkpoint(tmp_path,model=b[1],optimizer=b[2],scaler=t)['step']==2
        assert [update(b,t,True),update(b,t,True)]==expected
        equal(a[1].state_dict(),b[1].state_dict());equal(a[2].state_dict(),b[2].state_dict());equal(s.state_dict(),t.state_dict())
    finally:torch.use_deterministic_algorithms(old,warn_only=warn)
