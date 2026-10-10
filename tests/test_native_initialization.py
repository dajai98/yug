"""Initializer changes fresh tied weights only; checkpoint functions stay identical."""
import importlib.util
from pathlib import Path
import pytest
import torch
from yug.native.config import ModelConfig, estimate_parameter_count
from yug.native.model import YugTransformer

@pytest.fixture(scope='module')
def legacy():
    path=Path(__file__).resolve().parents[1]/'artifacts/model_initialization_gate_v1/legacy_model.py'
    spec=importlib.util.spec_from_file_location('pre_init_fix_model',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.YugTransformer

def config(**kw):
    values=dict(vocab_size=64,tokenizer_identity='sha256:'+'0'*64,hidden_size=32,num_layers=3,num_heads=4,ffn_size=96,context_length=32,dropout=0.0)
    values.update(kw)
    return ModelConfig(**values)

@pytest.mark.parametrize('options',[{}, {'tie_embeddings':False}, {'positional_encoding':'learned','bias':True}, {'normalization':'layernorm','activation':'swiglu'}])
def test_only_tied_embedding_changes_and_rng_is_preserved(legacy,options):
    cfg=config(**options)
    torch.manual_seed(17);old=legacy(cfg);old_rng=torch.get_rng_state().clone()
    torch.manual_seed(17);new=YugTransformer(cfg);new_rng=torch.get_rng_state().clone()
    assert torch.equal(old_rng,new_rng)
    assert old.state_dict().keys()==new.state_dict().keys()
    for name,value in old.state_dict().items():
        expected=value*cfg.hidden_size**-0.5 if cfg.tie_embeddings and name in ('token_embedding.weight','lm_head.weight') else value
        assert torch.equal(new.state_dict()[name],expected),name
    assert new.count_parameters()==old.count_parameters()==estimate_parameter_count(cfg).total
    assert (new.lm_head.weight is new.token_embedding.weight)==cfg.tie_embeddings

def test_old_checkpoint_replaces_initializer_without_output_change(legacy,tmp_path):
    cfg=config();torch.manual_seed(23);old=legacy(cfg).eval()
    path=tmp_path/'legacy_weights.pt';torch.save(old.state_dict(),path)
    torch.manual_seed(99);new=YugTransformer(cfg).eval()
    new.load_state_dict(torch.load(path,weights_only=True))
    assert new.lm_head.weight is new.token_embedding.weight
    ids=torch.tensor([[1,2,3,4,5]])
    with torch.no_grad():assert torch.equal(old(ids),new(ids))

def test_corrected_initialization_reproduces_weights_and_logits():
    cfg=config();torch.manual_seed(31);a=YugTransformer(cfg).eval()
    torch.manual_seed(31);b=YugTransformer(cfg).eval()
    assert all(torch.equal(x,b.state_dict()[k]) for k,x in a.state_dict().items())
    ids=torch.tensor([[1,3,5,7]])
    with torch.no_grad():assert torch.equal(a(ids),b(ids))

def test_corrected_tied_weights_receive_finite_gradient():
    m=YugTransformer(config());ids=torch.tensor([[1,2,3,4]])
    torch.nn.functional.cross_entropy(m(ids).reshape(-1,64),torch.tensor([2,3,4,5])).backward()
    assert m.lm_head.weight is m.token_embedding.weight
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
