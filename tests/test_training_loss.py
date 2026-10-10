"""Tests for training.loss (NB-07).

VERIFICATION STATUS: written without a PyTorch installation available.
NOT executed by the assistant. See training/__init__.py.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.loss import causal_lm_loss  # noqa: E402


def test_loss_is_scalar_and_finite():
    logits = torch.randn(2, 4, 10, requires_grad=True)
    targets = torch.randint(0, 10, (2, 4))

    loss = causal_lm_loss(logits, targets)

    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_loss_gradient_flows_to_logits():
    logits = torch.randn(2, 4, 10, requires_grad=True)
    targets = torch.randint(0, 10, (2, 4))

    loss = causal_lm_loss(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert torch.any(logits.grad != 0)


def test_perfect_prediction_gives_near_zero_loss():
    vocab_size = 5
    targets = torch.tensor([[0, 1, 2]])
    # One-hot-like huge logits at the correct target position.
    logits = torch.full((1, 3, vocab_size), -1e4)
    for t in range(3):
        logits[0, t, targets[0, t]] = 1e4

    loss = causal_lm_loss(logits, targets)
    assert loss.item() < 1e-3


def test_uniform_logits_give_expected_loss_magnitude():
    # Uniform logits over vocab_size classes -> cross-entropy should
    # equal ln(vocab_size), the well-known "random guessing" baseline.
    import math

    vocab_size = 8
    logits = torch.zeros(1, 3, vocab_size)
    targets = torch.randint(0, vocab_size, (1, 3))

    loss = causal_lm_loss(logits, targets)
    assert abs(loss.item() - math.log(vocab_size)) < 1e-4

@pytest.mark.parametrize('noncontiguous', [False, True])
def test_flattened_loss_matches_legacy_values_gradients_and_evaluation(noncontiguous):
    from training.loss import evaluation_nll_count
    torch.manual_seed(91)
    data=torch.randn(2,3,7,dtype=torch.float64)
    target=torch.tensor([[1,-100,3],[0,5,6]])
    if noncontiguous:
        data=data.transpose(0,1);target=target.transpose(0,1)
    actual=data.detach().requires_grad_();reference=data.detach().clone().requires_grad_()
    loss=causal_lm_loss(actual,target)
    expected=torch.nn.functional.cross_entropy(reference.transpose(1,2),target,ignore_index=-100)
    torch.testing.assert_close(loss,expected,rtol=1e-12,atol=1e-12)
    loss.backward();expected.backward()
    torch.testing.assert_close(actual.grad,reference.grad,rtol=1e-12,atol=1e-12)
    nll,count=evaluation_nll_count(data,target)
    selected=-data.log_softmax(-1)[target!=-100].gather(1,target[target!=-100,None]).sum().item()
    assert count==5
    assert nll==pytest.approx(selected,rel=1e-12)

def test_flattened_all_ignored_keeps_training_nan_and_evaluation_zero():
    from training.loss import evaluation_nll_count
    logits=torch.randn(2,3,7)
    target=torch.full((2,3),-100)
    assert torch.isnan(causal_lm_loss(logits,target))
    assert evaluation_nll_count(logits,target)==(0.0,0)

@pytest.mark.parametrize('evaluation',[False,True])
def test_flattening_rejects_wrong_position_shape_even_with_equal_numel(evaluation):
    from training.loss import evaluation_nll_count
    fn=evaluation_nll_count if evaluation else causal_lm_loss
    with pytest.raises(RuntimeError,match='aligned'):
        fn(torch.randn(2,3,7),torch.zeros(3,2,dtype=torch.long))

def test_both_losses_use_two_dimensional_cross_entropy(monkeypatch):
    from training import loss as module
    original=module.F.cross_entropy;calls=[]
    def checked(logits,targets,**kw):
        calls.append((tuple(logits.shape),tuple(targets.shape),kw['ignore_index']))
        return original(logits,targets,**kw)
    monkeypatch.setattr(module.F,'cross_entropy',checked)
    logits=torch.randn(2,3,7);target=torch.zeros(2,3,dtype=torch.long)
    module.causal_lm_loss(logits,target);module.evaluation_nll_count(logits,target)
    assert calls==[((6,7),(6,),-100)]*2
