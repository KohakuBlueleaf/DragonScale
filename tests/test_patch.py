"""Patch, remove and bake on stock torch models, for each install and a tied weight: preserve keeps
the function; zero rows learn through their gain; remove restores parameters, keys and forwards;
after the weights and gains move, eval, a deepcopy, the baked state and bake give the patched
output."""

import copy

import pytest
import torch
from torch import nn

import dragonscale as ds
from dragonscale import nn as dnn
from dragonscale.functional.rows import row_norms
from dragonscale.layer_types import LAYER_TYPES
from tests.models import LM, tokens


def _zero_writer() -> nn.Module:
    model = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))
    with torch.no_grad():
        model[2].weight.zero_()
    return model


MODELS = {
    "lm": (LM, tokens),
    "lm_tied": (lambda: LM(tied=True), tokens),
    "zero_writer": (_zero_writer, lambda: torch.randn(4, 16, dtype=torch.float64)),
}
CASES = [
    ("lm", "unit_gain"),
    ("lm", "parametrize_unit_gain"),
    ("lm_tied", "unit_gain"),
    ("lm_tied", "parametrize_unit_gain"),
    ("zero_writer", "unit_gain"),
]


def _build(name):
    torch.manual_seed(0)
    factory, inputs = MODELS[name]
    return factory().double().train(), inputs()


def _move(session) -> None:
    with torch.no_grad():
        for raw, _ in session.raw_weights():
            raw.add_(0.1 * torch.randn_like(raw))
        for gain in session.gains():
            gain.mul_(torch.rand_like(gain) + 0.5).add_(0.1 * torch.randn_like(gain))


@pytest.mark.parametrize("name,rules", CASES)
def test_patch_remove_bake(name, rules):
    model, x = _build(name)
    params = {k: id(p) for k, p in model.named_parameters()}
    state = {k: v.clone() for k, v in model.state_dict().items()}
    before = model(x)

    # Preserve keeps the function; rows that start at zero learn through their gain
    plan = ds.make_plan(model, rules)
    modules = dict(model.named_modules())
    zero = {
        p.path: row_norms(getattr(modules[p.path], LAYER_TYPES[p.layer_type].weight), p.spec) == 0
        for p in plan
        if p.tied_to is None
    }
    session = ds.Session(model, plan)
    assert torch.allclose(model(x), before, rtol=1e-10, atol=1e-12)
    model(x).square().sum().backward()
    for p in session.untied_layers():
        if p.gain is not None and zero[p.plan.path].any():
            assert (p.gain.grad[zero[p.plan.path]] != 0).all(), p.plan.path
    model.zero_grad(set_to_none=True)

    # Remove restores everything exactly
    session.remove()
    assert {k: id(p) for k, p in model.named_parameters()} == params
    assert model.state_dict().keys() == state.keys()
    assert all(torch.equal(model.state_dict()[k], v) for k, v in state.items())
    assert all("forward" not in vars(m) and "train" not in vars(m) for m in model.modules())

    # After a move: eval (cached), a deepcopy, the baked state and bake give the patched output
    session = ds.apply(model, rules)
    _move(session)
    patched = model(x)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), patched, rtol=1e-10, atol=1e-12)
        _move(session)
        patched = model(x)
    model.train()
    assert torch.allclose(model(x), patched, rtol=1e-10, atol=1e-12)

    clone = copy.deepcopy(model)
    with torch.no_grad():
        for raw, _ in ds.Session.from_model(clone).raw_weights():
            raw.add_(torch.randn_like(raw))
    assert torch.allclose(model(x), patched, rtol=1e-10, atol=1e-12)
    assert not torch.allclose(clone(x), patched)

    stock, _ = _build(name)
    stock.load_state_dict(session.state_dict("baked"))
    assert torch.allclose(stock(x), patched, rtol=1e-10, atol=1e-12)
    session.bake()
    assert model.state_dict().keys() == state.keys()
    assert torch.allclose(model(x), patched, rtol=1e-10, atol=1e-12)


def test_a_tied_pair_shares_one_gain():
    model, _ = _build("lm_tied")
    session = ds.apply(model, "unit_gain")
    assert model.embed.gain is model.head.gain
    assert len(session.gains()) == len(session.untied_layers())


@pytest.mark.parametrize("name,rules", CASES)
def test_canonicalize(name, rules):
    model, x = _build(name)
    session = ds.apply(model, rules)
    _move(session)
    patched = model(x)
    native = {k: v.clone() for k, v in session.state_dict("native").items()}
    params = {id(p) for p in model.parameters()}
    session.canonicalize()

    unit = [m for m in model.modules() if isinstance(m, dnn.UnitRows)]
    assert len(unit) == len(session.patched)
    assert {id(p) for p in model.parameters()} == params
    assert model.state_dict().keys() == native.keys()
    assert all(torch.equal(model.state_dict()[k], v) for k, v in native.items())
    assert torch.allclose(model(x), patched, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped unit rows run on CUDA")
def test_grouped_matches_per_layer():
    """The grouped forward gives the per-layer output and gradients, over two accumulating
    backward passes; remove drops its hooks."""
    torch.manual_seed(0)
    model = LM().float().cuda()
    session = ds.apply(model, "unit_gain")
    plain = copy.deepcopy(model)
    ds.patch.grouped.detach(plain)
    x = tokens().cuda()
    for m in (model, plain):
        for _ in range(2):
            m(x).pow(2).mean().backward()
    torch.testing.assert_close(model(x), plain(x), rtol=1e-5, atol=1e-6)
    for a, b in zip(model.parameters(), plain.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-6)
    session.remove()
    assert not model._forward_pre_hooks and not model._forward_hooks


class _Checkpointed(nn.Module):
    def __init__(self, reentrant: bool | None):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Linear(16, 16) for _ in range(3))
        self.unused = nn.Linear(16, 16)
        self.reentrant = reentrant

    def forward(self, x):
        for block in self.blocks:
            if self.reentrant is None:
                x = x + block(x)
            else:
                x = x + torch.utils.checkpoint.checkpoint(block, x, use_reentrant=self.reentrant)
        return x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped unit rows run on CUDA")
@pytest.mark.parametrize("reentrant", [None, True, False])
def test_grouped_checkpoint_and_unused(reentrant):
    """Under activation checkpointing the grouped model gives the per-layer gradients, and a layer
    the forward does not use gets no gradient."""
    torch.manual_seed(0)
    model = _Checkpointed(reentrant).cuda()
    ds.apply(model, "unit_gain")
    plain = copy.deepcopy(model)
    ds.patch.grouped.detach(plain)
    x = torch.randn(4, 16, device="cuda", requires_grad=True)
    for m in (model, plain):
        m(x).pow(2).mean().backward()
    assert model.unused.weight.grad is None and model.unused.gain.grad is None
    for a, b in zip(model.blocks.parameters(), plain.blocks.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-6)
