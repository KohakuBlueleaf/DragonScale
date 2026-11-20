"""Regression tests for the 0.2.0 fixes: weights read outside their layer's forward
(`nn.MultiheadAttention`), ops the analysis had no rule for, tracing with kernel backends, the
eval-mode weight cache, tied layers with differing options, and a layer read only by a
normalization."""

import dataclasses
import warnings

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import dragonscale as ds
from dragonscale.analysis import free_scales, suggest
from dragonscale.analysis.absorb import find_consumers
from dragonscale.problems import ProblemsError

# Weights read outside their own forward


def _mha_block() -> nn.Module:
    torch.manual_seed(0)
    return nn.TransformerEncoderLayer(16, 2, 32, dropout=0.0, batch_first=True).double()


def test_multihead_attention_is_patched_through_parametrize():
    layer, x = _mha_block(), torch.randn(2, 5, 16, dtype=torch.float64)
    before = layer(x)
    session = ds.apply(layer, "unit_gain")
    installs = {p.plan.path: p.plan.install for p in session.patched}
    assert installs == {
        "self_attn": "parametrize",
        "self_attn.out_proj": "parametrize",
        "linear1": "parametrize",
        "linear2": "parametrize",
    }
    torch.testing.assert_close(layer(x), before, rtol=1e-10, atol=1e-12)

    # Every gain takes effect, and the raw weights' scale does not
    gains = {p.plan.path: p.gain for p in session.patched}
    for path in installs:
        with torch.no_grad():
            gains[path].mul_(3.0)
        assert not torch.allclose(layer(x), before), path
        with torch.no_grad():
            gains[path].div_(3.0)
    with torch.no_grad():
        for raw, _ in session.raw_weights():
            raw.mul_(10.0)
    torch.testing.assert_close(layer(x), before, rtol=1e-10, atol=1e-12)


def test_encoder_layer_fast_path_uses_the_unit_rows():
    """In eval mode without gradients the encoder layer runs a fused kernel that reads the weights
    itself; it must see the unit-row weights."""
    layer, x = _mha_block().float(), torch.randn(2, 5, 16)
    session = ds.apply(layer, "unit_gain")
    with torch.no_grad():
        for gain in session.gains():
            gain.mul_(torch.rand_like(gain) + 0.5)
    expected = layer(x)
    layer.eval()
    with torch.no_grad():
        torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-5)


class _ReadsWeight(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = nn.Linear(8, 8)
        self.head = nn.Linear(8, 8)

    def forward(self, x):
        return F.linear(self.inner(x), self.head.weight)  # bypasses head's forward


def test_a_raw_weight_read_outside_its_forward_is_refused():
    model = _ReadsWeight()
    ds.apply(model, "unit_gain")
    with pytest.raises(ProblemsError, match="head: its raw weight was read outside"):
        model(torch.randn(2, 8))

    model = _ReadsWeight()
    ds.apply(model, [ds.Rule("head", install="parametrize"), ds.Rule(".*")])
    model(torch.randn(2, 8))  # the check passes; it runs once per mode
    model(torch.randn(2, 8))


def test_forward_install_on_a_parent_read_layer_is_refused():
    layer = _mha_block()
    with pytest.raises(ProblemsError, match="parent module reads its weight"):
        ds.apply(layer, [ds.Rule(r"self_attn\.out_proj", install="forward"), ds.Rule(".*")])


# Ops without a rule


def _report(model, x):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return free_scales(model.double(), x.double())


def test_in_place_relu_keeps_the_free_scale():
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    for inplace in (False, True):
        model = nn.Sequential(nn.Linear(8, 8, bias=False), nn.ReLU(inplace), nn.Linear(8, 8))
        report = _report(model, x)
        assert report.is_free({"0.weight": 1, "2.weight": -1}), inplace


class _Normalized(nn.Module):
    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.inner = nn.Linear(8, 8, bias=False)
        self.out = nn.Linear(8, 4)

    def forward(self, x):
        h = self.inner(x)
        match self.how:
            case "sqrt":
                h = h / h.square().sum(-1, keepdim=True).sqrt()
            case "norm":
                h = h / h.norm(dim=-1, keepdim=True)
            case "normalize":
                h = F.normalize(h, dim=-1)
            case "vector_norm":
                h = h / torch.linalg.vector_norm(h, dim=-1, keepdim=True)
        return self.out(h)


@pytest.mark.parametrize("how", ["sqrt", "norm", "normalize", "vector_norm"])
def test_vector_norms_have_rules(how):
    report = _report(_Normalized(how), torch.randn(4, 8))
    assert not report.unknown
    assert report.is_free({"inner.weight": 1})


class _MoreOps(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8, bias=False)
        self.b = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        a, b = self.a(x), self.b(x)
        h = torch.maximum(a, b) + torch.minimum(a, b)
        h = torch.addcmul(h, a, torch.ones_like(a))
        var, mean = torch.var_mean(h, dim=-1, keepdim=True)
        h = (h - mean) / (var + 1e-5).sqrt()
        h = h.relu_().clamp_(max=10.0)
        return torch.logsumexp(h, dim=-1)


def test_more_ops_have_rules():
    report = _report(_MoreOps(), torch.randn(4, 8))
    assert not report.unknown
    assert report.is_free({"a.weight": 1, "b.weight": 1})


class _Unknown(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)

    def forward(self, x):
        return torch.special.i0(self.a(x))


def test_unknown_ops_are_loud():
    model, x = _Unknown().double(), torch.randn(4, 8, dtype=torch.float64)
    with pytest.warns(UserWarning, match="without a degree rule"):
        report = free_scales(model, x)
    assert report.describe().startswith("WARNING")
    with pytest.raises(ValueError, match="allow_unknown"):
        suggest(model, x, level="absorb")
    suggest(model, x, level="absorb", allow_unknown=True)


# Tracing a patched model with a kernel backend


def test_trace_uses_plain_ops_whatever_the_backend():
    """A kernel that writes the unit-row weight into a new buffer hides the gain from a dispatch
    trace; the analysis computes the weight with plain torch ops instead."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8, elementwise_affine=False))
    model = nn.Sequential(model, nn.Linear(8, 4)).double()
    x = torch.randn(4, 8, dtype=torch.float64)
    session = ds.apply(model, "unit_gain")
    patched = session.patched[0]

    def opaque(w, view, row_dims, gain, radius):
        out = torch.empty_like(w)
        with torch.no_grad():
            norms = w.reshape(view).norm(dim=1)
            out.copy_(w * (gain * radius / norms)[:, None])
        return out, norms

    op = patched.unit_row_op(model[0][0].weight)
    patched._ops[x.device] = dataclasses.replace(op, forward=opaque)
    report = free_scales(model, x)
    assert report.is_free({"0.0.gain": 1, "0.0.bias": 1})


# Eval-mode cache


def test_eval_forward_sees_writes_through_data():
    torch.manual_seed(0)
    model = nn.Linear(8, 8).double()
    session = ds.apply(model, "unit_gain")
    x = torch.randn(2, 8, dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        first = model(x)
        model.gain.data.copy_(2 * model.gain.data)  # does not bump the version counter
        torch.testing.assert_close(model(x), 2 * first - model.bias)

        with session.cache_weights():
            cached = model(x)
            assert model(x) is not cached  # a new output, from the same cached weight
            model.gain.data.mul_(0.5)
            torch.testing.assert_close(model(x), cached)  # stale until invalidated
            session.invalidate()
            torch.testing.assert_close(model(x), first)


# Tied layers


def test_tied_layers_with_different_options_are_refused():
    torch.manual_seed(0)
    from tests.models import LM

    model = LM(tied=True)
    with pytest.raises(ProblemsError, match="shares its weight with embed"):
        ds.make_plan(model, [ds.Rule("embed", mode="unit", init="ones"), ds.Rule(".*")])
    plan = ds.make_plan(model, "unit_gain")  # only the layer types' default specs differ
    assert next(p for p in plan if p.path == "head").tied_to == "embed"


# A layer read only by a normalization


class _ConvNorm(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1, groups=4)
        self.norm = nn.GroupNorm(groups, 4, eps=1e-12, affine=False)
        self.out = nn.Conv2d(4, 2, 1)

    def forward(self, x):
        return self.out(self.norm(self.conv(x)))


def test_a_biased_layer_read_by_a_norm():
    """The conv's scale with its bias is free (a norm reads it). With one group per channel the
    rows can move to unit length on their own; with one group for all channels a per-row move
    changes the output and is refused, and the `absorb` level leaves the conv's gain."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 6, 6, dtype=torch.float64)
    model = _ConvNorm(4).double()
    report = free_scales(model, x)
    assert report.is_free({"conv.weight": 1, "conv.bias": 1})
    trace = ds.analysis.trace_degrees(model, x)
    assert find_consumers(model, "conv", trace, report) == []
    before = model(x)
    assert ds.analysis.absorb(model, "conv", x, into=[]) < 1e-9
    torch.testing.assert_close(model(x), before, rtol=1e-9, atol=1e-12)

    model = _ConvNorm(1).double()
    with pytest.raises(ValueError, match="changed the output"):
        ds.analysis.absorb(model, "conv", x, into=[])
    plan = suggest(model, x, level="absorb")
    assert "conv" not in dict(plan.moves)
    assert next(p for p in plan if p.path == "conv").mode == "unit_gain"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel backends run on CUDA")
def test_trace_on_cuda_matches_the_cpu():
    """With the Triton kernels as the default on CUDA, the analysis of a patched model gives the
    same free scales as on the CPU."""
    from tests.models import LM, tokens

    reports = []
    for device in ("cpu", "cuda"):
        torch.manual_seed(0)
        model = LM(depth=1).to(device)
        ds.apply(model, "unit_gain")
        reports.append(free_scales(model, tokens().to(device)))
    assert reports[0].summary() == reports[1].summary()
    assert torch.allclose(
        reports[0].basis @ reports[0].basis.T, reports[1].basis @ reports[1].basis.T
    )
