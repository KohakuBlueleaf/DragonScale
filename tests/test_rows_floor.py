"""The plain-torch unit-row paths (`compiled_unit_rows` for the default backend, `reference`) keep a
finite gradient for rows that weight decay has shrunk towards zero, and are unchanged, value and
gradient, for every row with a norm above the floor."""

import math

import pytest
import torch

from dragonscale.functional.rows import compiled_unit_rows, reference, unit_rows
from dragonscale.spec.rows import RowSpec
from dragonscale_kernels import ops
from dragonscale_kernels.layout import as_rows, reduced, wide


def _old_compiled(w, view, row_dims, radius, gain):
    """The formula before the fix: `sum(v^2).sqrt().clamp_min(FLOOR)`."""
    v = wide(w).reshape(view)
    dims = reduced(view, row_dims)
    norms = v.square().sum(dims, keepdim=True).sqrt() if dims else v.abs()
    scale = radius / norms.clamp_min(ops.FLOOR)
    if gain is not None:
        scale = scale * as_rows(gain.to(v.dtype), view, row_dims)
    return (v * scale).to(w.dtype).reshape(w.shape)


def _grads(fn, w0, gain0):
    w = w0.clone().requires_grad_(True)
    gain = gain0.clone().requires_grad_(True)
    out = fn(w, gain)
    torch.manual_seed(1)
    (out * torch.randn_like(out)).sum().backward()
    return out.detach(), w.grad, gain.grad


def _shrunk(decay: float, shape=(8, 768)) -> torch.Tensor:
    """A weight after `decay` = weight decay times the summed learning rate with no update."""
    torch.manual_seed(0)
    return torch.randn(shape) / math.sqrt(shape[1]) * math.exp(-decay)


SPEC = RowSpec()


@pytest.mark.parametrize("decay", [0.0, 30.0, 48.5, 49.0, 50.0, 55.0, 80.0])
def test_shrunk_rows_have_finite_gradients(decay):
    w0 = _shrunk(decay)
    gain0 = torch.ones(w0.shape[0])
    view, dims, radius = SPEC.view_of(w0.shape), tuple(SPEC.row_dims), SPEC.radius_of(w0.shape)
    paths = {
        "compiled": lambda w, g: compiled_unit_rows(w, view, dims, radius, g, None),
        "reference": lambda w, g: reference(w, SPEC, g),
        "eager": lambda w, g: unit_rows(w, SPEC, g),
    }
    for name, fn in paths.items():
        out, grad_w, grad_gain = _grads(fn, w0, gain0)
        assert torch.isfinite(out).all(), name
        assert torch.isfinite(grad_w).all(), name
        assert torch.isfinite(grad_gain).all(), name


def test_old_formula_was_not_finite():
    """Guards the regression test itself: the old formula fails on the same input."""
    w0 = _shrunk(55.0)
    gain0 = torch.ones(w0.shape[0])
    view, dims, radius = SPEC.view_of(w0.shape), tuple(SPEC.row_dims), SPEC.radius_of(w0.shape)
    _, grad_w, _ = _grads(lambda w, g: _old_compiled(w, view, dims, radius, g), w0, gain0)
    assert not torch.isfinite(grad_w).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64])
@pytest.mark.parametrize("shape", [(64, 32), (16, 8, 3, 3), (5,)])
def test_normal_range_is_bitwise_unchanged(dtype, shape):
    torch.manual_seed(0)
    spec = RowSpec()
    w0 = torch.randn(shape) * torch.logspace(-4, 2, shape[0]).reshape(-1, *[1] * (len(shape) - 1))
    w0 = w0.to(dtype)
    rows = spec.rows(w0.shape)
    gain0 = torch.rand(rows, dtype=torch.float32) + 0.5
    view, dims, radius = spec.view_of(w0.shape), tuple(spec.row_dims), spec.radius_of(w0.shape)
    new = _grads(lambda w, g: compiled_unit_rows(w, view, dims, radius, g, None), w0, gain0)
    old = _grads(lambda w, g: _old_compiled(w, view, dims, radius, g), w0, gain0)
    for a, b in zip(new, old, strict=True):
        assert torch.equal(a, b)
