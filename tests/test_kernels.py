"""Every kernel backend against the fp64 plain-torch reference, forward and backward.

For each available backend on each device (eager on the CPU; Triton, `torch.compile` and eager on
CUDA): the unit-row op of a Linear, Conv2d, ConvTranspose2d and Embedding weight in fp32, bf16 and
fp16, with rows of ordinary, tiny (1e-8) and huge (1e8) norm and an all-zero row, with and without
a gain. The output and both gradients must match the reference computed in fp64 from the same
inputs to within one rounding of the result dtype (relative to the largest entry of the row).
The row norms the op returns must be fp32 (fp64 for an fp64 weight) and match fp64 to fp32
precision whatever the weight dtype.
"""

import math
import sys

import pytest
import torch

from dragonscale.functional.rows import reference
from dragonscale.kernels import available, ops, resolve
from dragonscale.spec import defaults
from dragonscale.spec.rows import RowSpec

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DTYPES = {torch.float32: 2e-6, torch.bfloat16: 1.6e-2, torch.float16: 2e-3}
SHAPES = {
    "linear": ((48, 40), defaults.rows()),
    "linear_wide": ((24, 3000), defaults.rows()),  # several column tiles in the Triton kernel
    "conv": ((16, 8, 3, 3), defaults.rows()),
    "conv_transpose": ((8, 6, 3, 3), defaults.conv_transpose((8, 6, 3, 3), 2)),
    "embedding": ((100, 32), defaults.embedding()),
}


def _weight(shape, spec: RowSpec, scale_kind: str) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    w = torch.randn(shape, generator=g, dtype=torch.float64)
    view = spec.view_of(shape)
    rows = spec.rows(shape)
    factors = torch.ones(rows, dtype=torch.float64)
    if scale_kind == "tiny":
        factors[::2] = 1e-8
    elif scale_kind == "huge":
        factors[::2] = 1e8
    elif scale_kind == "zero_row":
        factors[0] = 0.0
    from dragonscale.kernels.layout import as_rows

    return (w.reshape(view) * as_rows(factors, view, spec.row_dims)).reshape(shape)


def _row_err(a: torch.Tensor, b: torch.Tensor, spec: RowSpec) -> float:
    """Largest error per row, over the largest entry of that row of `b`."""
    from dragonscale.kernels.layout import reduced

    view = spec.view_of(b.shape)
    dims = reduced(view, spec.row_dims)
    diff = (a.double() - b).reshape(view).abs()
    top = b.reshape(view).abs()
    if dims:
        diff, top = diff.amax(dims), top.amax(dims)
    return (diff / top.clamp_min(1e-300)).max().item()


def _cases():
    for device in DEVICES:
        for backend in available(ops.UNIT_ROWS, device):
            for name in SHAPES:
                for dtype in DTYPES:
                    for kind in ("normal", "tiny", "huge", "zero_row"):
                        if dtype == torch.float16 and kind == "huge":
                            continue  # 1e8 is not an fp16 number
                        yield device, backend, name, dtype, kind


@pytest.mark.parametrize("device,backend,name,dtype,kind", list(_cases()))
@pytest.mark.parametrize("with_gain", [True, False])
def test_backend_matches_fp64_reference(device, backend, name, dtype, kind, with_gain):
    shape, spec = SHAPES[name]
    view, row_dims, radius = spec.view_of(shape), tuple(spec.row_dims), spec.radius_of(shape)
    w64 = _weight(shape, spec, kind)
    g = torch.Generator().manual_seed(1)
    rows = spec.rows(shape)
    gain64 = (torch.rand(rows, generator=g, dtype=torch.float64) + 0.5) if with_gain else None
    grad64 = torch.randn(shape, generator=g, dtype=torch.float64)

    w = w64.to(dtype).to(device)
    gain = None if gain64 is None else gain64.to(dtype).to(device)
    grad = grad64.to(dtype).to(device)

    # The reference sees the same rounded inputs, in fp64
    leaves = [w.double().cpu().requires_grad_()]
    if gain is not None:
        leaves.append(gain.double().cpu().requires_grad_())
    ref = reference(leaves[0], spec, leaves[1] if gain is not None else None)
    ref_grads = torch.autograd.grad(ref, leaves, grad.double().cpu())

    out, norms = resolve(ops.UNIT_ROWS, device, backend)(w, view, row_dims, gain, radius)
    grad_w, grad_gain = resolve(ops.UNIT_ROWS_BWD, device, backend)(
        grad, w, view, row_dims, norms, gain, radius
    )
    assert out.dtype == dtype and grad_w.dtype == dtype
    assert norms.dtype == torch.float32
    tol = DTYPES[dtype]
    assert torch.isfinite(out).all()
    # The gradient of a zero row is radius / FLOOR times the incoming one: past the fp16 range
    # for fp16 weights (under fp16 AMP the weights are fp32)
    if not (dtype == torch.float16 and kind in ("tiny", "zero_row")):
        assert torch.isfinite(grad_w).all()

    # Norms in fp32 whatever the weight dtype
    ref_norms = w.double().cpu().reshape(view)
    from dragonscale.kernels.layout import reduced

    ref_norms = ref_norms.square().sum(reduced(view, row_dims)).sqrt().reshape(-1)
    nonzero = ref_norms > 0
    rel = (norms.double().cpu()[nonzero] - ref_norms[nonzero]).abs() / ref_norms[nonzero]
    assert rel.max().item() < 1e-6

    assert _row_err(out.cpu(), ref.detach(), spec) < tol
    # A zero row's direction is undefined; its gradient must still be finite
    if kind != "zero_row" and not (dtype == torch.float16 and kind == "tiny"):
        assert _row_err(grad_w.cpu(), ref_grads[0], spec) < 4 * tol
    if gain is not None:
        assert grad_gain is not None and torch.isfinite(grad_gain).all()
        err = (grad_gain.double().cpu() - ref_grads[1]).abs().max()
        assert err.item() <= 4 * tol * ref_grads[1].abs().max().item() + 1e-30


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("amp", ["fp16", "bf16"])
def test_patched_layer_under_amp_matches_fp64(device, amp):
    """fp32 weights under autocast (fp16 AMP with the gradient scaler, bf16 AMP): the unit-row
    weight is computed in fp32 and only the matmul runs in the low dtype, so the layer output is
    as close to fp64 as a plain layer's."""
    import copy

    import dragonscale as ds

    if device == "cpu" and amp == "fp16":
        pytest.skip("fp16 autocast on the CPU is not a training setting")
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    torch.manual_seed(0)
    plain = torch.nn.Linear(256, 512).to(device)
    patched = copy.deepcopy(plain)
    ds.apply(patched, "unit_gain")
    x = torch.randn(64, 256, device=device)
    exact = torch.nn.functional.linear(x.double(), plain.weight.double(), plain.bias.double())
    scaler = torch.amp.GradScaler(device, enabled=amp == "fp16")
    errors = {}
    for name, layer in (("plain", plain), ("patched", patched)):
        with torch.autocast(device, dtype):
            y = layer(x)
        errors[name] = (y.double() - exact).abs().max().item()
        scaler.scale(y.float().square().mean()).backward()
        assert all(torch.isfinite(p.grad).all() for p in layer.parameters())
    assert errors["patched"] <= 2 * errors["plain"] + 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_never_updated_rows_stay_finite_on_every_backend(device):
    """A row decayed far below the floor (as an unused embedding row under weight decay) keeps a
    finite output and finite gradients on every backend (the analytic backward)."""
    w = torch.randn(8, 768, dtype=torch.float32) / math.sqrt(768)
    w[::2] *= math.exp(-60.0)  # sum of squares underflows to 0 in fp32
    w = w.to(device)
    gain = torch.ones(8, device=device)
    grad = torch.randn_like(w)
    for backend in available(ops.UNIT_ROWS, device):
        out, norms = resolve(ops.UNIT_ROWS, device, backend)(w, tuple(w.shape), (0,), gain, 1.0)
        gw, gg = resolve(ops.UNIT_ROWS_BWD, device, backend)(
            grad, w, tuple(w.shape), (0,), norms, gain, 1.0
        )
        for t in (out, norms, gw, gg):
            assert torch.isfinite(t).all(), backend


@pytest.mark.skipif(sys.platform == "win32", reason="Inductor needs a C++ compiler")
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", [None, "eager"])
def test_compiled_patched_model_matches_eager(device, backend):
    """`torch.compile` of a patched model (plain torch ops for the default backend, the library's
    custom op for a pinned one) gives the eager output and gradients, including the zero rows."""
    import copy

    import dragonscale as ds
    from tests.models import LM, tokens

    torch.manual_seed(0)
    model = LM().to(device)
    with torch.no_grad():
        model.embed.weight[:3].zero_()
    ds.apply(model, "unit_gain", backend=backend)
    twin = copy.deepcopy(model)
    compiled = torch.compile(twin, fullgraph=False)
    x = tokens().to(device)
    out_eager, out_compiled = model(x), compiled(x)
    torch.testing.assert_close(out_compiled, out_eager, rtol=1e-4, atol=1e-5)
    out_eager.square().mean().backward()
    out_compiled.square().mean().backward()
    for (name, a), b in zip(model.named_parameters(), twin.parameters(), strict=True):
        assert torch.isfinite(b.grad).all(), name
        torch.testing.assert_close(b.grad, a.grad, rtol=1e-3, atol=1e-6, msg=name)
