"""Precision of the unit-row op per path against fp64: the output and both gradients, for fp32,
bf16 and fp16 weights, next to the same formula written naively in plain PyTorch at the weight's
dtype (`w / w.norm(dim) * radius * gain`, the norm accumulated in that dtype).

Columns: the mean and the largest error per row, relative to the largest entry of that row of the
fp64 result (`rel_mean`, `rel_max`), for the path and for the naive formula.

    python measure/ops/precision.py --out runs/bench
"""

import argparse
from pathlib import Path

import torch
from common import DTYPES, paths, unit_rows_fn, write

from dragonscale.functional.rows import reference
from dragonscale.kernels.layout import as_rows, reduced
from dragonscale.spec import defaults

SHAPES = {
    "linear 768x768": ((768, 768), defaults.rows()),
    "linear 3072x768": ((3072, 768), defaults.rows()),
    "linear 768x3072": ((768, 3072), defaults.rows()),
    "linear 4096x11008": ((4096, 11008), defaults.rows()),
    "embedding 50304x768": ((50304, 768), defaults.embedding()),
    "conv 256x256x3x3": ((256, 256, 3, 3), defaults.rows()),
    "depthwise 768x1x7x7": ((768, 1, 7, 7), defaults.rows()),
}


def naive(w, gain, spec):
    view = spec.view_of(w.shape)
    rows = w.reshape(view)
    norm = torch.linalg.vector_norm(rows, dim=reduced(view, spec.row_dims), keepdim=True)
    scaled = rows / norm * spec.radius_of(w.shape) * as_rows(gain, view, spec.row_dims)
    return scaled.reshape(w.shape)


def row_errors(out, ref, spec) -> tuple[float, float]:
    view = spec.view_of(ref.shape)
    dims = reduced(view, spec.row_dims)
    diff = (out.double() - ref).reshape(view).abs()
    top = ref.reshape(view).abs()
    if dims:
        diff, top = diff.amax(dims), top.amax(dims)
    rel = diff / top.clamp_min(1e-300)
    return rel.mean().item(), rel.max().item()


def grads(fn, w, gain, grad):
    leaves = (w.detach().clone().requires_grad_(), gain.detach().clone().requires_grad_())
    out = fn(*leaves)
    return (out.detach(), *torch.autograd.grad(out, leaves, grad))


def measure(name, dtype, device) -> list[dict]:
    shape, spec = SHAPES[name]
    g = torch.Generator().manual_seed(0)
    w = torch.randn(shape, generator=g).to(device, DTYPES[dtype])
    gain = (torch.rand(spec.rows(shape), generator=g) + 0.5).to(device, DTYPES[dtype])
    grad = torch.randn(shape, generator=g).to(device, DTYPES[dtype])
    ref = grads(lambda a, b: reference(a, spec, b), w.double(), gain.double(), grad.double())
    base = grads(lambda a, b: naive(a, b, spec), w, gain, grad)
    rows = []
    for path in paths(device):
        got = grads(unit_rows_fn(path, spec, shape, device), w, gain, grad)
        for part, a, b, r in zip(("out", "grad_w", "grad_gain"), got, base, ref, strict=True):
            if part == "grad_gain":
                err = ((a.double() - r).abs().max() / r.abs().max()).item()
                naive_err = ((b.double() - r).abs().max() / r.abs().max()).item()
                mean, worst, naive_mean, naive_worst = None, err, None, naive_err
            else:
                mean, worst = row_errors(a, r, spec)
                naive_mean, naive_worst = row_errors(b, r, spec)
            rows.append(
                {
                    "shape": name,
                    "dtype": dtype,
                    "path": path,
                    "part": part,
                    "rel_mean": mean,
                    "rel_max": worst,
                    "naive_rel_mean": naive_mean,
                    "naive_rel_max": naive_worst,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("runs/bench"))
    args = parser.parse_args()
    rows = [r for name in SHAPES for dtype in DTYPES for r in measure(name, dtype, args.device)]
    columns = ["shape", "dtype", "path", "part", "rel_mean", "rel_max", "naive_rel_mean"]
    write(rows, args.out, "precision", [*columns, "naive_rel_max"])


if __name__ == "__main__":
    main()
