"""Precision of each backend's unit-row op (forward and backward) at fp16 and bf16, next to the
same computation written in plain PyTorch. Both are measured against the fp64 result.

    kogine run measure/ops/precision.py --set DEVICE=cuda
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from common import append_row, run_dir, table

from dragonscale.functional.rows import reference
from dragonscale.spec import defaults
from dragonscale_kernels import available, ops, resolve
from dragonscale_kernels.error_stats import compare
from dragonscale_kernels.layout import as_rows, reduced
from dragonscale_kernels.registry import load_backends

SHAPES = {
    "linear 4096x1024 rows": ((4096, 1024), defaults.rows()),
    "linear 1024x4096 rows": ((1024, 4096), defaults.rows()),
    "linear 4096x1024 columns": ((4096, 1024), defaults.columns()),
    "conv 256x128x3x3 rows": ((256, 128, 3, 3), defaults.rows()),
    "embedding 32000x512": ((32000, 512), defaults.embedding()),
}
DTYPES = (torch.float16, torch.bfloat16)
DEVICE = "cuda"  # "cuda" | "cuda:<n>" | "cpu"


def plain_pytorch(w, gain, spec):
    """Unit rows as one would write them in PyTorch, at the input dtype."""
    view = spec.view_of(w.shape)
    rows = w.reshape(view)
    norm = rows.norm(dim=reduced(view, spec.row_dims), keepdim=True)
    radius = spec.radius_of(w.shape)
    return (rows / norm * radius * as_rows(gain, view, spec.row_dims)).reshape(w.shape)


def measure(name, shape, spec, dtype, device, backend) -> list[dict]:
    generator = torch.Generator(device="cpu").manual_seed(0)
    w64 = torch.randn(shape, generator=generator, dtype=torch.float64).to(device)
    gain64 = torch.rand(spec.rows(shape), generator=generator, dtype=torch.float64) + 0.5
    gain64 = gain64.to(device)
    grad64 = torch.randn(shape, generator=generator, dtype=torch.float64).to(device)
    w, gain, grad = w64.to(dtype), gain64.to(dtype), grad64.to(dtype)
    view, row_dims, radius = spec.view_of(shape), tuple(spec.row_dims), spec.radius_of(shape)

    # Ruler: fp64 reference and its autograd backward
    leaves64 = (w.double().requires_grad_(), gain.double().requires_grad_())
    ref = reference(leaves64[0], spec, leaves64[1])
    ref_gw, ref_gg = torch.autograd.grad(ref, leaves64, grad.double())

    # Baseline: plain PyTorch at the dtype
    leaves = (w.clone().requires_grad_(), gain.clone().requires_grad_())
    base = plain_pytorch(leaves[0], gain=leaves[1], spec=spec)
    base_gw, base_gg = torch.autograd.grad(base, leaves, grad)

    # Candidate: the backend's ops
    out, norms = resolve(ops.UNIT_ROWS, device, backend)(w, view, row_dims, gain, radius)
    gw, gg = resolve(ops.UNIT_ROWS_BWD, device, backend)(
        grad, w, view, row_dims, norms, gain, radius
    )

    rows = []
    for part, cand, plain, ruler in (
        ("forward", out, base, ref),
        ("grad weight", gw, base_gw, ref_gw),
        ("grad gain", gg, base_gg, ref_gg),
    ):
        report = compare(cand, plain, ruler, floor=1e-30)
        rows.append(
            {
                "case": name,
                "dtype": str(dtype).removeprefix("torch."),
                "backend": backend,
                "part": part,
                "rel_mean": report["candidate"]["rel_mean"],
                "rel_mean_pytorch": report["baseline"]["rel_mean"],
                "worst_ratio": max(report["ratio"].values()),
                "mark": report["mark"],
                "report": report,
            }
        )
    return rows


def main() -> None:
    load_backends()
    out = run_dir("precision") / f"{DEVICE.replace(':', '')}.jsonl"
    out.unlink(missing_ok=True)
    rows = []
    for backend in available(ops.UNIT_ROWS, DEVICE):
        for name, (shape, spec) in SHAPES.items():
            for dtype in DTYPES:
                for row in measure(name, shape, spec, dtype, DEVICE, backend):
                    append_row(out, row)
                    rows.append(row)
    columns = ["case", "dtype", "backend", "part", "rel_mean", "rel_mean_pytorch", "worst_ratio"]
    print(table(rows, [*columns, "mark"]))
    print(f"\nrows: {out}")


if __name__ == "__main__":
    main()
