"""Speed of the unit-row op per path: forward, and forward + backward through autograd, median
over timed calls, on weight shapes of common models. Weights in fp32 (the master weights a layer
reads under fp16 AMP or bf16 AMP), bf16 and fp16.

The op is memory-bound: it reads the weight and writes one of the same size. `gb_s` is the bytes
the forward and backward must move (read w, gain, grad; write out, norms, grad_w, grad_gain)
over their time.

    python measure/ops/speed.py --out runs/bench
"""

import argparse
from pathlib import Path

import torch
from common import DTYPES, median_ms, paths, unit_rows_fn, write

from dragonscale.spec import defaults

SHAPES = {
    "linear 768x768": ((768, 768), defaults.rows()),
    "linear 3072x768": ((3072, 768), defaults.rows()),
    "linear 768x3072": ((768, 3072), defaults.rows()),
    "linear 4096x4096": ((4096, 4096), defaults.rows()),
    "linear 11008x4096": ((11008, 4096), defaults.rows()),
    "embedding 50304x768": ((50304, 768), defaults.embedding()),
    "conv 256x256x3x3": ((256, 256, 3, 3), defaults.rows()),
    "depthwise 768x1x7x7": ((768, 1, 7, 7), defaults.rows()),
}


def measure(name: str, dtype: str, path: str, device: str) -> dict:
    shape, spec = SHAPES[name]
    w = torch.randn(shape, device=device, dtype=DTYPES[dtype], requires_grad=True)
    gain = torch.rand(spec.rows(shape), device=device, dtype=DTYPES[dtype]) + 0.5
    gain.requires_grad_(True)
    grad = torch.randn_like(w)
    fn = unit_rows_fn(path, spec, shape, device)

    def forward():
        with torch.no_grad():
            fn(w, gain)

    def both():
        out = fn(w, gain)
        torch.autograd.grad(out, (w, gain), grad)

    fwd, total = median_ms(forward), median_ms(both)
    weight = w.numel() * w.element_size()
    rows = gain.numel() * (gain.element_size() + 4)
    moved = 5 * weight + 3 * rows
    return {
        "shape": name,
        "dtype": dtype,
        "path": path,
        "fwd_ms": fwd,
        "fwd_bwd_ms": total,
        "gb_s": moved / total / 1e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("runs/bench"))
    args = parser.parse_args()
    rows = []
    for name in SHAPES:
        for dtype in DTYPES:
            group = [measure(name, dtype, p, args.device) for p in paths(args.device)]
            best = min(r["fwd_bwd_ms"] for r in group)
            for r in group:
                r["vs_best"] = r["fwd_bwd_ms"] / best
            rows += group
    columns = ["shape", "dtype", "path", "fwd_ms", "fwd_bwd_ms", "vs_best", "gb_s"]
    write(rows, args.out, "speed", columns)


if __name__ == "__main__":
    main()
