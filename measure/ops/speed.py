"""Speed of the unit-row ops per backend: forward (`unit_rows`) and backward (`unit_rows_bwd`),
median over `ROUNDS` timed calls, for layer shapes from the experiments and a large LM.

Both ops are memory-bound, so speed is reported as effective bandwidth: the bytes an op must move
divided by its time, and that as a share of the card's peak (`PEAK_GB_S`). The forward reads w and
gain and writes out and norms; the backward reads grad, w, norms and gain and writes grad_w and
grad_gain.

    kogine run measure/ops/speed.py
    kogine run measure/ops/speed.py --set DTYPE=bf16

Weights are fp32 by default: the master weights a patched layer reads under autocast.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from common import append_row, run_dir, table

from dragonscale.spec import defaults
from dragonscale_kernels import available, ops, resolve
from dragonscale_kernels.registry import load_backends

DEVICE = "cuda"
DTYPE = "fp32"  # "fp32" | "bf16" | "fp16"
ROUNDS = 200
WARMUP = 20
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
PEAK_GB_S = {  # spec memory bandwidth
    "NVIDIA GeForce RTX 4090": 1008,
    "NVIDIA GeForce RTX 3090": 936,
    "NVIDIA GeForce RTX 2080 Ti": 616,
    "Tesla V100-SXM2-16GB": 900,
    "Tesla PG503-216": 900,
}
SHAPES = {
    "byte LM qkv/out 256x256": (256, 256),
    "byte LM gate/up 768x256": (768, 256),
    "byte LM down 256x768": (256, 768),
    "JiT q/k/v/proj 768x768": (768, 768),
    "JiT gate/up 2048x768": (2048, 768),
    "JiT down 768x2048": (768, 2048),
    "LM 4096x4096": (4096, 4096),
    "LM up 11008x4096": (11008, 4096),
    "LM down 4096x11008": (4096, 11008),
    "embedding 32000x4096": (32000, 4096),
}


def median_ms(fn) -> float:
    for _ in range(WARMUP):
        fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ROUNDS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ROUNDS)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends, strict=True))
    return times[len(times) // 2]


def measure(name: str, shape: tuple[int, ...], dtype: torch.dtype, backend: str) -> dict:
    spec = defaults.embedding() if name.startswith("embedding") else defaults.rows()
    view, row_dims, radius = spec.view_of(shape), tuple(spec.row_dims), spec.radius_of(shape)
    w = torch.randn(shape, device=DEVICE, dtype=dtype)
    gain = torch.rand(spec.rows(shape), device=DEVICE, dtype=dtype) + 0.5
    grad = torch.randn_like(w)
    fwd = resolve(ops.UNIT_ROWS, DEVICE, backend)
    bwd = resolve(ops.UNIT_ROWS_BWD, DEVICE, backend)
    _, norms = fwd(w, view, row_dims, gain, radius)
    fwd_ms = median_ms(lambda: fwd(w, view, row_dims, gain, radius))
    bwd_ms = median_ms(lambda: bwd(grad, w, view, row_dims, norms, gain, radius))
    weight = w.numel() * w.element_size()
    rows = gain.numel() * gain.element_size() + norms.numel() * norms.element_size()
    fwd_bytes = 2 * weight + rows
    bwd_bytes = 3 * weight + 2 * rows
    peak = PEAK_GB_S.get(torch.cuda.get_device_name(DEVICE))
    fwd_gb_s = fwd_bytes / fwd_ms / 1e6
    bwd_gb_s = bwd_bytes / bwd_ms / 1e6
    return {
        "case": name,
        "backend": backend,
        "dtype": DTYPE,
        "fwd_ms": fwd_ms,
        "bwd_ms": bwd_ms,
        "total_ms": fwd_ms + bwd_ms,
        "fwd_gb_s": fwd_gb_s,
        "bwd_gb_s": bwd_gb_s,
        "fwd_peak": fwd_gb_s / peak if peak else None,
        "bwd_peak": bwd_gb_s / peak if peak else None,
    }


def main() -> None:
    load_backends()
    dtype = DTYPES[DTYPE]
    backends = available(ops.UNIT_ROWS, DEVICE)
    gpu = torch.cuda.get_device_name(DEVICE)
    out = run_dir("speed") / f"{gpu.replace(' ', '_')}-{DTYPE}.jsonl"
    out.unlink(missing_ok=True)
    rows = []
    for name, shape in SHAPES.items():
        by_backend = {}
        for backend in backends:
            row = {**measure(name, shape, dtype, backend), "gpu": gpu}
            append_row(out, row)
            by_backend[backend] = row
        best = min(by_backend.values(), key=lambda r: r["total_ms"])
        for row in by_backend.values():
            row["vs_best"] = row["total_ms"] / best["total_ms"]
            rows.append(row)
    print(gpu)
    columns = ["case", "backend", "fwd_ms", "bwd_ms", "vs_best", "fwd_gb_s", "bwd_gb_s"]
    columns += ["fwd_peak", "bwd_peak"]
    print(table(rows, columns))
    print(f"\nrows: {out}")


if __name__ == "__main__":
    main()
