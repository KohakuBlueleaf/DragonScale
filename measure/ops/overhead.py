"""Time and peak memory of one forward + backward through a patched layer against the stock layer,
per backend, for Linear and Conv2d shapes.

    kogine run measure/ops/overhead.py --set DEVICE=cuda --set DTYPE=bf16
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from common import append_row, run_dir, table
from torch import nn

import dragonscale as ds
from dragonscale_kernels import available, ops
from dragonscale_kernels.registry import load_backends

ROUNDS = 20
DEVICE = "cuda"
DTYPE = "bf16"  # "fp32" | "bf16" | "fp16"
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
CASES = {
    "linear 4096x4096, 8192 tokens": (lambda: nn.Linear(4096, 4096), (8192, 4096)),
    "linear 4096x11008, 8192 tokens": (lambda: nn.Linear(4096, 11008), (8192, 4096)),
    "linear 11008x4096, 8192 tokens": (lambda: nn.Linear(11008, 4096), (8192, 11008)),
    "conv2d 256->256 3x3, 8x64x64": (lambda: nn.Conv2d(256, 256, 3, padding=1), (8, 256, 64, 64)),
}


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def step_cost(layer: nn.Module, x: torch.Tensor) -> tuple[float, float]:
    """Mean ms of forward + backward over `ROUNDS` after a warm call, and peak MiB (CUDA only)."""
    device = x.device
    layer(x).float().square().mean().backward()
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(ROUNDS):
        layer.zero_grad(set_to_none=True)
        layer(x).float().square().mean().backward()
    sync(device)
    ms = (time.perf_counter() - start) * 1000 / ROUNDS
    if device.type != "cuda":
        return ms, float("nan")
    return ms, torch.cuda.max_memory_allocated(device) / 2**20


def measure(name: str, device: torch.device, dtype: torch.dtype, backend: str) -> dict:
    factory, shape = CASES[name]
    torch.manual_seed(0)
    layer = factory().to(device, dtype)
    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    stock_ms, stock_mib = step_cost(layer, x)
    ds.apply(layer, "unit_gain", backend=backend)
    patched_ms, patched_mib = step_cost(layer, x)
    return {
        "case": name,
        "backend": backend,
        "stock_ms": stock_ms,
        "patched_ms": patched_ms,
        "time_ratio": patched_ms / stock_ms,
        "stock_mib": stock_mib,
        "patched_mib": patched_mib,
    }


def main() -> None:
    device, dtype = torch.device(DEVICE), DTYPES[DTYPE]
    load_backends()
    out = run_dir("overhead") / f"{DEVICE.replace(':', '')}-{DTYPE}.jsonl"
    out.unlink(missing_ok=True)
    rows = []
    for backend in available(ops.UNIT_ROWS, device):
        for name in CASES:
            row = {**measure(name, device, dtype, backend), "dtype": DTYPE}
            append_row(out, row)
            rows.append(row)
    columns = ["case", "backend", "stock_ms", "patched_ms", "time_ratio"]
    print(table(rows, [*columns, "stock_mib", "patched_mib"]))
    print(f"\nrows: {out}")


if __name__ == "__main__":
    main()
