"""Time of one forward + backward through a unit-row layer against the plain layer, per kernel
backend, with fp32 weights under autocast as in training: the two are timed in alternating blocks of
`BLOCK` queued steps, median over `ROUNDS` blocks each.

    kogine run measure/ops/overhead.py
    kogine run measure/ops/overhead.py --set DEVICE=cuda:2 --set AMP=fp16
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from common import append_row, run_dir, table
from torch import nn

import dragonscale as ds
from dragonscale_kernels import available, ops
from dragonscale_kernels.registry import load_backends

ROUNDS = 30
BLOCK = 10  # steps queued per timed block, as a training loop queues them
WARMUP = 10
DEVICE = "cuda:0"
AMP = "fp16"  # autocast dtype: "fp16" | "bf16" | "fp32" (no autocast)
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
CASES = {  # name: (layer factory, input shape); same weight shapes as measure/ops/speed.py
    "linear 256->768, 16384 tokens": (lambda: nn.Linear(256, 768), (16384, 256)),
    "linear 768->2048, 16384 tokens": (lambda: nn.Linear(768, 2048), (16384, 768)),
    "linear 4096->11008, 8192 tokens": (lambda: nn.Linear(4096, 11008), (8192, 4096)),
    "conv2d 256->256 3x3, 8x64x64": (lambda: nn.Conv2d(256, 256, 3, padding=1), (8, 256, 64, 64)),
}


def stepper(layer: nn.Module, x: torch.Tensor):
    """One forward + backward under autocast."""
    amp = AMP != "fp32"

    def step():
        layer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", DTYPES[AMP], enabled=amp):
            y = layer(x)
        y.float().square().mean().backward()

    return step


def median_ms(times: list[float]) -> float:
    return sorted(times)[len(times) // 2]


def measure(name: str, device: torch.device, backend: str) -> dict:
    """Plain and patched twins timed in alternating rounds, so both see the same clocks."""
    factory, shape = CASES[name]
    torch.manual_seed(0)
    plain = factory().to(device)
    patched = copy.deepcopy(plain)
    ds.apply(patched, "unit_gain", backend=backend)
    x = torch.randn(shape, device=device, requires_grad=True)
    steps = {"plain": stepper(plain, x), "patched": stepper(patched, x)}
    for step in steps.values():
        for _ in range(WARMUP):
            step()
    times = {key: [] for key in steps}
    for _ in range(ROUNDS):
        for key, step in steps.items():
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(BLOCK):
                step()
            end.record()
            end.synchronize()
            times[key].append(start.elapsed_time(end) / BLOCK)
    plain_ms, patched_ms = median_ms(times["plain"]), median_ms(times["patched"])
    return {
        "case": name,
        "backend": backend,
        "plain_ms": plain_ms,
        "patched_ms": patched_ms,
        "relative_speed": plain_ms / patched_ms,
    }


def main() -> None:
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)  # the timing events record on the current device
    load_backends()
    gpu = torch.cuda.get_device_name(device)
    out = run_dir("overhead") / f"{gpu.replace(' ', '_')}-{AMP}.jsonl"
    out.unlink(missing_ok=True)
    rows = []
    for backend in available(ops.UNIT_ROWS, device):
        for name in CASES:
            row = {**measure(name, device, backend), "amp": AMP, "gpu": gpu}
            append_row(out, row)
            rows.append(row)
    print(gpu)
    print(table(rows, ["case", "backend", "plain_ms", "patched_ms", "relative_speed"]))
    print(f"\nrows: {out}")


if __name__ == "__main__":
    main()
