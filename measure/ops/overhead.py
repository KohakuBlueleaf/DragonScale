"""Training-step time of a patched layer (or a stack of layers) against the plain one: forward
and backward with fp32 weights in fp32, under bf16 AMP, and under fp16 AMP (autocast with the
gradient scaler). The two are timed in alternating blocks, median over blocks.

Settings per case: eager mode with each kernel backend, eager mode with the grouped launch
(stacks only), and `torch.compile` of both the plain and the patched module (default backend).

    python measure/ops/overhead.py --out runs/bench
"""

import argparse
import copy
from pathlib import Path

import torch
from common import median_ms, write
from torch import nn

import dragonscale as ds
from dragonscale.kernels import available, ops


class Stack(nn.Module):
    """`depth` transformer-sized blocks of plain linears (q, k, v, out, fc1, fc2), no attention:
    many small weights, as the grouped launch targets."""

    def __init__(self, dim=768, depth=12):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.extend(nn.Linear(dim, dim) for _ in range(4))
            self.layers.extend([nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)])

    def forward(self, x):
        for i in range(0, len(self.layers), 6):
            q, k, v, o, fc1, fc2 = self.layers[i : i + 6]
            x = x + o(q(x) + k(x) + v(x))
            x = x + fc2(torch.relu(fc1(x)))
        return x


CASES = {  # name: (module factory, input factory)
    "linear 768->3072, 16k tokens": (lambda: nn.Linear(768, 3072), lambda: torch.randn(16384, 768)),
    "linear 3072->768, 16k tokens": (
        lambda: nn.Linear(3072, 768),
        lambda: torch.randn(16384, 3072),
    ),
    "linear 4096->11008, 8k tokens": (
        lambda: nn.Linear(4096, 11008),
        lambda: torch.randn(8192, 4096),
    ),
    "embedding 50304x768, 16k tokens": (
        lambda: nn.Embedding(50304, 768),
        lambda: torch.randint(0, 50304, (16384,)),
    ),
    "conv 3x3 256, 8x64x64": (
        lambda: nn.Conv2d(256, 256, 3, padding=1),
        lambda: torch.randn(8, 256, 64, 64),
    ),
    "depthwise 7x7 768, 32x14x14": (
        lambda: nn.Conv2d(768, 768, 7, padding=3, groups=768),
        lambda: torch.randn(32, 768, 14, 14),
    ),
    "stack of 72 linears (12 blocks, dim 768), 4k tokens": (Stack, lambda: torch.randn(4096, 768)),
}
PRECISIONS = ("fp32", "bf16 AMP", "fp16 AMP")


def stepper(module: nn.Module, x: torch.Tensor, precision: str):
    """One forward and backward; fp16 AMP scales the loss with a gradient scaler."""
    dtype = {"bf16 AMP": torch.bfloat16, "fp16 AMP": torch.float16}.get(precision)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16 AMP")

    def step():
        module.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype or torch.float32, enabled=dtype is not None):
            y = module(x)
        scaler.scale(y.float().square().mean()).backward()

    return step


def settings(name: str) -> list[tuple[str, str | None, bool, bool]]:
    """(label, backend, group, compiled) per setting."""
    out = []
    for backend in available(ops.UNIT_ROWS, "cuda"):
        out.append((f"eager, {backend}", backend, False, False))
    if name.startswith("stack") and "triton" in available(ops.UNIT_ROWS_GROUPED, "cuda"):
        out.append(("eager, grouped", None, True, False))
    out.append(("torch.compile", None, False, True))
    return out


def measure(name: str, precision: str, label: str, backend, group: bool, compiled: bool) -> dict:
    factory, inputs = CASES[name]
    torch.manual_seed(0)
    plain = factory().cuda()
    patched = copy.deepcopy(plain)
    ds.apply(patched, "unit_gain", backend=backend, group=group)
    x = inputs().cuda()
    if x.is_floating_point():
        x.requires_grad_(True)
    if compiled:
        plain, patched = torch.compile(plain), torch.compile(patched)
    steps = {"plain": stepper(plain, x, precision), "patched": stepper(patched, x, precision)}
    times = {key: [] for key in steps}
    for _ in range(5):  # alternate, so both see the same clocks
        for key, step in steps.items():
            times[key].append(median_ms(step, rounds=20, warmup=5))
    plain_ms, patched_ms = sorted(times["plain"])[2], sorted(times["patched"])[2]
    return {
        "case": name,
        "precision": precision,
        "setting": label,
        "plain_ms": plain_ms,
        "patched_ms": patched_ms,
        "overhead": patched_ms / plain_ms - 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/bench"))
    args = parser.parse_args()
    rows = []
    for name in CASES:
        for precision in PRECISIONS:
            for setting in settings(name):
                rows.append(measure(name, precision, *setting))
    write(
        rows,
        args.out,
        "overhead",
        ["case", "precision", "setting", "plain_ms", "patched_ms", "overhead"],
    )


if __name__ == "__main__":
    main()
