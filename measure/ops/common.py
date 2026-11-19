"""Shared by the benchmarks: the unit-row paths compared, timing, and output tables.

Paths of the unit-row op, each differentiable (forward and backward through autograd):

| path | what runs |
| --- | --- |
| `triton` | the Triton kernels (the default on CUDA when Triton is installed) |
| `compile` | `torch.compile` of the plain-torch ops, called as kernels |
| `eager` | the plain-torch ops (fp32 accumulation), one autograd node |
| `inductor` | the plain-torch formula traced into a `torch.compile` graph, as in a compiled model |
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dragonscale.functional.rows import compiled_unit_rows, unit_row_op  # noqa: E402
from dragonscale.kernels import available, ops  # noqa: E402
from dragonscale.spec.rows import RowSpec  # noqa: E402

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
torch._dynamo.config.cache_size_limit = 1024  # one compiled graph per shape and dtype


def paths(device: str) -> list[str]:
    """Every unit-row path available on `device`."""
    found = available(ops.UNIT_ROWS, device)
    return found + (["inductor"] if device.startswith("cuda") else [])


def unit_rows_fn(path: str, spec: RowSpec, shape, device: str):
    """`f(w, gain) -> unit-row weight` for one path."""
    if path == "inductor":
        view, dims, radius = spec.view_of(shape), tuple(spec.row_dims), spec.radius_of(shape)

        def plain(w, gain):
            return compiled_unit_rows(w, view, dims, radius, gain, None)

        return torch.compile(plain, dynamic=False)
    return unit_row_op(spec, shape, torch.device(device), path)


def median_ms(fn, rounds: int = 50, warmup: int = 10) -> float:
    """Median time of `fn` in milliseconds over `rounds` calls, by CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rounds)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rounds)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends, strict=True))
    return times[len(times) // 2]


def table(rows: list[dict], columns: list[str]) -> str:
    """A markdown table; floats in 4 significant digits."""

    def cell(value) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        return "-" if value is None else str(value)

    lines = ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    lines += ["| " + " | ".join(cell(r.get(c)) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


def write(rows: list[dict], out: Path, name: str, columns: list[str]) -> None:
    """`<out>/<name>.jsonl` and a markdown table in `<out>/<name>.md`, also printed."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{name}.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
    text = table(rows, columns)
    (out / f"{name}.md").write_text(text + "\n", encoding="utf-8")
    print(f"\n## {name} ({torch.cuda.get_device_name()})\n\n{text}")
