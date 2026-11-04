"""Triton grouped row ops: `unit_rows_grouped` and `unit_rows_grouped_bwd` over many weights in one
launch per (dtype, column tile).

Every weight is a contiguous `[rows, cols]` tensor with rows first. A table holds, per weight, its
pointers, shape and first program; each program finds its weight in the table and handles
`BLOCK_R` of its rows in one pass (the row fits one `BLOCK_C` tile).
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from dragonscale_kernels.backends.triton.common import ONE_PASS_BYTES, TILE
from dragonscale_kernels.ops import FLOOR

FWD_FIELDS = 8  # w, out, gain (0: none), norms, rows, cols, first program, unused
BWD_FIELDS = (
    10  # grad, w, norms, gain (0: none), grad_w, grad_gain, rows, cols, first program, unused
)
TL_DTYPES = {torch.float32: tl.float32, torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}


@triton.jit
def _find(table, n_tensors, FIELDS: tl.constexpr, START: tl.constexpr):
    """The table entry whose program range holds this program."""
    pid = tl.program_id(0)
    t = 0
    for i in range(n_tensors):
        if pid >= tl.load(table + i * FIELDS + START):
            t = i
    return t, pid - tl.load(table + t * FIELDS + START)


@triton.jit
def _grouped_fwd(table, radii, n_tensors, floor, W_T: tl.constexpr, G_T: tl.constexpr,
                 BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):  # fmt: skip
    t, block = _find(table, n_tensors, 8, 6)
    entry = table + t * 8
    w_ptr = tl.load(entry + 0).to(tl.pointer_type(W_T))
    out_ptr = tl.load(entry + 1).to(tl.pointer_type(W_T))
    gain_addr = tl.load(entry + 2)
    n_ptr = tl.load(entry + 3).to(tl.pointer_type(tl.float32))
    rows = tl.load(entry + 4)
    cols = tl.load(entry + 5)
    radius = tl.load(radii + t)

    r = block * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.arange(0, BLOCK_C)
    rm = r < rows
    m = rm[:, None] & (c < cols)[None, :]
    offsets = r[:, None] * cols + c[None, :]
    x = tl.load(w_ptr + offsets, mask=m, other=0.0).to(tl.float32)
    n = tl.sqrt(tl.sum(x * x, axis=1))
    tl.store(n_ptr + r, n, mask=rm)
    scale = radius / tl.maximum(n, floor)
    if gain_addr != 0:
        scale *= tl.load(gain_addr.to(tl.pointer_type(G_T)) + r, mask=rm, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, (x * scale[:, None]).to(W_T), mask=m)


@triton.jit
def _grouped_bwd(table, radii, n_tensors, floor, W_T: tl.constexpr, G_T: tl.constexpr,
                 BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):  # fmt: skip
    t, block = _find(table, n_tensors, 10, 8)
    entry = table + t * 10
    g_ptr = tl.load(entry + 0).to(tl.pointer_type(W_T))
    w_ptr = tl.load(entry + 1).to(tl.pointer_type(W_T))
    n_ptr = tl.load(entry + 2).to(tl.pointer_type(tl.float32))
    gain_addr = tl.load(entry + 3)
    gw_ptr = tl.load(entry + 4).to(tl.pointer_type(W_T))
    gg_addr = tl.load(entry + 5)
    rows = tl.load(entry + 6)
    cols = tl.load(entry + 7)
    radius = tl.load(radii + t)

    r = block * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.arange(0, BLOCK_C)
    rm = r < rows
    m = rm[:, None] & (c < cols)[None, :]
    offsets = r[:, None] * cols + c[None, :]
    x = tl.load(w_ptr + offsets, mask=m, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offsets, mask=m, other=0.0).to(tl.float32)
    n = tl.maximum(tl.load(n_ptr + r, mask=rm, other=1.0), floor)
    dot = tl.sum(x * g, axis=1) / n
    coef = radius / n
    if gain_addr != 0:
        coef *= tl.load(gain_addr.to(tl.pointer_type(G_T)) + r, mask=rm, other=0.0).to(tl.float32)
        tl.store(gg_addr.to(tl.pointer_type(G_T)) + r, (radius * dot).to(G_T), mask=rm)
    out = coef[:, None] * (g - (x / n[:, None]) * dot[:, None])
    tl.store(gw_ptr + offsets, out.to(W_T), mask=m)


def fits(w: Tensor) -> bool:
    """Whether `w` (as `[rows, cols]`) can join a grouped launch."""
    return (
        w.is_cuda
        and w.dim() == 2
        and w.is_contiguous()
        and w.dtype in TL_DTYPES
        and triton.next_power_of_2(max(w.shape[1], 1)) * 4 <= ONE_PASS_BYTES
    )


def _buckets(ws: list[Tensor], gains: list[Tensor | None]) -> dict[tuple, list[int]]:
    """Weight indices by (weight dtype, gain dtype, column tile)."""
    out: dict[tuple, list[int]] = {}
    for i, (w, gain) in enumerate(zip(ws, gains, strict=True)):
        gain_dtype = gain.dtype if gain is not None else w.dtype
        key = (w.dtype, gain_dtype, triton.next_power_of_2(max(w.shape[1], 1)))
        out.setdefault(key, []).append(i)
    return out


def _tile(block_c: int) -> tuple[int, int]:
    block_r = max(1, TILE // block_c)
    return block_r, min(16, max(4, block_r * block_c // 1024))


def unit_rows_grouped(
    ws: list[Tensor], gains: list[Tensor | None], radii: list[float]
) -> tuple[list[Tensor], list[Tensor]]:
    outs = [torch.empty_like(w) for w in ws]
    norms = [torch.empty(w.shape[0], device=w.device, dtype=torch.float32) for w in ws]
    for (w_dtype, g_dtype, block_c), members in _buckets(ws, gains).items():
        block_r, warps = _tile(block_c)
        rows, start = [], 0
        for i in members:
            w, gain = ws[i], gains[i]
            rows += [w.data_ptr(), outs[i].data_ptr(), gain.data_ptr() if gain is not None else 0]
            rows += [norms[i].data_ptr(), w.shape[0], w.shape[1], start, 0]
            start += triton.cdiv(w.shape[0], block_r)
        device = ws[members[0]].device
        table = torch.tensor(rows, dtype=torch.int64).to(device, non_blocking=True)
        radius = torch.tensor([radii[i] for i in members], dtype=torch.float32)
        with torch.cuda.device(device):
            _grouped_fwd[(start,)](
                table, radius.to(device, non_blocking=True), len(members), FLOOR,
                TL_DTYPES[w_dtype], TL_DTYPES[g_dtype], block_r, block_c, num_warps=warps,
            )  # fmt: skip
    return outs, norms


def unit_rows_grouped_bwd(
    grads: list[Tensor],
    ws: list[Tensor],
    norms: list[Tensor],
    gains: list[Tensor | None],
    radii: list[float],
) -> tuple[list[Tensor], list[Tensor | None]]:
    grads = [g.contiguous() for g in grads]
    grad_ws = [torch.empty_like(w) for w in ws]
    grad_gains = [torch.empty_like(g) if g is not None else None for g in gains]
    for (w_dtype, g_dtype, block_c), members in _buckets(ws, gains).items():
        block_r, warps = _tile(block_c)
        rows, start = [], 0
        for i in members:
            w, gain, grad_gain = ws[i], gains[i], grad_gains[i]
            rows += [grads[i].data_ptr(), w.data_ptr(), norms[i].data_ptr()]
            rows += [gain.data_ptr() if gain is not None else 0, grad_ws[i].data_ptr()]
            rows += [grad_gain.data_ptr() if grad_gain is not None else 0]
            rows += [w.shape[0], w.shape[1], start, 0]
            start += triton.cdiv(w.shape[0], block_r)
        device = ws[members[0]].device
        table = torch.tensor(rows, dtype=torch.int64).to(device, non_blocking=True)
        radius = torch.tensor([radii[i] for i in members], dtype=torch.float32)
        with torch.cuda.device(device):
            _grouped_bwd[(start,)](
                table, radius.to(device, non_blocking=True), len(members), FLOOR,
                TL_DTYPES[w_dtype], TL_DTYPES[g_dtype], block_r, block_c, num_warps=warps,
            )  # fmt: skip
    return grad_ws, grad_gains
