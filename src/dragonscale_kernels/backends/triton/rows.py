"""Triton row ops (contracts in `dragonscale_kernels.ops`) over a strided `[rows, cols]` view.

Each program owns `BLOCK_R` whole rows. With `ONE_PASS` a row fits in one `BLOCK_C` tile and is
read once; otherwise the columns are walked in `BLOCK_C` chunks, one pass for the reductions and one
for the writes. Accumulation is fp32, fp64 for fp64 inputs.
"""

import functools

import torch
import triton
import triton.language as tl
from torch import Tensor

from dragonscale_kernels.backends.triton.common import as_2d, from_2d, rows_cols, tiles
from dragonscale_kernels.ops import FLOOR


def _acc_dtype(t: Tensor) -> torch.dtype:
    return torch.float64 if t.dtype == torch.float64 else torch.float32


def on_device(fn):
    """Launch under the device of the first tensor argument, not the current device."""

    @functools.wraps(fn)
    def run(*args, **kwargs):
        first = next(a for a in args if isinstance(a, Tensor))
        with torch.cuda.device(first.device):
            return fn(*args, **kwargs)

    return run


@triton.jit
def _sumsq(w_ptr, rows_idx, row_mask, cols, s_r, s_c, BLOCK_C: tl.constexpr, WIDE: tl.constexpr):
    """Per-row sum of squares of `w` for the rows `rows_idx`."""
    acc_t = tl.float64 if WIDE else tl.float32
    acc = tl.zeros((rows_idx.shape[0],), acc_t)
    for c0 in range(0, cols, BLOCK_C):
        c = c0 + tl.arange(0, BLOCK_C)
        m = row_mask[:, None] & (c < cols)[None, :]
        x = tl.load(w_ptr + rows_idx[:, None] * s_r + c[None, :] * s_c, mask=m, other=0.0)
        x = x.to(acc_t)
        acc += tl.sum(x * x, axis=1)
    return acc


@triton.jit
def _norm_kernel(w_ptr, n_ptr, rows, cols, s_r, s_c, BLOCK_R: tl.constexpr,
                 BLOCK_C: tl.constexpr, WIDE: tl.constexpr):  # fmt: skip
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    rm = r < rows
    acc = _sumsq(w_ptr, r, rm, cols, s_r, s_c, BLOCK_C, WIDE)
    tl.store(n_ptr + r, tl.sqrt(acc), mask=rm)


@triton.jit
def _unit_kernel(w_ptr, g_ptr, out_ptr, n_ptr, rows, cols, s_r, s_c, o_r, o_c, radius, floor,
                 HAS_GAIN: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
                 WIDE: tl.constexpr, ONE_PASS: tl.constexpr):  # fmt: skip
    acc_t = tl.float64 if WIDE else tl.float32
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    rm = r < rows
    if ONE_PASS:
        c = tl.arange(0, BLOCK_C)
        m = rm[:, None] & (c < cols)[None, :]
        x = tl.load(w_ptr + r[:, None] * s_r + c[None, :] * s_c, mask=m, other=0.0).to(acc_t)
        n = tl.sqrt(tl.sum(x * x, axis=1))
        tl.store(n_ptr + r, n, mask=rm)
        scale = radius / tl.maximum(n, floor)
        if HAS_GAIN:
            scale *= tl.load(g_ptr + r, mask=rm, other=0.0).to(acc_t)
        y = (x * scale[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + r[:, None] * o_r + c[None, :] * o_c, y, mask=m)
        return
    n = tl.sqrt(_sumsq(w_ptr, r, rm, cols, s_r, s_c, BLOCK_C, WIDE))
    tl.store(n_ptr + r, n, mask=rm)
    scale = radius / tl.maximum(n, floor)
    if HAS_GAIN:
        scale *= tl.load(g_ptr + r, mask=rm, other=0.0).to(acc_t)

    # out = w * scale per row
    for c0 in range(0, cols, BLOCK_C):
        c = c0 + tl.arange(0, BLOCK_C)
        m = rm[:, None] & (c < cols)[None, :]
        x = tl.load(w_ptr + r[:, None] * s_r + c[None, :] * s_c, mask=m, other=0.0).to(acc_t)
        y = (x * scale[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + r[:, None] * o_r + c[None, :] * o_c, y, mask=m)


@triton.jit
def _bwd_kernel(g_ptr, w_ptr, n_ptr, gain_ptr, gw_ptr, gg_ptr, rows, cols, g_r, g_c, s_r, s_c,
                o_r, o_c, radius, floor, HAS_GAIN: tl.constexpr, BLOCK_R: tl.constexpr,
                BLOCK_C: tl.constexpr, WIDE: tl.constexpr, ONE_PASS: tl.constexpr):  # fmt: skip
    acc_t = tl.float64 if WIDE else tl.float32
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    rm = r < rows
    n = tl.maximum(tl.load(n_ptr + r, mask=rm, other=1.0).to(acc_t), floor)
    if ONE_PASS:
        c = tl.arange(0, BLOCK_C)
        m = rm[:, None] & (c < cols)[None, :]
        x = tl.load(w_ptr + r[:, None] * s_r + c[None, :] * s_c, mask=m, other=0.0).to(acc_t)
        g = tl.load(g_ptr + r[:, None] * g_r + c[None, :] * g_c, mask=m, other=0.0).to(acc_t)
        dot = tl.sum(x * g, axis=1) / n
        coef = radius / n
        if HAS_GAIN:
            coef *= tl.load(gain_ptr + r, mask=rm, other=0.0).to(acc_t)
            tl.store(gg_ptr + r, (radius * dot).to(gg_ptr.dtype.element_ty), mask=rm)
        out = coef[:, None] * (g - (x / n[:, None]) * dot[:, None])
        tl.store(gw_ptr + r[:, None] * o_r + c[None, :] * o_c,
                 out.to(gw_ptr.dtype.element_ty), mask=m)  # fmt: skip
        return

    # <w, G> per row
    dot = tl.zeros((BLOCK_R,), acc_t)
    for c0 in range(0, cols, BLOCK_C):
        c = c0 + tl.arange(0, BLOCK_C)
        m = rm[:, None] & (c < cols)[None, :]
        x = tl.load(w_ptr + r[:, None] * s_r + c[None, :] * s_c, mask=m, other=0.0).to(acc_t)
        g = tl.load(g_ptr + r[:, None] * g_r + c[None, :] * g_c, mask=m, other=0.0).to(acc_t)
        dot += tl.sum(x * g, axis=1)
    dot = dot / n
    coef = radius / n
    if HAS_GAIN:
        coef *= tl.load(gain_ptr + r, mask=rm, other=0.0).to(acc_t)
        tl.store(gg_ptr + r, (radius * dot).to(gg_ptr.dtype.element_ty), mask=rm)

    # grad_w = coef * (G - u <u, G>), u = w / n
    for c0 in range(0, cols, BLOCK_C):
        c = c0 + tl.arange(0, BLOCK_C)
        m = rm[:, None] & (c < cols)[None, :]
        x = tl.load(w_ptr + r[:, None] * s_r + c[None, :] * s_c, mask=m, other=0.0).to(acc_t)
        g = tl.load(g_ptr + r[:, None] * g_r + c[None, :] * g_c, mask=m, other=0.0).to(acc_t)
        out = coef[:, None] * (g - (x / n[:, None]) * dot[:, None])
        tl.store(gw_ptr + r[:, None] * o_r + c[None, :] * o_c,
                 out.to(gw_ptr.dtype.element_ty), mask=m)  # fmt: skip


@triton.jit
def _project_kernel(t_ptr, w_ptr, out_ptr, rows, cols, t_r, t_c, s_r, s_c, o_r, o_c, floor,
                    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr, WIDE: tl.constexpr):  # fmt: skip
    acc_t = tl.float64 if WIDE else tl.float32
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    rm = r < rows
    sq = tl.zeros((BLOCK_R,), acc_t)
    dot = tl.zeros((BLOCK_R,), acc_t)
    for c0 in range(0, cols, BLOCK_C):
        c = c0 + tl.arange(0, BLOCK_C)
        m = rm[:, None] & (c < cols)[None, :]
        x = tl.load(w_ptr + r[:, None] * s_r + c[None, :] * s_c, mask=m, other=0.0).to(acc_t)
        t = tl.load(t_ptr + r[:, None] * t_r + c[None, :] * t_c, mask=m, other=0.0).to(acc_t)
        sq += tl.sum(x * x, axis=1)
        dot += tl.sum(x * t, axis=1)
    coef = dot / tl.maximum(sq, floor * floor)

    # t - w <w, t> / |w|^2
    for c0 in range(0, cols, BLOCK_C):
        c = c0 + tl.arange(0, BLOCK_C)
        m = rm[:, None] & (c < cols)[None, :]
        x = tl.load(w_ptr + r[:, None] * s_r + c[None, :] * s_c, mask=m, other=0.0).to(acc_t)
        t = tl.load(t_ptr + r[:, None] * t_r + c[None, :] * t_c, mask=m, other=0.0).to(acc_t)
        out = t - x * coef[:, None]
        tl.store(out_ptr + r[:, None] * o_r + c[None, :] * o_c,
                 out.to(out_ptr.dtype.element_ty), mask=m)  # fmt: skip


def _grid(rows: int, block_r: int) -> tuple[int]:
    return (triton.cdiv(rows, block_r),)


@on_device
def row_norm(w: Tensor, view: tuple[int, ...], row_dims: tuple[int, ...]) -> Tensor:
    w2, _ = as_2d(w, view, row_dims)
    rows, cols = rows_cols(view, row_dims)
    norms = torch.empty(rows, device=w.device, dtype=_acc_dtype(w))
    block_r, block_c, warps, _ = tiles(rows, cols, w.element_size())
    _norm_kernel[_grid(rows, block_r)](
        w2, norms, rows, cols, *w2.stride(), block_r, block_c, w.dtype == torch.float64,
        num_warps=warps,
    )  # fmt: skip
    return norms


def _scaled_rows(w, view, row_dims, gain, radius, out_2d=None):
    """`w` scaled per row by `gain * radius / norm`, written into `out_2d` (else a new tensor)."""
    w2, _ = as_2d(w, view, row_dims)
    rows, cols = rows_cols(view, row_dims)
    out = torch.empty((rows, cols), device=w.device, dtype=w.dtype) if out_2d is None else out_2d
    norms = torch.empty(rows, device=w.device, dtype=_acc_dtype(w))
    block_r, block_c, warps, one_pass = tiles(rows, cols, w.element_size())
    gain_arg = gain.contiguous() if gain is not None else norms
    _unit_kernel[_grid(rows, block_r)](
        w2, gain_arg, out, norms, rows, cols, *w2.stride(), *out.stride(), float(radius),
        FLOOR, gain is not None, block_r, block_c, w.dtype == torch.float64, one_pass,
        num_warps=warps,
    )  # fmt: skip
    return out, norms


@on_device
def unit_rows(w, view, row_dims, gain, radius):
    out2, norms = _scaled_rows(w, view, row_dims, gain, radius)
    return from_2d(out2, view, row_dims, w.shape), norms


@on_device
def unit_rows_bwd(grad, w, view, row_dims, norms, gain, radius):
    g2, _ = as_2d(grad, view, row_dims)
    w2, _ = as_2d(w, view, row_dims)
    rows, cols = rows_cols(view, row_dims)
    grad_w = torch.empty((rows, cols), device=w.device, dtype=w.dtype)
    grad_gain = None if gain is None else torch.empty_like(gain)
    block_r, block_c, warps, one_pass = tiles(rows, cols, w.element_size())
    _bwd_kernel[_grid(rows, block_r)](
        g2, w2, norms, gain if gain is not None else norms, grad_w,
        grad_gain if grad_gain is not None else norms, rows, cols, *g2.stride(),
        *w2.stride(), *grad_w.stride(), float(radius), FLOOR, gain is not None, block_r,
        block_c, w.dtype == torch.float64, one_pass, num_warps=warps,
    )  # fmt: skip
    return from_2d(grad_w, view, row_dims, w.shape), grad_gain


@on_device
def project(t, w, view, row_dims):
    t2, _ = as_2d(t, view, row_dims)
    w2, _ = as_2d(w, view, row_dims)
    rows, cols = rows_cols(view, row_dims)
    out = torch.empty((rows, cols), device=t.device, dtype=t.dtype)
    block_r, block_c, warps, _ = tiles(rows, cols, max(t.element_size(), w.element_size()))
    _project_kernel[_grid(rows, block_r)](
        t2, w2, out, rows, cols, *t2.stride(), *w2.stride(), *out.stride(), FLOOR, block_r,
        block_c, (t.dtype == torch.float64) or (w.dtype == torch.float64), num_warps=warps,
    )  # fmt: skip
    return from_2d(out, view, row_dims, t.shape)


@on_device
def rescale_rows_(w, view, row_dims, radius):
    with torch.no_grad():
        w2, aliased = as_2d(w, view, row_dims)
        if aliased:
            _, norms = _scaled_rows(w, view, row_dims, None, radius, out_2d=w2)
            return norms
        out, norms = unit_rows(w, view, row_dims, None, radius)
        w.copy_(out)
        return norms


@on_device
def project_rescale_rows_(w, moments, view, row_dims, radius):
    with torch.no_grad():
        for moment in moments:
            moment.copy_(project(moment, w, view, row_dims))
        return rescale_rows_(w, view, row_dims, radius)
