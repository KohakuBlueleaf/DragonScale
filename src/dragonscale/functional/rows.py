"""Unit-row weights per `RowSpec`: the autograd reference and the kernel-op path.

`gain` is flat `[rows]` (`channel`) or one element (`scalar`), broadcast over every row.

The kernel-op path is an autograd Function in eager mode. Under `torch.compile` it is plain torch
ops for the default and `compile` backends, which Inductor fuses, and the custom ops
`dragonscale::unit_rows` / `dragonscale::unit_rows_bwd` for a pinned kernel backend, which trace as
one opaque op each.
"""

import functools
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from dragonscale.spec.rows import RowSpec
from dragonscale_kernels import ops, resolve
from dragonscale_kernels.layout import as_rows, reduced, wide


def _gain_rows(gain: Tensor | None, rows: int) -> Tensor | None:
    """A channel gain as is, a scalar gain expanded to `rows`."""
    if gain is None or gain.numel() == rows:
        return gain
    return gain.reshape(1).expand(rows)


def row_norms(w: Tensor, spec: RowSpec) -> Tensor:
    """Row norms, flat `[rows]`, in at least fp32; differentiable."""
    view = spec.view_of(w.shape)
    v = wide(w).reshape(view)
    dims = reduced(view, spec.row_dims)
    out = v.square().sum(dims).sqrt() if dims else v.abs()
    return out.reshape(-1)


def reference(w: Tensor, spec: RowSpec, gain: Tensor | None = None) -> Tensor:
    """`gain * radius * w / norm` per row in plain torch ops."""
    view = spec.view_of(w.shape)
    rows = spec.rows(w.shape)
    v = wide(w).reshape(view)
    scale = spec.radius_of(w.shape) / row_norms(w, spec).clamp_min(ops.FLOOR)
    gain = _gain_rows(gain, rows)
    if gain is not None:
        scale = scale * gain.to(scale.dtype)
    return (v * as_rows(scale, view, spec.row_dims)).to(w.dtype).reshape(w.shape)


def gain_scaled(w: Tensor, spec: RowSpec, gain: Tensor | None = None) -> Tensor:
    """The weight of rows kept at the radius by the optimizer: `gain * w` (`w` when no gain)."""
    gain = _gain_rows(gain, spec.rows(w.shape))
    if gain is None:
        return w
    view = spec.view_of(w.shape)
    scaled = wide(w).reshape(view) * as_rows(gain.to(wide(w).dtype), view, spec.row_dims)
    return scaled.to(w.dtype).reshape(w.shape)


def fill_zero_rows_(w: Tensor, spec: RowSpec) -> Tensor | None:
    """In place: every all-zero row gets a random direction at the radius; returns the flat
    `[rows]` mask of the filled rows, or None when no row was zero."""
    with torch.no_grad():
        view = spec.view_of(w.shape)
        zero = row_norms(w, spec) <= ops.FLOOR
        if not bool(zero.any()):
            return None
        draw = torch.randn(view, device=w.device, dtype=wide(w).dtype)
        draw = draw * as_rows(
            spec.radius_of(w.shape) / row_norms(draw, spec).clamp_min(ops.FLOOR),
            view,
            spec.row_dims,
        )
        mask = as_rows(zero, view, spec.row_dims)
        filled = torch.where(mask, draw, wide(w).reshape(view))
        w.copy_(filled.to(w.dtype).reshape(w.shape))
        return zero


def zero_rows_(w: Tensor, spec: RowSpec, mask: Tensor) -> None:
    """In place: the rows in the flat `[rows]` `mask` set to zero."""
    with torch.no_grad():
        view = spec.view_of(w.shape)
        kept = wide(w).reshape(view).masked_fill(as_rows(mask, view, spec.row_dims), 0)
        w.copy_(kept.to(w.dtype).reshape(w.shape))


def preserving_gain(w: Tensor, spec: RowSpec) -> Tensor:
    """The channel gain with `reference(w, spec, gain) == w`: row norm over radius."""
    with torch.no_grad():
        return row_norms(w, spec) / spec.radius_of(w.shape)


@dataclass(frozen=True)
class UnitRowOp:
    """The unit-row op of one (spec, weight shape, device), bound to kernel implementations."""

    view: tuple[int, ...]
    row_dims: tuple[int, ...]
    radius: float
    rows: int
    forward: object
    backward: object

    backend: str | None = None

    def __call__(self, w: Tensor, gain: Tensor | None = None) -> Tensor:
        gain = _gain_rows(gain, self.rows)
        if torch.compiler.is_compiling():
            return compiled_unit_rows(w, self.view, self.row_dims, self.radius, gain, self.backend)
        return UnitRowsFn.apply(w, gain, self)


class UnitRowsFn(torch.autograd.Function):
    """Unit rows through the kernel ops; saves the raw weight and fp32 norms, not the output."""

    @staticmethod
    def forward(ctx, w: Tensor, gain: Tensor | None, op: UnitRowOp) -> Tensor:
        out, norms = op.forward(w, op.view, op.row_dims, gain, op.radius)
        ctx.save_for_backward(w, norms, gain)
        ctx.op = op
        return out

    @staticmethod
    def backward(ctx, grad: Tensor):
        w, norms, gain = ctx.saved_tensors
        op = ctx.op
        grad_w, grad_gain = op.backward(
            grad.contiguous(), w, op.view, op.row_dims, norms, gain, op.radius
        )
        return grad_w, grad_gain, None


def compiled_unit_rows(
    w: Tensor,
    view: tuple[int, ...],
    row_dims: tuple[int, ...],
    radius: float,
    gain: Tensor | None,
    backend: str | None,
) -> Tensor:
    """The op under `torch.compile`: plain torch ops for the default and `compile` backends, for
    Inductor to fuse and differentiate; the opaque custom op for a pinned kernel backend."""
    if backend in (None, "compile"):
        v = wide(w).reshape(view)
        dims = reduced(view, row_dims)
        norms = v.square().sum(dims, keepdim=True).sqrt() if dims else v.abs()
        scale = radius / norms.clamp_min(ops.FLOOR)
        if gain is not None:
            scale = scale * as_rows(gain.to(v.dtype), view, row_dims)
        return (v * scale).to(w.dtype).reshape(w.shape)
    if gain is not None:
        gain = gain.contiguous()
    out, _ = unit_rows_op(w, gain, list(view), list(row_dims), radius, backend)
    return out


def unit_row_op(
    spec: RowSpec,
    shape: tuple[int, ...],
    device: torch.device | str,
    backend: str | None = None,
) -> UnitRowOp:
    """The `UnitRowOp` of `spec` for a weight of `shape` on `device`."""
    return UnitRowOp(
        view=spec.view_of(shape),
        row_dims=tuple(spec.row_dims),
        radius=spec.radius_of(shape),
        rows=spec.rows(shape),
        forward=resolve(ops.UNIT_ROWS, device, backend),
        backward=resolve(ops.UNIT_ROWS_BWD, device, backend),
        backend=backend,
    )


@functools.lru_cache(maxsize=None)
def _bound(op: str, device: torch.device, backend: str):
    return resolve(op, device, backend or None)


def _acc_dtype(t: Tensor) -> torch.dtype:
    return torch.float64 if t.dtype == torch.float64 else torch.float32


@torch.library.custom_op("dragonscale::unit_rows", mutates_args=())
def unit_rows_op(
    w: Tensor,
    gain: Tensor | None,
    view: list[int],
    row_dims: list[int],
    radius: float,
    backend: str,
) -> tuple[Tensor, Tensor]:
    """`unit_rows` of the kernel registry as one op: (out, norms)."""
    fn = _bound(ops.UNIT_ROWS, w.device, backend)
    out, norms = fn(w, tuple(view), tuple(row_dims), gain, radius)
    return out.clone() if out is w else out, norms


@unit_rows_op.register_fake
def _(w, gain, view, row_dims, radius, backend):
    rows = math.prod(view[d] for d in row_dims)
    return torch.empty_like(w), w.new_empty(rows, dtype=_acc_dtype(w))


@torch.library.custom_op("dragonscale::unit_rows_bwd", mutates_args=())
def unit_rows_bwd_op(
    grad: Tensor,
    w: Tensor,
    norms: Tensor,
    gain: Tensor | None,
    view: list[int],
    row_dims: list[int],
    radius: float,
    backend: str,
) -> tuple[Tensor, Tensor]:
    """`unit_rows_bwd` as one op: (grad_w, grad_gain); grad_gain is empty without a gain."""
    fn = _bound(ops.UNIT_ROWS_BWD, w.device, backend)
    grad_w, grad_gain = fn(grad.contiguous(), w, tuple(view), tuple(row_dims), norms, gain, radius)
    return grad_w, grad_gain if grad_gain is not None else w.new_empty(0)


@unit_rows_bwd_op.register_fake
def _(grad, w, norms, gain, view, row_dims, radius, backend):
    return torch.empty_like(w), (torch.empty_like(gain) if gain is not None else w.new_empty(0))


def _setup(ctx, inputs, output):
    w, gain, view, row_dims, radius, backend = inputs
    ctx.save_for_backward(w, output[1], gain)
    ctx.spec = (view, row_dims, radius, backend)


def _backward(ctx, grad_out, grad_norms):
    w, norms, gain = ctx.saved_tensors
    view, row_dims, radius, backend = ctx.spec
    grad_w, grad_gain = unit_rows_bwd_op(grad_out, w, norms, gain, view, row_dims, radius, backend)
    return grad_w, (grad_gain if gain is not None else None), None, None, None, None


unit_rows_op.register_autograd(_backward, setup_context=_setup)


_OPS: dict[tuple, UnitRowOp] = {}


def unit_rows(
    w: Tensor,
    spec: RowSpec | None = None,
    gain: Tensor | None = None,
    backend: str | None = None,
) -> Tensor:
    """`gain * radius * w / norm` per row through the kernel ops, differentiable; `spec` None is
    output rows at radius 1. The op is bound once per (spec, shape, device, backend)."""
    spec = spec or RowSpec()
    key = (spec, tuple(w.shape), w.device, backend)
    op = _OPS.get(key)
    if op is None:
        op = _OPS[key] = unit_row_op(spec, w.shape, w.device, backend)
    return op(w, gain)
