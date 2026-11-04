"""Row-view and channel layout helpers shared by every backend."""

import torch
from torch import Tensor


def wide(t: Tensor) -> Tensor:
    """`t` in at least fp32 (fp64 stays fp64)."""
    return t.to(torch.promote_types(t.dtype, torch.float32))


def reduced(view: tuple[int, ...], row_dims: tuple[int, ...]) -> tuple[int, ...]:
    """The axes of `view` that are not row dims."""
    return tuple(axis for axis in range(len(view)) if axis not in row_dims)


def row_shape(view: tuple[int, ...], row_dims: tuple[int, ...]) -> tuple[int, ...]:
    """`view` with every reduced axis set to 1: the broadcast shape of a per-row value."""
    return tuple(size if axis in row_dims else 1 for axis, size in enumerate(view))


def as_rows(values: Tensor, view: tuple[int, ...], row_dims: tuple[int, ...]) -> Tensor:
    """Flat `[rows]` values reshaped to broadcast over the view."""
    return values.reshape(row_shape(view, row_dims))


def row_dot(a: Tensor, b: Tensor, dims: tuple[int, ...]) -> Tensor:
    """Per-row inner product over `dims`, kept in the broadcast shape."""
    prod = a * b
    return prod.sum(dims, keepdim=True) if dims else prod


def channel_shape(t: Tensor, axis: int) -> list[int]:
    """Broadcast shape of a per-channel vector along `axis` of `t`."""
    shape = [1] * t.dim()
    shape[axis % t.dim()] = -1
    return shape


def other_axes(t: Tensor, axis: int) -> tuple[int, ...]:
    """Every axis of `t` but `axis`."""
    return tuple(a for a in range(t.dim()) if a != axis % t.dim())
