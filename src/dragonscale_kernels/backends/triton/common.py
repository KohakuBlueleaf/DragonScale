"""Row views as strided 2D tensors, and tile sizes, for the Triton row kernels."""

import math

import triton
from torch import Tensor

from dragonscale_kernels.layout import reduced

TILE = 4096
MAX_COLS = 2048
ONE_PASS_BYTES = 64 * 1024  # a row up to this size is read once


def rows_cols(view: tuple[int, ...], row_dims: tuple[int, ...]) -> tuple[int, int]:
    rows = math.prod(view[axis] for axis in row_dims)
    return rows, math.prod(view) // rows


def as_2d(t: Tensor, view: tuple[int, ...], row_dims: tuple[int, ...]) -> tuple[Tensor, bool]:
    """`t` as `[rows, cols]` (row axes first): a view when strides allow, else a copy; the flag
    says whether the result aliases `t`."""
    rows, cols = rows_cols(view, row_dims)
    permuted = t.reshape(view).permute(*row_dims, *reduced(view, row_dims))
    try:
        return permuted.view(rows, cols), True
    except RuntimeError:
        return permuted.reshape(rows, cols), False


def from_2d(t2: Tensor, view: tuple[int, ...], row_dims: tuple[int, ...], shape) -> Tensor:
    """Inverse of `as_2d` for a `[rows, cols]` result: a tensor of `shape` in view order."""
    order = (*row_dims, *reduced(view, row_dims))
    inverse = [order.index(axis) for axis in range(len(view))]
    permuted = t2.reshape([view[axis] for axis in order])
    return permuted.permute(*inverse).reshape(shape)


def tiles(rows: int, cols: int, element_size: int = 4) -> tuple[int, int, int, bool]:
    """(rows per program, column block, warps, one pass) for a `[rows, cols]` row op: a row that
    fits `ONE_PASS_BYTES` (counted at fp32 or wider) is one tile, read once."""
    width = triton.next_power_of_2(max(cols, 1))
    one_pass = width * max(element_size, 4) <= ONE_PASS_BYTES
    block_c = width if one_pass else MAX_COLS
    block_r = max(1, min(TILE // block_c, triton.next_power_of_2(max(rows, 1))))
    warps = min(16, max(4, block_r * block_c // 1024))
    return block_r, block_c, warps, one_pass
