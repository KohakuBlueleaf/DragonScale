"""Triton backend: fused row kernels on CUDA (sm70 and newer)."""

import triton  # noqa: F401

from dragonscale.kernels import ops
from dragonscale.kernels.backends.triton import grouped, rows
from dragonscale.kernels.constraints import requires
from dragonscale.kernels.registry import register

CUDA = requires(("cuda",), min_sm=70)
IMPLS = {
    ops.ROW_NORM: rows.row_norm,
    ops.UNIT_ROWS: rows.unit_rows,
    ops.UNIT_ROWS_BWD: rows.unit_rows_bwd,
    ops.UNIT_ROWS_GROUPED: grouped.unit_rows_grouped,
    ops.UNIT_ROWS_GROUPED_BWD: grouped.unit_rows_grouped_bwd,
}

for _op, _fn in IMPLS.items():
    register(_op, "triton", _fn, CUDA)
