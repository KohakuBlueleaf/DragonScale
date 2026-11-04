"""Triton backend: fused row kernels on CUDA (sm70 and newer)."""

import triton  # noqa: F401

from dragonscale_kernels import ops
from dragonscale_kernels.backends.triton import rows
from dragonscale_kernels.constraints import requires
from dragonscale_kernels.registry import register

CUDA = requires(("cuda",), min_sm=70)
IMPLS = {
    ops.ROW_NORM: rows.row_norm,
    ops.UNIT_ROWS: rows.unit_rows,
    ops.UNIT_ROWS_BWD: rows.unit_rows_bwd,
    ops.PROJECT: rows.project,
    ops.RESCALE_ROWS: rows.rescale_rows_,
    ops.PROJECT_RESCALE_ROWS: rows.project_rescale_rows_,
}

for _op, _fn in IMPLS.items():
    register(_op, "triton", _fn, CUDA)
