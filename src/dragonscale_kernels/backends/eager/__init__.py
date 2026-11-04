"""Eager backend: every op in plain torch."""

from dragonscale_kernels import ops
from dragonscale_kernels.backends.eager import rows, scale
from dragonscale_kernels.registry import register

IMPLS = {
    ops.ROW_NORM: rows.row_norm,
    ops.UNIT_ROWS: rows.unit_rows,
    ops.UNIT_ROWS_BWD: rows.unit_rows_bwd,
    ops.SCALE_ROWS: rows.scale_rows,
    ops.PROJECT: rows.project,
    ops.RESCALE_ROWS: rows.rescale_rows_,
    ops.PROJECT_RESCALE_ROWS: rows.project_rescale_rows_,
    ops.OUT_SCALE: scale.out_scale,
    ops.OUT_SCALE_BWD: scale.out_scale_bwd,
    ops.IN_SCALE: scale.in_scale,
    ops.IN_SCALE_BWD: scale.in_scale_bwd,
}

for _op, _fn in IMPLS.items():
    register(_op, "eager", _fn)
