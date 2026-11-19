"""Eager backend: every op in plain torch."""

from dragonscale_kernels import ops
from dragonscale_kernels.backends.eager import rows
from dragonscale_kernels.registry import register

IMPLS = {
    ops.ROW_NORM: rows.row_norm,
    ops.UNIT_ROWS: rows.unit_rows,
    ops.UNIT_ROWS_BWD: rows.unit_rows_bwd,
}

for _op, _fn in IMPLS.items():
    register(_op, "eager", _fn)
