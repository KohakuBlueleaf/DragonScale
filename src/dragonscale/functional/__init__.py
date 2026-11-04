"""Unit-row weights as functions: the autograd reference, the kernel-op path, layer forwards."""

from dragonscale.functional.layers import (
    conv,
    conv_transpose,
    embedding,
    linear,
    transposed_linear,
)
from dragonscale.functional.rows import (
    UnitRowOp,
    UnitRowsFn,
    gain_scaled,
    preserving_gain,
    reference,
    row_norms,
    unit_row_op,
    unit_rows,
)

__all__ = [
    "UnitRowOp",
    "UnitRowsFn",
    "conv",
    "conv_transpose",
    "embedding",
    "linear",
    "preserving_gain",
    "reference",
    "gain_scaled",
    "row_norms",
    "transposed_linear",
    "unit_row_op",
    "unit_rows",
]
