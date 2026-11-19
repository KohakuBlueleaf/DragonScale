"""Op names and their contracts, implemented by every backend.

A weight is read through a row view: `view` is a shape with the weight's element count,
`row_dims` the sorted axes of `view` that index a row; every other axis is reduced. Norms are
fp32 (fp64 for an fp64 weight), flat `[rows]` in row-major order of the row dims. A norm is
floored at `FLOOR`; nothing is added to it.

row_norm(w, view, row_dims) -> norms[rows]
    the L2 norm of every row.

unit_rows(w, view, row_dims, gain[rows] | None, radius) -> (out[w.shape], norms[rows])
    out = gain * radius * w / norm per row, in w.dtype; gain None means 1.

unit_rows_bwd(grad[w.shape], w, view, row_dims, norms, gain | None, radius)
    -> (grad_w[w.shape], grad_gain[rows] | None)
    the gradient of `unit_rows`: grad_w = (c / n) (G - u <u, G>) with u = w / n and
    c = gain * radius, tangent to every row; grad_gain = radius <u, G>.

unit_rows_grouped(ws: list, gains: list, radii: list) -> (outs: list, norms: list)
    `unit_rows` of every contiguous `[rows, cols]` weight in `ws` (rows first), in few launches.
unit_rows_grouped_bwd(grads, ws, norms, gains, radii) -> (grad_ws: list, grad_gains: list)
    `unit_rows_bwd` of each; a grad_gain is None where its gain is None.
"""

ROW_NORM = "row_norm"
UNIT_ROWS = "unit_rows"
UNIT_ROWS_BWD = "unit_rows_bwd"
UNIT_ROWS_GROUPED = "unit_rows_grouped"
UNIT_ROWS_GROUPED_BWD = "unit_rows_grouped_bwd"

OPS = (
    ROW_NORM,
    UNIT_ROWS,
    UNIT_ROWS_BWD,
    UNIT_ROWS_GROUPED,
    UNIT_ROWS_GROUPED_BWD,
)
FLOOR = 1e-12
