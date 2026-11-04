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

scale_rows(w, view, row_dims, scale[rows]) -> out[w.shape]
    w * scale per row, in w.dtype.

out_scale(y, scale[c], bias[c] | None, axis) -> out[y.shape]
    y * scale + bias along channel axis `axis`, rounded once to y.dtype.

out_scale_bwd(grad, y, scale, has_bias, axis) -> (grad_y, grad_scale[c], grad_bias[c] | None)
    grad_y = grad * scale; grad_scale = sum(grad * y) and grad_bias = sum(grad) over every axis
    but `axis`, in at least fp32. `y` is the input of `out_scale`.

in_scale(x, scale[c], axis) -> out[x.shape]
    x * scale along channel axis `axis`.

in_scale_bwd(grad, x, scale, axis) -> (grad_x, grad_scale[c])
    grad_x = grad * scale; grad_scale = sum(grad * x) in at least fp32.

project(t[w.shape], w, view, row_dims) -> out[w.shape]
    t minus its component along every row of w: t - u <u, t>, in t.dtype.

rescale_rows_(w, view, row_dims, radius) -> norms[rows]
    in place: every row of w rescaled to norm `radius`; returns the norms before.

project_rescale_rows_(w, moments: list, view, row_dims, radius) -> norms[rows]
    in place: every tensor in `moments` projected tangent to w's rows, then `rescale_rows_(w)`.
"""

ROW_NORM = "row_norm"
UNIT_ROWS = "unit_rows"
UNIT_ROWS_BWD = "unit_rows_bwd"
SCALE_ROWS = "scale_rows"
OUT_SCALE = "out_scale"
OUT_SCALE_BWD = "out_scale_bwd"
IN_SCALE = "in_scale"
IN_SCALE_BWD = "in_scale_bwd"
PROJECT = "project"
RESCALE_ROWS = "rescale_rows_"
PROJECT_RESCALE_ROWS = "project_rescale_rows_"

OPS = (
    ROW_NORM,
    UNIT_ROWS,
    UNIT_ROWS_BWD,
    SCALE_ROWS,
    OUT_SCALE,
    OUT_SCALE_BWD,
    IN_SCALE,
    IN_SCALE_BWD,
    PROJECT,
    RESCALE_ROWS,
    PROJECT_RESCALE_ROWS,
)
FLOOR = 1e-12
