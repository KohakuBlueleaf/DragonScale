"""Eager row ops (contracts in `dragonscale_kernels.ops`): rows reduced in fp32, fp64 kept."""

from torch import Tensor

from dragonscale_kernels.layout import as_rows, reduced, row_dot, wide
from dragonscale_kernels.ops import FLOOR


def row_norm(w: Tensor, view: tuple[int, ...], row_dims: tuple[int, ...]) -> Tensor:
    v = wide(w).reshape(view)
    dims = reduced(view, row_dims)
    norms = v.square().sum(dims).sqrt() if dims else v.abs()
    return norms.reshape(-1)


def unit_rows(
    w: Tensor,
    view: tuple[int, ...],
    row_dims: tuple[int, ...],
    gain: Tensor | None,
    radius: float,
) -> tuple[Tensor, Tensor]:
    v = wide(w).reshape(view)
    norms = row_norm(w, view, row_dims)
    scale = radius / norms.clamp_min(FLOOR)
    if gain is not None:
        scale = scale * gain.to(scale.dtype)
    out = (v * as_rows(scale, view, row_dims)).to(w.dtype).reshape(w.shape)
    return out, norms


def unit_rows_bwd(
    grad: Tensor,
    w: Tensor,
    view: tuple[int, ...],
    row_dims: tuple[int, ...],
    norms: Tensor,
    gain: Tensor | None,
    radius: float,
) -> tuple[Tensor, Tensor | None]:
    g = wide(grad).reshape(view)
    v = wide(w).reshape(view)
    n = as_rows(norms.clamp_min(FLOOR).to(v.dtype), view, row_dims)
    u = v / n

    # Tangent part of the incoming gradient, scaled by gain * radius / norm
    dot = row_dot(g, u, reduced(view, row_dims))
    c = radius if gain is None else radius * as_rows(gain.to(v.dtype), view, row_dims)
    grad_w = (c / n) * (g - u * dot)
    grad_gain = None if gain is None else (radius * dot).reshape(-1).to(gain.dtype)
    return grad_w.to(w.dtype).reshape(w.shape), grad_gain
