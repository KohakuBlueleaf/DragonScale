"""Eager channel scales (contracts in `dragonscale_kernels.ops`): vector math in fp32."""

from torch import Tensor

from dragonscale_kernels.layout import channel_shape, other_axes, wide


def out_scale(y: Tensor, scale: Tensor, bias: Tensor | None, axis: int) -> Tensor:
    shape = channel_shape(y, axis)
    out = wide(y) * wide(scale).reshape(shape)
    if bias is not None:
        out = out + wide(bias).reshape(shape)
    return out.to(y.dtype)


def out_scale_bwd(
    grad: Tensor, y: Tensor, scale: Tensor, has_bias: bool, axis: int
) -> tuple[Tensor, Tensor, Tensor | None]:
    g = wide(grad)
    dims = other_axes(y, axis)
    grad_y = (g * wide(scale).reshape(channel_shape(y, axis))).to(y.dtype)
    grad_scale = (g * wide(y)).sum(dims)
    grad_bias = g.sum(dims) if has_bias else None
    return grad_y, grad_scale, grad_bias


def in_scale(x: Tensor, scale: Tensor, axis: int) -> Tensor:
    return (wide(x) * wide(scale).reshape(channel_shape(x, axis))).to(x.dtype)


def in_scale_bwd(grad: Tensor, x: Tensor, scale: Tensor, axis: int) -> tuple[Tensor, Tensor]:
    g = wide(grad)
    grad_x = (g * wide(scale).reshape(channel_shape(x, axis))).to(x.dtype)
    grad_scale = (g * wide(x)).sum(other_axes(x, axis))
    return grad_x, grad_scale
