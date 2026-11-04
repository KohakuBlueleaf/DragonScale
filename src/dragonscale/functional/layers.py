"""Stock layer forwards with the weight and bias given: `(module, x, weight, bias, ...) -> y`."""

import torch.nn.functional as F
from torch import Tensor, nn

CONV_TRANSPOSE = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}


def linear(module: nn.Linear, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    return F.linear(x, weight, bias)


def transposed_linear(module: nn.Module, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    """`x @ weight + bias` for a weight stored `[in, out]` (transformers' `Conv1D`)."""
    return F.linear(x, weight.t(), bias)


def conv(module: nn.modules.conv._ConvNd, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    return module._conv_forward(x, weight, bias)


def conv_transpose(
    module: nn.modules.conv._ConvTransposeNd,
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    output_size: list[int] | None = None,
) -> Tensor:
    spatial = len(module.kernel_size)
    padding = module._output_padding(
        x,
        output_size,
        module.stride,
        module.padding,
        module.kernel_size,
        spatial,
        module.dilation,
    )
    return CONV_TRANSPOSE[spatial](
        x, weight, bias, module.stride, module.padding, padding, module.groups, module.dilation
    )


def embedding(module: nn.Embedding, x: Tensor, weight: Tensor, bias: None = None) -> Tensor:
    return F.embedding(
        x,
        weight,
        module.padding_idx,
        module.max_norm,
        module.norm_type,
        module.scale_grad_by_freq,
        module.sparse,
    )
