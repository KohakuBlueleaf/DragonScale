"""`Gain`: a learned per-channel (or scalar) scale."""

import torch
from torch import Tensor, nn

from dragonscale_kernels.layout import channel_shape


class Gain(nn.Module):
    """`x * gain` along `axis`; `channels=1` is one scalar for every channel."""

    def __init__(self, channels: int, axis: int = -1, init: float = 1.0, device=None, dtype=None):
        super().__init__()
        self.axis = axis
        self.weight = nn.Parameter(torch.full((channels,), float(init), device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.weight.reshape(channel_shape(x, self.axis)).to(x.dtype)

    def extra_repr(self) -> str:
        return f"channels={self.weight.numel()}, axis={self.axis}"
