"""Forwards of the `forward` install: `(patched, x, *args, **kwargs) -> y`.

`eval_forward` is the eval-mode forward: the cached weight under no grad, else the training
forward.
"""

import torch
from torch import Tensor


def train_forward(patched, x: Tensor, *args, **kwargs) -> Tensor:
    """The layer with its unit-row weight."""
    module = patched.module
    weight = patched.weight(getattr(module, patched.layer_type.weight), patched.gain)
    return patched.layer_type.forward(module, x, weight, patched.bias(), *args, **kwargs)


def eval_forward(patched, x: Tensor, *args, **kwargs) -> Tensor:
    """Eval mode: the cached weight when no gradient is recorded."""
    if torch.is_grad_enabled():
        return train_forward(patched, x, *args, **kwargs)
    module = patched.module
    weight = patched.cached_weight(getattr(module, patched.layer_type.weight), patched.gain)
    return patched.layer_type.forward(module, x, weight, patched.bias(), *args, **kwargs)
