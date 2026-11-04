"""Forwards of the `forward` install, one per method: `(patched, x, *args, **kwargs) -> y`.

`eval_forward` is the eval-mode forward: the cached weight under no grad, else the method's
forward.
"""

import torch
from torch import Tensor


def method_forward(patched, x: Tensor, *args, **kwargs) -> Tensor:
    """The layer with the weight its method gives (`normalize` or `rescale`)."""
    module = patched.module
    weight = patched.weight(getattr(module, patched.layer_type.weight), patched.gain)
    return patched.layer_type.forward(module, x, weight, patched.bias(), *args, **kwargs)


def eval_forward(patched, x: Tensor, *args, **kwargs) -> Tensor:
    """Eval mode: the cached weight when no gradient is recorded."""
    if torch.is_grad_enabled():
        return method_forward(patched, x, *args, **kwargs)
    module = patched.module
    weight = patched.cached_weight(getattr(module, patched.layer_type.weight), patched.gain)
    return patched.layer_type.forward(module, x, weight, patched.bias(), *args, **kwargs)
