"""The first forward of a patched model in each mode checks that no op reads a raw weight directly.

A layer patched with `install="forward"` replaces its module's forward. A parent module that reads
the layer's weight itself (`nn.MultiheadAttention` reads `out_proj.weight`) bypasses that forward
and computes with the raw weight: the patch would have no effect on that path. The check runs one
forward under a dispatch mode that records every op argument that is a raw weight, outside the
unit-row computation itself, and raises if there is one. It runs once per (training, grad) mode,
and never under `torch.compile`.
"""

import threading

import torch
from torch import Tensor, nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from dragonscale.plan.validate import MARKER
from dragonscale.problems import raise_problems

PROBED = "_ds_probed"
_STATE = threading.local()


def probing() -> bool:
    return getattr(_STATE, "mode", None) is not None


class inside_unit_rows:
    """Marks the computation of a unit-row weight: its reads of the raw weight are expected."""

    def __enter__(self):
        _STATE.inside = getattr(_STATE, "inside", 0) + 1

    def __exit__(self, *exc):
        _STATE.inside -= 1


class RawReads(TorchDispatchMode):
    """Records the path of every patched layer whose raw weight reaches an op directly."""

    def __init__(self, raw: dict[int, str]):
        super().__init__()
        self.raw = raw
        self.found: set[str] = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if getattr(_STATE, "inside", 0) == 0:
            for t in tree_leaves((args, kwargs)):
                if isinstance(t, Tensor) and id(t) in self.raw:
                    self.found.add(self.raw[id(t)])
        return func(*args, **(kwargs or {}))


def _raw_weights(model: nn.Module) -> dict[int, str]:
    found = {}
    for module in model.modules():
        patched = vars(module).get(MARKER)
        if patched is not None and patched.plan.install == "forward":
            found[id(patched.raw_weight())] = patched.plan.path or "<root>"
    return found


def enter(model: nn.Module, args) -> None:
    """Forward pre-hook: start the check on the first forward of this mode."""
    if torch.compiler.is_compiling() or probing():
        return
    key = (model.training, torch.is_grad_enabled())
    probed = vars(model).setdefault(PROBED, set())
    if key in probed:
        return
    probed.add(key)
    raw = _raw_weights(model)
    if not raw:
        return
    mode = RawReads(raw)
    mode.__enter__()
    _STATE.mode = mode


def leave(model: nn.Module, args, output) -> None:
    """Forward hook: end the check and raise for every raw weight read outside its layer."""
    mode = getattr(_STATE, "mode", None)
    if mode is None or torch.compiler.is_compiling():
        return
    _STATE.mode = None
    mode.__exit__(None, None, None)
    raise_problems(
        [
            f"{path}: its raw weight was read outside its own forward (by a parent module or "
            "the model's code), so the patch did not apply there; patch it with "
            "install='parametrize'"
            for path in sorted(mode.found)
        ]
    )


def attach(model: nn.Module) -> None:
    if any(hook is enter for hook in model._forward_pre_hooks.values()):
        return
    model.register_forward_pre_hook(enter)
    model.register_forward_hook(leave, always_call=True)


def detach(model: nn.Module) -> None:
    for hooks, fn in ((model._forward_pre_hooks, enter), (model._forward_hooks, leave)):
        for key in [key for key, hook in hooks.items() if hook is fn]:
            del hooks[key]
    vars(model).pop(PROBED, None)
