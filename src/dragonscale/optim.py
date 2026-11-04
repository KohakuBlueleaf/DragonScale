"""Optimizer side of unit rows, as step hooks on any torch optimizer.

`attach` returns a `TangentGradientDecorator`: before each step it projects the gradients tangent
to the rows; after it, it projects the first moments and (with `rescale`) puts the rows back at
the radius. `reset_raw_norms` puts raw rows at the radius once, keeping the function.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from dragonscale.patch.session import Session
from dragonscale.spec.rows import RowSpec
from dragonscale_common.log import get_logger
from dragonscale_kernels import ops, resolve

log = get_logger("optim")

FIRST_MOMENTS: dict[type, tuple[str, ...]] = {
    torch.optim.Adam: ("exp_avg",),
    torch.optim.AdamW: ("exp_avg",),
    torch.optim.NAdam: ("exp_avg",),
    torch.optim.RAdam: ("exp_avg",),
    torch.optim.Adamax: ("exp_avg",),
    torch.optim.SGD: ("momentum_buffer",),
    torch.optim.RMSprop: ("momentum_buffer",),
    torch.optim.Adagrad: (),
}
if hasattr(torch.optim, "Muon"):
    FIRST_MOMENTS[torch.optim.Muon] = ("momentum_buffer",)


def first_moments(optimizer: torch.optim.Optimizer) -> tuple[str, ...] | None:
    """The first-moment state keys of `optimizer`'s class or its nearest listed base class."""
    for cls in type(optimizer).__mro__:
        if cls in FIRST_MOMENTS:
            return FIRST_MOMENTS[cls]
    return None


def _pairs(weights: Session | Sequence[tuple[nn.Parameter, RowSpec]]):
    return weights.raw_weights() if isinstance(weights, Session) else list(weights)


@dataclass(frozen=True)
class TrackedWeight:
    """One raw weight the hooks keep: its row view, radius and bound ops."""

    param: nn.Parameter
    view: tuple[int, ...]
    row_dims: tuple[int, ...]
    radius: float
    project: object
    project_rescale_rows: object


def _tracked(param: nn.Parameter, spec: RowSpec, backend: str | None) -> TrackedWeight:
    shape = param.shape
    return TrackedWeight(
        param,
        spec.view_of(shape),
        tuple(spec.row_dims),
        spec.radius_of(shape),
        resolve(ops.PROJECT, param.device, backend),
        resolve(ops.PROJECT_RESCALE_ROWS, param.device, backend),
    )


class TangentGradientDecorator:
    """The step hooks on one optimizer; `remove` detaches them."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        weights: Sequence[tuple[nn.Parameter, RowSpec]],
        project_grad: bool = True,
        rescale: bool = True,
        backend: str | None = None,
    ):
        self.optimizer = optimizer
        self.tracked = [_tracked(p, spec, backend) for p, spec in weights]
        self.rescale = rescale
        keys = first_moments(optimizer)
        if keys is None:
            log.warning_once(
                type(optimizer).__name__,
                "unknown optimizer: gradients projected, moments left alone",
                optimizer=type(optimizer).__name__,
            )
        self.first = keys or ()
        self.handles = []
        if project_grad:
            self.handles.append(optimizer.register_step_pre_hook(self._pre))
        self.handles.append(optimizer.register_step_post_hook(self._post))

    def _pre(self, optimizer, args, kwargs) -> None:
        with torch.no_grad():
            for t in self.tracked:
                grad = t.param.grad
                if grad is not None:
                    grad.copy_(t.project(grad, t.param, t.view, t.row_dims))

    def _post(self, optimizer, args, kwargs) -> None:
        with torch.no_grad():
            for t in self.tracked:
                state = optimizer.state.get(t.param, {})
                moments = [
                    state[k]
                    for k in self.first
                    if isinstance(state.get(k), torch.Tensor) and state[k].shape == t.param.shape
                ]
                if self.rescale:
                    t.project_rescale_rows(t.param, moments, t.view, t.row_dims, t.radius)
                else:
                    for m in moments:
                        m.copy_(t.project(m, t.param, t.view, t.row_dims))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def attach(
    optimizer: torch.optim.Optimizer,
    weights: Session | Sequence[tuple[nn.Parameter, RowSpec]],
    project_grad: bool = True,
    rescale: bool = True,
    backend: str | None = None,
) -> TangentGradientDecorator:
    """Hooks keeping `weights` (a session's raw weights, or `(param, spec)` pairs) tangent."""
    return TangentGradientDecorator(optimizer, _pairs(weights), project_grad, rescale, backend)


def reset_raw_norms(
    weights: Session | Sequence[tuple[nn.Parameter, RowSpec]], backend: str | None = None
) -> None:
    """Every raw weight's rows back to their radius; the function is unchanged."""
    for param, spec in _pairs(weights):
        rescale = resolve(ops.RESCALE_ROWS, param.device, backend)
        rescale(param, spec.view_of(param.shape), tuple(spec.row_dims), spec.radius_of(param.shape))
