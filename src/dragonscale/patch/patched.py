"""`PatchedLayer`: one layer plan on its module (held weakly), with its gain and kernel ops."""

import weakref
from functools import partial

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from dragonscale.functional.rows import (
    UnitRowOp,
    fill_zero_rows_,
    gain_scaled,
    preserving_gain,
    unit_row_op,
    zero_rows_,
)
from dragonscale.layer_types import LAYER_TYPES, LayerType
from dragonscale.nn.layers import WEIGHT_FNS
from dragonscale.patch.forwards import eval_forward, method_forward
from dragonscale.plan.layer_plan import LayerPlan
from dragonscale.plan.validate import MARKER

GAIN = "gain"


class UnitRowWeight(nn.Module):
    """The parametrization of the `parametrize` install: `weight` reads as its unit-row version,
    cached in eval mode under no grad."""

    def __init__(self, patched: "PatchedLayer", gain: nn.Parameter | None, training: bool):
        super().__init__()
        self.patched = patched
        self.gain = gain
        self.train(training)

    def train(self, mode: bool = True) -> "UnitRowWeight":
        super().train(mode)
        self.weight_fn = self.patched.weight if mode else self.patched.eval_weight
        return self

    def forward(self, raw: Tensor) -> Tensor:
        return self.weight_fn(raw, self.gain)


def train_and_set_forward(patched: "PatchedLayer", mode: bool = True) -> nn.Module:
    """`Module.train` for a `forward` install: the mode changes, then the forward is set again."""
    module = patched.module
    nn.Module.train(module, mode)
    patched.set_forward(mode)
    return module


def _rebuild(
    plan: LayerPlan,
    layer_type: str,
    backend: str | None,
    module: nn.Module,
    tied_to: "PatchedLayer | None",
    gain: nn.Parameter | None,
) -> "PatchedLayer":
    """A `PatchedLayer` rebuilt by deepcopy or unpickling, on the copied module."""
    patched = PatchedLayer(plan, module, LAYER_TYPES[layer_type], backend, tied_to)
    patched._gain = gain
    return patched


def _own_class(module: nn.Module) -> None:
    """Give a parametrized module a class of its own."""
    cls = type(module)
    body = {k: v for k, v in vars(cls).items() if k not in ("__dict__", "__weakref__")}
    module.__class__ = type(cls.__name__, cls.__bases__, body)


def _reset(module: nn.Module) -> None:
    """The module's own parameter reset (`reset_parameters` or `_reset_parameters`)."""
    reset = getattr(module, "reset_parameters", None) or module._reset_parameters
    reset()


class PatchedLayer:
    """One layer plan applied to its module; the session holds it, the module holds its forward.

    A layer `tied_to` another uses that layer's gain and writes nothing.
    """

    def __init__(
        self,
        plan: LayerPlan,
        module: nn.Module,
        layer_type: LayerType,
        backend: str | None = None,
        tied_to: "PatchedLayer | None" = None,
    ):
        self.plan = plan
        self.layer_type = layer_type
        self.backend = backend
        self.tied_to = tied_to
        self.ref = weakref.ref(module)
        self._ops: dict[torch.device, UnitRowOp] = {}
        self._gain: nn.Parameter | None = None
        self._filled: tuple[Tensor, int] | None = None
        self._cache: tuple[tuple, Tensor] | None = None

    def __reduce__(self):
        """Copies and pickles rebuild around the copied module; kernel ops are not kept."""
        args = (
            self.plan,
            self.layer_type.name,
            self.backend,
            self.module,
            self.tied_to,
            self.gain,
        )
        return _rebuild, args

    @property
    def module(self) -> nn.Module:
        return self.ref()

    @property
    def gain(self) -> nn.Parameter | None:
        """The gain as the module holds it now."""
        if self._gain is None:
            return None
        module = self.module
        if self.plan.install == "parametrize":
            return module.parametrizations[self.layer_type.weight][0].gain
        return module._parameters.get(GAIN, self._gain)

    def bias(self) -> Tensor | None:
        name = self.layer_type.bias
        return None if name is None else getattr(self.module, name, None)

    def raw_weight(self) -> nn.Parameter:
        """The trained parameter behind the weight, for either install."""
        module = self.module
        if self.plan.install == "parametrize":
            return module.parametrizations[self.layer_type.weight].original
        return getattr(module, self.layer_type.weight)

    def unit_row_op(self, raw: Tensor) -> UnitRowOp:
        """The unit-row op for `raw`'s device, bound once per device."""
        op = self._ops.get(raw.device)
        if op is None:
            op = unit_row_op(self.plan.spec, raw.shape, raw.device, self.backend)
            self._ops[raw.device] = op
        return op

    def set_backend(self, backend: str | None) -> None:
        self.backend = backend
        self._ops.clear()

    def weight(self, raw: Tensor, gain: Tensor | None) -> Tensor:
        """The weight the layer computes with, from a raw weight and a gain, per method."""
        if self.plan.method == "rescale":
            return gain_scaled(raw, self.plan.spec, gain)
        return self.unit_row_op(raw)(raw, gain)

    def unit_weight(self) -> Tensor:
        return self.weight(self.raw_weight(), self.gain)

    def cached_weight(self, raw: Tensor, gain: Tensor | None) -> Tensor:
        """The weight without autograd, kept until the raw weight or the gain changes."""
        if self.tied_to is not None:
            return self.tied_to.cached_weight(raw, gain)
        key = (raw.data_ptr(), raw._version, raw.device, raw.dtype)
        if gain is not None:
            key += (gain.data_ptr(), gain._version)
        if self._cache is None or self._cache[0] != key:
            with torch.no_grad():
                self._cache = (key, self.weight(raw, gain).detach())
        return self._cache[1]

    def eval_weight(self, raw: Tensor, gain: Tensor | None) -> Tensor:
        """Eval mode: the cached weight when no gradient is recorded."""
        if torch.is_grad_enabled():
            return self.weight(raw, gain)
        return self.cached_weight(raw, gain)

    def set_forward(self, training: bool) -> None:
        """Set the forward of the current mode; training mode drops the cache."""
        forward = method_forward if training else eval_forward
        self.module.forward = partial(forward, self)
        if training:
            self._cache = None

    def _initial_gain(self, raw: Tensor) -> nn.Parameter | None:
        """The gain the layer starts with, per its init."""
        plan = self.plan
        if plan.mode != "unit_gain":
            return None
        if plan.init in ("preserve", "reset") and plan.gain == "channel":
            start = preserving_gain(raw, plan.spec)
        else:
            rows = plan.spec.rows(raw.shape) if plan.gain == "channel" else 1
            start = torch.ones(rows, device=raw.device)
        return nn.Parameter(start.to(raw.dtype))

    def install(self) -> None:
        """Init, then the forward or the parametrization."""
        module = self.module
        if self.tied_to is not None:
            self._gain = self.tied_to.gain
        else:
            if self.plan.init == "reset":
                _reset(module)
            raw = getattr(module, self.layer_type.weight)
            self._gain = self._initial_gain(raw.detach())

            # Zero rows get a direction; with gain 0 the function is unchanged
            mask = fill_zero_rows_(raw, self.plan.spec)
            self._filled = None if mask is None else (mask, -1)

        match self.plan.install:
            case "forward":
                if self._gain is not None:
                    module.register_parameter(GAIN, self._gain)
                self.set_forward(module.training)
                module.train = partial(train_and_set_forward, self)
            case "parametrize":
                parametrize.register_parametrization(
                    module,
                    self.layer_type.weight,
                    UnitRowWeight(self, self._gain, module.training),
                )
        if self._filled is not None:
            self._filled = (self._filled[0], self.raw_weight()._version)
        setattr(module, MARKER, self)

    def bake(self) -> None:
        """The `bake` install: write the unit-row weight once; nothing stays installed."""
        if self.tied_to is not None:
            return
        if self.plan.init == "reset":
            _reset(self.module)
        raw = getattr(self.module, self.layer_type.weight)
        gain = self._initial_gain(raw.detach())
        with torch.no_grad():
            raw.copy_(self.weight(raw.detach(), gain))

    def canonicalize(self) -> None:
        """Uninstall, then turn the module into its `dragonscale.nn` unit-row class in place, with
        the same weight, bias and gain parameters."""
        module, gain, plan = self.module, self.gain, self.plan
        self.uninstall(restore_zero_rows=False)
        module.__class__ = self.layer_type.unit_class
        module.spec = plan.spec
        module.mode = plan.mode
        module._ops = {}
        module._weight_fn = WEIGHT_FNS[plan.mode]
        module.register_parameter(GAIN, gain)

    def uninstall(self, restore_zero_rows: bool = True) -> None:
        """Undo `install`: the module forwards and serializes as before it."""
        module = self.module
        filled = self._filled
        unchanged = filled is not None and self.raw_weight()._version == filled[1]
        match self.plan.install:
            case "forward":
                vars(module).pop("forward", None)
                vars(module).pop("train", None)
                module._parameters.pop(GAIN, None)
            case "parametrize":
                _own_class(module)
                parametrize.remove_parametrizations(
                    module, self.layer_type.weight, leave_parametrized=False
                )
        vars(module).pop(MARKER, None)

        # Filled rows go back to zero when no step has changed the weight since install
        if restore_zero_rows and unchanged:
            zero_rows_(getattr(module, self.layer_type.weight), self.plan.spec, filled[0])
        self._filled = None
        self._gain = None
        self._cache = None
        self._ops.clear()
