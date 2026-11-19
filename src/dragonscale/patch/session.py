"""`Session`: every change one plan made to one model, and the only way to undo or bake it."""

import weakref
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from torch import Tensor, nn

from dragonscale.layer_types import LAYER_TYPES
from dragonscale.log import get_logger
from dragonscale.patch import grouped, probe
from dragonscale.patch import state as formats
from dragonscale.patch.patched import PatchedLayer
from dragonscale.plan.build import make_plan
from dragonscale.plan.layer_plan import Plan
from dragonscale.plan.rules import Rule
from dragonscale.plan.validate import MARKER, validate
from dragonscale.problems import raise_problems
from dragonscale.spec.rows import RowSpec

log = get_logger("patch")


def warn_changed_output(model: nn.Module, plan: Plan) -> None:
    """Log the layers that start with another output than the model had: `mode="unit"` with
    `init="preserve"` (no gain to set), and a number as `init` (the gain starts at it)."""
    no_gain = [
        layer.path
        for layer in plan
        if layer.init == "preserve" and layer.tied_to is None and layer.mode == "unit"
    ]
    fixed = [
        layer.path
        for layer in plan
        if isinstance(layer.init, int | float) and layer.tied_to is None
    ]
    if no_gain:
        log.warning(
            "init 'preserve' with mode 'unit' has no gain to set; the output changes",
            layers=no_gain,
        )
    if fixed:
        log.warning("the gain starts at the given number; the output changes", layers=fixed)


class Session:
    """Applies `plan` to `model` on construction; `remove` or `bake` ends it. With `group`, every
    forward of `model` computes the eligible layers' weights together (`dragonscale.patch.grouped`).
    """

    def __init__(
        self, model: nn.Module, plan: Plan, backend: str | None = None, group: bool = True
    ):
        raise_problems(validate(model, plan))
        warn_changed_output(model, plan)
        self.plan = plan
        self._model = weakref.ref(model)
        self.patched: list[PatchedLayer] = []
        self.closed = False
        modules = dict(model.named_modules())
        by_path: dict[str, PatchedLayer] = {}
        for layer in plan:
            tied_to = None if layer.tied_to is None else by_path[layer.tied_to]
            layer_type = LAYER_TYPES[layer.layer_type]
            patched = PatchedLayer(layer, modules[layer.path], layer_type, backend, tied_to)
            by_path[layer.path] = patched
            if layer.install == "bake":
                patched.bake()
                continue
            patched.install()
            self.patched.append(patched)
        if group:
            grouped.attach(model)
        probe.attach(model)

    @classmethod
    def from_model(cls, model: nn.Module) -> "Session":
        """The session of a model patched elsewhere (a deepcopy, an unpickled model)."""
        patched = [
            found
            for _, module in model.named_modules()
            if (found := vars(module).get(MARKER)) is not None
        ]
        if not patched:
            raise ValueError("the model has no patched layer")
        session = cls.__new__(cls)
        session.plan = Plan(tuple(p.plan for p in patched))
        session._model = weakref.ref(model)
        session.patched = patched
        session.closed = False
        return session

    @property
    def model(self) -> nn.Module:
        return self._model()

    def _check_open(self) -> None:
        if self.closed:
            raise RuntimeError("the session was removed or baked")

    def untied_layers(self) -> list[PatchedLayer]:
        """Every patched layer not tied to an earlier one."""
        return [p for p in self.patched if p.tied_to is None]

    def gains(self) -> list[nn.Parameter]:
        """The gains the session added, once each."""
        return [p.gain for p in self.untied_layers() if p.gain is not None]

    def raw_weights(self) -> list[tuple[nn.Parameter, RowSpec]]:
        """Every trained weight once, with its row spec."""
        return [(p.raw_weight(), p.plan.spec) for p in self.untied_layers()]

    def set_backend(self, backend: str | None) -> None:
        """Bind every layer's kernel ops to `backend` (None: priority order)."""
        for p in self.patched:
            p.set_backend(backend)

    @contextmanager
    def cache_weights(self) -> Iterator[None]:
        """Inside this block, a forward in eval mode without gradients reuses each layer's
        unit-row weight until its raw weight or gain changes. A change is seen through the
        tensor's version counter, which writes through `.data` do not bump: after such a write,
        call `invalidate()`."""
        layers = list(self.patched)
        for p in layers:
            p.caching = True
        try:
            yield
        finally:
            for p in layers:
                p.caching = False
                p.invalidate()

    def invalidate(self) -> None:
        """Drop every cached unit-row weight (see `cache_weights`)."""
        for p in self.patched:
            p.invalidate()

    def state_dict(self, format: str = "raw") -> dict[str, Tensor]:
        """The model's state in `format` (`raw`, `baked` or `native`)."""
        self._check_open()
        raw = formats.to_raw(self.model.state_dict(), self.patched)
        match format:
            case "raw":
                return raw
            case "baked":
                return formats.to_baked(raw, self.patched)
            case "native":
                raise_problems(formats.native_errors(self.patched))
                return raw
        raise ValueError(f"format {format!r} not in {formats.FORMATS}")

    def load_state_dict(self, state: dict[str, Tensor], format: str = "raw", strict: bool = True):
        """Load a state in `format` into the patched model."""
        self._check_open()
        match format:
            case "raw":
                raw = state
            case "baked":
                raw = formats.from_baked(state, self.patched)
            case "native":
                raise_problems(formats.native_errors(self.patched))
                raw = state
            case _:
                raise ValueError(f"format {format!r} not in {formats.FORMATS}")
        return self.model.load_state_dict(formats.from_raw(raw, self.patched), strict=strict)

    def bake(self) -> None:
        """Write every unit-row weight into its stock module and end the session."""
        self._check_open()
        weights = [(p, p.unit_weight().detach()) for p in self.untied_layers()]
        for p in reversed(self.patched):
            p.uninstall()
        for p, weight in weights:
            getattr(p.module, p.layer_type.weight).data.copy_(weight)
        grouped.detach(self.model)
        probe.detach(self.model)
        self.closed = True

    def canonicalize(self) -> None:
        """Turn every patched module into its `dragonscale.nn` unit-row class in place, keeping
        its parameters and output, and end the session. The model then saves `native` keys."""
        self._check_open()
        raise_problems(formats.native_errors(self.patched))
        for p in reversed(self.patched):
            p.canonicalize()
        grouped.detach(self.model)
        probe.detach(self.model)
        self.closed = True

    def remove(self) -> None:
        """Undo every change; the weights keep what training and init wrote."""
        self._check_open()
        for p in reversed(self.patched):
            p.uninstall()
        grouped.detach(self.model)
        probe.detach(self.model)
        self.closed = True

    def describe(self) -> str:
        return self.plan.describe()


def apply(
    model: nn.Module,
    plan: Plan | Sequence[Rule] | str,
    backend: str | None = None,
    group: bool = True,
) -> Session:
    """Apply a plan, rule list or preset name to `model`; `group` as in `Session`."""
    if not isinstance(plan, Plan):
        plan = make_plan(model, plan)
    return Session(model, plan, backend, group)
