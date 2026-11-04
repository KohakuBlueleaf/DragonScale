"""`absorb`: move a source's per-channel scale into its consumers' input columns, keeping the
function; `absorb_plan` makes a plan's moves and absorbs its `init="absorb"` layers.

The source is a module with a layer type (its rows are rescaled to the radius) or a norm with an
affine (its weight is divided by its magnitude). Consumers are given, or found by
`find_consumers`: the modules with a layer type whose first input carries the source's degree,
taken in call order until the move is a free direction. The move must be free on the given inputs,
and the output must not change.
"""

from collections.abc import Sequence
from dataclasses import replace

import torch
from torch import Tensor, nn

from dragonscale.analysis.report import FreeScales, solve
from dragonscale.analysis.trace import DegreeTrace, trace_degrees
from dragonscale.functional.rows import row_norms
from dragonscale.layer_types import layer_type_of
from dragonscale.plan.layer_plan import Plan
from dragonscale.spec.rows import RowSpec
from dragonscale_common.problems import raise_problems
from dragonscale_kernels.layout import as_rows
from dragonscale_kernels.ops import FLOOR


def _source(
    module: nn.Module, spec: RowSpec | None
) -> tuple[Tensor, Tensor | None, RowSpec | None]:
    """The source weight, its bias, and its row spec (None for a norm). A layer's spec must index
    rows by output channel: rows along `out_axis`, or the layer type's own default."""
    layer_type = layer_type_of(module)
    if layer_type is not None:
        default = layer_type.spec(module)
        spec = spec or default
        plain = spec.view is None and tuple(spec.row_dims) == (layer_type.out_axis,)
        same = (tuple(spec.row_dims), spec.view) == (tuple(default.row_dims), default.view)
        if not (plain or same):
            raise ValueError(f"{layer_type.name}: absorb needs rows along the output channels")
        bias = getattr(module, layer_type.bias, None) if layer_type.bias else None
        return getattr(module, layer_type.weight), bias, spec
    weight = getattr(module, "weight", None)
    if not isinstance(weight, Tensor) or weight.dim() != 1:
        raise ValueError(f"{type(module).__name__} is neither a layer type nor an affine norm")
    return weight, getattr(module, "bias", None), None


def _outputs(model: nn.Module, inputs, kwargs) -> list[Tensor]:
    out = model(*inputs, **kwargs)
    flat = (
        out.values()
        if isinstance(out, dict)
        else (out if isinstance(out, tuple | list) else (out,))
    )
    return [t.detach().clone() for t in flat if isinstance(t, Tensor) and t.is_floating_point()]


def _input_weight(module: nn.Module, channels: int) -> tuple[Tensor, int] | None:
    """The weight of a module with an input-channel axis of size `channels`, and that axis."""
    layer_type = layer_type_of(module)
    if layer_type is None or layer_type.in_axis is None:
        return None
    weight = getattr(module, layer_type.weight)
    return (weight, layer_type.in_axis) if weight.shape[layer_type.in_axis] == channels else None


def _coeffs(params: dict[int, str], weight: Tensor, bias, consumers) -> dict[str, float]:
    """Source weight (and bias) up, every consumer weight down."""
    coeffs = {params[id(weight)]: 1.0}
    if isinstance(bias, Tensor):
        coeffs[params[id(bias)]] = 1.0
    for cw, _ in consumers:
        coeffs[params[id(cw)]] = coeffs.get(params[id(cw)], 0.0) - 1.0
    return coeffs


def _tied(model: nn.Module, weight: Tensor) -> list[str]:
    """Every module that holds `weight` directly."""
    return [p for p, m in model.named_modules() if any(q is weight for q in m.parameters(False))]


def find_consumers(
    model: nn.Module, source: str, trace: DegreeTrace, report: FreeScales
) -> list[str] | None:
    """The consumers of `source` in call order, added until the move is free; None when there is
    no such set, the source weight is tied, or its rows are not along the output channels."""
    modules = dict(model.named_modules())
    params = {id(p): n for n, p in model.named_parameters()}
    try:
        weight, bias, spec = _source(modules[source], None)
    except ValueError:
        return None
    if len(_tied(model, weight)) > 1:
        return None
    channels = weight.numel() if spec is None else spec.rows(weight.shape)
    k = trace.names.index(params[id(weight)])
    chosen: list[str] = []
    for path, degree in trace.module_inputs.items():
        if path == source or abs(degree[k].item()) < 1e-9:
            continue
        if _input_weight(modules[path], channels) is None:
            continue
        chosen.append(path)
        consumers = [_input_weight(modules[p], channels) for p in chosen]
        if report.contains(_coeffs(params, weight, bias, consumers)):
            return chosen
    return None


def _restore_modes(modes: list[tuple[nn.Module, bool]]) -> None:
    """Each module back to its training mode, parents first."""
    for module, training in modes:
        if module.training != training:
            module.train(training)


def absorb(
    model: nn.Module,
    source: str,
    *inputs,
    into: Sequence[str] | None = None,
    spec: RowSpec | None = None,
    rtol: float = 1e-4,
    **kwargs,
) -> float:
    """Move `source`'s channel scale into `into` (None: found); returns the largest relative
    output change. Runs the model in eval mode with TF32 off, and restores both after."""
    modes = [(m, m.training) for m in model.modules()]
    tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        return _absorb(model, source, inputs, into, spec, rtol, kwargs)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        _restore_modes(modes)


@torch.no_grad()
def _absorb(
    model: nn.Module,
    source: str,
    inputs: tuple,
    into: Sequence[str] | None,
    spec: RowSpec | None,
    rtol: float,
    kwargs: dict,
) -> float:
    modules = dict(model.named_modules())
    params = {id(p): n for n, p in model.named_parameters()}
    weight, bias, spec = _source(modules[source], spec)
    holders = _tied(model, weight)
    if len(holders) > 1:
        raise ValueError(f"{source}: its weight is tied to {holders}; absorb does not move it")
    scale = weight.abs() if spec is None else row_norms(weight, spec) / spec.radius_of(weight.shape)
    channels = scale.numel()

    # Consumers: given, or found from the degree trace
    trace = trace_degrees(model, *inputs, **kwargs)
    report = solve(trace)
    if into is None:
        into = find_consumers(model, source, trace, report)
        if into is None:
            raise ValueError(f"{source}: no set of consumers makes a free direction; pass `into`")
    consumers, problems = [], []
    for path in into:
        found = _input_weight(modules[path], channels)
        if found is None:
            problems.append(f"{path}: no input-channel axis of size {channels}")
        else:
            consumers.append(found)
    raise_problems(problems)
    if not report.contains(_coeffs(params, weight, bias, consumers)):
        raise ValueError(f"absorbing {source} into {list(into)} is not a free direction")

    before = _outputs(model, inputs, kwargs)
    saved = [
        weight.clone(),
        None if bias is None else bias.clone(),
        *(cw.clone() for cw, _ in consumers),
    ]
    factor = torch.where(scale > FLOOR, scale, torch.ones_like(scale))

    # Source channels divided by their scale, consumer columns multiplied by it
    if spec is None:
        weight.div_(factor.to(weight.dtype))
    else:
        view = spec.view_of(weight.shape)
        rows = weight.reshape(view) / as_rows(factor.to(weight.dtype), view, spec.row_dims)
        weight.copy_(rows.reshape(weight.shape))
    if isinstance(bias, Tensor):
        bias.div_(factor.to(bias.dtype))
    for cw, axis in consumers:
        shape = [1] * cw.dim()
        shape[axis] = -1
        cw.mul_(factor.to(cw.dtype).reshape(shape))

    # The output must not change; undo otherwise
    after = _outputs(model, inputs, kwargs)
    change = max(
        ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()
        for a, b in zip(after, before, strict=True)
    )
    if change > rtol:
        for target, value in zip([weight, bias, *(cw for cw, _ in consumers)], saved, strict=True):
            if value is not None:
                target.copy_(value)
        raise ValueError(f"absorbing {source} into {list(into)} changed the output by {change:.2e}")
    return change


def absorb_plan(model: nn.Module, plan: Plan, *inputs, rtol: float = 1e-4, **kwargs) -> Plan:
    """Make the plan's moves in order, then absorb every other `init="absorb"` layer; returns the
    plan without moves, ready for `ds.apply`. A moved norm's affine weight is frozen."""
    modules = dict(model.named_modules())
    layers = {layer.path: layer for layer in plan}
    moved = set()
    for source, into in plan.moves:
        layer = layers.get(source)
        spec = None if layer is None else layer.spec
        absorb(model, source, *inputs, into=into, spec=spec, rtol=rtol, **kwargs)
        if layer_type_of(modules[source]) is None:
            modules[source].weight.requires_grad_(False)
        moved.add(source)
    for layer in plan:
        if layer.init == "absorb" and layer.tied_to is None and layer.path not in moved:
            absorb(model, layer.path, *inputs, spec=layer.spec, rtol=rtol, **kwargs)
    return replace(plan, moves=())
