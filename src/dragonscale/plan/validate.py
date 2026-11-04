"""`validate(model, plan)`: every reason the plan cannot be applied to the model."""

import torch
from torch import nn
from torch.nn.utils import parametrize

from dragonscale.functional.rows import row_norms
from dragonscale.layer_types import LAYER_TYPES
from dragonscale.nn.layers import UnitRows
from dragonscale.plan.layer_plan import TIED_OPTIONS, LayerPlan, Plan

MARKER = "_dragonscale_patched"


def _layer_errors(layer: LayerPlan, module: nn.Module) -> list[str]:
    """Why `module` cannot take `layer`."""
    where = layer.path or "<root>"
    layer_type = LAYER_TYPES.get(layer.layer_type)
    if layer_type is None:
        return [f"{where}: unknown layer type {layer.layer_type!r}"]
    if getattr(module, MARKER, None) is not None:
        return [f"{where}: already patched"]
    if isinstance(module, UnitRows):
        return [f"{where}: already a unit-row module"]

    errors = []
    weight = getattr(module, layer_type.weight, None)
    if parametrize.is_parametrized(module, layer_type.weight):
        return [f"{where}: {layer_type.weight} already has a parametrization"]
    if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
        errors.append(f"{where}: {layer_type.weight} is not a floating-point tensor")
    else:
        errors += [f"{where}: {p}" for p in layer.spec.problems(weight.shape)]
    if layer.install == "forward" and layer_type.forward is None:
        errors.append(f"{where}: {layer_type.name} has no forward; use install 'parametrize'")
    if layer.install == "forward" and "forward" in vars(module):
        errors.append(f"{where}: already has an instance forward; use install 'parametrize'")
    if layer.mode == "unit_gain" and layer.install == "forward" and hasattr(module, "gain"):
        errors.append(f"{where}: already has an attribute 'gain'")
    preserving = layer.init in ("preserve", "reset")
    if preserving and layer.mode == "unit_gain" and layer.gain == "scalar":
        errors.append(f"{where}: init {layer.init!r} needs a channel gain")
    if layer.init == "absorb" and not errors and not _at_radius(weight, layer):
        errors.append(
            f"{where}: init 'absorb' needs its row norms moved into the consumers first: "
            "ds.analysis.absorb_plan(model, plan, *inputs)"
        )
    return errors


def _at_radius(weight: torch.Tensor, layer: LayerPlan) -> bool:
    """Whether every row of `weight` has the layer's radius (relative 1e-6)."""
    radius = layer.spec.radius_of(weight.shape)
    norms = row_norms(weight.detach(), layer.spec)
    return not bool(((norms - radius).abs() > 1e-6 * radius).any())


def _holders(model: nn.Module) -> dict[int, list[str]]:
    """Parameter id -> the paths of every module that holds it directly."""
    found: dict[int, list[str]] = {}
    for path, module in model.named_modules():
        for param in module.parameters(recurse=False):
            found.setdefault(id(param), []).append(path)
    return found


def _tie_errors(model: nn.Module, plan: Plan) -> list[str]:
    """Tied weights: every holder is planned, each later one is tied to the first, same options."""
    modules = dict(model.named_modules())
    holders = _holders(model)
    planned = {layer.path for layer in plan}
    first: dict[int, LayerPlan] = {}
    errors = []
    for layer in plan:
        module, layer_type = modules.get(layer.path), LAYER_TYPES.get(layer.layer_type)
        if module is None or layer_type is None:
            continue
        weight = getattr(module, layer_type.weight, None)
        where = layer.path or "<root>"
        for other in holders.get(id(weight), ()):
            if other != layer.path and other not in planned:
                errors.append(f"{where}: its weight is tied to {other}, which is not planned")

        earlier = first.setdefault(id(weight), layer)
        if earlier is layer:
            if layer.tied_to is not None:
                errors.append(f"{where}: tied_to {layer.tied_to!r}, which reads another weight")
            continue
        if layer.tied_to != earlier.path:
            errors.append(f"{where}: reads the weight of {earlier.path}; set tied_to")
        for name in TIED_OPTIONS:
            if getattr(layer, name) != getattr(earlier, name):
                errors.append(f"{where}: {name} differs from {earlier.path}, its tied layer")
        if (layer.install == "bake") != (earlier.install == "bake"):
            errors.append(f"{where}: bakes or not unlike {earlier.path}, its tied layer")
    return errors


def validate(model: nn.Module, plan: Plan) -> list[str]:
    """Every error of `plan` on `model`; empty when it can be applied."""
    modules = dict(model.named_modules())
    errors = _tie_errors(model, plan)
    if plan.moves:
        errors.append(
            f"{len(plan.moves)} scale move(s) not made: "
            "ds.analysis.absorb_plan(model, plan, *inputs)"
        )
    seen = set()
    for layer in plan:
        where = layer.path or "<root>"
        if layer.path in seen:
            errors.append(f"{where}: listed twice")
        seen.add(layer.path)
        module = modules.get(layer.path)
        if module is None:
            errors.append(f"{where}: no such module")
            continue
        errors += _layer_errors(layer, module)
    return errors
