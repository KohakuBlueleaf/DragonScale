"""State-dict formats of a patched model.

| format | holds | loads into |
| --- | --- | --- |
| `raw` | raw weight and `gain` under `<path>.weight` / `<path>.gain` | a patched model |
| `baked` | the unit-row weight under `<path>.weight`, no gain | the stock model |
| `native` | the `raw` keys | a model of `dragonscale.nn` unit-row modules |
"""

import torch
from torch import Tensor

from dragonscale.functional.rows import fill_zero_rows_, preserving_gain
from dragonscale.patch.patched import GAIN, PatchedLayer
from dragonscale_common.problems import raise_problems

FORMATS = ("raw", "baked", "native")


def key(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def native_errors(patched: list[PatchedLayer]) -> list[str]:
    """Every patched layer with no `dragonscale.nn` unit-row equivalent."""
    errors = []
    for p in patched:
        where = p.plan.path or "<root>"
        if p.layer_type.unit_class is None:
            errors.append(f"{where}: {p.layer_type.name} has no unit-row module")
        if p.plan.method != "normalize":
            errors.append(f"{where}: method {p.plan.method!r} has no unit-row module")
        if p.plan.install == "parametrize" and hasattr(p.module, GAIN):
            errors.append(f"{where}: already has an attribute 'gain'")
    return errors


def renames(patched: list[PatchedLayer]) -> dict[str, str]:
    """Module state key -> raw key, for every key the `parametrize` install moves."""
    out = {}
    for p in patched:
        if p.plan.install != "parametrize":
            continue
        path, weight = p.plan.path, p.layer_type.weight
        out[key(path, f"parametrizations.{weight}.original")] = key(path, weight)
        out[key(path, f"parametrizations.{weight}.0.{GAIN}")] = key(path, GAIN)
    return out


def to_raw(state: dict[str, Tensor], patched: list[PatchedLayer]) -> dict[str, Tensor]:
    names = renames(patched)
    return {names.get(k, k): v for k, v in state.items()}


def from_raw(state: dict[str, Tensor], patched: list[PatchedLayer]) -> dict[str, Tensor]:
    back = {raw: actual for actual, raw in renames(patched).items()}
    return {back.get(k, k): v for k, v in state.items()}


def to_baked(state: dict[str, Tensor], patched: list[PatchedLayer]) -> dict[str, Tensor]:
    """A raw state with every layer's weight replaced by its unit-row weight and no gain."""
    out = dict(state)
    with torch.no_grad():
        for p in patched:
            out[key(p.plan.path, p.layer_type.weight)] = p.unit_weight().detach()
            out.pop(key(p.plan.path, GAIN), None)
    return out


def from_baked(state: dict[str, Tensor], patched: list[PatchedLayer]) -> dict[str, Tensor]:
    """A raw state that computes the baked weights: raw = baked, gain = its row norms; a zero row
    gets a random direction with gain 0."""
    out = dict(state)
    problems = []
    for p in patched:
        plan = p.plan
        if plan.mode == "unit_gain" and plan.gain == "scalar":
            problems.append(f"{plan.path}: a baked weight does not load into a scalar gain")
            continue
        if plan.method == "rescale":
            problems.append(
                f"{plan.path}: a baked weight does not load into rows the optimizer rescales"
            )
            continue
        if plan.mode == "unit_gain":
            weight_key = key(plan.path, p.layer_type.weight)
            baked = state[weight_key]
            out[key(plan.path, GAIN)] = preserving_gain(baked, plan.spec).to(baked.dtype)
            raw = baked.clone()
            if fill_zero_rows_(raw, plan.spec) is not None:
                out[weight_key] = raw
    raise_problems(problems)
    return out
