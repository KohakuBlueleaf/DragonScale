"""`make_plan`: the first rule claiming a module decides its layer plan."""

from collections.abc import Sequence

from torch import nn

from dragonscale.layer_types import layer_type_of
from dragonscale.plan.layer_plan import TIED_OPTIONS, LayerPlan, Plan
from dragonscale.plan.presets import preset
from dragonscale.plan.rules import Rule


def make_plan(model: nn.Module, rules: Sequence[Rule] | str) -> Plan:
    """The plan of `model` under `rules` (or a preset name); modules without a layer type are
    skipped. A layer whose weight an earlier layer already reads is `tied_to` it and takes its
    options."""
    rules = preset(rules) if isinstance(rules, str) else list(rules)
    planned = []
    first: dict[int, LayerPlan] = {}
    for path, module in model.named_modules():
        layer_type = layer_type_of(module)
        if layer_type is None:
            continue
        rule = next((r for r in rules if r.claims(path, module)), None)
        if rule is None or rule.mode == "plain":
            continue
        spec = rule.spec if rule.spec is not None else layer_type.spec
        spec = spec(module) if callable(spec) else spec
        install = rule.install
        if install == "auto":
            install = "forward" if layer_type.forward is not None else "parametrize"
        layer = LayerPlan(
            path=path,
            layer_type=layer_type.name,
            mode=rule.mode,
            spec=spec,
            gain=rule.gain,
            init=rule.init,
            install=install,
            method=rule.method,
        )

        # A weight an earlier layer already reads: tie to it
        weight = getattr(module, layer_type.weight, None)
        earlier = first.get(id(weight))
        if earlier is not None:
            layer = layer.with_options(
                tied_to=earlier.path, **{name: getattr(earlier, name) for name in TIED_OPTIONS}
            )
        else:
            first[id(weight)] = layer
        planned.append(layer)
    return Plan(tuple(planned))
