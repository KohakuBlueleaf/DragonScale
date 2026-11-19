"""`make_plan`: the first rule claiming a module decides its layer plan."""

from collections.abc import Sequence

from torch import nn

from dragonscale.layer_types import layer_type_of, read_by_parent
from dragonscale.plan.layer_plan import TIED_OPTIONS, LayerPlan, Plan
from dragonscale.plan.presets import preset
from dragonscale.plan.rules import Rule
from dragonscale.problems import raise_problems


def make_plan(model: nn.Module, rules: Sequence[Rule] | str) -> Plan:
    """The plan of `model` under `rules` (or a preset name); modules without a layer type are
    skipped. `install="auto"` is `parametrize` for a layer whose weight its parent reads itself
    (`dragonscale.layer_types.READ_BY_PARENT`). A layer whose weight an earlier layer already reads is `tied_to` it and takes its
    options."""
    rules = preset(rules) if isinstance(rules, str) else list(rules)
    planned, problems = [], []
    first: dict[int, LayerPlan] = {}
    by_parent = read_by_parent(model)
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
            plain_forward = layer_type.forward is not None and path not in by_parent
            install = "forward" if plain_forward else "parametrize"
        layer = LayerPlan(
            path=path,
            layer_type=layer_type.name,
            mode=rule.mode,
            spec=spec,
            gain=rule.gain,
            init=rule.init,
            install=install,
        )

        # A weight an earlier layer already reads: tie to it
        weight = getattr(module, layer_type.weight, None)
        earlier = first.get(id(weight))
        if earlier is not None:
            # Options a rule sets must agree; a layer type's default spec yields to the first's
            names = ("mode", "gain", "init") + (("spec",) if rule.spec is not None else ())
            differ = [n for n in names if getattr(layer, n) != getattr(earlier, n)]
            if differ:
                problems.append(
                    f"{path}: shares its weight with {earlier.path} but its rule gives a "
                    f"different {', '.join(differ)}; give both the same options"
                )
            layer = layer.with_options(
                tied_to=earlier.path, **{name: getattr(earlier, name) for name in TIED_OPTIONS}
            )
        else:
            first[id(weight)] = layer
        planned.append(layer)
    raise_problems(problems)
    return Plan(tuple(planned))
