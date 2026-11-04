"""`suggest`: the plan of one level.

| level | plan |
| --- | --- |
| `report` | empty; the report (with its architecture notes) is the result |
| `minimal` | `unit_gain` on every module whose weight is a matrix in a dangerous direction |
| `matrices` | `unit_gain` on every module with a layer type; affines and biases kept |
| `absorb` | norm affines moved into their consumers and frozen; layers whose rows can move |
| | into their consumers absorbed (`mode="unit"`, `init="absorb"`), every other one `unit_gain` |

`absorb` needs `ds.analysis.absorb_plan(model, plan, *inputs)` before `ds.apply`. Every level
keeps the output at the start.
"""

import re
from dataclasses import replace

from torch import nn

from dragonscale.analysis.absorb import find_consumers
from dragonscale.analysis.report import FreeScales, solve
from dragonscale.analysis.trace import trace_degrees
from dragonscale.layer_types import layer_type_of
from dragonscale.plan.build import make_plan
from dragonscale.plan.layer_plan import Plan
from dragonscale.plan.rules import Rule
from dragonscale_common.problems import raise_problems

LEVELS = ("report", "minimal", "matrices", "absorb")


def holders(model: nn.Module) -> dict[int, list[str]]:
    """Parameter id -> the path of every module with a layer type whose weight it is."""
    found: dict[int, list[str]] = {}
    for path, module in model.named_modules():
        layer_type = layer_type_of(module)
        weight = None if layer_type is None else getattr(module, layer_type.weight, None)
        if weight is not None:
            found.setdefault(id(weight), []).append(path)
    return found


def _minimal(model: nn.Module, report: FreeScales) -> Plan:
    params = dict(model.named_parameters())
    held = holders(model)
    paths, problems = set(), []
    for name in sorted({n for d in report.dangerous for n in d.fast_ends}):
        found = held.get(id(params[name]))
        if not found:
            problems.append(f"{name}: in a dangerous direction, held by no module with a type")
            continue
        paths.update(found)
    raise_problems(problems)
    rules = [Rule(re.escape(path)) for path in sorted(paths)] + [Rule(".*", mode="plain")]
    return make_plan(model, rules)


def _norm_affine(module: nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    return layer_type_of(module) is None and isinstance(weight, nn.Parameter) and weight.dim() == 1


def _absorbed(model: nn.Module, inputs, kwargs) -> Plan:
    trace = trace_degrees(model, *inputs, **kwargs)
    report = solve(trace)
    modules = dict(model.named_modules())

    # Moves in call order: a producer's move rescales its consumers' columns before they move
    moves, absorbed = [], set()
    for path in trace.module_inputs:
        module = modules[path]
        if not _norm_affine(module) and layer_type_of(module) is None:
            continue
        into = find_consumers(model, path, trace, report)
        if into is None:
            continue
        moves.append((path, tuple(into)))
        if layer_type_of(module) is not None:
            absorbed.add(path)

    rules = []
    for path, module in modules.items():
        if layer_type_of(module) is None:
            continue
        options = {"mode": "unit", "init": "absorb"} if path in absorbed else {}
        rules.append(Rule(re.escape(path), **options))
    plan = make_plan(model, [*rules, Rule(".*", mode="plain")])
    return replace(plan, moves=tuple(moves))


def suggest(
    model: nn.Module,
    *inputs,
    level: str = "matrices",
    report: FreeScales | None = None,
    **kwargs,
) -> Plan:
    """The plan of `level` for `model` on `inputs`; `report` is reused by `minimal`."""
    match level:
        case "report":
            return Plan()
        case "minimal":
            return _minimal(model, report or solve(trace_degrees(model, *inputs, **kwargs)))
        case "matrices":
            return make_plan(model, "unit_gain")
        case "absorb":
            return _absorbed(model, inputs, kwargs)
    raise ValueError(f"level {level!r} not in {LEVELS}")
