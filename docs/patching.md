# Patching

`ds.patch_model(model, *inputs)` runs three steps. Run them yourself to look at or edit the plan
in between:

```python
import torch
from torch import nn

import dragonscale as ds

torch.manual_seed(0)
net = nn.Sequential(nn.Embedding(32, 16), nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 32))
t = torch.randint(0, 32, (4, 8))
plan = ds.analysis.suggest(net, t)  # the plan: one LayerPlan per layer, moves, notes
plan = ds.analysis.absorb_plan(net, plan, t)  # make the moves; the output does not change
session = ds.apply(net, plan)  # patch
session.remove()
```

`patch_model` takes the same options as `suggest` (`level`, `gains`, `output_gain`, `roles`,
`allow_unknown`) and as `apply` (`backend`, `group`).

## Plans and rules

A plan is data: one `LayerPlan` per layer. `ds.make_plan(model, rules)` matches `ds.Rule(match,
**options)` against module paths (a regex) or types, first match wins. `ds.apply` takes a plan, a
rule list or a preset name (`unit_gain`, `unit`, `linear_unit_gain`, `conv_unit_gain`,
`parametrize_unit_gain`).

| option | values | meaning |
| --- | --- | --- |
| `mode` | `unit_gain`, `unit`, `plain` | unit rows times a gain, unit rows alone, untouched |
| `gain` | `channel`, `scalar` | one gain per row, or one for the layer |
| `init` | `preserve`, `ones`, `reset`, `absorb`, a number | gain = row length (same output), 1, re-initialize then preserve, rows already moved by `absorb_plan`, or this value |
| `install` | `forward`, `parametrize`, `bake` | see the README |
| `spec` | a `ds.RowSpec` | which dimensions form a row, and the radius |

```python
model = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 8))
x = torch.randn(4, 8)
before = model(x)
session = ds.apply(model, [ds.Rule("0", mode="unit_gain"), ds.Rule(".*", mode="plain")])
assert torch.allclose(model(x), before, atol=1e-6)
print(session.describe().splitlines()[0])
session.remove()
```

## Checkpoints

| call | result |
| --- | --- |
| `session.state_dict("raw")` | raw weights and gains, under the module's own keys plus `<path>.gain` |
| `session.state_dict("baked")` | unit-row weights under the stock keys; loads into the stock model |
| `session.load_state_dict(state, format)` | reads either |
| `session.bake()` | stock modules with the unit-row weights, same output |
| `session.canonicalize()` | the patched modules become `ds.nn` modules in place |
| `session.remove()` | the patch undone; the weights keep what training wrote |

## Modules for new models

`ds.nn` has the same layers as modules (`UnitRowLinear`, `UnitRowConv2d`, `UnitRowEmbedding`, ...),
`ds.nn.Gain`, and `ds.nn.QKNorm` (q and k RMS-normalized per head, a gain on q only).
`ds.functional.unit_rows(w, spec, gain)` does it for a bare weight.

```python
q, k = torch.randn(2, 4, 10, 16), torch.randn(2, 4, 10, 16)
q, k = ds.nn.QKNorm(16)(q, k)
w = ds.functional.unit_rows(torch.randn(32, 8))
print(round(w.norm(dim=1).mean().item(), 4))
```

```text
1.0
```

`ds.fold_layerscale(session, {"blocks.0.mlp.fc2": "blocks.0.ls2"})` multiplies a LayerScale factor
into the gain (and bias) of the layer before it and replaces the LayerScale module with
`nn.Identity`; the output does not change.

## Monitor

`ds.monitor.Monitor(session, angles=False)`: `start()` before a forward, `read()` after it gives,
per patched layer, the output RMS and largest entry, the raw row lengths (min, mean, max), the
gains (min, median, max) and, with `angles=True`, the angle each row turned since the last read.

```python
session = ds.apply(model, "unit_gain")
monitor = ds.monitor.Monitor(session)
monitor.start()
model(x)
print(sorted(monitor.read())[0])
```

```text
0/gain_max
```
