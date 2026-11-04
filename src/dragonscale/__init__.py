"""DragonScale: unit-row weights and free-scale analysis.

import dragonscale as ds

report = ds.analysis.free_scales(model, x)   # which scales the loss cannot see
session = ds.apply(model, "unit_gain")       # every Linear / Conv / Embedding, same function
optimizer = torch.optim.AdamW(model.parameters())  # gains are module parameters
hooks = ds.optim.attach(optimizer, session)  # tangent steps, rows kept at the radius
...
session.bake()                               # stock modules with the unit-row weights
"""

from dragonscale import analysis, functional, layer_types, monitor, nn, optim, spec
from dragonscale.patch import Session, apply
from dragonscale.plan import PRESETS, LayerPlan, Plan, Rule, make_plan, preset, validate
from dragonscale.spec import RowSpec, defaults

__version__ = "0.1.0"

__all__ = [
    "PRESETS",
    "LayerPlan",
    "Plan",
    "RowSpec",
    "Rule",
    "Session",
    "analysis",
    "apply",
    "defaults",
    "functional",
    "layer_types",
    "make_plan",
    "monitor",
    "nn",
    "optim",
    "preset",
    "spec",
    "validate",
]
