"""Free-scale analysis: degree propagation over a real forward, the free directions it leaves,
the scale-Jacobian spectrum, the plan that removes the dangerous ones, and scale moves.

    report = ds.analysis.free_scales(model, x)
    print(report.describe())
    plan = ds.analysis.suggest(model, x, level="absorb")
    session = ds.apply(model, ds.analysis.absorb_plan(model, plan, x))
"""

from dragonscale.analysis.absorb import absorb, absorb_plan
from dragonscale.analysis.jacobian import ScaleJacobian, scale_jacobian
from dragonscale.analysis.report import Direction, FreeScales, free_scales
from dragonscale.analysis.rules import register_degree_rule
from dragonscale.analysis.suggest import suggest
from dragonscale.analysis.trace import DegreeTrace, trace_degrees

__all__ = [
    "DegreeTrace",
    "Direction",
    "FreeScales",
    "ScaleJacobian",
    "absorb",
    "absorb_plan",
    "free_scales",
    "register_degree_rule",
    "scale_jacobian",
    "suggest",
    "trace_degrees",
]
