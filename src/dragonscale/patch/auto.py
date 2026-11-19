"""`patch_model`: analysis, plan and patch in one call."""

from torch import nn

from dragonscale.analysis.absorb import absorb_plan
from dragonscale.analysis.suggest import suggest
from dragonscale.patch.session import Session, apply


def patch_model(
    model: nn.Module,
    *inputs,
    level: str = "selective",
    gains: str = "selective",
    output_gain: float | None = None,
    roles: dict[str, str] | None = None,
    allow_unknown: bool = False,
    backend: str | None = None,
    group: bool = True,
    model_kwargs: dict | None = None,
) -> Session:
    """Trace `model(*inputs, **model_kwargs)`, build the plan of `level`
    (`ds.analysis.suggest`), make its moves (`ds.analysis.absorb_plan`) and patch the model
    (`ds.apply`). The session's plan keeps the plan's notes."""
    kwargs = model_kwargs or {}
    plan = suggest(model, *inputs, level=level, gains=gains, output_gain=output_gain, roles=roles,
                   allow_unknown=allow_unknown, **kwargs)  # fmt: skip
    return apply(model, absorb_plan(model, plan, *inputs, **kwargs), backend=backend, group=group)
