"""Named rule lists.

| preset | layers |
| --- | --- |
| `unit_gain` | every supported module: unit rows, channel gain, same function at start |
| `unit` | every supported module: unit rows, no gain (the function changes) |
| `linear_unit_gain` | Linear only |
| `conv_unit_gain` | Conv1d / 2d / 3d only |
| `parametrize_unit_gain` | every supported module through a parametrized `weight` |
"""

from torch import nn

from dragonscale.plan.rules import Rule

PRESETS: dict[str, tuple[Rule, ...]] = {
    "unit_gain": (Rule(".*"),),
    "unit": (Rule(".*", mode="unit", init="ones"),),
    "linear_unit_gain": (Rule(nn.Linear),),
    "conv_unit_gain": (Rule((nn.Conv1d, nn.Conv2d, nn.Conv3d)),),
    "parametrize_unit_gain": (Rule(".*", install="parametrize"),),
}


def preset(name: str) -> list[Rule]:
    """The rules of a named preset."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; presets: {', '.join(PRESETS)}")
    return list(PRESETS[name])
