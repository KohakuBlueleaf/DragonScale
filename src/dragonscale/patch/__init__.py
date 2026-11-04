"""Apply a plan to a model: replaced forwards, parametrized weights or baked weights, in a
`Session`."""

from dragonscale.patch.patched import PatchedLayer
from dragonscale.patch.session import Session, apply

__all__ = ["PatchedLayer", "Session", "apply"]
