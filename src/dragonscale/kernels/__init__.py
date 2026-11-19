"""Kernel ops: `resolve(op, device)` binds one implementation at build time.

Op contracts live in `ops`. Importing this package never imports triton.
"""

from dragonscale.kernels.errors import NoImplementationError
from dragonscale.kernels.registry import available, register, resolve

__all__ = ["NoImplementationError", "available", "register", "resolve"]
