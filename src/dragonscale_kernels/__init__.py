"""Kernel ops: `resolve(op, device)` binds one implementation at build time.

Op contracts live in `ops`. Importing this package never imports triton or tilelang.
"""

from dragonscale_kernels.errors import NoImplementationError
from dragonscale_kernels.registry import available, register, resolve

__all__ = ["NoImplementationError", "available", "register", "resolve"]
