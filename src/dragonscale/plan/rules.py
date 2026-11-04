"""`Rule`: which modules it claims and the options it gives them."""

import re
from collections.abc import Callable
from dataclasses import dataclass

from torch import nn

from dragonscale.spec.rows import RowSpec

Matcher = str | type | tuple[type, ...] | Callable[[str, nn.Module], bool]


@dataclass(frozen=True)
class Rule:
    """Claims modules by `match`: a regex the full path must match, a module type (or tuple of
    types), or a predicate `(path, module) -> bool`. `mode="plain"` claims and leaves untouched.

    `spec` None is the layer type's default; a callable takes the module. `install="auto"` is
    `forward` where the layer type has a forward, else `parametrize`.
    """

    match: Matcher
    mode: str = "unit_gain"
    gain: str = "channel"
    spec: RowSpec | Callable[[nn.Module], RowSpec] | None = None
    init: str = "preserve"
    install: str = "auto"
    method: str = "normalize"

    def claims(self, path: str, module: nn.Module) -> bool:
        """Whether this rule claims the module at `path`."""
        match self.match:
            case str():
                return re.fullmatch(self.match, path) is not None
            case type() | tuple():
                return isinstance(module, self.match)
            case _:
                return bool(self.match(path, module))
