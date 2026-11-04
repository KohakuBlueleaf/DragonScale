"""`Registry` and `build`: names resolve to factories once, at build time."""

import difflib
from collections.abc import Callable
from typing import Any

from dragonscale_common.imports import import_object


class Registry:
    """Name -> object map for one family; dotted paths also resolve."""

    def __init__(self, family: str):
        self.family = family
        self._items: dict[str, Any] = {}

    def register(self, name: str | None = None) -> Callable:
        """Decorator adding a class or factory under `name` (its `__name__` by default)."""

        def wrap(obj):
            self.add(name or obj.__name__, obj)
            return obj

        return wrap

    def add(self, name: str, obj: Any, replace: bool = False) -> None:
        """Add `obj`; a duplicate name raises unless `replace`."""
        if name in self._items and not replace:
            raise KeyError(f"duplicate {self.family} {name!r}")
        self._items[name] = obj

    def get(self, name: str) -> Any:
        """The entry for `name` or a dotted path; an unknown name lists close ones."""
        if name in self._items:
            return self._items[name]
        if "." in name or ":" in name:
            return import_object(name)
        close = difflib.get_close_matches(name, self._items, n=3)
        hint = f"; did you mean {', '.join(close)}?" if close else ""
        raise KeyError(f"unknown {self.family} {name!r}{hint}")

    def names(self) -> list[str]:
        return sorted(self._items)

    def items(self) -> list[tuple[str, Any]]:
        return sorted(self._items.items())

    def __contains__(self, name: str) -> bool:
        return name in self._items


def build(choice: Any, registry: Registry, key: str = "name", **defaults) -> Any:
    """Resolve a name, dotted path, `{key: name, ...}`, class or instance; None stays None."""
    match choice:
        case None:
            return None
        case str():
            return registry.get(choice)(**defaults)
        case dict():
            choice = dict(choice)
            factory = registry.get(choice.pop(key))
            return factory(**{**defaults, **choice})
        case type():
            return choice(**defaults)
        case _:
            return choice
