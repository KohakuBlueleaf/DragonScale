"""`RowSpec`: which slices of a weight are normalized, and to what radius."""

import math
from dataclasses import dataclass
from typing import Self

RADII = ("sqrt_fan_in",)


@dataclass(frozen=True)
class RowSpec:
    """View the weight as `view` (its own shape when None), index rows by the `row_dims` axes,
    normalize over every other axis to norm `radius`.

    `radius` is a float or `"sqrt_fan_in"` (unit-RMS entries). `row_dims=()` is one Frobenius row.
    """

    row_dims: tuple[int, ...] = (0,)
    view: tuple[int, ...] | None = None
    radius: float | str = 1.0

    def view_of(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """The row view of a weight of `shape`."""
        return tuple(shape) if self.view is None else tuple(self.view)

    def problems(self, shape: tuple[int, ...]) -> list[str]:
        """Why this spec cannot read a weight of `shape`; empty when it can."""
        view = self.view_of(shape)
        problems = []
        if math.prod(view) != math.prod(shape):
            problems.append(f"view {view} does not hold a weight of shape {tuple(shape)}")
        if list(self.row_dims) != sorted(set(self.row_dims)):
            problems.append(f"row_dims {self.row_dims} is not sorted and unique")
        if any(not 0 <= axis < len(view) for axis in self.row_dims):
            problems.append(f"row_dims {self.row_dims} outside the {len(view)} axes of {view}")
        if isinstance(self.radius, str) and self.radius not in RADII:
            problems.append(f"radius {self.radius!r} not a float or one of {RADII}")
        return problems

    def rows(self, shape: tuple[int, ...]) -> int:
        """Number of rows of a weight of `shape`."""
        view = self.view_of(shape)
        rows = 1
        for axis in self.row_dims:
            rows *= view[axis]
        return rows

    def fan_in(self, shape: tuple[int, ...]) -> int:
        """Elements per row of a weight of `shape`."""
        return math.prod(shape) // self.rows(shape)

    def radius_of(self, shape: tuple[int, ...]) -> float:
        """The row norm target for a weight of `shape`."""
        if self.radius == "sqrt_fan_in":
            return math.sqrt(self.fan_in(shape))
        return float(self.radius)

    def to_dict(self) -> dict:
        return {
            "row_dims": list(self.row_dims),
            "view": self.view and list(self.view),
            "radius": self.radius,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        view = data.get("view")
        return cls(tuple(data["row_dims"]), None if view is None else tuple(view), data["radius"])
