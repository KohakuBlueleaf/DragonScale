"""`LayerPlan` and `Plan`: what happens to one layer, and to every layer of one model."""

from dataclasses import dataclass, field, replace
from typing import Self

from dragonscale.spec.rows import RowSpec

TIED_OPTIONS = ("mode", "spec", "gain", "init", "method")


@dataclass(frozen=True)
class LayerPlan:
    """One layer to patch: its path, layer type name and every resolved option.

    `tied_to` names the earlier layer whose weight this one also reads; the two share one gain
    and every option in `TIED_OPTIONS`.
    """

    path: str
    layer_type: str
    mode: str
    spec: RowSpec
    gain: str = "channel"
    init: str = "preserve"
    install: str = "forward"
    method: str = "normalize"
    tied_to: str | None = None

    def to_dict(self) -> dict:
        return {**self.__dict__, "spec": self.spec.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(**{**data, "spec": RowSpec.from_dict(data["spec"])})

    def with_options(self, **options) -> Self:
        return replace(self, **options)


@dataclass(frozen=True)
class Plan:
    """Every layer plan of one model, in module order, and the scale moves to make before it
    is applied (`moves`: source path and consumer paths, run by `ds.analysis.absorb_plan`).
    An empty plan changes nothing."""

    layers: tuple[LayerPlan, ...] = field(default_factory=tuple)
    moves: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def to_dict(self) -> dict:
        return {
            "layers": [layer.to_dict() for layer in self.layers],
            "moves": [[source, list(into)] for source, into in self.moves],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        layers = tuple(LayerPlan.from_dict(row) for row in data["layers"])
        moves = tuple((source, tuple(into)) for source, into in data.get("moves", ()))
        return cls(layers, moves)

    def describe(self) -> str:
        """One line per move and per layer: path, type, mode, gain, init, install, method, spec
        and tie."""
        lines = [f"{len(self.moves)} move(s), {len(self.layers)} layer(s)"]
        for source, into in self.moves:
            lines.append(f"  move {source} -> {', '.join(into)}")
        for p in self.layers:
            spec = f"row_dims={p.spec.row_dims} view={p.spec.view} radius={p.spec.radius}"
            gain = p.gain if p.mode == "unit_gain" else "-"
            tied = f" tied_to={p.tied_to}" if p.tied_to is not None else ""
            lines.append(
                f"  {p.path or '<root>'}: {p.layer_type} {p.mode} gain={gain} init={p.init} "
                f"install={p.install} method={p.method} {spec}{tied}"
            )
        return "\n".join(lines)
