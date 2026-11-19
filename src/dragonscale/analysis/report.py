"""`free_scales`: the free scales of a model from one degree trace, in four classes.

A free scale is a direction `s` over the log-scales of the parameter tensors along which the
model's output does not change: `A s = 0`, with `A` the constraints of every op of the trace and
the anchors (constraints set by adding a fixed-scale term). The free scales form the space
`ker A`; counts are dimensions of subspaces, so they do not depend on how a basis is chosen.

| class | space | count |
| --- | --- | --- |
| change a hidden state | `ker A` minus the part with `D s = 0` | `rank(D N)`, `N` a basis of `ker A` |
| change no hidden state | `ker A` with `D s = 0` (a raw weight behind unit rows, a gain before a norm) | `dim ker A - rank(D N)` |
| weakly pinned | free under every constraint but the anchors | `dim ker A0 - dim ker A` |
| ops without a rule | ops the trace treated as fixing the scale of their inputs | count per op |

`D` stacks the degree of every data-dependent tensor of the trace (every hidden state). A weakly
pinned direction is held only by a fixed-scale term added to a scaled one; once the scaled term
outgrows it, the direction is nearly free.
"""

import warnings
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from dragonscale.analysis.trace import DegreeTrace, trace_degrees

TOL = 1e-9


@dataclass(frozen=True)
class Direction:
    """One readable free direction: the log-scale coefficient of every parameter in it, whether
    it changes a hidden state, and the sources of the anchors that hold it (weakly pinned only)."""

    coeffs: dict[str, float]
    changes_hidden_state: bool
    anchors: tuple[str, ...] = ()

    def describe(self) -> str:
        terms = " ".join(f"{c:+g} {name}" for name, c in self.coeffs.items())
        held = f"  (held by: {'; '.join(self.anchors)})" if self.anchors else ""
        return f"{terms}{held}"


@dataclass(frozen=True)
class FreeScales:
    """The free scales of one model on one input.

    `changing` and `unchanging` are readable bases (reduced row-echelon form) of the free scales
    that change a hidden state and of those that change none; `weak` of the weakly pinned
    directions; `basis` an orthonormal basis of every free scale. `notes` name what a weight
    patch cannot change and the model code can."""

    names: list[str]
    shapes: list[tuple[int, ...]]
    basis: Tensor
    changing: list[Direction]
    unchanging: list[Direction]
    weak: list[Direction]
    unknown: dict[str, int]
    notes: list[str]
    constraints: int

    @property
    def count(self) -> int:
        """Every free scale: `dim ker A`."""
        return self.basis.shape[1] if self.basis.numel() else 0

    def vector(self, coeffs: dict[str, float]) -> Tensor:
        """A direction over every parameter from `{name: coefficient}`."""
        out = torch.zeros(len(self.names), dtype=torch.float64)
        for name, c in coeffs.items():
            out[self.names.index(name)] = c
        return out

    def contains(self, coeffs: dict[str, float]) -> bool:
        """Whether the direction `{name: coefficient}` is a free scale."""
        return in_span(self.basis, self.vector(coeffs))

    def summary(self) -> dict[str, int]:
        """The four counts."""
        return {
            "change_hidden_state": len(self.changing),
            "change_none": len(self.unchanging),
            "weakly_pinned": len(self.weak),
            "ops_without_rule": sum(self.unknown.values()),
        }

    def describe(self) -> str:
        lines = []
        if self.unknown:
            lines.append(unknown_message(self.unknown))
        lines.append(
            f"{self.count} free scale(s): {len(self.changing)} change a hidden state, "
            f"{len(self.unchanging)} change none; {len(self.weak)} weakly pinned; "
            f"{sum(self.unknown.values())} op call(s) without a rule"
        )
        if self.changing:
            lines.append("free scales that change a hidden state:")
            lines += [f"  {d.describe()}" for d in self.changing]
        if self.unchanging:
            alone = [d for d in self.unchanging if len(d.coeffs) == 1]
            lines.append("free scales that change no hidden state:")
            lines += [f"  {d.describe()}" for d in self.unchanging if d not in alone]
            if alone:
                names = ", ".join(next(iter(d.coeffs)) for d in alone)
                lines.append(f"  one parameter each ({len(alone)}): {names}")
        if self.weak:
            lines.append("weakly pinned (held only by adding a fixed-scale term):")
            lines += [f"  {d.describe()}" for d in self.weak]
        if self.notes:
            lines.append("for the model code:")
            lines += [f"  - {note}" for note in self.notes]
        return "\n".join(lines)


def unknown_message(unknown: dict[str, int]) -> str:
    ops = ", ".join(f"{name} (x{count})" for name, count in sorted(unknown.items()))
    return (
        f"WARNING: {len(unknown)} op(s) without a degree rule were treated as fixing the scale "
        f"of their inputs, which can hide free scales: {ops}. Add a rule with "
        "ds.analysis.register_degree_rule."
    )


def in_span(basis: Tensor, s: Tensor) -> bool:
    if basis.numel() == 0:
        return bool(s.abs().max() < TOL)
    residual = s - basis @ (basis.T @ s)
    return bool(residual.norm() <= 1e-7 * max(s.norm().item(), 1.0))


def null_space(a: Tensor) -> Tensor:
    """An orthonormal basis of `{s : a s = 0}`, `[groups, free]`."""
    gram = a.T @ a
    values, vectors = torch.linalg.eigh(gram)
    scale = max(values.abs().max().item(), 1.0)
    return vectors[:, values <= 1e-10 * scale]


def complement(inner: Tensor, outer: Tensor) -> Tensor:
    """An orthonormal basis of the part of span(`outer`) orthogonal to span(`inner`)."""
    if outer.numel() == 0:
        return outer
    rest = outer - inner @ (inner.T @ outer) if inner.numel() else outer
    u, sigma, _ = torch.linalg.svd(rest, full_matrices=False)
    return u[:, sigma > 1e-8 * max(sigma.max().item(), 1.0)] if sigma.numel() else rest


def echelon(basis: Tensor) -> Tensor:
    """The rows of `basis.T` in reduced row-echelon form: one readable vector per direction."""
    m = basis.T.clone()
    rows, cols = m.shape
    pivot_row = 0
    for col in range(cols):
        if pivot_row == rows:
            break
        best = pivot_row + m[pivot_row:, col].abs().argmax().item()
        if m[best, col].abs() < 1e-8:
            continue
        m[[pivot_row, best]] = m[[best, pivot_row]]
        m[pivot_row] /= m[pivot_row, col].clone()
        for r in range(rows):
            if r != pivot_row:
                m[r] -= m[r, col] * m[pivot_row]
        pivot_row += 1
    m[m.abs() < 1e-9] = 0.0
    return m


def directions_of(trace: DegreeTrace, basis: Tensor, anchors: bool = False) -> list[Direction]:
    """Readable directions spanning `basis`, each with whether it changes a hidden state and
    (with `anchors`) the anchors it moves against."""
    if basis.numel() == 0 or basis.shape[1] == 0:
        return []
    degrees = trace.degrees()
    anchor_rows = trace.anchor_matrix()
    out = []
    for row in echelon(basis):
        coeffs = {n: round(c, 6) for n, c in zip(trace.names, row.tolist(), strict=True) if c}
        changes = bool((degrees @ row).abs().max() > TOL)
        held = []
        if anchors and trace.anchors:
            for source, value in zip(
                trace.anchor_sources, (anchor_rows @ row).tolist(), strict=True
            ):
                if abs(value) > TOL and source not in held:
                    held.append(source)
        out.append(Direction(coeffs, changes, tuple(held)))
    return out


def model_notes(trace: DegreeTrace, basis: Tensor, weak: list[Direction]) -> list[str]:
    """What the model code can change and a weight patch cannot: attention calls whose q and k
    trade scale, and weakly pinned directions that change a hidden state."""
    notes = []
    if basis.numel():
        free_qk = [
            i for i, (dq, _) in enumerate(trace.attention) if (basis.T @ dq).abs().max() > TOL
        ]
        if free_qk:
            notes.append(
                f"attention calls {free_qk}: q and k trade scale freely; normalize q and k "
                "(QK-norm, ds.nn.QKNorm) with a gain on q only"
            )
    for d in weak:
        if d.changes_hidden_state:
            notes.append(
                f"{', '.join(d.coeffs)} are held only by a fixed-scale term "
                f"({'; '.join(d.anchors)}); once they outgrow it their scale is nearly free. "
                "Typically a normalization in front of the output hides the residual stream's "
                "scale from the loss: let the output layer read the stream directly"
            )
    return notes


def solve(trace: DegreeTrace) -> FreeScales:
    """The free scales of a degree trace, in their classes."""
    hard = trace.matrix()
    full = torch.cat([hard, trace.anchor_matrix()])
    basis = null_space(full)
    degrees = trace.degrees()
    unchanging = null_space(torch.cat([full, degrees])) if basis.numel() else basis
    changing = complement(unchanging, basis)
    weak_basis = complement(basis, null_space(hard))
    weak = directions_of(trace, weak_basis, anchors=True)
    return FreeScales(
        names=trace.names,
        shapes=trace.shapes,
        basis=basis,
        changing=directions_of(trace, changing),
        unchanging=directions_of(trace, unchanging),
        weak=weak,
        unknown=dict(trace.unknown),
        notes=model_notes(trace, basis, weak),
        constraints=len(trace.constraints) + len(trace.anchors),
    )


def free_scales(model: nn.Module, *inputs, **kwargs) -> FreeScales:
    """Trace `model(*inputs, **kwargs)` once and return its free scales. Ops without a degree
    rule are listed in the report and raise a warning."""
    report = solve(trace_degrees(model, *inputs, **kwargs))
    if report.unknown:
        warnings.warn(unknown_message(report.unknown), stacklevel=2)
    return report
