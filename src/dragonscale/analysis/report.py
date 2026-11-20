"""`free_scales`: the free scales of a model from one degree trace.

A free scale is a way to rescale whole parameter tensors whose effect on the output is zero, or
goes to zero as the scale grows. Written as a direction `s` over the log-scales of the parameter
tensors, there are two kinds, both counted as free:

- exact: the output is unchanged at every scale, `A s = 0` with `A` the constraints of every op
  of the trace and the anchors (`ker A`);
- held only by a fixed-scale term: the output changes only because a fixed-scale term (a
  fixed-scale input, a constant, a buffer) is added to the scaled one. The change is about the
  ratio fixed / scaled, so it vanishes as the scaled term grows: `A0 s = 0`, with `A0` the
  constraints without the anchors (`ker A0` beyond `ker A`).

The free scales form the space `ker A0`. They are split into two classes, each with its exact and
held part; counts are dimensions of subspaces, so they do not depend on how a basis is chosen.

| class | count |
| --- | --- |
| change a hidden state | `dim ker A0 - dim ker [A0; D]` |
| change no hidden state (a raw weight behind unit rows, a gain before a norm) | `dim ker [A0; D]` |
| of either class, exact | `dim ker A` split the same way with `A` |
| ops without a rule (treated as fixing the scale of their inputs) | count per op |

`D` stacks the degree of every data-dependent tensor of the trace (every hidden state).
"""

import warnings
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from dragonscale.analysis.trace import DegreeTrace, trace_degrees

TOL = 1e-9
HELD_MARK = "*"


@dataclass(frozen=True)
class Direction:
    """One readable free direction: the log-scale coefficient of every parameter in it, whether
    it changes a hidden state, and the fixed-scale terms that hold it (empty when exact)."""

    coeffs: dict[str, float]
    changes_hidden_state: bool
    held_by: tuple[str, ...] = ()

    @property
    def exact(self) -> bool:
        """Whether the output is unchanged at every scale along this direction."""
        return not self.held_by

    def describe(self) -> str:
        terms = " ".join(f"{c:+g} {name}" for name, c in self.coeffs.items())
        held = f"  {HELD_MARK} held by: {'; '.join(self.held_by)}" if self.held_by else ""
        return f"{terms}{held}"


@dataclass(frozen=True)
class FreeScales:
    """The free scales of one model on one input.

    `changing` and `unchanging` are readable bases (reduced row-echelon form) of the free scales
    that change a hidden state and of those that change none, exact directions first, then those
    held only by a fixed-scale term. `exact` and `held` list each kind. `basis` is an orthonormal
    basis of every free scale (`ker A0`), `exact_basis` of the exact ones (`ker A`). `notes` name
    what a weight patch cannot change and the model code can."""

    names: list[str]
    shapes: list[tuple[int, ...]]
    basis: Tensor
    exact_basis: Tensor
    changing: list[Direction]
    unchanging: list[Direction]
    unknown: dict[str, int]
    notes: list[str]
    constraints: int

    @property
    def count(self) -> int:
        """Every free scale: `dim ker A0`."""
        return _width(self.basis)

    @property
    def exact(self) -> list[Direction]:
        """The free scales that leave the output unchanged at every scale."""
        return [d for d in self.changing + self.unchanging if d.exact]

    @property
    def held(self) -> list[Direction]:
        """The free scales held only by a fixed-scale term."""
        return [d for d in self.changing + self.unchanging if not d.exact]

    def vector(self, coeffs: dict[str, float]) -> Tensor:
        """A direction over every parameter from `{name: coefficient}`."""
        out = torch.zeros(len(self.names), dtype=torch.float64)
        for name, c in coeffs.items():
            out[self.names.index(name)] = c
        return out

    def is_free(self, coeffs: dict[str, float], exact: bool = False) -> bool:
        """Whether the direction `{name: coefficient}` is a free scale (with `exact`, one that
        leaves the output unchanged at every scale)."""
        return in_span(self.exact_basis if exact else self.basis, self.vector(coeffs))

    def summary(self) -> dict[str, int]:
        """The counts."""
        return {
            "change_hidden_state": len(self.changing),
            "change_none": len(self.unchanging),
            "held_by_fixed_term": len(self.held),
            "ops_without_rule": sum(self.unknown.values()),
        }

    def describe(self) -> str:
        lines = []
        if self.unknown:
            lines.append(unknown_message(self.unknown))
        lines.append(
            f"{self.count} free scale(s): {len(self.changing)} change a hidden state, "
            f"{len(self.unchanging)} change none; {len(self.held)} of them held only by a "
            f"fixed-scale term; {sum(self.unknown.values())} op call(s) without a rule"
        )
        if self.changing:
            lines.append("free scales that change a hidden state:")
            lines += [f"  {d.describe()}" for d in self.changing]
        if self.unchanging:
            alone = [d for d in self.unchanging if len(d.coeffs) == 1 and d.exact]
            lines.append("free scales that change no hidden state:")
            lines += [f"  {d.describe()}" for d in self.unchanging if d not in alone]
            if alone:
                names = ", ".join(next(iter(d.coeffs)) for d in alone)
                lines.append(f"  one parameter each ({len(alone)}): {names}")
        if self.held:
            lines.append(
                f"{HELD_MARK} held only by a fixed-scale term: the output changes only by about "
                "fixed / scaled, so the hold weakens as the scaled term grows."
            )
        if self.notes:
            lines.append("for the model code:")
            lines += [f"  - {note}" for note in self.notes]
        return "\n".join(lines)


def _width(basis: Tensor) -> int:
    return basis.shape[1] if basis.numel() else 0


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


def directions_of(trace: DegreeTrace, basis: Tensor, held: bool = False) -> list[Direction]:
    """Readable directions spanning `basis`, each with whether it changes a hidden state and
    (with `held`) the fixed-scale terms that hold it."""
    if basis.numel() == 0 or basis.shape[1] == 0:
        return []
    degrees = trace.degrees()
    anchor_rows = trace.anchor_matrix()
    out = []
    for row in echelon(basis):
        values = (anchor_rows @ row).tolist()
        if held and sum(values) < 0:  # the sign along which the scaled terms grow
            row, values = -row, [-v for v in values]
        coeffs = {n: round(c, 6) for n, c in zip(trace.names, row.tolist(), strict=True) if c}
        changes = bool((degrees @ row).abs().max() > TOL)
        sources = []
        if held and trace.anchors:
            for source, value in zip(trace.anchor_sources, values, strict=True):
                if abs(value) > TOL and source not in sources:
                    sources.append(source)
        out.append(Direction(coeffs, changes, tuple(sources)))
    return out


def model_notes(trace: DegreeTrace, basis: Tensor, held: list[Direction]) -> list[str]:
    """What the model code can change and a weight patch cannot: attention calls whose q and k
    trade scale, and free scales held only by a fixed-scale term that change a hidden state."""
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
    for d in held:
        if not d.changes_hidden_state:
            continue
        head = (
            f"{' '.join(f'{c:+g} {n}' for n, c in d.coeffs.items())}: a free scale held only by "
            f"a fixed-scale term ({'; '.join(d.held_by)}); the hold weakens as the scaled term "
            "grows. "
        )
        readers = [n for n, c in d.coeffs.items() if c < 0]
        if all(n.endswith("bias") for n in d.coeffs):
            notes.append(
                head + "These biases are added to a fixed-scale term and nothing else holds "
                "them; weight decay on them, or no bias, keeps them in place"
            )
        elif readers:
            notes.append(
                head + f"{', '.join(readers)} shrink as the others grow: give one of these "
                "tensors a fixed scale (drop its gain)"
            )
        else:
            notes.append(
                head + "Typically a norm in front of the output: with gains on the layers that "
                "write to the stream this leaves the stream's scale free (held only by the "
                "fixed-scale input); to remove it, drop those gains or remove the norm"
            )
    return notes


def _span(a: Tensor, b: Tensor) -> Tensor:
    """An orthonormal basis of span(`a`) + span(`b`)."""
    if not a.numel():
        return b
    return torch.cat([a, complement(a, b)], dim=1) if b.numel() else a


def solve(trace: DegreeTrace) -> FreeScales:
    """The free scales of a degree trace, in their classes."""
    hard = trace.matrix()
    full = torch.cat([hard, trace.anchor_matrix()])
    degrees = trace.degrees()
    exact = null_space(full)
    free = null_space(hard)
    # Exact directions: as before, split by whether they change a hidden state
    exact_none = null_space(torch.cat([full, degrees])) if exact.numel() else exact
    exact_changing = complement(exact_none, exact)
    # Held directions: the rest of ker A0, split the same way
    none = null_space(torch.cat([hard, degrees])) if free.numel() else free
    held_none = complement(exact_none, none)
    held_changing = complement(_span(exact, none), free)
    changing = directions_of(trace, exact_changing) + directions_of(trace, held_changing, held=True)
    unchanging = directions_of(trace, exact_none) + directions_of(trace, held_none, held=True)
    held = [d for d in changing + unchanging if not d.exact]
    return FreeScales(
        names=trace.names,
        shapes=trace.shapes,
        basis=free,
        exact_basis=exact,
        changing=changing,
        unchanging=unchanging,
        unknown=dict(trace.unknown),
        notes=model_notes(trace, exact, held),
        constraints=len(trace.constraints) + len(trace.anchors),
    )


def free_scales(model: nn.Module, *inputs, **kwargs) -> FreeScales:
    """Trace `model(*inputs, **kwargs)` once and return its free scales. Ops without a degree
    rule are listed in the report and raise a warning."""
    report = solve(trace_degrees(model, *inputs, **kwargs))
    if report.unknown:
        warnings.warn(unknown_message(report.unknown), stacklevel=2)
    return report
