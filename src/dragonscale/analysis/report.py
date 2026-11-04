"""`free_scales`: the null space of a degree trace's constraints, as named directions, each
labelled parameter-only, activation-moving, or dangerous (activation-moving with a matrix in it).

A weakly pinned direction is free under every constraint except anchors: only a fixed-scale addend
holds it, and once the scaled term outgrows that addend the direction is nearly free."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from dragonscale.analysis.trace import DegreeTrace, trace_degrees

TOL = 1e-9


@dataclass(frozen=True)
class Direction:
    """One free direction: the log-scale coefficient of every group in it, and its matrices
    (`fast_ends`)."""

    coeffs: dict[str, float]
    moves_activations: bool
    fast_ends: tuple[str, ...]
    anchors: tuple[str, ...] = ()

    @property
    def dangerous(self) -> bool:
        return self.moves_activations and bool(self.fast_ends)

    def describe(self) -> str:
        terms = " ".join(f"{c:+g}*{name}" for name, c in self.coeffs.items())
        label = "dangerous" if self.dangerous else (
            "activation-moving" if self.moves_activations else "parameter-only"
        )  # fmt: skip
        held = f" (held by: {'; '.join(self.anchors)})" if self.anchors else ""
        return f"{label}: {terms}{held}"


@dataclass(frozen=True)
class FreeScales:
    """Every free direction of one model on one input, in reduced row-echelon form, the weakly
    pinned directions with the anchors that hold them (`weak`), and the architecture notes a weight
    patch cannot address (`notes`)."""

    names: list[str]
    shapes: list[tuple[int, ...]]
    directions: list[Direction]
    basis: Tensor
    constraints: int
    unknown: dict[str, int]
    notes: list[str]
    weak: list[Direction]

    @property
    def dangerous(self) -> list[Direction]:
        return [d for d in self.directions if d.dangerous]

    def vector(self, coeffs: dict[str, float]) -> Tensor:
        """A direction over every group from `{name: coefficient}`."""
        out = torch.zeros(len(self.names), dtype=torch.float64)
        for name, c in coeffs.items():
            out[self.names.index(name)] = c
        return out

    def contains(self, coeffs: dict[str, float]) -> bool:
        """Whether the direction `{name: coefficient}` is free."""
        s = self.vector(coeffs)
        if self.basis.numel() == 0:
            return bool(s.abs().max() < TOL)
        residual = s - self.basis @ torch.linalg.lstsq(self.basis, s[:, None]).solution[:, 0]
        return bool(residual.norm() <= 1e-7 * max(s.norm().item(), 1.0))

    def describe(self) -> str:
        lines = [
            f"{len(self.directions)} free direction(s), {len(self.dangerous)} dangerous, "
            f"{self.constraints} constraints over {len(self.names)} groups"
        ]
        alone = [d for d in self.directions if len(d.coeffs) == 1 and not d.moves_activations]
        lines += [f"  {d.describe()}" for d in self.directions if d not in alone]
        if alone:
            names = ", ".join(next(iter(d.coeffs)) for d in alone)
            lines.append(f"  parameter-only, one group each ({len(alone)}): {names}")
        if self.weak:
            lines.append(f"{len(self.weak)} weakly pinned direction(s), held only by anchors:")
            lines += [f"  {d.describe()}" for d in self.weak]
        if self.unknown:
            lines.append(f"  ops without a rule (treated as nonlinear): {dict(self.unknown)}")
        if self.notes:
            lines.append("architecture:")
            lines += [f"  {note}" for note in self.notes]
        return "\n".join(lines)


def null_space(a: Tensor) -> Tensor:
    """An orthonormal basis of `{s : a s = 0}`, `[groups, free]`."""
    gram = a.T @ a
    values, vectors = torch.linalg.eigh(gram)
    scale = max(values.abs().max().item(), 1.0)
    return vectors[:, values <= 1e-10 * scale]


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


def directions_of(result: DegreeTrace, basis: Tensor) -> list[Direction]:
    """Readable directions spanning `basis`, labelled, each with the anchors it moves against."""
    degrees = result.degrees()
    anchors = result.anchor_matrix()
    out = []
    for row in echelon(basis):
        coeffs = {n: round(c, 6) for n, c in zip(result.names, row.tolist(), strict=True) if c}
        moves = bool((degrees @ row).abs().max() > TOL)
        fast_ends = tuple(n for n in coeffs if len(result.shapes[result.names.index(n)]) >= 2)
        held = []
        if result.anchors:
            for source, value in zip(result.anchor_sources, (anchors @ row).tolist(), strict=True):
                if abs(value) > TOL and source not in held:
                    held.append(source)
        out.append(Direction(coeffs, moves, fast_ends, tuple(held)))
    return out


def complement(inner: Tensor, outer: Tensor) -> Tensor:
    """An orthonormal basis of the part of span(`outer`) orthogonal to span(`inner`)."""
    if outer.numel() == 0:
        return outer
    rest = outer - inner @ (inner.T @ outer) if inner.numel() else outer
    u, sigma, _ = torch.linalg.svd(rest, full_matrices=False)
    return u[:, sigma > 1e-8 * max(sigma.max().item(), 1.0)] if sigma.numel() else rest


def architecture_notes(
    result: DegreeTrace, basis: Tensor, directions: list[Direction], weak: list[Direction]
) -> list[str]:
    """Attention calls whose q and k trade scale freely, parameters of two or more modules that
    scale the residual stream together (an activation-moving direction with every coefficient
    positive), and dangerous weakly pinned directions."""
    notes = []
    if basis.numel():
        free_qk = [
            i for i, (dq, _) in enumerate(result.attention) if (basis.T @ dq).abs().max() > TOL
        ]
        if free_qk:
            notes.append(
                f"attention calls {free_qk}: q and k trade scale freely; "
                "normalize q and k (QK-norm) with a gain on one side only"
            )
    for d in directions:
        modules = {name.rpartition(".")[0] for name in d.coeffs}
        if d.moves_activations and len(modules) >= 2 and all(c > 0 for c in d.coeffs.values()):
            notes.append(
                f"{', '.join(d.coeffs)} scale the residual stream together; "
                "give the model input a fixed scale (an embedding: mode 'unit')"
            )
    for d in weak:
        if d.dangerous:
            notes.append(
                f"{', '.join(d.coeffs)} are held only by a fixed-scale addend "
                f"({'; '.join(d.anchors)}); once they outgrow it this scale is nearly free. Let "
                "the output read this stream directly, without a norm in between, so the loss "
                "sets its scale"
            )
    return notes


def solve(result: DegreeTrace) -> FreeScales:
    """The free directions of a degree trace, and its weakly pinned ones."""
    hard = result.matrix()
    basis = null_space(torch.cat([hard, result.anchor_matrix()]))
    weak_basis = complement(basis, null_space(hard))
    directions = [
        Direction(d.coeffs, d.moves_activations, d.fast_ends) for d in directions_of(result, basis)
    ]
    weak = directions_of(result, weak_basis) if weak_basis.numel() else []
    return FreeScales(
        result.names,
        result.shapes,
        directions,
        basis,
        len(result.constraints) + len(result.anchors),
        dict(result.unknown),
        architecture_notes(result, basis, directions, weak),
        weak,
    )


def free_scales(model: nn.Module, *inputs, **kwargs) -> FreeScales:
    """Trace `model(*inputs, **kwargs)` once and return its free directions."""
    return solve(trace_degrees(model, *inputs, **kwargs))
