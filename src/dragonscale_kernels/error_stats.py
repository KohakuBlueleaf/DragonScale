"""Precision report: a candidate and the PyTorch baseline, both against an fp64 reference."""

import torch
from torch import Tensor
from torch.nn.functional import cosine_similarity

MARKED = ("abs_max", "abs_mean", "abs_median", "rel_max", "rel_mean", "rel_median", "cos_deficit")
CLEAR_ISSUE = 10.0
LARGE_DIFF = 2.0


def stats(out: Tensor, ref: Tensor) -> dict[str, float | bool]:
    """Finiteness, cosine to `ref`, and max / mean / median of the abs and rel error."""
    a = out.detach().double().flatten()
    b = ref.detach().double().flatten()
    diff = (a - b).abs()
    nonzero = b != 0
    rel = diff[nonzero] / b[nonzero].abs() if nonzero.any() else diff.new_zeros(1)
    cos = 1.0 if a.equal(b) else cosine_similarity(a, b, dim=0).item()

    row: dict[str, float | bool] = {
        "finite": bool(torch.isfinite(a).all()),
        "cos": cos,
        "cos_deficit": 1.0 - cos,
    }
    for name, values in (("abs", diff), ("rel", rel)):
        row[f"{name}_max"] = values.max().item()
        row[f"{name}_mean"] = values.mean().item()
        row[f"{name}_median"] = values.median().item()
    return row


def mark(candidate: dict, baseline: dict, ratios: dict[str, float]) -> str | None:
    """`clear issue`, `large diff` or None, from how far the candidate sits past the baseline."""
    if baseline["finite"] and not candidate["finite"]:
        return "clear issue"
    worst = max(ratios[key] for key in MARKED)
    if worst >= CLEAR_ISSUE:
        return "clear issue"
    return "large diff" if worst >= LARGE_DIFF else None


def compare(out: Tensor, baseline: Tensor, ref64: Tensor, floor: float = 1e-300) -> dict:
    """The report row: `candidate` and `baseline` against `ref64`, their ratios and the mark.

    `floor` bounds each baseline stat from below before dividing.
    """
    candidate = stats(out, ref64)
    plain = stats(baseline, ref64)
    ratios = {key: candidate[key] / max(plain[key], floor) for key in MARKED}
    return {
        "candidate": candidate,
        "baseline": plain,
        "ratio": ratios,
        "mark": mark(candidate, plain, ratios),
    }
