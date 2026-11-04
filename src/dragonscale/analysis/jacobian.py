"""`scale_jacobian`: `J[m, k] = <theta_k, grad_theta_k (rho_m . f(x))>` over random output
projections; `J s` is how fast the output moves along the log-scale direction `s`."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils._pytree import tree_leaves

from dragonscale.analysis.trace import parameter_groups


@dataclass(frozen=True)
class ScaleJacobian:
    """`matrix` is `[projections, groups]`; its singular values and right singular vectors."""

    names: list[str]
    matrix: Tensor
    singular: Tensor
    directions: Tensor

    def pinning(self, s: Tensor) -> float:
        """`|J s| / (|J| |s|)`: 0 for a free direction, small for a weakly pinned one."""
        scale = self.matrix.norm().item() * s.norm().item()
        return (self.matrix @ s).norm().item() / max(scale, 1e-300)


def scale_jacobian(
    model: nn.Module, *inputs, projections: int = 32, seed: int = 0, **kwargs
) -> ScaleJacobian:
    """The scale Jacobian of `model(*inputs, **kwargs)` over `projections` random projections;
    the column of a parameter without `requires_grad` is 0."""
    names, params = parameter_groups(model)
    trained = [p for p in params if p.requires_grad]
    with torch.enable_grad():
        out = model(*inputs, **kwargs)
        flat = torch.cat(
            [
                t.reshape(-1)
                for t in tree_leaves(out)
                if isinstance(t, Tensor) and t.is_floating_point()
            ]
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        rows = []
        for _ in range(projections):
            rho = torch.randn(flat.shape, generator=generator, dtype=flat.dtype).to(flat.device)
            grads = torch.autograd.grad(
                (flat * rho).sum(), trained, retain_graph=True, allow_unused=True
            )
            by_param = {id(p): g for p, g in zip(trained, grads, strict=True)}
            row = [
                0.0 if by_param.get(id(p)) is None else (by_param[id(p)] * p).sum().item()
                for p in params
            ]
            rows.append(row)
    matrix = torch.tensor(rows, dtype=torch.float64)
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    return ScaleJacobian(names, matrix, singular, vh)
