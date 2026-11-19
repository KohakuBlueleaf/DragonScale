"""Optimizer parameter groups for a patched model.

    optimizer = torch.optim.AdamW(ds.optim.param_groups(session, weight_decay=0.1), lr=1e-3)

Weight decay goes on the raw weights of the patched layers and on every other parameter with two
or more dimensions; gains, biases, normalization affines and every other parameter with fewer
than two dimensions get none. Parameters that do not require a gradient are left out.
"""

from torch import nn

from dragonscale.patch.session import Session


def param_groups(
    session: Session, weight_decay: float, model: nn.Module | None = None, **overrides
) -> list[dict]:
    """Two groups over `model` (the session's model by default): `{"params": decayed,
    "weight_decay": weight_decay}` and `{"params": rest, "weight_decay": 0.0}`; `overrides`
    (for example `lr`) go into both. A group without parameters is left out."""
    model = session.model if model is None else model
    gains = {id(g) for g in session.gains()}
    raw = {id(p) for p, _ in session.raw_weights()}
    decay, rest, seen = [], [], set()
    for param in model.parameters():
        if id(param) in seen or not param.requires_grad:
            continue
        seen.add(id(param))
        if id(param) in raw or (param.dim() >= 2 and id(param) not in gains):
            decay.append(param)
        else:
            rest.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay, **overrides},
        {"params": rest, "weight_decay": 0.0, **overrides},
    ]
    return [g for g in groups if g["params"]]
