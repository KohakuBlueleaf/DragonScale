"""`fold_layerscale`: move LayerScale factors into the gains of the layers they follow."""

from collections.abc import Mapping

import torch
from torch import nn

from dragonscale.patch.session import Session
from dragonscale.problems import raise_problems


@torch.no_grad()
def fold_layerscale(session: Session, layerscales: Mapping[str, str]) -> None:
    """For each `{writer: layerscale}` pair of module paths, multiply the writer's gain (and bias)
    by the LayerScale factor (its one parameter, per channel or scalar) and replace the LayerScale module
    with `nn.Identity`. The model's output does not change; the gain takes the factor's place."""
    model = session.model
    patched = {p.plan.path: p for p in session.patched}
    problems, folds = [], []
    for writer, path in layerscales.items():
        layer = patched.get(writer)
        if layer is None or layer.gain is None:
            problems.append(f"{writer}: not a patched layer with a gain")
            continue
        module = model.get_submodule(path)
        params = list(module.parameters(recurse=False))
        if len(params) != 1:
            problems.append(f"{path}: expected one LayerScale parameter, found {len(params)}")
            continue
        factor = params[0].detach().reshape(-1)
        if factor.numel() not in (1, layer.gain.numel()):
            problems.append(
                f"{path}: {factor.numel()} factors for the {layer.gain.numel()} gains of {writer}"
            )
            continue
        folds.append((layer, path, factor))
    raise_problems(problems)
    for layer, path, factor in folds:
        layer.gain.mul_(factor.to(layer.gain.dtype))
        bias = layer.bias()
        if bias is not None:
            bias.mul_(factor.to(bias.dtype))
        parent, _, name = path.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, name, nn.Identity())
