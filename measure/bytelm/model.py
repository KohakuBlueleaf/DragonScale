"""A pre-norm byte-level LM with separate q / k / v / out / gate / up / down linears, and the three
presets the experiments compare.

| preset | model | patch |
| --- | --- | --- |
| `stock` | RMSNorm with affine | none |
| `matrices` | `stock` | `suggest(level="matrices")`: unit rows and a channel gain on every layer |
| `arch` | norms without affine, QK-norm with a gain on q only | `suggest(level="absorb")` and |
| | | `absorb_plan`; q, k (followed by QK-norm) and both embeddings unit rows without gain |
"""

import dataclasses
import math
import re

import torch
import torch.nn.functional as F
from torch import nn

import dragonscale as ds

VOCAB = 256
PRESETS = ("stock", "matrices", "arch")
FIXED_SCALE = re.compile(r"(embed|pos|blocks\.\d+\.(q|k))")


@dataclasses.dataclass(frozen=True)
class Shape:
    depth: int = 8
    dim: int = 256
    heads: int = 4
    context: int = 256


class Block(nn.Module):
    """Pre-norm attention and SwiGLU MLP; `stats` collects magnitudes when not None."""

    def __init__(self, shape: Shape, affine: bool, qk_norm: bool):
        super().__init__()
        dim, hidden, head = shape.dim, 3 * shape.dim, shape.dim // shape.heads
        self.heads = shape.heads
        self.norm1 = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=affine)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(head, eps=1e-6) if qk_norm else None
        self.k_norm = nn.RMSNorm(head, eps=1e-6, elementwise_affine=False) if qk_norm else None
        self.out = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=affine)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.stats: dict | None = None

    def forward(self, x):
        batch, length, dim = x.shape
        h = self.norm1(x)

        def split(t):
            return t.view(batch, length, self.heads, -1).transpose(1, 2)

        q, k, v = split(self.q(h)), split(self.k(h)), split(self.v(h))
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        mixed = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out(mixed.transpose(1, 2).reshape(batch, length, dim))
        h = self.norm2(x)
        gate, up = self.gate(h), self.up(h)
        x = x + self.down(F.silu(gate) * up)
        if self.stats is not None:
            self.stats.update(magnitudes(x, q, k, gate, up))
        return x


@torch.no_grad()
def magnitudes(x, q, k, gate, up) -> dict:
    """Stream RMS / max / largest channel RMS, MLP hidden max, attention-logit max."""
    x = x.float()
    length = q.shape[-2]
    logits = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    mask = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
    return {
        "stream_rms": x.square().mean().sqrt(),
        "stream_max": x.abs().max(),
        "channel_max": x.square().mean(dim=(0, 1)).sqrt().max(),
        "hidden_max": torch.maximum(gate.float().abs().max(), up.float().abs().max()),
        "logit_max": logits.masked_select(mask).abs().max(),
    }


class LM(nn.Module):
    def __init__(self, shape: Shape, affine: bool, qk_norm: bool):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, shape.dim)
        self.pos = nn.Embedding(shape.context, shape.dim)
        self.blocks = nn.ModuleList(Block(shape, affine, qk_norm) for _ in range(shape.depth))
        self.norm = nn.RMSNorm(shape.dim, eps=1e-6, elementwise_affine=affine)
        self.head = nn.Linear(shape.dim, VOCAB, bias=False)
        with torch.no_grad():
            self.pos.weight.mul_(0.1)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    module.weight.normal_(0, module.in_features**-0.5)
            for block in self.blocks:
                block.out.weight.mul_((2 * shape.depth) ** -0.5)
                block.down.weight.mul_((2 * shape.depth) ** -0.5)

    def forward(self, tokens):
        x = self.embed(tokens) + self.pos.weight[: tokens.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def build(preset: str, shape: Shape, sample: torch.Tensor, device: torch.device):
    """The model of `preset` on `device`, and its session (None for `stock`)."""
    arch = preset == "arch"
    model = LM(shape, affine=not arch, qk_norm=arch).to(device)
    match preset:
        case "stock":
            return model, None
        case "matrices":
            return model, ds.apply(model, ds.analysis.suggest(model, sample, level="matrices"))
        case "arch":
            plan = ds.analysis.suggest(model, sample, level="absorb")
            plan = ds.analysis.absorb_plan(model, plan, sample)
            layers = tuple(
                (
                    dataclasses.replace(p, mode="unit", init="ones")
                    if FIXED_SCALE.fullmatch(p.path)
                    else p
                )
                for p in plan
            )
            return model, ds.apply(model, dataclasses.replace(plan, layers=layers))
    raise ValueError(f"preset {preset!r} not in {PRESETS}")
