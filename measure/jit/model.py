"""JiT, after the official `model_jit.py` (github.com/LTH14/JiT): bottleneck patch embedding,
fixed 2D sin-cos positions, 2D RoPE, QK-norm, SwiGLU, adaLN-Zero, 32 in-context class tokens from
block `in_context_start`, x-prediction.

Written as separate linears where the official code fuses them (q / k / v, the SwiGLU gate and
up, each adaLN chunk); the function is the same. With `adaln` False (the experiments' shape) the
time + class embedding is one token in front of the image tokens from block 0 instead of adaLN-Zero,
so the model has no gates, no modulation and no zero-initialized layer.

| preset | model | patch |
| --- | --- | --- |
| `stock` | RMSNorm with affine, QK-norm with a gain on q and on k | none |
| `matrices` | `stock` | `suggest(level="matrices")`: unit rows and a channel gain on every layer |
| `arch` | norms without affine, QK-norm with a gain on q only | `suggest(level="absorb")` and |
| | | `absorb_plan`; without the final norm, every input enters through a norm without affine, |
| | | the layers right before those norms and the output layer have unit rows and no gain, and |
| | | the conditioning and in-context tokens are read-only (attended to, never written) |
"""

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn

import dragonscale as ds

PRESETS = ("stock", "matrices", "arch")
CHUNKS = ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")


@dataclasses.dataclass(frozen=True)
class Shape:
    """JiT-B/16 by default."""

    image: int = 256
    patch: int = 16
    depth: int = 12
    dim: int = 768
    heads: int = 12
    bottleneck: int = 128
    in_context: int = 32
    in_context_start: int = 4
    classes: int = 102

    @property
    def grid(self) -> int:
        return self.image // self.patch


def sincos_2d(dim: int, grid: int) -> torch.Tensor:
    """[grid * grid, dim] fixed positions: half the channels from the row, half from the column."""

    def axis(d: int, pos: torch.Tensor) -> torch.Tensor:
        omega = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float64) / (d / 2))
        out = pos.flatten().double()[:, None] * omega[None]
        return torch.cat([out.sin(), out.cos()], dim=1)

    rows, cols = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    return torch.cat([axis(dim // 2, rows), axis(dim // 2, cols)], dim=1).float()


def rope_2d(head_dim: int, grid: int, prefix: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos / sin [prefix + grid * grid, head_dim] for interleaved pairs: the first half of the
    channels rotate with the row, the second with the column; `prefix` tokens do not rotate."""
    half = head_dim // 2
    freqs = 1.0 / 10000 ** (torch.arange(0, half, 2, dtype=torch.float32) / half)
    pos = torch.arange(grid, dtype=torch.float32)
    rows, cols = torch.meshgrid(pos, pos, indexing="ij")
    angles = torch.cat(
        [rows.flatten()[:, None] * freqs, cols.flatten()[:, None] * freqs], dim=1
    ).repeat_interleave(2, dim=1)
    angles = torch.cat([torch.zeros(prefix, head_dim), angles], dim=0)
    return angles.cos(), angles.sin()


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    pairs = x.unflatten(-1, (-1, 2))
    rotated = torch.stack([-pairs[..., 1], pairs[..., 0]], dim=-1).flatten(-2)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, shape: Shape, qk_affine: tuple[bool, bool]):
        super().__init__()
        dim, head = shape.dim, shape.dim // shape.heads
        self.heads = shape.heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.q_norm = nn.RMSNorm(head, eps=1e-6, elementwise_affine=qk_affine[0])
        self.k_norm = nn.RMSNorm(head, eps=1e-6, elementwise_affine=qk_affine[1])
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, cos, sin, stats: dict | None):
        batch, length, dim = x.shape

        def split(t):
            return t.view(batch, length, self.heads, -1).transpose(1, 2)

        q = rotate(self.q_norm(split(self.q(x))), cos, sin)
        k = rotate(self.k_norm(split(self.k(x))), cos, sin)
        v = split(self.v(x))
        if stats is not None:
            with torch.no_grad():
                logits = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
                stats["logit_max"] = logits.abs().max()
        mixed = F.scaled_dot_product_attention(q, k, v)
        return self.proj(mixed.transpose(1, 2).reshape(batch, length, dim))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden)
        self.up = nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Modulation(nn.Module):
    """adaLN: SiLU, then one linear per chunk (`CHUNKS`; `shift` and `scale` in the final
    layer)."""

    def __init__(self, dim: int, names: tuple[str, ...]):
        super().__init__()
        self.names = names
        for name in names:
            setattr(self, name, nn.Linear(dim, dim))

    def forward(self, c) -> dict:
        c = F.silu(c)
        return {name: getattr(self, name)(c).unsqueeze(1) for name in self.names}


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Block(nn.Module):
    """A pre-norm attention and SwiGLU block; with `adaln`, adaLN-Zero modulation and gates."""

    def __init__(
        self,
        shape: Shape,
        affine: bool,
        qk_affine: tuple[bool, bool],
        adaln: bool,
        read_only_prefix: bool = False,
    ):
        super().__init__()
        self.read_only_prefix = read_only_prefix
        self.norm1 = nn.RMSNorm(shape.dim, eps=1e-6, elementwise_affine=affine)
        self.attn = Attention(shape, qk_affine)
        self.norm2 = nn.RMSNorm(shape.dim, eps=1e-6, elementwise_affine=affine)
        self.mlp = SwiGLU(shape.dim, int(shape.dim * 4 * 2 / 3))
        self.modulation = Modulation(shape.dim, CHUNKS) if adaln else None
        self.stats: dict | None = None
        self.prefix = 0  # tokens in front of the image tokens, set by the model before each call

    def forward(self, x, c, cos, sin):
        stats = self.stats
        if self.modulation is None:
            m = {}
            attn = self.attn(self.norm1(x), cos, sin, stats)
            if self.read_only_prefix and self.prefix:
                # Prefix tokens are read (keys and values) but receive no writes
                attn = torch.cat(
                    [torch.zeros_like(attn[:, : self.prefix]), attn[:, self.prefix :]], 1
                )
            x = x + attn
            mlp = self.mlp(self.norm2(x))
            if self.read_only_prefix and self.prefix:
                mlp = torch.cat([torch.zeros_like(mlp[:, : self.prefix]), mlp[:, self.prefix :]], 1)
            x = x + mlp
        else:
            m = self.modulation(c)
            h = modulate(self.norm1(x), m["shift_msa"], m["scale_msa"])
            attn = self.attn(h, cos, sin, stats)
            x = x + m["gate_msa"] * attn
            mlp = self.mlp(modulate(self.norm2(x), m["shift_mlp"], m["scale_mlp"]))
            x = x + m["gate_mlp"] * mlp
        if stats is not None:
            with torch.no_grad():
                f, n = x.float(), self.prefix
                stats["stream_rms"] = f.square().mean().sqrt()
                stats["stream_max"] = f.abs().max()
                stats["channel_max"] = f.square().mean(dim=(0, 1)).sqrt().max()
                stats["attn_out_rms"] = attn.float().square().mean().sqrt()
                stats["mlp_out_rms"] = mlp.float().square().mean().sqrt()
                image = f[:, n:]
                stats["image_rms"] = image.square().mean().sqrt()
                stats["image_max"] = image.abs().max()
                stats["image_mlp_rms"] = mlp[:, n:].float().square().mean().sqrt()
                stats["image_attn_rms"] = attn[:, n:].float().square().mean().sqrt()
                if n:
                    stats["prefix_rms"] = f[:, :n].square().mean().sqrt()
                    stats["prefix_max"] = f[:, :n].abs().max()
                    stats["cond_rms"] = f[:, 0].square().mean().sqrt()
                for name, value in m.items():
                    stats[f"{name}_max"] = value.float().abs().max()
        return x


class JiT(nn.Module):
    """JiT; `adaln` False conditions by tokens: the time + class embedding is one token in front of
    the image tokens from block 0, and nothing is zero-initialized. `final_norm` False lets the
    output layer read the residual stream directly. `input_norm` puts every entry into the stream
    (the image tokens, the conditioning token, the in-context tokens) through a norm without affine,
    so each enters at a fixed scale; the layers right before those norms then carry no bias, and
    the prefix tokens are read-only: blocks attend to them but write only into the image tokens."""

    def __init__(
        self,
        shape: Shape,
        affine: bool = True,
        qk_affine=(True, True),
        adaln: bool = True,
        final_norm: bool = True,
        input_norm: bool = False,
    ):
        super().__init__()
        self.shape = shape
        self.adaln = adaln
        self.prefix = 0 if adaln else 1
        dim = shape.dim
        entry_bias = not input_norm
        self.t_embed = nn.Sequential(
            nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim, bias=entry_bias)
        )
        self.y_embed = nn.Embedding(shape.classes + 1, dim)
        self.patch1 = nn.Conv2d(3, shape.bottleneck, shape.patch, shape.patch, bias=False)
        self.patch2 = nn.Conv2d(shape.bottleneck, dim, 1, bias=entry_bias)
        self.register_buffer("pos", sincos_2d(dim, shape.grid), persistent=False)
        self.in_context_pos = nn.Embedding(shape.in_context, dim)
        head = dim // shape.heads
        for name, prefix in (("rope", self.prefix), ("rope_ctx", self.prefix + shape.in_context)):
            cos, sin = rope_2d(head, shape.grid, prefix)
            self.register_buffer(f"{name}_cos", cos, persistent=False)
            self.register_buffer(f"{name}_sin", sin, persistent=False)
        self.blocks = nn.ModuleList(
            Block(shape, affine, qk_affine, adaln, read_only_prefix=input_norm)
            for _ in range(shape.depth)
        )
        self.norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=affine) if final_norm else None
        self.input_norm = (
            nn.RMSNorm(dim, eps=1e-6, elementwise_affine=False) if input_norm else None
        )
        self.final_modulation = Modulation(dim, ("shift", "scale")) if adaln else None
        self.out = nn.Linear(dim, shape.patch * shape.patch * 3)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for conv in (self.patch1, self.patch2):
            nn.init.xavier_uniform_(conv.weight.view(conv.weight.shape[0], -1))
        if self.patch2.bias is not None:
            nn.init.zeros_(self.patch2.bias)
        nn.init.normal_(self.y_embed.weight, std=0.02)
        nn.init.normal_(self.in_context_pos.weight, std=0.02)
        nn.init.normal_(self.t_embed[0].weight, std=0.02)
        nn.init.normal_(self.t_embed[2].weight, std=0.02)
        if not self.adaln:
            return
        for modulation in [b.modulation for b in self.blocks] + [self.final_modulation]:
            for name in modulation.names:
                nn.init.zeros_(getattr(modulation, name).weight)
                nn.init.zeros_(getattr(modulation, name).bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def time_features(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.cos(), args.sin()], dim=-1)

    def forward(self, z, t, y):
        s = self.shape
        y_emb = self.y_embed(y)
        c = self.t_embed(self.time_features(t).to(z.dtype)) + y_emb
        x = self.patch2(self.patch1(z)).flatten(2).transpose(1, 2) + self.pos.to(z.dtype)
        enter = self.input_norm if self.input_norm is not None else (lambda v: v)
        x = enter(x)
        if not self.adaln:
            c = enter(c)
            x = torch.cat([c.unsqueeze(1), x], dim=1)
        cos, sin = self.rope_cos.to(z.dtype), self.rope_sin.to(z.dtype)
        prefix = self.prefix
        for i, block in enumerate(self.blocks):
            if i == s.in_context_start:
                positions = torch.arange(s.in_context, device=z.device)
                ctx = enter(y_emb.unsqueeze(1) + self.in_context_pos(positions)).to(x.dtype)
                x = torch.cat([x[:, : self.prefix], ctx, x[:, self.prefix :]], dim=1)
                cos, sin = self.rope_ctx_cos.to(z.dtype), self.rope_ctx_sin.to(z.dtype)
                prefix = self.prefix + s.in_context
            block.prefix = prefix
            x = block(x, c, cos, sin)
        x = x[:, self.prefix + s.in_context :]
        if self.norm is not None:
            x = self.norm(x)
        if self.final_modulation is not None:
            m = self.final_modulation(c)
            x = modulate(x, m["shift"], m["scale"])
        x = self.out(x)
        x = x.view(z.shape[0], s.grid, s.grid, s.patch, s.patch, 3)
        return torch.einsum("nhwpqc->nchpwq", x).reshape(z.shape)


def build(
    preset: str,
    shape: Shape,
    sample: tuple,
    device: torch.device,
    adaln: bool = False,
    final_norm: bool = True,
):
    """The model of `preset` on `device`, and its session (None for `stock`); `sample` is one
    (z, t, y) model input for the analysis."""
    arch = preset == "arch"
    model = JiT(
        shape,
        affine=not arch,
        qk_affine=(True, not arch),
        adaln=adaln,
        final_norm=final_norm,
        input_norm=arch and not final_norm,
    ).to(device)
    match preset:
        case "stock":
            return model, None
        case "matrices":
            return model, ds.apply(model, ds.analysis.suggest(model, *sample, level="matrices"))
        case "arch":
            plan = ds.analysis.absorb_plan(
                model, ds.analysis.suggest(model, *sample, level="absorb"), *sample
            )
            if not final_norm:
                # The output layer and the layers right before the input norms keep no gain
                fixed = ("out", "patch2", "t_embed.2", "y_embed", "in_context_pos")
                layers = tuple(
                    dataclasses.replace(p, mode="unit", init="ones") if p.path in fixed else p
                    for p in plan
                )
                plan = dataclasses.replace(plan, layers=layers)
            return model, ds.apply(model, plan)
    raise ValueError(f"preset {preset!r} not in {PRESETS}")
