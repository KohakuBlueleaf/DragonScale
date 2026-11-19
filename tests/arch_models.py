"""Small models with the method's model-side rules, for the `selective` level's tests: a language
model, a diffusion transformer with conditioning tokens, a ViT student with a DINO-style head, and
a ConvNeXt student with two projections. Module names follow the models the method was developed
on, so that the expected plans read the same.

Rules on the model side: norms without affine, QK-norm with a gain on q only, every input entering
through a fixed-scale path, and (unless `final_norm`) the output reading the residual stream
directly.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class QKAttention(nn.Module):
    """Separate q, k, v, out; q and k RMS-normalized per head, a gain on q only."""

    def __init__(self, dim: int, heads: int, bias: bool = False, causal: bool = False):
        super().__init__()
        self.heads, self.causal = heads, causal
        self.q = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, dim, bias=bias)
        self.v = nn.Linear(dim, dim, bias=bias)
        self.out = nn.Linear(dim, dim, bias=bias)
        self.q_norm = nn.RMSNorm(dim // heads, eps=1e-6)
        self.k_norm = nn.RMSNorm(dim // heads, eps=1e-6, elementwise_affine=False)

    def forward(self, x):
        b, n, d = x.shape

        def split(t):
            return t.view(b, n, self.heads, -1).transpose(1, 2)

        q, k = self.q_norm(split(self.q(x))), self.k_norm(split(self.k(x)))
        mixed = F.scaled_dot_product_attention(q, k, split(self.v(x)), is_causal=self.causal)
        return self.out(mixed.transpose(1, 2).reshape(b, n, d))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, bias: bool = False):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=bias)
        self.up = nn.Linear(dim, hidden, bias=bias)
        self.down = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Pre-norm block; `prefix` leading tokens are read and never written."""

    def __init__(self, dim, heads, hidden, bias=False, causal=False, norm=nn.RMSNorm):
        super().__init__()
        self.norm1 = norm(dim, elementwise_affine=False)
        self.attn = QKAttention(dim, heads, bias, causal)
        self.norm2 = norm(dim, elementwise_affine=False)
        self.mlp = SwiGLU(dim, hidden, bias)
        self.prefix = 0

    def write(self, x, update):
        if self.prefix:
            update = torch.cat(
                [torch.zeros_like(update[:, : self.prefix]), update[:, self.prefix :]], 1
            )
        return x + update

    def forward(self, x):
        x = self.write(x, self.attn(self.norm1(x)))
        return self.write(x, self.mlp(self.norm2(x)))


class LM(nn.Module):
    def __init__(self, vocab=64, dim=32, depth=2, heads=2, context=16, final_norm=False):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(context, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, 3 * dim, causal=True) for _ in range(depth))
        self.norm = nn.RMSNorm(dim, elementwise_affine=False) if final_norm else nn.Identity()
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, tokens):
        x = self.embed(tokens) + self.pos(torch.arange(tokens.shape[1]))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


class DiT(nn.Module):
    """Patch embedding through a bottleneck, a fixed position table, a time + class token in
    front, `in_context` class tokens from block `start`; every entry through a norm without
    affine; prefix tokens read-only; x-prediction output reshaped to the image."""

    def __init__(self, image=32, patch=8, dim=32, depth=2, heads=2, bottleneck=8, in_context=4,
                 start=1, classes=5, final_norm=False):  # fmt: skip
        super().__init__()
        self.patch, self.grid, self.start = patch, image // patch, start
        self.t_embed = nn.Sequential(nn.Linear(16, dim), nn.SiLU(), nn.Linear(dim, dim, bias=False))
        self.y_embed = nn.Embedding(classes, dim)
        self.patch1 = nn.Conv2d(3, bottleneck, patch, patch, bias=False)
        self.patch2 = nn.Conv2d(bottleneck, dim, 1, bias=False)
        self.register_buffer("pos", torch.randn(self.grid**2, dim) * 0.5)
        self.in_context_pos = nn.Embedding(in_context, dim)
        self.input_norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.blocks = nn.ModuleList(Block(dim, heads, 2 * dim, bias=True) for _ in range(depth))
        self.norm = nn.RMSNorm(dim, elementwise_affine=False) if final_norm else nn.Identity()
        self.out = nn.Linear(dim, patch * patch * 3)

    def forward(self, z, t, y):
        freqs = torch.exp(-math.log(1e4) * torch.arange(8) / 8)
        features = torch.cat([torch.sin(t[:, None] * freqs), torch.cos(t[:, None] * freqs)], -1)
        y_emb = self.y_embed(y)
        c = self.t_embed(features) + y_emb
        x = self.patch2(self.patch1(z)).flatten(2).transpose(1, 2) + self.pos
        x = torch.cat([self.input_norm(c).unsqueeze(1), self.input_norm(x)], 1)
        prefix = 1
        for i, block in enumerate(self.blocks):
            if i == self.start:
                ctx = y_emb.unsqueeze(1) + self.in_context_pos(
                    torch.arange(self.in_context_pos.num_embeddings)
                )
                x = torch.cat([x[:, :1], self.input_norm(ctx), x[:, 1:]], 1)
                prefix = 1 + ctx.shape[1]
            block.prefix = prefix
            x = block(x)
        x = self.out(self.norm(x[:, prefix:]))
        x = x.view(z.shape[0], self.grid, self.grid, self.patch, self.patch, 3)
        return torch.einsum("nhwpqc->nchpwq", x).reshape(z.shape)


class GELUBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = QKAttention(dim, heads, bias=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Module()
        self.mlp.fc1, self.mlp.fc2 = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm2(x))))


class WeightNormLinear(nn.Module):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        self.weight_v = nn.Parameter(torch.randn(fan_out, fan_in) * 0.02)
        self.weight_g = nn.Parameter(torch.ones(fan_out, 1))

    def forward(self, x):
        v = self.weight_v
        return F.linear(x, self.weight_g * v / v.square().sum(1, keepdim=True).sqrt())


class DINOHead(nn.Module):
    def __init__(self, dim, out=64, hidden=48, bottleneck=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, bottleneck))  # fmt: skip
        self.last_layer = WeightNormLinear(bottleneck, out)

    def forward(self, x):
        return self.last_layer(F.normalize(self.mlp(x), dim=-1))


class ViTStudent(nn.Module):
    """ViT backbone (tokens through a LayerNorm without affine) and a DINO-style head on the
    class token and on every patch token."""

    def __init__(self, image=32, patch=16, dim=32, depth=2, heads=2, final_norm=False):
        super().__init__()
        grid = image // patch
        self.backbone = nn.Module()
        bb = self.backbone
        bb.patch_embed = nn.Conv2d(3, dim, patch, patch)
        bb.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 1e-2)
        bb.pos_embed = nn.Parameter(torch.randn(1, grid**2 + 1, dim) * 0.02)
        bb.input_norm = nn.LayerNorm(dim, elementwise_affine=False)
        bb.blocks = nn.ModuleList(GELUBlock(dim, heads) for _ in range(depth))
        bb.norm = nn.LayerNorm(dim, elementwise_affine=False) if final_norm else nn.Identity()
        self.head = DINOHead(dim)

    def forward(self, images):
        bb = self.backbone
        t = bb.patch_embed(images).flatten(2).transpose(1, 2)
        t = torch.cat([bb.cls_token.expand(t.shape[0], -1, -1), t], 1)
        t = bb.input_norm(t + bb.pos_embed)
        for block in bb.blocks:
            t = block(t)
        t = bb.norm(t)
        return self.head(t[:, 0]), self.head(t[:, 1:])


class ChannelNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.fc1, self.fc2 = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)

    def forward(self, x):
        y = self.dwconv(x).permute(0, 2, 3, 1)
        y = self.fc2(F.gelu(self.fc1(self.norm(y))))
        return x + y.permute(0, 3, 1, 2)


class ConvNeXtStudent(nn.Module):
    """Stem conv and a norm without affine (the fixed-scale entry), stages of blocks, stage
    transitions reading the raw stream, and two linear projections of the last map (pooled and
    per position)."""

    def __init__(self, depths=(1, 1, 2, 1), dims=(8, 16, 32, 64), out=24, final_norm=False):
        super().__init__()
        self.backbone = nn.Module()
        bb = self.backbone
        bb.stem = nn.Module()
        bb.stem.conv, bb.stem.norm = nn.Conv2d(3, dims[0], 4, 4), ChannelNorm(dims[0])
        bb.downs = nn.ModuleList()
        for i in range(3):
            down = nn.Module()
            down.conv = nn.Conv2d(dims[i], dims[i + 1], 2, 2)
            bb.downs.append(down)
        bb.stages = nn.ModuleList(
            nn.Sequential(*(ConvNeXtBlock(d) for _ in range(n)))
            for n, d in zip(depths, dims, strict=True)
        )
        bb.norm = nn.LayerNorm(dims[-1], elementwise_affine=False) if final_norm else nn.Identity()
        self.cls_proj, self.patch_proj = nn.Linear(dims[-1], out), nn.Linear(dims[-1], out)

    def forward(self, images):
        bb = self.backbone
        x = bb.stem.norm(bb.stem.conv(images))
        for i, stage in enumerate(bb.stages):
            if i:
                x = bb.downs[i - 1].conv(x)
            x = stage(x)
        pooled = bb.norm(x.mean((-2, -1)))
        tokens = bb.norm(x.flatten(2).transpose(1, 2))
        return self.cls_proj(pooled), self.patch_proj(tokens)
