"""A stock pre-norm LM from stock torch modules (Embedding, RMSNorm, Linear, SDPA, SwiGLU)."""

import torch
import torch.nn.functional as F
from torch import nn


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int, affine: bool):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=affine)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=affine)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        b, n, d = x.shape
        h = self.norm1(x)
        q, k, v = (
            p(h).view(b, n, self.heads, -1).transpose(1, 2) for p in (self.q, self.k, self.v)
        )
        mixed = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out(mixed.transpose(1, 2).reshape(b, n, d))
        h = self.norm2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class LM(nn.Module):
    def __init__(self, vocab=32, dim=16, depth=2, heads=2, hidden=32, affine=True, tied=False):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, hidden, affine) for _ in range(depth))
        self.norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=affine)
        self.head = nn.Linear(dim, vocab, bias=False)
        if tied:
            self.head.weight = self.embed.weight
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name.endswith("norm1.weight") or name.endswith("norm2.weight"):
                    param.uniform_(0.5, 1.5)

    def forward(self, tokens):
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def tokens(vocab: int = 32, shape=(2, 6)) -> torch.Tensor:
    return torch.randint(0, vocab, shape, generator=torch.Generator().manual_seed(1))
