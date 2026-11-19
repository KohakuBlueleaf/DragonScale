"""`QKNorm`: q and k normalized per head, with a learned gain on q only."""

from torch import Tensor, nn


class QKNorm(nn.Module):
    """RMS norm of q and of k over their last axis (one head's channels); a learned gain on q
    only. A gain on both sides would leave their product's split between them free.

        q, k = self.qk_norm(q, k)   # [batch, heads, tokens, head_dim] each
    """

    def __init__(self, head_dim: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.q_norm = nn.RMSNorm(head_dim, eps=eps, device=device, dtype=dtype)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps, elementwise_affine=False)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        return self.q_norm(q), self.k_norm(k)
