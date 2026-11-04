"""Smoke test of an installed wheel on the CPU: imports, eager ops, a session round trip."""

import torch
from torch import nn

import dragonscale as ds
from dragonscale_kernels import available, ops


def main() -> None:
    assert "site-packages" in ds.__file__, ds.__file__
    assert "eager" in available(ops.UNIT_ROWS, "cpu")

    layer = nn.TransformerEncoderLayer(16, 2, 32, dropout=0.0, batch_first=True).train()
    x = torch.randn(2, 5, 16)
    before = layer(x)
    session = ds.apply(layer, "unit_gain")
    assert torch.allclose(layer(x), before, rtol=1e-4, atol=1e-5)
    session.bake()
    assert torch.allclose(layer(x), before, rtol=1e-4, atol=1e-5)
    print(f"dragonscale {ds.__version__} at {ds.__file__}: {len(session.plan)} sites")


if __name__ == "__main__":
    main()
