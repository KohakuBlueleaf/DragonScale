"""Parameter groups from a session, the monitor's readings, and a number as gain init."""

import math

import pytest
import torch
from torch import nn

import dragonscale as ds
from tests.models import LM, tokens


def test_param_groups_decay_raw_weights_only():
    torch.manual_seed(0)
    model = LM(depth=1)
    model.extra = nn.Parameter(torch.randn(3, 3))  # an unpatched matrix: decayed
    session = ds.apply(model, "unit_gain")
    with torch.no_grad():
        model.blocks[0].norm1.weight.requires_grad_(False)  # frozen: left out
    groups = ds.optim.param_groups(session, weight_decay=0.1, lr=1e-3)
    decay, rest = groups
    assert decay["weight_decay"] == 0.1 and rest["weight_decay"] == 0.0
    assert decay["lr"] == rest["lr"] == 1e-3
    ids = lambda ps: {id(p) for p in ps}  # noqa: E731
    raw = ids(p for p, _ in session.raw_weights())
    assert ids(decay["params"]) == raw | {id(model.extra)}
    assert ids(session.gains()) <= ids(rest["params"])
    assert id(model.blocks[0].norm1.weight) not in ids(decay["params"]) | ids(rest["params"])
    every = ids(p for p in model.parameters() if p.requires_grad)
    assert ids(decay["params"]) | ids(rest["params"]) == every
    torch.optim.AdamW(groups).step()  # a valid argument


def test_monitor_readings():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    session = ds.apply(model, "unit_gain")
    monitor = ds.monitor.Monitor(session, angles=True)
    x = torch.randn(5, 8)
    monitor.start()
    out = model(x)
    first = monitor.read()
    assert first["2/out_max"] == pytest.approx(out.abs().max().item())
    assert first["2/out_rms"] == pytest.approx(out.square().mean().sqrt().item())
    norms = model[0].weight.norm(dim=1)
    assert first["0/raw_norm_max"] == pytest.approx(norms.max().item(), rel=1e-6)
    assert first["0/gain_max"] == pytest.approx(norms.max().item(), rel=1e-6)
    assert "0/angle_median" not in first

    # Turn row 0 of the first layer by a known angle in its own plane
    with torch.no_grad():
        w = model[0].weight
        u = w[0] / w[0].norm()
        v = torch.randn(8)
        v = v - (v @ u) * u
        v = v / v.norm()
        w[0] = w[0].norm() * (math.cos(0.01) * u + math.sin(0.01) * v)
    second = monitor.read()
    assert second["0/angle_max"] == pytest.approx(0.01, rel=1e-4)
    assert second["2/angle_max"] == pytest.approx(0.0, abs=1e-6)


def test_number_as_gain_init():
    torch.manual_seed(0)
    model = nn.Linear(8, 4)
    session = ds.apply(model, [ds.Rule(".*", init=0.0)])
    assert torch.equal(session.gains()[0], torch.zeros(4))
    torch.testing.assert_close(model(torch.randn(2, 8)), model.bias.expand(2, 4))
    with pytest.raises(ds.problems.ProblemsError, match="has none"):
        ds.apply(nn.Linear(8, 4), [ds.Rule(".*", mode="unit", init=2.0)])


def test_lm_param_groups_train():
    """A few AdamW steps on the patched LM with the groups: finite, loss goes down."""
    torch.manual_seed(0)
    model, x = LM(), tokens()
    session = ds.apply(model, "unit_gain")
    optimizer = torch.optim.AdamW(ds.optim.param_groups(session, weight_decay=0.1), lr=1e-2)
    losses = []
    for _ in range(20):
        logits = model(x[:, :-1])
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert all(math.isfinite(v) for v in losses) and losses[-1] < losses[0]
