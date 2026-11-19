"""The README and docs run as written: every `python` block of a file is executed in order in one
namespace, and every line of each `text` block appears in what the blocks print. The numbers the
README quotes about the example are recomputed."""

import contextlib
import copy
import io
import re
from pathlib import Path

import pytest
import torch
from torch import nn

import dragonscale as ds

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def blocks(path: Path, language: str) -> list[str]:
    return re.findall(rf"^```{language}\n(.*?)^```", path.read_text(encoding="utf-8"), re.M | re.S)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_code_blocks_run(path):
    printed = io.StringIO()
    namespace = {"__name__": "__readme__"}
    with contextlib.redirect_stdout(printed):
        for code in blocks(path, "python"):
            exec(compile(code, str(path), "exec"), namespace)
    out = printed.getvalue()
    for text in blocks(path, "text"):
        for line in text.strip().splitlines():
            assert line.rstrip() in out, f"{path.name}: {line!r} not printed"


def _example():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Embedding(256, 64),
        nn.TransformerEncoderLayer(64, nhead=4, dropout=0.0, batch_first=True, norm_first=True),
        nn.Linear(64, 256),
    )
    tokens = torch.randint(0, 256, (8, 33))
    return model, tokens[:, :-1]


def test_readme_numbers():
    """The output change the README quotes for the quick start, and the layers it names."""
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    model, x = _example()
    with torch.no_grad():
        before = model(x)
    for level in ("absorb", "selective"):
        patched = copy.deepcopy(model)
        plan = ds.analysis.suggest(patched, x, level=level)
        ds.apply(patched, ds.analysis.absorb_plan(patched, plan, x))
        with torch.no_grad():
            change = (patched(x) - before).abs().max().item()
        summary = ds.analysis.free_scales(patched, x).summary()
        if level == "absorb":
            assert change < 1e-5
        else:
            rms = before.square().mean().sqrt().item()
            assert f"changes by up to {change:.1f}, against an output RMS of {rms:.2f}" in readme
            assert any("output at the start of: 0, 2" in n for n in plan.notes)
            assert summary["change_hidden_state"] == 0
