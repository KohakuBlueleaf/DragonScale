"""Analysis on a one-block stock pre-norm LM: the free pairs worked out by hand are found and
change a hidden state, the SiLU-pinned pair is not free; a fixed-scale input, gains on the
writers and a final norm leave the stream's scale free, held only by the input; suggest + apply leaves no free scale with a matrix that changes a
hidden state; absorb keeps the function and refuses a pinned move."""

import pytest
import torch

import dragonscale as ds
from dragonscale.analysis import absorb, free_scales, suggest
from tests import arch_models as am
from tests.models import LM, tokens

B = "blocks.0"
PAIRS = {
    "up/down": {f"{B}.up.weight": 1, f"{B}.down.weight": -1},
    "v/out": {f"{B}.v.weight": 1, f"{B}.out.weight": -1},
    "q/k": {f"{B}.q.weight": 1, f"{B}.k.weight": -1},
    "norm/readers": {
        f"{B}.norm1.weight": 1,
        f"{B}.q.weight": -1,
        f"{B}.k.weight": -1,
        f"{B}.v.weight": -1,
    },
    "final norm/head": {"norm.weight": 1, "head.weight": -1},
    "stream": {"embed.weight": 1, f"{B}.out.weight": 1, f"{B}.down.weight": 1},
}
PINNED = {f"{B}.gate.weight": 1, f"{B}.down.weight": -1}


def _model() -> LM:
    torch.manual_seed(0)
    return LM(depth=1).double()


def test_free_scales_before_and_after_suggest():
    model, x = _model(), tokens()
    report = free_scales(model, x)
    assert not report.unknown
    for name, coeffs in PAIRS.items():
        assert report.is_free(coeffs), name
    assert not report.is_free(PINNED)
    assert {f for d in report.changing for f in d.coeffs} >= {n for c in PAIRS.values() for n in c}
    assert report.summary() == {
        "change_hidden_state": 7,
        "change_none": 0,
        "held_by_fixed_term": 0,
        "ops_without_rule": 0,
    }

    ds.apply(model, suggest(model, x, level="minimal", report=report))
    after = free_scales(model, x)
    assert _matrices_in_changing(after) == set()
    assert after.is_free({f"{B}.up.weight": 1})


def _matrices_in_changing(report) -> set[str]:
    shapes = dict(zip(report.names, report.shapes, strict=True))
    return {n for d in report.changing for n in d.coeffs if len(shapes[n]) >= 2}


def test_absorb_keeps_the_function_and_refuses_pinned_moves():
    model, x = _model(), tokens()
    block = model.blocks[0]
    assert absorb(model, f"{B}.norm1", x, into=[f"{B}.q", f"{B}.k", f"{B}.v"]) < 1e-12
    assert torch.equal(block.norm1.weight, torch.ones_like(block.norm1.weight))
    assert absorb(model, f"{B}.up", x, into=[f"{B}.down"]) < 1e-12
    assert torch.allclose(block.up.weight.norm(dim=1), torch.ones(block.up.out_features).double())

    before = {k: v.clone() for k, v in model.state_dict().items()}
    for source, into in ((f"{B}.gate", [f"{B}.down"]), (f"{B}.norm2", [f"{B}.up"])):
        with pytest.raises(ValueError, match="not a free direction"):
            absorb(model, source, x, into=into)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in before.items())


def test_absorb_finds_its_consumers():
    found, x = _model(), tokens()
    given = _model()
    absorb(found, f"{B}.norm1", x)
    absorb(given, f"{B}.norm1", x, into=[f"{B}.q", f"{B}.k", f"{B}.v"])
    absorb(found, f"{B}.up", x)
    absorb(given, f"{B}.up", x, into=[f"{B}.down"])
    assert all(torch.equal(found.state_dict()[k], v) for k, v in given.state_dict().items())


def test_init_absorb_keeps_the_function():
    model, x = _model(), tokens()
    before = model(x)
    rules = [ds.Rule(f"{B}\\.up", mode="unit", init="absorb"), ds.Rule(".*", mode="plain")]
    plan = ds.make_plan(model, rules)
    with pytest.raises(ValueError, match="absorb_plan"):
        ds.Session(model, plan)
    session = ds.apply(model, ds.analysis.absorb_plan(model, plan, x))
    assert session.gains() == []
    assert torch.allclose(model(x), before, rtol=1e-10, atol=1e-12)


def test_suggest_levels():
    model, x = _model(), tokens()
    names = ("q", "k", "v", "out", "gate", "up", "down")
    layers = {"embed", "head", *(f"{B}.{n}" for n in names)}
    assert len(suggest(model, x, level="report")) == 0
    assert {p.path for p in suggest(model, x, level="minimal")} <= layers
    assert {p.path for p in suggest(model, x, level="matrices")} == layers

    plan = suggest(model, x, level="absorb")
    absorbed = {p.path for p in plan if p.mode == "unit" and p.init == "absorb"}
    assert absorbed == {f"{B}.v", f"{B}.up"}
    assert {p.path for p in plan if p.mode == "unit_gain"} == layers - absorbed
    assert dict(plan.moves) == {
        f"{B}.norm1": (f"{B}.q", f"{B}.k", f"{B}.v"),
        f"{B}.norm2": (f"{B}.gate", f"{B}.up"),
        "norm": ("head",),
        f"{B}.v": (f"{B}.out",),
        f"{B}.up": (f"{B}.down",),
    }


def test_absorb_level_keeps_the_function():
    model, x = _model(), tokens()
    before = model(x)
    plan = suggest(model, x, level="absorb")
    with pytest.raises(ValueError, match="absorb_plan"):
        ds.Session(model, plan)
    ds.apply(model, ds.analysis.absorb_plan(model, plan, x))
    assert torch.allclose(model(x), before, rtol=1e-10, atol=1e-12)
    assert _matrices_in_changing(free_scales(model, x)) == set()
    for norm in (model.blocks[0].norm1, model.blocks[0].norm2, model.norm):
        assert not norm.weight.requires_grad


def test_model_notes():
    notes = free_scales(_model(), tokens()).notes
    assert any("attention calls [0]" in note for note in notes)


WRITERS = {f"{B}.out.gain", f"{B}.down.gain"}


def _fixed_input_model(final_norm: bool = True, writer_gains: bool = True) -> LM:
    """Fixed-scale (unit rows, no gain) input and readers, gains (or none) on the layers that write
    to the residual stream, and a final norm (or the head reading the stream directly)."""
    torch.manual_seed(0)
    model = LM(depth=1, affine=False, final_norm=final_norm).double()
    writers = ds.Rule(r".*\.(out|down)", mode="unit_gain" if writer_gains else "unit", init="ones")
    rules = [writers, ds.Rule(r"embed|head|.*\.(q|k|v|gate|up)", mode="unit", init="ones")]
    ds.apply(model, [*rules, ds.Rule(".*", mode="plain")])
    return model


def test_fixed_scale_input_behind_a_final_norm_leaves_the_stream_free():
    """h = e + c B: after the final norm the output changes only by about |e| / |c B|."""
    report = free_scales(_fixed_input_model(), tokens())
    assert report.summary()["change_hidden_state"] == 1
    assert report.summary()["held_by_fixed_term"] == 1
    (stream,) = report.changing
    assert set(stream.coeffs) == WRITERS
    assert not stream.exact and stream.held_by == (f"{B}: aten::add with output of embed",)
    assert report.held == [stream] and stream not in report.exact
    assert report.is_free(dict.fromkeys(WRITERS, 1.0))
    assert not report.is_free(dict.fromkeys(WRITERS, 1.0), exact=True)
    assert "1 of them held only by a fixed-scale term" in report.describe()
    assert any("drop those gains or remove the norm" in note for note in report.notes)


def test_no_free_stream_without_the_final_norm_or_the_writer_gains():
    for options in ({"final_norm": False}, {"writer_gains": False}):
        report = free_scales(_fixed_input_model(**options), tokens())
        assert report.summary()["change_hidden_state"] == 0, options
        assert report.held == [], options
    report = free_scales(_fixed_input_model(final_norm=False), tokens())
    assert not report.is_free(dict.fromkeys(WRITERS, 1.0))


def test_held_scale_fades_like_one_over_the_scale():
    """The relative output change for a small rescaling of every writer gain falls like 1/c as
    the writer gains are multiplied by c."""
    model = _fixed_input_model()
    params = dict(model.named_parameters())
    _relative_changes(model, [params[n] for n in sorted(WRITERS)], (tokens(),))


class PatchModel(torch.nn.Module):
    """A JiT-like model: patches through a norm without affine into the stream, conditioning
    tokens (through the same kind of norm) put in front from the second block, read and never
    written, a final norm (or none) and an output layer."""

    def __init__(self, final_norm: bool = True, dim: int = 16, patch: int = 4, classes: int = 3):
        super().__init__()
        self.patch = torch.nn.Conv2d(3, dim, patch, patch, bias=False)
        self.cond = torch.nn.Embedding(classes, dim)
        self.in_norm = torch.nn.RMSNorm(dim, elementwise_affine=False)
        self.blocks = torch.nn.ModuleList(am.Block(dim, 2, 2 * dim) for _ in range(2))
        self.norm = torch.nn.RMSNorm(dim, elementwise_affine=False) if final_norm else None
        self.out = torch.nn.Linear(dim, patch * patch * 3)

    def forward(self, images, labels):
        x = self.in_norm(self.patch(images).flatten(2).transpose(1, 2))
        cond = self.in_norm(self.cond(labels)).unsqueeze(1)
        for i, block in enumerate(self.blocks):
            if i == 1:
                x = torch.cat([cond, x], 1)
                block.prefix = 1
            x = block(x)
        x = x[:, 1:]
        return self.out(self.norm(x) if self.norm is not None else x)


G = torch.Generator().manual_seed(0)
PATCH_INPUTS = (torch.randn(2, 3, 8, 8, generator=G, dtype=torch.float64), torch.tensor([0, 2]))
PATCH_WRITERS = {f"blocks.{i}.{n}.gain" for i in (0, 1) for n in ("attn.out", "mlp.down")}


def _patch_model(
    final_norm: bool = True, writer_gains: bool = True, output_gain: bool = True
) -> PatchModel:
    torch.manual_seed(0)
    model = PatchModel(final_norm).double()
    writers = r"blocks\.\d\.(attn\.out|mlp\.down)"
    rules = [
        ds.Rule(writers, mode="unit_gain" if writer_gains else "unit", init="ones"),
        ds.Rule("out", mode="unit_gain" if output_gain else "unit", init="ones"),
        ds.Rule(".*", mode="unit", init="ones"),
    ]
    ds.apply(model, ds.make_plan(model, rules))
    return model


def test_norm_output_entry_holds_the_stream():
    """The fixed-scale input is a norm output, and fixed-scale tokens are put in front of the
    stream by a concatenation: the stream's scale is free, held only by those terms."""
    report = free_scales(_patch_model(), *PATCH_INPUTS)
    assert report.summary()["change_hidden_state"] == 1, report.describe()
    (stream,) = report.changing
    assert not stream.exact
    assert PATCH_WRITERS <= set(stream.coeffs), report.describe()
    assert stream.held_by == (
        "blocks.0: aten::add with output of in_norm",
        "model: aten::cat with output of in_norm",
    )
    for options in ({"final_norm": False, "output_gain": False}, {"writer_gains": False}):
        report = free_scales(_patch_model(**options), *PATCH_INPUTS)
        assert report.summary()["change_hidden_state"] == 0, (options, report.describe())
    # Without the final norm, a gain on the output layer trades with the stream: h = e + c B read
    # by g / c gives g e / c + g B, held only by e
    report = free_scales(_patch_model(final_norm=False), *PATCH_INPUTS)
    (stream,) = report.changing
    assert stream.coeffs["out.gain"] == -1 and not stream.exact


def test_norm_output_entry_hold_fades_like_one_over_the_scale():
    model = _patch_model()
    params = dict(model.named_parameters())
    _relative_changes(model, [params[n] for n in sorted(PATCH_WRITERS)], PATCH_INPUTS)


def _relative_changes(model, gains, inputs) -> list[float]:
    """The relative output change for a small rescaling of `gains`, with the gains multiplied by
    c in 1, 4, 16, 64: it falls like 1/c."""
    start = [g.detach().clone() for g in gains]
    eps, changes = 1e-4, []
    with torch.no_grad():
        for c in (1.0, 4.0, 16.0, 64.0):
            for g, g0 in zip(gains, start, strict=True):
                g.copy_(c * g0)
            before = model(*inputs)
            for g in gains:
                g.mul_(1 + eps)
            after = model(*inputs)
            changes.append(((after - before).norm() / before.norm()).item() / eps)
    assert all(b < a for a, b in zip(changes, changes[1:], strict=False)), changes
    assert changes[-1] < changes[0] / 8, changes
    return changes


def test_counts_do_not_depend_on_the_basis():
    """Two free scales that change a hidden state through the same tensor and one that changes
    none: the counts are the dimensions rank(D N) and dim ker A - rank(D N)."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.LayerNorm(4, elementwise_affine=False),
        torch.nn.Linear(4, 2),
    ).double()
    x = torch.randn(3, 4, dtype=torch.float64)
    report = free_scales(model, x)
    # 0 and 1 trade (changes the hidden state between them); 0 + 1 together is cut by the norm
    assert report.count == 2
    assert report.summary()["change_hidden_state"] == 2
    ds.apply(model, [ds.Rule("1", mode="unit", init="ones"), ds.Rule(".*", mode="plain")])
    report = free_scales(model, x)
    assert report.summary()["change_hidden_state"] == 1  # 0's scale, cut by unit rows of 1
    assert report.summary()["change_none"] == 1  # 1's raw weight
