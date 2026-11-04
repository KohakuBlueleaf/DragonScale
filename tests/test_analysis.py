"""Analysis on a one-block stock pre-norm LM: the free pairs worked out by hand are found and are
dangerous, the SiLU-pinned pair is not; a fixed-scale input behind a final norm leaves the writers
weakly pinned; suggest + apply leaves none dangerous; absorb keeps the function and refuses a
pinned move."""

import pytest
import torch

import dragonscale as ds
from dragonscale.analysis import absorb, free_scales, suggest
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
        assert report.contains(coeffs), name
    assert not report.contains(PINNED)
    assert {f for d in report.dangerous for f in d.fast_ends} >= {
        n for c in PAIRS.values() for n in c if "norm" not in n
    }

    ds.apply(model, suggest(model, x, level="minimal", report=report))
    after = free_scales(model, x)
    assert after.dangerous == []
    assert after.contains({f"{B}.up.weight": 1})


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
    assert free_scales(model, x).dangerous == []
    for norm in (model.blocks[0].norm1, model.blocks[0].norm2, model.norm):
        assert not norm.weight.requires_grad


def test_architecture_notes():
    notes = free_scales(_model(), tokens()).notes
    assert any("attention calls [0]" in note for note in notes)
    assert any("scale the residual stream together" in note for note in notes)


@pytest.mark.parametrize("final_norm", [True, False])
def test_fixed_scale_input_behind_a_final_norm_is_weakly_pinned(final_norm):
    torch.manual_seed(0)
    model, x = LM(depth=1, affine=False, final_norm=final_norm).double(), tokens()
    fixed = r"embed|.*\.(q|k|v|gate|up)" + ("" if final_norm else "|head")
    rules = [ds.Rule(fixed, mode="unit", init="ones"), ds.Rule(".*", mode="plain")]
    ds.apply(model, rules)
    report = free_scales(model, x)
    writers = {f"{B}.out.weight", f"{B}.down.weight"}
    if final_norm:
        assert [set(d.coeffs) for d in report.weak] == [writers]
        assert report.weak[0].dangerous
        assert report.weak[0].anchors == (f"{B}: aten::add",)
        assert any("held only by a fixed-scale addend" in note for note in report.notes)
    else:
        assert report.weak == []
