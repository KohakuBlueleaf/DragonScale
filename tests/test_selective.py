"""The `selective` level on four small models built with the model-side rules: it patches the
layers in the modes the method prescribes (written out by hand below, as used for the models the
method was developed on), gives the same output at step 0 as that hand-written patch, and leaves
no free scale that changes a hidden state in the patched part of the model."""

import re
import warnings

import pytest
import torch

import dragonscale as ds
from tests import arch_models as am

B = r"(backbone\.)?blocks\.\d+"
G = torch.Generator().manual_seed(2)
TOKENS = torch.randint(0, 64, (2, 12), generator=G)
IMAGES = torch.randn(2, 3, 32, 32, generator=G)
DIFFUSION = (IMAGES, torch.rand(2, generator=G), torch.randint(0, 5, (2,), generator=G))

# name: (model factory, inputs, suggest options, layers without gain, output gain start)
CASES = {
    "lm": (am.LM, (TOKENS,), {}, rf"embed|pos|head|{B}\.attn\.(q|k|v)|{B}\.mlp\.up", None),
    "lm_final_norm": (
        lambda: am.LM(final_norm=True),
        (TOKENS,),
        {},
        rf"embed|pos|{B}\.attn\.(q|k|v)|{B}\.mlp\.up",
        None,
    ),
    "dit": (
        am.DiT,
        DIFFUSION,
        {"output_gain": 0.18},
        rf"t_embed\.2|y_embed|patch1|patch2|in_context_pos|{B}\.attn\.(q|k|v)|{B}\.mlp\.up",
        0.18,
    ),
    "dit_final_norm": (
        lambda: am.DiT(final_norm=True),
        DIFFUSION,
        {},
        rf"t_embed\.2|y_embed|patch1|patch2|in_context_pos|{B}\.attn\.(q|k|v)|{B}\.mlp\.up",
        None,
    ),
    "vit": (
        am.ViTStudent,
        (IMAGES,),
        {"roles": {"plain": r"head\.(mlp\.[1-9]|last_layer).*"}},
        rf"backbone\.patch_embed|{B}\.attn\.(q|k|v)|head\.mlp\.0",
        None,
    ),
    "vit_final_norm": (
        lambda: am.ViTStudent(final_norm=True),
        (IMAGES,),
        {"roles": {"plain": r"head\.(mlp\.[1-9]|last_layer).*"}},
        rf"backbone\.patch_embed|{B}\.attn\.(q|k|v)",
        None,
    ),
    "convnext": (
        am.ConvNeXtStudent,
        (IMAGES,),
        {"output_gain": 1.0},
        r"backbone\.(stem\.conv|downs\.\d\.conv|stages\.\d\.\d\.dwconv)",
        1.0,
    ),
    "convnext_final_norm": (
        lambda: am.ConvNeXtStudent(final_norm=True),
        (IMAGES,),
        {"output_gain": 1.0},
        r"backbone\.(stem\.conv|downs\.\d\.conv|stages\.\d\.\d\.dwconv)",
        1.0,
    ),
}
OUTPUTS = r"head|out|cls_proj|patch_proj"


def _build(factory):
    torch.manual_seed(0)
    return factory()


def _by_hand(model, inputs, unit: str, output_gain, plain: str | None) -> ds.Session:
    """The `absorb` plan, then unit rows without gain on the given layers and an output gain
    start: the patch written by hand."""
    plan = ds.analysis.suggest(model, *inputs, level="absorb", allow_unknown=True)
    rules = []
    for p in plan:
        if plain and re.fullmatch(plain, p.path):
            rules.append(ds.Rule(re.escape(p.path), mode="plain"))
        elif re.fullmatch(unit, p.path):
            init = "absorb" if p.init == "absorb" else "ones"
            rules.append(ds.Rule(re.escape(p.path), mode="unit", init=init, spec=p.spec))
        else:
            rules.append(ds.Rule(re.escape(p.path), mode=p.mode, init=p.init, spec=p.spec))
    hand = ds.make_plan(model, [*rules, ds.Rule(".*", mode="plain")])
    plan = ds.Plan(hand.layers, plan.moves)
    session = ds.apply(model, ds.analysis.absorb_plan(model, plan, *inputs))
    if output_gain is not None:
        with torch.no_grad():
            for p in session.patched:
                if re.fullmatch(OUTPUTS, p.plan.path) and p.gain is not None:
                    p.gain.fill_(output_gain)
    return session


def _output(model, inputs):
    with torch.no_grad():
        out = model(*inputs)
    return torch.cat([t.flatten() for t in (out if isinstance(out, tuple) else (out,))])


@pytest.mark.parametrize("name", list(CASES))
def test_selective_matches_the_method(name):
    factory, inputs, options, unit, output_gain = CASES[name]
    model = _build(factory)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = ds.analysis.suggest(model, *inputs, level="selective", **options)
    session = ds.apply(model, ds.analysis.absorb_plan(model, plan, *inputs))

    no_gain = {p.plan.path for p in session.patched if p.plan.mode == "unit"}
    patched = {p.plan.path for p in session.patched}
    assert no_gain == {p for p in patched if re.fullmatch(unit, p)}, plan.describe()

    plain = options.get("roles", {}).get("plain")
    reference = _build(factory)
    _by_hand(reference, inputs, unit, output_gain, plain)
    assert {p.plan.path for p in ds.Session.from_model(reference).patched} == patched
    torch.testing.assert_close(_output(model, inputs), _output(reference, inputs))

    report = ds.analysis.free_scales(model, *inputs)
    assert not report.unknown
    remaining = {n for d in report.changing for n in d.coeffs}
    if name.startswith("vit"):
        # The head's layers left unpatched by the `plain` role: its bottleneck layer, read only
        # through an L2 normalization, keeps a free scale; the backbone has none
        assert remaining == {"head.mlp.4.weight", "head.mlp.4.bias"}
    else:
        assert remaining == set()
    if report.weak:
        assert any("normalization in front of the output" in n for n in plan.notes)
    if name in ("lm_final_norm", "vit_final_norm"):
        assert report.weak


def test_gains_all_and_none():
    model, inputs = _build(am.LM), (TOKENS,)
    absorb = ds.analysis.suggest(model, *inputs, level="absorb")
    every = ds.analysis.suggest(model, *inputs, level="selective", gains="all")
    assert [(p.path, p.mode) for p in every] == [(p.path, p.mode) for p in absorb]
    none = ds.analysis.suggest(model, *inputs, level="selective", gains="none", output_gain=0.5)
    assert {p.path for p in none if p.mode == "unit_gain"} == {"head"}
    assert next(p for p in none if p.path == "head").init == 0.5


def test_roles_override_the_analysis():
    model, inputs = _build(am.LM), (TOKENS,)
    plan = ds.analysis.suggest(model, *inputs, roles={"qk": r"blocks\.0\.mlp\.gate"})
    assert next(p for p in plan if p.path == "blocks.0.mlp.gate").mode == "unit"
    with pytest.raises(ValueError, match="roles"):
        ds.analysis.suggest(model, *inputs, roles={"writer": "x"})


class _WithModule(torch.nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim, elementwise_affine=False)
        self.fc = torch.nn.Linear(dim, dim)
        self.ls = _Gamma(dim)

    def forward(self, x):
        return x + self.ls(self.fc(self.norm(x)))


class _Gamma(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.rand(dim) * 1e-2)

    def forward(self, x):
        return x * self.gamma


def test_fold_layerscale_keeps_the_output():
    torch.manual_seed(0)
    model, x = _WithModule().double(), torch.randn(3, 16, dtype=torch.float64)
    before = model(x)
    session = ds.apply(model, "unit_gain")
    ds.fold_layerscale(session, {"fc": "ls"})
    assert isinstance(model.ls, torch.nn.Identity)
    torch.testing.assert_close(model(x), before)
    with pytest.raises(ds.problems.ProblemsError, match="not a patched layer"):
        ds.fold_layerscale(session, {"norm": "ls"})


def test_qk_norm_leaves_no_free_qk_scale():
    class Attention(torch.nn.Module):
        def __init__(self, dim=16, heads=2):
            super().__init__()
            self.heads = heads
            self.q, self.k = torch.nn.Linear(dim, dim), torch.nn.Linear(dim, dim)
            self.v, self.out = torch.nn.Linear(dim, dim), torch.nn.Linear(dim, dim)
            self.qk_norm = ds.nn.QKNorm(dim // heads)

        def forward(self, x):
            b, n, d = x.shape
            q, k, v = (
                p(x).view(b, n, self.heads, -1).transpose(1, 2) for p in (self.q, self.k, self.v)
            )
            q, k = self.qk_norm(q, k)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            return self.out(y.transpose(1, 2).reshape(b, n, d))

    torch.manual_seed(0)
    model, x = Attention(), torch.randn(2, 5, 16)
    report = ds.analysis.free_scales(model, x)
    assert not any("QK-norm" in note for note in report.notes)
    plan = ds.analysis.suggest(model, x)
    assert {p.path for p in plan if p.mode == "unit"} >= {"q", "k"}


def test_frozen_parameters_are_constants():
    """After `absorb_plan` freezes a moved norm's weight, the analysis treats it as a constant:
    the stock encoder layers reach no free scale that changes a hidden state."""
    torch.manual_seed(0)
    layer = dict(
        d_model=32, nhead=2, dim_feedforward=64, dropout=0.0, batch_first=True, norm_first=True
    )
    model = torch.nn.Sequential(
        torch.nn.Embedding(64, 32),
        torch.nn.TransformerEncoderLayer(**layer),
        torch.nn.TransformerEncoderLayer(**layer),
        torch.nn.Linear(32, 64),
    )
    x = torch.randint(0, 64, (2, 12))
    plan = ds.analysis.suggest(model, x)
    ds.apply(model, ds.analysis.absorb_plan(model, plan, x))
    assert not model[1].norm1.weight.requires_grad
    assert ds.analysis.free_scales(model, x).summary()["change_hidden_state"] == 0
    for p in model.parameters():
        p.requires_grad_(False)
    assert ds.analysis.free_scales(model, x).names  # nothing trained: every parameter counts
