"""The `selective` level: unit rows on every layer, and a gain only where the loss sees its scale.

Starting from the `absorb` plan (unit rows and a gain on every layer, pair moves made), gains are
taken away in this order, each decision read from a trace of the model patched so far:

1. `roles`: layers named `embed` or `qk` lose their gain; `output` layers lose it, or start it at
   `output_gain`; `plain` layers are not patched.
2. Entries: a layer whose input depends on no parameter (an embedding lookup, a patch embedding)
   and whose gain is in a free scale that changes a hidden state. The input then enters with a
   fixed scale.
3. Free on its own: a gain (with its layer's bias) whose scale is free by itself: a layer read only by a normalization (q and k in front of a QK-norm, a layer in front
   of a norm), or held only by a fixed-scale term added to it.
4. A stream that starts nowhere: in an exact free scale that changes a hidden state with every
   coefficient positive, the gain of the layer called first (the one that starts the stream).
5. Readers: in a free scale that changes a hidden state, the layers with a
   negative coefficient read that scale (the output layer reads the residual stream) and lose
   their gain; a reader whose output is a model output starts its gain at `output_gain` instead,
   when that is given.

Steps 3 to 5 repeat until no gain is taken away. A layer that loses its gain gets unit rows
without one (`mode="unit"`, `init="ones"`, or `init="absorb"` when its rows moved into its
consumers): its output at the start changes unless its rows were moved. `gains="all"` keeps
every gain (the `absorb` plan), `gains="none"` takes away every gain except `output_gain`'s.

"Free" covers both kinds of free scale: exact, and held only by a fixed-scale term
(`analysis.report`). The plan's notes list what remains for the model code: exact free scales that
change a hidden state (with the parameters in them), those held only by a fixed-scale term
(typically a norm in front of the output), and attention calls without QK-norm.
"""

import copy
import re
from dataclasses import replace

import torch
from torch import nn

from dragonscale.analysis.absorb import absorb_plan
from dragonscale.analysis.report import FreeScales, in_span, null_space, solve
from dragonscale.analysis.trace import DegreeTrace, trace_degrees  # noqa: F401
from dragonscale.log import get_logger
from dragonscale.patch.session import apply
from dragonscale.plan.build import make_plan
from dragonscale.plan.layer_plan import LayerPlan, Plan
from dragonscale.plan.rules import Rule

ROLES = ("embed", "output", "qk", "plain")
GAINS = ("selective", "all", "none")
MAX_ROUNDS = 64
log = get_logger("analysis")


def _matches(pattern: str | None, path: str) -> bool:
    return pattern is not None and re.fullmatch(pattern, path) is not None


class _State:
    """The layer plans being decided, by path, and what each lost or kept."""

    def __init__(self, base: Plan, roles: dict[str, str], output_gain: float | None):
        self.layers: dict[str, LayerPlan] = {p.path: p for p in base}
        self.groups: dict[str, list[str]] = {}
        for p in base:
            root = p.tied_to or p.path
            self.groups.setdefault(root, []).append(p.path)
        self.output_gain = output_gain
        self.fixed: set[str] = set()  # decided, not to be looked at again
        self.reasons: dict[str, str] = {}
        for path in list(self.layers):
            if _matches(roles.get("plain"), path):
                del self.layers[path]
                continue
            for role in ("embed", "qk"):
                if _matches(roles.get(role), path):
                    self.drop(path, f"role {role}")
            if _matches(roles.get("output"), path):
                self.reader(path, "role output")

    def tied(self, path: str) -> list[str]:
        root = self.layers[path].tied_to or path
        return [p for p in self.groups.get(root, [path]) if p in self.layers]

    def drop(self, path: str, reason: str) -> None:
        for p in self.tied(path):
            layer = self.layers[p]
            init = "absorb" if layer.init == "absorb" else "ones"
            self.layers[p] = layer.with_options(mode="unit", init=init)
            self.fixed.add(p)
            self.reasons[p] = reason

    def reader(self, path: str, reason: str) -> None:
        if self.output_gain is None:
            self.drop(path, reason)
            return
        for p in self.tied(path):
            self.layers[p] = self.layers[p].with_options(mode="unit_gain", init=self.output_gain)
            self.fixed.add(p)
            self.reasons[p] = f"{reason}, gain starts at {self.output_gain:g}"

    def has_gain(self, path: str) -> bool:
        return self.layers[path].mode == "unit_gain"

    def plan(self, base: Plan) -> Plan:
        return Plan(tuple(self.layers[p.path] for p in base if p.path in self.layers))


def _rules(plan: Plan) -> list[Rule]:
    rules = [
        Rule(
            re.escape(p.path),
            mode=p.mode,
            gain=p.gain,
            spec=p.spec,
            init=p.init,
            install=p.install,
        )
        for p in plan
    ]
    return [*rules, Rule(".*", mode="plain")]


def _trace(work: nn.Module, plan: Plan, inputs, kwargs):
    """Trace `work` patched with `plan` (then unpatched). Returns the trace, and per layer path
    with a gain the parameter names of its gain and of its bias."""
    session = apply(work, make_plan(work, _rules(plan)))
    try:
        names = {id(p): n for n, p in work.named_parameters()}
        gain_of, bias_of = {}, {}
        for p in session.untied_layers():
            if p.gain is None:
                continue
            gain_of[p.plan.path] = names[id(p.gain)]
            bias = p.bias()
            if isinstance(bias, nn.Parameter):
                bias_of[p.plan.path] = names[id(bias)]
        trace = trace_degrees(work, *inputs, **kwargs)
    finally:
        session.remove()
    return trace, gain_of, bias_of


class _Spaces:
    """The free scales of one trace: exact (`ker A`), all of them (`ker A0`), and the report for
    their readable directions."""

    def __init__(self, trace: DegreeTrace):
        # A bias held only by the fixed-scale product it is added to: no choice of gains changes
        # that hold, so the decisions read the trace with those anchors as constraints
        trace = trace.biases_fixed()
        self.trace = trace
        self.names = trace.names
        hard = trace.matrix()
        self.exact = null_space(torch.cat([hard, trace.anchor_matrix()]))
        self.loose = null_space(hard)
        self.report: FreeScales = solve(trace)

    def vector(self, coeffs: dict[str, float]):
        s = torch.zeros(len(self.names), dtype=torch.float64)
        for name, c in coeffs.items():
            s[self.names.index(name)] = c
        return s

    def changing(self) -> list[dict[str, float]]:
        """Readable free scales (both kinds) that change a hidden state."""
        return [d.coeffs for d in self.report.changing]

    def exact_changing(self) -> list[dict[str, float]]:
        """Readable exact free scales that change a hidden state."""
        return [d.coeffs for d in self.report.changing if d.exact]


def _selective(
    model: nn.Module,
    inputs: tuple,
    kwargs: dict,
    base: Plan,
    gains: str,
    output_gain: float | None,
    roles: dict[str, str],
) -> Plan:
    work = copy.deepcopy(model)
    absorb_plan(work, base, *inputs, **kwargs)
    layers_only = replace(base, moves=())
    state = _State(layers_only, roles, output_gain)

    if gains == "none":
        outputs = set()
        if output_gain is not None:
            outputs = _trace(work, state.plan(layers_only), inputs, kwargs)[0].outputs
        for path in list(state.layers):
            if path not in state.fixed and state.has_gain(path):
                if path in outputs:
                    state.reader(path, "output layer")
                else:
                    state.drop(path, "gains='none'")
    elif gains == "selective":
        _decide(work, layers_only, state, inputs, kwargs)

    if output_gain is not None and gains != "all":
        trace, _, _ = _trace(work, state.plan(layers_only), inputs, kwargs)
        for path in trace.outputs:
            if path in state.layers and state.has_gain(path) and path not in state.fixed:
                state.reader(path, "output layer")
    plan = state.plan(layers_only)
    notes = _notes(work, plan, inputs, kwargs, state)
    for note in notes:
        log.info(note)
    return Plan(plan.layers, base.moves, tuple(notes))


def _decide(work, base: Plan, state: _State, inputs, kwargs) -> None:
    # 2. Entries, read on the plan with every gain
    trace, gain_of, _ = _trace(work, state.plan(base), inputs, kwargs)
    spaces = _Spaces(trace)
    in_changing = {n for coeffs in spaces.changing() for n in coeffs}
    for path in trace.entries:
        if path in state.layers and path not in state.fixed and state.has_gain(path):
            if gain_of.get(path) in in_changing:
                state.drop(path, "entry: its input depends on no parameter")

    for _ in range(MAX_ROUNDS):
        trace, gain_of, bias_of = _trace(work, state.plan(base), inputs, kwargs)
        spaces = _Spaces(trace)
        layer_of = {name: path for path, name in gain_of.items()}
        open_paths = [
            p
            for p in trace.module_inputs
            if p in gain_of and p not in state.fixed and state.has_gain(p)
        ]

        # 3. Free on its own (either kind)
        alone = []
        for path in open_paths:
            coeffs = {gain_of[path]: 1.0}
            if path in bias_of:
                coeffs[bias_of[path]] = 1.0
            if in_span(spaces.loose, spaces.vector(coeffs)):
                alone.append(path)
        for path in alone:
            state.drop(path, "its scale is free on its own")
        if alone:
            continue

        # 4. A stream that starts nowhere: the first-called gain of an all-positive free scale
        order = {p: i for i, p in enumerate(trace.module_inputs)}
        starts = set()
        for coeffs in spaces.exact_changing():
            if any(c < 0 for c in coeffs.values()):
                continue
            paths = [layer_of[n] for n in coeffs if n in layer_of and layer_of[n] in open_paths]
            if paths:
                starts.add(min(paths, key=order.__getitem__))
        for path in starts:
            state.drop(path, "it starts a stream whose scale is free")
        if starts:
            continue

        # 5. Readers: negative coefficients against an all-positive rest
        readers = set()
        for coeffs in spaces.changing():
            negative = [n for n, c in coeffs.items() if c < 0]
            if not negative or not all(n in layer_of for n in negative):
                continue
            readers.update(layer_of[n] for n in negative if layer_of[n] in open_paths)
        for path in readers:
            if path in trace.outputs:
                state.reader(path, "the output layer reads a scale the loss does not fix")
            else:
                state.drop(path, "it reads a scale the loss does not fix")
        if readers:
            continue
        break


def _notes(work, plan: Plan, inputs, kwargs, state: _State) -> list[str]:
    notes = []
    changed = sorted(p for p, r in state.reasons.items() if state.layers[p].init == "ones")
    if changed:
        notes.append(
            "unit rows without a gain change the output at the start of: " + ", ".join(changed)
        )
    trace, _, _ = _trace(work, plan, inputs, kwargs)
    report = solve(trace)
    for d in report.changing:
        if not d.exact:
            continue  # in report.notes
        notes.append(
            "a free scale that changes a hidden state remains: "
            + " ".join(f"{c:+g} {n}" for n, c in d.coeffs.items())
            + "; give the parameters in it a fixed scale in the model code"
        )
    notes += report.notes
    return notes


def selective(
    model: nn.Module,
    inputs: tuple,
    kwargs: dict,
    base: Plan,
    gains: str = "selective",
    output_gain: float | None = None,
    roles: dict[str, str] | None = None,
) -> Plan:
    """The `selective` plan (module docstring) from the `absorb` plan `base`."""
    if gains not in GAINS:
        raise ValueError(f"gains {gains!r} not in {GAINS}")
    roles = dict(roles or {})
    unknown = sorted(set(roles) - set(ROLES))
    if unknown:
        raise ValueError(f"roles {unknown} not in {ROLES}")
    return _selective(model, inputs, kwargs, base, gains, output_gain, roles)
