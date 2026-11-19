"""`trace_degrees`: run one forward under a dispatch mode and record, per aten op, the degree of
every output tensor and the constraints the op puts on a log-scale direction over parameters.

Every parameter is one group (tied ones once); inputs carry degree 0 and mark the tensors that
depend on data. Unit-row weights are computed with plain torch ops during the trace
(`plain_unit_rows`), whatever the kernel backend, so that every op is seen.
"""

from collections import Counter
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from dragonscale.analysis.rules import (
    ATTENTION,
    ZEROS,
    Anchor,
    Call,
    nonlinear,
    passthrough,
    rule_of,
    zero_like,
)
from dragonscale.functional.rows import plain_unit_rows
from dragonscale.layer_types import layer_type_of

ATTENTION_OPS = {f"aten::{name}" for name in ATTENTION}


@dataclass
class DegreeTrace:
    """What one forward gave: groups, constraints, anchors (constraints set by a fixed-scale
    addend), the degrees of data-dependent tensors, the degree of each module's first input
    (`module_inputs`, by path, in first-call order), the q and k degrees of every attention
    call (`attention`), and the modules whose first input depends on no parameter (`entries`:
    an embedding lookup, a patch embedding of the input), and the modules with a layer type whose
    output reaches a model output through view and layout ops only (`outputs`). A source is the module path and op that set the constraint.
    """

    names: list[str]
    shapes: list[tuple[int, ...]]
    constraints: list[Tensor] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    anchors: list[Tensor] = field(default_factory=list)
    anchor_sources: list[str] = field(default_factory=list)
    activations: list[Tensor] = field(default_factory=list)
    unknown: Counter = field(default_factory=Counter)
    module_inputs: dict[str, Tensor] = field(default_factory=dict)
    attention: list[tuple[Tensor, Tensor]] = field(default_factory=list)
    entries: set[str] = field(default_factory=set)
    outputs: set[str] = field(default_factory=set)

    def matrix(self) -> Tensor:
        """The stacked constraints, `[constraints, groups]` (one zero row when there is none)."""
        k = len(self.names)
        return torch.stack(self.constraints) if self.constraints else torch.zeros(1, k).double()

    def anchor_matrix(self) -> Tensor:
        """The stacked anchors, `[anchors, groups]` (one zero row when there is none)."""
        k = len(self.names)
        return torch.stack(self.anchors) if self.anchors else torch.zeros(1, k).double()

    def degrees(self) -> Tensor:
        """The stacked degrees of every data-dependent tensor, `[tensors, groups]`."""
        k = len(self.names)
        return torch.stack(self.activations) if self.activations else torch.zeros(1, k).double()


class _DegreeRecorder(TorchDispatchMode):
    """Propagates degree vectors through every aten op of one forward."""

    def __init__(self, trace: DegreeTrace, params: list[Tensor]):
        super().__init__()
        self.trace = trace
        self.zero = zero_like(len(params))
        self.degree: dict[int, Tensor] = {}
        self.data: set[int] = set()
        self.touched: set[int] = {id(p) for p in params}
        self.zeros: set[int] = set()  # tensors known to be exactly zero: any degree fits them
        self.source: dict[int, str] = {}  # tensor -> the layer whose output it is a view of
        self.alive: list[Tensor] = []
        self.paths: list[str] = []
        for k, param in enumerate(params):
            self.degree[id(param)] = self.zero.clone().index_fill_(0, torch.tensor(k), 1.0)
            self.alive.append(param)

    def mark_input(self, tensor: Tensor) -> None:
        self.data.add(id(tensor))
        self.alive.append(tensor)

    def degree_of(self, tensor: Tensor) -> Tensor:
        return self.degree.get(id(tensor), self.zero)

    def constrain(self, vector: Tensor | Anchor, op: str) -> None:
        anchor = isinstance(vector, Anchor)
        vector = vector.vector if anchor else vector
        if vector.abs().max().item() <= 1e-12:
            return
        source = f"{(self.paths[-1] or 'model') if self.paths else 'model'}: {op}"
        if anchor:
            self.trace.anchors.append(vector)
            self.trace.anchor_sources.append(source)
        else:
            self.trace.constraints.append(vector)
            self.trace.sources.append(source)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = func._schema.name
        inputs = [t for t in tree_leaves((args, kwargs)) if isinstance(t, Tensor)]
        outputs = [t for t in tree_leaves(out) if isinstance(t, Tensor)]
        if not outputs:
            return out

        # Degrees of the outputs and the op's constraints
        call = Call(name, tuple(args), kwargs, self.zero, self.degree_of, self.zeros)
        if name in ATTENTION_OPS:
            self.trace.attention.append((call.deg(args[0]), call.deg(args[1])))
        rule = rule_of(name)
        if rule is None:
            self.trace.unknown[name] += 1
            rule = nonlinear
        degrees, constraints = rule(call)

        # Exact zeros and layer outputs pass through views and layout ops
        first = next((t for t in inputs), None)
        if name.removeprefix("aten::").rstrip("_") in ZEROS or (
            rule is passthrough and first is not None and id(first) in self.zeros
        ):
            self.zeros.update(id(t) for t in outputs)
        if rule is passthrough and first is not None and id(first) in self.source:
            for t in outputs:
                self.source[id(t)] = self.source[id(first)]
        for vector in constraints:
            self.constrain(vector, name)
        degrees = degrees if isinstance(degrees, list) else [degrees] * len(outputs)

        # Record them; data and parameter dependence flow from any input
        dependent = any(id(t) in self.data for t in inputs)
        touched = any(id(t) in self.touched for t in inputs)
        for tensor, degree in zip(outputs, degrees, strict=False):
            self.alive.append(tensor)
            if dependent:
                self.data.add(id(tensor))
            if touched:
                self.touched.add(id(tensor))
            if not tensor.is_floating_point():
                continue
            self.degree[id(tensor)] = degree
            if dependent:
                self.trace.activations.append(degree)
        return out


def parameter_groups(model: nn.Module) -> tuple[list[str], list[Tensor]]:
    """Every parameter once (tied ones under their first name)."""
    names, params = [], []
    for name, param in model.named_parameters():
        names.append(name)
        params.append(param)
    return names, params


def trace_degrees(model: nn.Module, *inputs, **kwargs) -> DegreeTrace:
    """Run `model(*inputs, **kwargs)` once and record its degrees and constraints."""
    names, params = parameter_groups(model)
    result = DegreeTrace(names, [tuple(p.shape) for p in params])
    recorder = _DegreeRecorder(result, params)
    for value in tree_leaves((inputs, kwargs)):
        if isinstance(value, Tensor):
            recorder.mark_input(value)

    # The degree of each module's first input, at its first call; the module being run
    def enter(path: str):
        def hook(module: nn.Module, args) -> None:
            recorder.paths.append(path)
            if args and isinstance(args[0], Tensor) and path not in result.module_inputs:
                result.module_inputs[path] = recorder.degree_of(args[0])
                if id(args[0]) not in recorder.touched:
                    result.entries.add(path)

        return hook

    def leave(module: nn.Module, args, out) -> None:
        path = recorder.paths.pop()
        if layer_type_of(module) is not None:
            for t in tree_leaves(out):
                if isinstance(t, Tensor):
                    recorder.source[id(t)] = path

    handles = []
    for path, module in model.named_modules():
        handles.append(module.register_forward_pre_hook(enter(path)))
        handles.append(module.register_forward_hook(leave))
    try:
        with torch.enable_grad(), plain_unit_rows(), recorder:
            out = model(*inputs, **kwargs)
    finally:
        for handle in handles:
            handle.remove()

    # Every output's degree is constrained to 0
    for tensor in tree_leaves(out):
        if isinstance(tensor, Tensor) and tensor.is_floating_point():
            recorder.constrain(recorder.degree_of(tensor), "output")
            if id(tensor) in recorder.source:
                result.outputs.add(recorder.source[id(tensor)])
    return result
