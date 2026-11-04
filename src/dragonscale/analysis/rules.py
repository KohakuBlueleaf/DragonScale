"""Degree rules per aten op: how a log-scale direction over parameter groups moves each output,
and which combinations an op needs fixed (constraints).

A rule is `rule(call) -> (outputs, constraints)`: `call.deg(t)` is the degree vector of a tensor
argument (zero for a scalar or a constant), `outputs` one degree per tensor output (a single value
applies to all), `constraints` the vectors whose dot with the direction must be 0.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

EPS_SCALAR = 1e-3


@dataclass
class Call:
    """One op call as the rules see it: the op name, its arguments and their degrees."""

    name: str
    args: tuple
    kwargs: dict
    zero: Tensor
    degree_of: Callable[[Tensor], Tensor]

    def deg(self, value) -> Tensor:
        return self.degree_of(value) if isinstance(value, Tensor) else self.zero

    def tensors(self) -> list[Tensor]:
        """Every floating tensor among the positional arguments and lists of them."""
        found = []
        for arg in self.args:
            for item in arg if isinstance(arg, list | tuple) else (arg,):
                if isinstance(item, Tensor) and item.is_floating_point():
                    found.append(item)
        return found

    def arg(self, index: int, name: str | None = None, default=None):
        if index < len(self.args):
            return self.args[index]
        return self.kwargs.get(name, default) if name else default


DegreeRule = Callable[[Call], tuple]
RULES: dict[str, DegreeRule] = {}


def register_degree_rule(names: str | tuple[str, ...], rule: DegreeRule) -> None:
    """Add (or replace) the rule of one or more `aten::name` ops."""
    for name in (names,) if isinstance(names, str) else names:
        RULES[name] = rule


def nonlinear(call: Call):
    """Output degree 0; every floating input must keep its scale."""
    return call.zero, [call.deg(t) for t in call.tensors()]


def passthrough(call: Call):
    """The first tensor argument's degree, unchanged."""
    first = next((a for a in call.args if isinstance(a, Tensor)), None)
    return call.deg(first), []


def constant(call: Call):
    return call.zero, []


def product(call: Call):
    """Degrees add: matmul, mul, outer products."""
    return sum((call.deg(a) for a in call.args if isinstance(a, Tensor)), call.zero), []


def additive(call: Call):
    """Both sides must move together; a scalar side is a constant unless it is an epsilon."""
    a, b = call.arg(0), call.arg(1, "other")
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        return call.deg(a), [call.deg(a) - call.deg(b)]
    tensor, scalar = (a, b) if isinstance(a, Tensor) else (b, a)
    small = isinstance(scalar, int | float) and abs(scalar) <= EPS_SCALAR
    return call.deg(tensor), [] if small else [call.deg(tensor)]


def matmul_add(call: Call):
    """`addmm(bias, a, b)` and its relatives: the product's degree, the bias must match it."""
    bias, a, b = call.args[0], call.args[1], call.args[2]
    out = call.deg(a) + call.deg(b)
    return out, [call.deg(bias) - out]


def linear(call: Call):
    x, w, bias = call.arg(0), call.arg(1), call.arg(2, "bias")
    out = call.deg(x) + call.deg(w)
    return out, [] if bias is None else [call.deg(bias) - out]


def convolution(call: Call):
    x, w, bias = call.args[0], call.args[1], call.arg(2, "bias")
    out = call.deg(x) + call.deg(w)
    return out, [] if bias is None else [call.deg(bias) - out]


def divide(call: Call):
    return call.deg(call.arg(0)) - call.deg(call.arg(1, "other")), []


def power(call: Call):
    base, exponent = call.arg(0), call.arg(1, "exponent")
    if isinstance(exponent, int | float) and isinstance(base, Tensor):
        return exponent * call.deg(base), []
    return nonlinear(call)


def scaled(factor: float) -> DegreeRule:
    def rule(call: Call):
        return factor * call.deg(call.arg(0)), []

    return rule


def concatenate(call: Call):
    """Every piece must move together (cat, stack)."""
    pieces = [t for t in call.args[0] if isinstance(t, Tensor)]
    first = call.deg(pieces[0])
    return first, [call.deg(t) - first for t in pieces[1:]]


def select_where(call: Call):
    a, b = call.arg(1), call.arg(2)
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        return call.deg(a), [call.deg(a) - call.deg(b)]
    tensor = a if isinstance(a, Tensor) else b
    return call.deg(tensor), []


def copy_into(call: Call):
    return call.deg(call.args[1]), []


def embedding(call: Call):
    return call.deg(call.args[0]), []


def normalized(
    call: Call, weight_index: int, bias_index: int | None, stats: tuple[int, ...]
) -> tuple:
    """A scale-invariant norm: output degree is the weight's (0 without one), the bias must match
    it; the statistic outputs carry the input's degree times `stats` (mean 1, rstd -1)."""
    x = call.args[0]
    weight = call.arg(weight_index)
    bias = call.arg(bias_index) if bias_index is not None else None
    out = call.deg(weight) if isinstance(weight, Tensor) else call.zero
    constraints = [call.deg(bias) - out] if isinstance(bias, Tensor) else []
    return [out, *(s * call.deg(x) for s in stats)], constraints


def running_norm(call: Call):
    """A batch norm on running statistics: the output carries the input's and the weight's
    degrees, the bias must match."""
    x, weight, bias = call.args[0], call.arg(1), call.arg(2)
    out = call.deg(x) + (call.deg(weight) if isinstance(weight, Tensor) else call.zero)
    return out, [call.deg(bias) - out] if isinstance(bias, Tensor) else []


def batch_norm(call: Call):
    """`native_batch_norm`: scale-invariant in training, on running statistics otherwise."""
    training = call.arg(5, "training", True)
    return normalized(call, 1, 2, ()) if training else running_norm(call)


def attention(call: Call):
    """SDPA: the output carries v's degree; q and k may only trade."""
    q, k, v = call.args[0], call.args[1], call.args[2]
    return [call.deg(v), call.zero], [call.deg(q) + call.deg(k)]


PASS = (
    "view", "_unsafe_view", "reshape", "t", "transpose", "permute", "expand", "clone", "contiguous",
    "slice", "select", "unsqueeze", "squeeze", "flatten", "unflatten", "split", "split_with_sizes",
    "chunk", "unbind", "narrow", "alias", "detach", "_to_copy", "to", "lift_fresh", "index_select",
    "gather", "repeat", "roll", "flip", "neg", "relu", "abs", "leaky_relu", "max_pool2d",
    "max_pool2d_with_indices", "max_pool1d", "max_pool3d_with_indices", "avg_pool1d", "avg_pool2d",
    "avg_pool3d", "_adaptive_avg_pool2d", "_adaptive_avg_pool3d", "adaptive_avg_pool1d",
    "constant_pad_nd", "pad", "upsample_nearest1d", "upsample_nearest2d", "upsample_nearest3d",
    "upsample_bilinear2d", "_upsample_nearest_exact2d", "pixel_shuffle", "pixel_unshuffle",
    "native_dropout", "dropout", "sum", "mean", "amax", "amin", "max", "min", "clamp", "clamp_min",
    "clamp_max", "masked_fill", "index", "tril", "triu", "diagonal", "as_strided", "cumsum",
    "expand_as", "view_as", "squeeze_", "unsqueeze_", "transpose_", "sort", "topk",
)  # fmt: skip
NONLINEAR = (
    "silu", "gelu", "tanh", "sigmoid", "exp", "log", "log1p", "expm1", "sin", "cos", "erf",
    "softplus", "elu", "mish", "hardswish", "hardsigmoid", "hardtanh", "logsigmoid", "_softmax",
    "softmax", "_log_softmax", "log_softmax", "sign", "atan", "atan2", "erfinv", "selu", "celu",
)  # fmt: skip
CONSTANT = (
    "zeros", "ones", "full", "empty", "arange", "zeros_like", "ones_like", "full_like",
    "empty_like", "randn", "rand", "randn_like", "rand_like", "scalar_tensor", "fill_", "zero_",
    "bernoulli_", "new_zeros", "new_ones", "new_empty", "new_full", "eye", "linspace",
)  # fmt: skip
ATTENTION = (
    "_scaled_dot_product_flash_attention", "_scaled_dot_product_flash_attention_for_cpu",
    "_scaled_dot_product_efficient_attention", "_scaled_dot_product_cudnn_attention",
    "_scaled_dot_product_attention_math",
)  # fmt: skip

for _name in PASS:
    register_degree_rule(f"aten::{_name}", passthrough)
for _name in NONLINEAR:
    register_degree_rule(f"aten::{_name}", nonlinear)
for _name in CONSTANT:
    register_degree_rule(f"aten::{_name}", constant)
for _name in ATTENTION:
    register_degree_rule(f"aten::{_name}", attention)
register_degree_rule(
    tuple(f"aten::{n}" for n in ("mul", "mul_", "mm", "bmm", "matmul", "dot", "mv", "outer")),
    product,
)
register_degree_rule(
    tuple(f"aten::{n}" for n in ("add", "add_", "sub", "sub_", "rsub", "lerp", "lerp_")),
    additive,
)
register_degree_rule(("aten::addmm", "aten::addmv", "aten::baddbmm", "aten::addbmm"), matmul_add)
register_degree_rule("aten::linear", linear)
register_degree_rule(
    ("aten::convolution", "aten::_convolution", "aten::conv2d", "aten::conv1d"), convolution
)
register_degree_rule(("aten::div", "aten::div_"), divide)
register_degree_rule(("aten::pow", "aten::pow_"), power)
register_degree_rule("aten::sqrt", scaled(0.5))
register_degree_rule("aten::rsqrt", scaled(-0.5))
register_degree_rule("aten::reciprocal", scaled(-1.0))
register_degree_rule("aten::square", scaled(2.0))
register_degree_rule("aten::var", scaled(2.0))
register_degree_rule("aten::std", scaled(1.0))
register_degree_rule(("aten::cat", "aten::stack"), concatenate)
register_degree_rule("aten::where", select_where)
register_degree_rule("aten::copy_", copy_into)
register_degree_rule("aten::embedding", embedding)
register_degree_rule("aten::native_layer_norm", lambda c: normalized(c, 2, 3, (1, -1)))
register_degree_rule("aten::native_group_norm", lambda c: normalized(c, 1, 2, (1, -1)))
register_degree_rule("aten::native_batch_norm", batch_norm)
register_degree_rule(
    ("aten::_native_batch_norm_legit", "aten::_batch_norm_with_update"),
    lambda c: normalized(c, 1, 2, ()),
)
register_degree_rule(
    ("aten::_native_batch_norm_legit_no_training", "aten::_batch_norm_no_update"), running_norm
)
register_degree_rule(
    ("aten::_fused_rms_norm", "aten::rms_norm"), lambda c: normalized(c, 2, None, (-1,))
)
register_degree_rule("aten::promote_types", constant)


def rule_of(name: str) -> DegreeRule | None:
    return RULES.get(name)


def zero_like(k: int) -> Tensor:
    return torch.zeros(k, dtype=torch.float64)
