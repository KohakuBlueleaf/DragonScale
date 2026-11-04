"""`LayerType`: what the patcher does differently per module class, one record per class; and the
records of the stock torch layers and of transformers' `Conv1D`.

A class is given by itself or by its `module.QualName` string, for a package that is not imported.
"""

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn

from dragonscale.functional import layers
from dragonscale.nn import layers as unit
from dragonscale.spec import defaults
from dragonscale.spec.rows import RowSpec


@dataclass(frozen=True)
class LayerType:
    """How one module class is patched.

    `forward(module, x, weight, bias, *args, **kwargs)` runs the layer with the weight given, or
    is None when only the `parametrize` install applies. `out_axis` and `in_axis` are the weight
    axes of the output and input channels (`in_axis` None: no input channel). `unit_class` is the
    `dragonscale.nn` class the module becomes on `Session.canonicalize` (None: it has none).
    """

    name: str
    module: type | str
    weight: str
    spec: Callable[[nn.Module], RowSpec]
    forward: Callable | None
    bias: str | None = "bias"
    out_axis: int = 0
    in_axis: int | None = 1
    unit_class: type | None = None


LAYER_TYPES: dict[str, LayerType] = {}
BY_CLASS: dict[type | str, LayerType] = {}


def qualified(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def register_layer_type(layer_type: LayerType) -> LayerType:
    """Add (or replace) the record of a module class."""
    LAYER_TYPES[layer_type.name] = layer_type
    BY_CLASS[layer_type.module] = layer_type
    return layer_type


def layer_type_of(module: nn.Module) -> LayerType | None:
    """The record of `module`'s class or its nearest registered base class."""
    for cls in type(module).__mro__:
        found = BY_CLASS.get(cls) or BY_CLASS.get(qualified(cls))
        if found is not None:
            return found
    return None


def _rows(module: nn.Module) -> RowSpec:
    return defaults.rows()


def _columns(module: nn.Module) -> RowSpec:
    return defaults.columns()


def _conv_transpose(module: nn.Module) -> RowSpec:
    return defaults.conv_transpose(module.weight.shape, module.groups)


def _embedding(module: nn.Module) -> RowSpec:
    return defaults.embedding()


register_layer_type(
    LayerType("linear", nn.Linear, "weight", _rows, layers.linear, unit_class=unit.UnitRowLinear)
)
CONVS = (
    (1, nn.Conv1d, unit.UnitRowConv1d),
    (2, nn.Conv2d, unit.UnitRowConv2d),
    (3, nn.Conv3d, unit.UnitRowConv3d),
)
for _dims, _cls, _unit in CONVS:
    register_layer_type(
        LayerType(f"conv{_dims}d", _cls, "weight", _rows, layers.conv, unit_class=_unit)
    )
CONV_TRANSPOSES = (
    (1, nn.ConvTranspose1d, unit.UnitRowConvTranspose1d),
    (2, nn.ConvTranspose2d, unit.UnitRowConvTranspose2d),
    (3, nn.ConvTranspose3d, unit.UnitRowConvTranspose3d),
)
for _dims, _cls, _unit in CONV_TRANSPOSES:
    register_layer_type(
        LayerType(
            f"conv_transpose{_dims}d",
            _cls,
            "weight",
            _conv_transpose,
            layers.conv_transpose,
            out_axis=1,
            in_axis=0,
            unit_class=_unit,
        )
    )
register_layer_type(
    LayerType(
        "embedding",
        nn.Embedding,
        "weight",
        _embedding,
        layers.embedding,
        bias=None,
        in_axis=None,
        unit_class=unit.UnitRowEmbedding,
    )
)
register_layer_type(
    LayerType(
        "hf_conv1d",
        "transformers.pytorch_utils.Conv1D",
        "weight",
        _columns,
        layers.transposed_linear,
        out_axis=1,
        in_axis=0,
    )
)
