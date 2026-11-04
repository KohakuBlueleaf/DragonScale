"""Stock layers with `weight` read through unit rows: `mode` plain, unit or unit_gain.

The raw `weight` stays the trained parameter; `gain` (unit_gain only) holds the row scale.
`init="preserve"` starts the gain at the row norms, so the layer starts as its stock version;
`init="ones"` starts it at 1.
"""

import torch
from torch import Tensor, nn

from dragonscale.functional import layers as F
from dragonscale.functional.rows import preserving_gain, unit_row_op
from dragonscale.spec import defaults
from dragonscale.spec.rows import RowSpec


def _plain_weight(module: nn.Module) -> Tensor:
    return module.weight


def _unit_weight(module: nn.Module) -> Tensor:
    op = module._ops.get(module.weight.device)
    if op is None:
        op = module._ops[module.weight.device] = unit_row_op(
            module.spec, module.weight.shape, module.weight.device
        )
    return op(module.weight, module.gain)


WEIGHT_FNS = {"plain": _plain_weight, "unit": _unit_weight, "unit_gain": _unit_weight}


class UnitRows:
    """Mixin over a stock layer: `unit_weight()` is the weight its forward uses."""

    def setup_unit(
        self, mode: str, spec: RowSpec | None, gain: str = "channel", init: str = "preserve"
    ) -> None:
        """Pick the weight function once; register `gain` (None unless unit_gain)."""
        self.spec = spec or self.default_spec()
        problems = self.spec.problems(self.weight.shape)
        if problems:
            raise ValueError("; ".join(problems))

        self.mode = mode
        self._ops: dict[torch.device, object] = {}
        self._weight_fn = WEIGHT_FNS[mode]
        if mode != "unit_gain":
            self.register_parameter("gain", None)
            return
        rows = self.spec.rows(self.weight.shape) if gain == "channel" else 1
        start = torch.ones(rows, device=self.weight.device, dtype=self.weight.dtype)
        if init == "preserve":
            start = preserving_gain(self.weight, self.spec)
            start = start if gain == "channel" else start.mean().reshape(1)
        self.gain = nn.Parameter(start.to(self.weight.dtype))

    def default_spec(self) -> RowSpec:
        return defaults.rows()

    def unit_weight(self) -> Tensor:
        """The weight the forward uses."""
        return self._weight_fn(self)


class UnitRowLinear(UnitRows, nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        mode: str = "unit_gain",
        spec: RowSpec | None = None,
        gain: str = "channel",
        init: str = "preserve",
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.setup_unit(mode, spec, gain, init)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(self, x, self.unit_weight(), self.bias)


class _UnitConv(UnitRows):
    def __init__(
        self,
        *args,
        mode: str = "unit_gain",
        spec: RowSpec | None = None,
        gain: str = "channel",
        init: str = "preserve",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.setup_unit(mode, spec, gain, init)

    def forward(self, x: Tensor) -> Tensor:
        return F.conv(self, x, self.unit_weight(), self.bias)


class UnitRowConv1d(_UnitConv, nn.Conv1d):
    pass


class UnitRowConv2d(_UnitConv, nn.Conv2d):
    pass


class UnitRowConv3d(_UnitConv, nn.Conv3d):
    pass


class _UnitConvTranspose(_UnitConv):
    def default_spec(self) -> RowSpec:
        return defaults.conv_transpose(self.weight.shape, self.groups)

    def forward(self, x: Tensor, output_size: list[int] | None = None) -> Tensor:
        return F.conv_transpose(self, x, self.unit_weight(), self.bias, output_size)


class UnitRowConvTranspose1d(_UnitConvTranspose, nn.ConvTranspose1d):
    pass


class UnitRowConvTranspose2d(_UnitConvTranspose, nn.ConvTranspose2d):
    pass


class UnitRowConvTranspose3d(_UnitConvTranspose, nn.ConvTranspose3d):
    pass


class UnitRowEmbedding(UnitRows, nn.Embedding):
    """Token rows of unit RMS by default (`mode="unit"`, radius sqrt_fan_in)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *args,
        mode: str = "unit",
        spec: RowSpec | None = None,
        gain: str = "channel",
        init: str = "preserve",
        **kwargs,
    ):
        super().__init__(num_embeddings, embedding_dim, *args, **kwargs)
        self.setup_unit(mode, spec, gain, init)

    def default_spec(self) -> RowSpec:
        return defaults.embedding()

    def forward(self, x: Tensor) -> Tensor:
        return F.embedding(self, x, self.unit_weight())
