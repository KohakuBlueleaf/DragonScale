"""Compile backend: `torch.compile` of each eager op on CUDA."""

import importlib.util

import torch

from dragonscale_kernels import ops
from dragonscale_kernels.backends.eager import IMPLS
from dragonscale_kernels.constraints import requires
from dragonscale_kernels.registry import register

if importlib.util.find_spec("triton") is None:
    raise ImportError("torch.compile on CUDA needs triton")

IN_PLACE = (ops.RESCALE_ROWS, ops.PROJECT_RESCALE_ROWS)
CUDA = requires(("cuda",))

for _op, _fn in IMPLS.items():
    if _op not in IN_PLACE:
        register(_op, "compile", torch.compile(_fn, dynamic=True), CUDA)
