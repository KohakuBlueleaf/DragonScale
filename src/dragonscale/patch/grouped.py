"""Grouped unit rows: before a model's forward, every eligible patched layer's weight is computed
in one autograd node (a few kernel launches), and each layer reads its weight from that node.

A layer is eligible when it normalizes, is not tied, has rows first over a contiguous plain
weight with storage, has a channel gain or none, and its backend is Triton or unset with Triton
available. The rest compute their weight per layer as before. Grouping runs, and a stashed weight
is read, only with gradients on, no saved-tensor hook active (checkpointing, offloading) and
outside `torch.compile`. A layer the forward does not use gets no gradient.
"""

import torch
from torch import Tensor, nn

from dragonscale.plan.validate import MARKER
from dragonscale_kernels import available, ops, resolve

STASH = "_ds_grouped"
GROUPED_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
ONE_PASS_BYTES = 64 * 1024  # the grouped kernel reads each row in one tile


def triton_width(cols: int) -> int:
    return 1 << max(cols - 1, 0).bit_length()


class GroupedUnitRowsFn(torch.autograd.Function):
    """Unit rows of `n` weights as one node. `meta` is (n, gain present per weight, radius per
    weight, rows per weight); the inputs are the raws, then the present gains."""

    @staticmethod
    def forward(ctx, meta, *tensors):
        ctx.set_materialize_grads(False)
        n, has_gain, radii, rows = meta
        raws, rest = list(tensors[:n]), iter(tensors[n:])
        gains = [next(rest) if flag else None for flag in has_gain]
        ws = [raw.reshape(r, -1) for raw, r in zip(raws, rows, strict=True)]
        device = raws[0].device
        outs, norms = resolve(ops.UNIT_ROWS_GROUPED, device, "triton")(ws, gains, radii)
        ctx.meta = meta
        ctx.save_for_backward(*raws, *norms, *(g for g in gains if g is not None))
        return tuple(out.reshape(raw.shape) for out, raw in zip(outs, raws, strict=True))

    @staticmethod
    def backward(ctx, *grads):
        n, has_gain, radii, rows = ctx.meta
        saved = ctx.saved_tensors
        raws, norms, rest = saved[:n], saved[n : 2 * n], iter(saved[2 * n :])
        gains = [next(rest) if flag else None for flag in has_gain]
        used = [i for i, g in enumerate(grads) if g is not None]
        grad_ws: list[Tensor | None] = [None] * n
        grad_gains: list[Tensor | None] = [None] * n
        if used:
            got_w, got_gain = resolve(ops.UNIT_ROWS_GROUPED_BWD, raws[0].device, "triton")(
                [grads[i].reshape(rows[i], -1) for i in used],
                [raws[i].reshape(rows[i], -1) for i in used],
                [norms[i] for i in used],
                [gains[i] for i in used],
                [radii[i] for i in used],
            )
            for i, grad_w, grad_gain in zip(used, got_w, got_gain, strict=True):
                grad_ws[i], grad_gains[i] = grad_w.reshape(raws[i].shape), grad_gain
        return None, *grad_ws, *(grad_gains[i] for i in range(n) if has_gain[i])


def _eligible(patched) -> tuple[Tensor, Tensor | None, int, float] | None:
    """(raw, gain, rows, radius) when `patched` can join the group, else None."""
    plan = patched.plan
    if plan.method != "normalize" or patched.tied_to is not None or plan.install == "bake":
        return None
    if tuple(plan.spec.row_dims) != (0,) or patched.backend not in (None, "triton"):
        return None
    raw, gain = patched.raw_weight(), patched.gain
    if type(raw.data) is not Tensor or not raw.is_cuda or not raw.is_contiguous():
        return None
    if raw.untyped_storage().nbytes() < raw.numel() * raw.element_size():
        return None
    rows = plan.spec.rows(raw.shape)
    if rows == 0 or raw.numel() == 0:
        return None
    if gain is not None and (gain.numel() != rows or not gain.is_contiguous()):
        return None
    cols = raw.numel() // rows
    if raw.dtype not in GROUPED_DTYPES or triton_width(cols) * 4 > ONE_PASS_BYTES:
        return None
    return raw, gain, rows, plan.spec.radius_of(raw.shape)


def plain_autograd() -> bool:
    """Gradients on, no saved-tensor hook active and not compiling: a stash may be used."""
    return (
        torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and torch._C._autograd._top_saved_tensors_default_hooks(False) is None
    )


def _layers(model: nn.Module) -> list:
    return [p for m in model.modules() if (p := vars(m).get(MARKER)) is not None]


def prepare(model: nn.Module, args) -> None:
    """Forward pre-hook: compute every eligible layer's weight in one node and stash it."""
    if torch.compiler.is_compiling() or not plain_autograd():
        return
    group = []
    for patched in _layers(model):
        found = _eligible(patched)
        if found is not None:
            group.append((patched, *found))
    if len(group) < 2:
        return
    device = group[0][1].device
    if any(raw.device != device for _, raw, *_ in group):
        return
    if "triton" not in available(ops.UNIT_ROWS_GROUPED, device):
        return
    raws = [raw for _, raw, _, _, _ in group]
    gains = [gain for _, _, gain, _, _ in group]
    meta = (
        len(group),
        tuple(g is not None for g in gains),
        [radius for *_, radius in group],
        tuple(rows for *_, rows, _ in group),
    )
    outs = GroupedUnitRowsFn.apply(meta, *raws, *(g for g in gains if g is not None))
    for (patched, raw, gain, _, _), out in zip(group, outs, strict=True):
        patched.stash = (raw, raw._version, gain, gain._version if gain is not None else 0, out)
    vars(model)[STASH] = [patched for patched, *_ in group]


def clear(model: nn.Module, args, output) -> None:
    """Forward hook: drop the stashed weights."""
    for patched in vars(model).pop(STASH, ()):
        patched.stash = None


def attach(model: nn.Module) -> None:
    """Group the patched layers of `model` on every forward of `model`."""
    if any(hook is prepare for hook in model._forward_pre_hooks.values()):
        return
    model.register_forward_pre_hook(prepare)
    model.register_forward_hook(clear, always_call=True)


def detach(model: nn.Module) -> None:
    for hooks, fn in ((model._forward_pre_hooks, prepare), (model._forward_hooks, clear)):
        for key in [key for key, hook in hooks.items() if hook is fn]:
            del hooks[key]
