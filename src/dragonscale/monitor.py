"""`Monitor`: per-layer scale readings of a session, read with one transfer per device.

    monitor = ds.monitor.Monitor(session, angles=True)
    for step in ...:
        monitor.start()           # the next forward records every patched layer's output
        loss = loss_fn(model(x))
        ...
        optimizer.step()
        stats = monitor.read()    # {"<path>/out_rms": ..., "<path>/angle_median": ...}

| key | value |
| --- | --- |
| `out_rms`, `out_max` | RMS and largest absolute value of the layer's output on the last forward after `start` |
| `gain_min`, `gain_median`, `gain_max` | of the absolute gain (layers with a gain) |
| `raw_norm_min`, `raw_norm_mean`, `raw_norm_max` | of the raw weight's row norms |
| `angle_median`, `angle_max` | with `angles=True`: the angle in radians each row turned since the previous `read` (absent on the first) |

Angles are computed as `2 asin(|u - u'| / 2)` of the unit rows, which resolves small angles.
"""

from collections import defaultdict

import torch
from torch import Tensor, nn

from dragonscale.functional.rows import row_norms
from dragonscale.kernels.layout import as_rows, wide
from dragonscale.patch.session import Session


def unit_directions(w: Tensor, spec) -> Tensor:
    """The rows of `w` as unit vectors in fp32 or wider, in the row view."""
    view = spec.view_of(w.shape)
    norms = row_norms(w, spec).clamp_min(1e-30)
    return wide(w).reshape(view) / as_rows(norms, view, spec.row_dims)


def row_angles(u: Tensor, previous: Tensor, spec, shape) -> Tensor:
    """Flat `[rows]` angles between two sets of unit rows in the same view."""
    from dragonscale.kernels.layout import reduced

    view = spec.view_of(shape)
    diff = (u - previous).square()
    dims = reduced(view, spec.row_dims)
    chord = (diff.sum(dims) if dims else diff).sqrt().reshape(-1)
    return 2 * torch.asin((chord / 2).clamp(max=1.0))


class Monitor:
    """Readings of every patched layer of `session`; forward hooks exist only between `start`
    and `read`. With `angles`, a copy of every raw weight's unit rows is kept between reads."""

    def __init__(self, session: Session, angles: bool = False):
        self.session = session
        self.angles = angles
        self.handles = []
        self.outputs: dict[str, Tensor] = {}
        self.previous: dict[str, Tensor] = {}

    def start(self) -> None:
        """Record the output magnitude of every patched layer on the next forward."""
        self.stop()
        for patched in self.session.patched:
            hook = self._hook(patched.plan.path)
            self.handles.append(patched.module.register_forward_hook(hook))

    def _hook(self, path: str):
        def record(module: nn.Module, args, out) -> None:
            if isinstance(out, Tensor) and out.is_floating_point():
                flat = out.detach().float()
                self.outputs[path] = torch.stack([flat.square().mean().sqrt(), flat.abs().max()])

        return record

    def stop(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @torch.no_grad()
    def read(self) -> dict[str, float]:
        """Every reading as a flat dict; stops the hooks."""
        self.stop()
        keys: dict[torch.device, list[str]] = defaultdict(list)
        values: dict[torch.device, list[Tensor]] = defaultdict(list)

        def add(key: str, value: Tensor) -> None:
            keys[value.device].append(key)
            values[value.device].append(value.float().reshape(()))

        for path, pair in self.outputs.items():
            add(f"{path}/out_rms", pair[0])
            add(f"{path}/out_max", pair[1])
        for patched in self.session.untied_layers():
            path, spec = patched.plan.path, patched.plan.spec
            raw = patched.raw_weight().detach()
            norms = row_norms(raw, spec)
            add(f"{path}/raw_norm_min", norms.min())
            add(f"{path}/raw_norm_mean", norms.mean())
            add(f"{path}/raw_norm_max", norms.max())
            if patched.gain is not None:
                gain = patched.gain.detach().abs()
                add(f"{path}/gain_min", gain.min())
                add(f"{path}/gain_median", gain.median())
                add(f"{path}/gain_max", gain.max())
            if self.angles:
                u = unit_directions(raw, spec)
                before = self.previous.get(path)
                if before is not None and before.shape == u.shape:
                    angle = row_angles(u, before, spec, raw.shape)
                    add(f"{path}/angle_median", angle.median())
                    add(f"{path}/angle_max", angle.max())
                self.previous[path] = u
        self.outputs.clear()

        # One transfer per device
        out: dict[str, float] = {}
        for device, names in keys.items():
            out.update(zip(names, torch.stack(values[device]).tolist(), strict=True))
        return out
