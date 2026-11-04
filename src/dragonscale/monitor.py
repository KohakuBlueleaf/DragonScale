"""`Monitor`: per-layer scale readings of a session, read with one transfer per device.

    monitor = ds.monitor.Monitor(session)
    monitor.start()           # the next forward records every layer's output magnitude
    model(x)
    stats = monitor.read()    # {"<path>/out_rms": ..., "<path>/gain_max": ...}

| key | value |
| --- | --- |
| `out_rms`, `out_max` | the layer's output on the last forward after `start` |
| `gain_max`, `gain_median` | the absolute gain |
| `raw_norm_max`, `raw_norm_min` | the raw weight's row norms |
"""

from collections import defaultdict

import torch
from torch import Tensor, nn

from dragonscale.functional.rows import row_norms
from dragonscale.patch.session import Session


class Monitor:
    """Readings of every patched layer of `session`; forward hooks exist only between `start`
    and `read`."""

    def __init__(self, session: Session):
        self.session = session
        self.handles = []
        self.outputs: dict[str, Tensor] = {}

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
            path = patched.plan.path
            norms = row_norms(patched.raw_weight().detach(), patched.plan.spec)
            add(f"{path}/raw_norm_max", norms.max())
            add(f"{path}/raw_norm_min", norms.min())
            if patched.gain is not None:
                gain = patched.gain.detach().abs()
                add(f"{path}/gain_max", gain.max())
                add(f"{path}/gain_median", gain.median())
        self.outputs.clear()

        # One transfer per device
        out: dict[str, float] = {}
        for device, names in keys.items():
            out.update(zip(names, torch.stack(values[device]).tolist(), strict=True))
        return out
