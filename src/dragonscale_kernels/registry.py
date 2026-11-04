"""Op registry: backends register implementations, `resolve` returns one plain callable.

Backends load on the first `resolve`; a backend whose import fails is marked unavailable.
"""

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from dragonscale_common.log import get_logger
from dragonscale_kernels.errors import NoImplementationError
from dragonscale_kernels.hardware import arch_of

BACKENDS = ("triton", "tilelang", "compile", "eager")
log = get_logger("kernels")


@dataclass
class Impl:
    """One backend's implementation of one op; `supports(arch)` is the reason it cannot serve."""

    fn: Callable
    supports: Callable[[str], str | None] | None = None


@dataclass
class State:
    """Everything registered in this process: op -> backend -> implementations, newest first."""

    impls: dict[str, dict[str, list[Impl]]] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)
    loaded: bool = False


STATE = State()


def register(
    op: str, backend: str, fn: Callable, supports: Callable[[str], str | None] | None = None
) -> None:
    """Add `fn` to `backend`'s implementations of `op`."""
    if backend not in BACKENDS:
        raise ValueError(f"backend {backend!r} not in {BACKENDS}")
    STATE.impls.setdefault(op, {}).setdefault(backend, []).insert(0, Impl(fn, supports))


def load_backends() -> None:
    """Import every backend package once; an ImportError marks that backend unavailable."""
    if STATE.loaded:
        return
    STATE.loaded = True
    for name in BACKENDS:
        try:
            importlib.import_module(f"dragonscale_kernels.backends.{name}")
        except ImportError as err:
            STATE.unavailable[name] = str(err)


def _pick(op: str, backend: str, arch: str) -> tuple[Impl | None, str]:
    """The first of `backend`'s `op` implementations serving `arch`, or why there is none."""
    if backend in STATE.unavailable:
        return None, f"unavailable ({STATE.unavailable[backend]})"
    reasons = []
    for impl in STATE.impls.get(op, {}).get(backend, ()):
        reason = None if impl.supports is None else impl.supports(arch)
        if reason is None:
            return impl, ""
        reasons.append(reason)
    return None, "; ".join(reasons) or "not implemented"


def resolve(op: str, device: torch.device | str, backend: str | None = None) -> Callable:
    """The implementation of `op` on `device`; a pinned `backend` is the only option."""
    load_backends()
    arch = arch_of(device)
    reasons = []
    for name in (backend,) if backend else BACKENDS:
        impl, why = _pick(op, name, arch)
        if impl is not None:
            log.debug("kernel bound", op=op, arch=arch, backend=name)
            return impl.fn
        reasons.append(f"{name}: {why}")
    raise NoImplementationError(f"{op} on {arch}: " + "; ".join(reasons))


def available(op: str, device: torch.device | str) -> list[str]:
    """Every backend that can serve `op` on `device`, in priority order."""
    load_backends()
    arch = arch_of(device)
    return [name for name in BACKENDS if _pick(op, name, arch)[0] is not None]
