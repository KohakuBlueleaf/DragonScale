"""Pick one candidate per (kernel, problem key): a stored winner, else time every candidate.

Tables are JSON per card, arch and torch version. Mode `off` binds the first candidate.
"""

import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from dragonscale_common.log import get_logger
from dragonscale_kernels.errors import NoImplementationError
from dragonscale_kernels.hardware import arch_of

CACHE_ENV = "DRAGONSCALE_KERNEL_CACHE"
MODE_ENV = "DRAGONSCALE_TUNING"
MODES = ("on", "off")
ROUNDS = 5
log = get_logger("kernels.tuning")
_STATE = {"mode": os.environ.get(MODE_ENV, "on")}
_CHOSEN: dict[tuple[str, str, str], str] = {}


def set_mode(mode: str) -> None:
    """Set the tuning mode and forget every settled choice."""
    if mode not in MODES:
        raise ValueError(f"tuning mode {mode!r} not in {MODES}")
    _STATE["mode"] = mode
    reset()


def get_mode() -> str:
    return _STATE["mode"]


def reset() -> None:
    """Forget every choice settled in this process."""
    _CHOSEN.clear()


def bucket(size: int) -> int:
    """Smallest power of two at least `size`."""
    return 1 << max(int(size) - 1, 0).bit_length()


def cache_dir() -> Path:
    """Root of the tables: `CACHE_ENV`, else under the user cache."""
    root = os.environ.get(CACHE_ENV)
    if root:
        return Path(root)
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "dragonscale_kernels"


def card_key(device: torch.device | str) -> str:
    """Table folder of `device`: card, arch and torch version."""
    device = torch.device(device)
    arch = arch_of(device)
    card = torch.cuda.get_device_name(device) if device.type == "cuda" else arch
    slug = re.sub(r"[^a-z0-9]+", "-", card.lower()).strip("-")
    return f"{slug}-{arch}-torch{torch.__version__}"


def table_path(kernel: str, device: torch.device | str) -> Path:
    return cache_dir() / "tuning" / card_key(device) / f"{kernel}.json"


def read_table(path: Path) -> dict[str, dict]:
    """The rows at `path`; a missing or unreadable file reads as empty."""
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return rows if isinstance(rows, dict) else {}


def write_row(path: Path, key: str, row: dict) -> None:
    """Merge `row` into the table atomically; a failed write logs and keeps going."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = {**read_table(path), key: row}
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(rows, file, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except OSError as err:
        log.warning("cannot write tuning table", path=path, error=err)


def time_ms(fn: Callable, make_args: Callable[[], tuple], device: torch.device) -> float:
    """Mean wall time of `fn` over `ROUNDS` calls after one warm call, synced per device."""
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda *_: None)
    fn(*make_args())
    sync(device)
    total = 0.0
    for _ in range(ROUNDS):
        args = make_args()
        sync(device)
        start = time.perf_counter()
        fn(*args)
        sync(device)
        total += time.perf_counter() - start
    return total * 1000 / ROUNDS


def choose(
    kernel: str,
    key: str,
    device: torch.device | str,
    candidates: Sequence[tuple[str, Callable]],
    make_args: Callable[[], tuple],
) -> Callable:
    """The fastest of `candidates` (name, fn) for problem `key`; `make_args` builds fresh inputs."""
    if not candidates:
        raise NoImplementationError(f"{kernel}: no candidates")
    if get_mode() == "off":
        return candidates[0][1]
    device = torch.device(device)
    by_name = dict(candidates)
    slot = (kernel, card_key(device), key)

    # A choice settled in this process, then a stored winner
    if slot in _CHOSEN:
        return by_name[_CHOSEN[slot]]
    path = table_path(kernel, device)
    stored = read_table(path).get(key, {}).get("name")
    if stored in by_name:
        _CHOSEN[slot] = stored
        return by_name[stored]

    # Time every candidate; one that raises is skipped with its reason
    times, failures = {}, {}
    for name, fn in candidates:
        try:
            times[name] = time_ms(fn, make_args, device)
        except Exception as err:  # noqa: BLE001
            failures[name] = f"{type(err).__name__}: {err}"
    if not times:
        raise NoImplementationError(f"{kernel} {key}: every candidate failed: {failures}")
    best = min(times, key=times.get)
    write_row(path, key, {"name": best, "ms": times, "failed": failures})
    _CHOSEN[slot] = best
    return by_name[best]
