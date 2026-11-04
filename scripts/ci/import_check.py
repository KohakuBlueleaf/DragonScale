"""Import every module of the three packages, each in its own interpreter: inside one process the
first successful import hides a cycle in everything imported after it. A backend package is
skipped when its library is not installed; nothing imports it then.

Usage:
    python scripts/ci/import_check.py
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PACKAGES = ("dragonscale", "dragonscale_common", "dragonscale_kernels")
BACKENDS = {  # backend package: the library it needs
    "dragonscale_kernels.backends.triton": "triton",
    "dragonscale_kernels.backends.compile": "triton",
    "dragonscale_kernels.backends.tilelang": "tilelang",
}


def module_names():
    for package in PACKAGES:
        for path in sorted((SRC / package).rglob("*.py")):
            parts = list(path.relative_to(SRC).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            yield ".".join(parts)


def skip_reason(name: str) -> str | None:
    for root, library in BACKENDS.items():
        if (name == root or name.startswith(f"{root}.")) and importlib.util.find_spec(
            library
        ) is None:
            return f"{library} not installed"
    return None


def main() -> int:
    failed, skipped, checked = [], 0, 0
    for name in module_names():
        reason = skip_reason(name)
        if reason:
            skipped += 1
            print(f"skip {name} ({reason})")
            continue
        checked += 1
        proc = subprocess.run(
            [sys.executable, "-c", f"import {name}"], cwd=ROOT, capture_output=True, text=True
        )
        if proc.returncode != 0:
            failed.append(name)
            print(f"FAIL {name}\n{proc.stderr}")
        else:
            print(f"ok   {name}")
    print(f"\n{checked - len(failed)}/{checked} modules import cleanly, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
