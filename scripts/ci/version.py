"""Read, stamp and bump the version. `[project] version` in pyproject.toml is the only source;
`write` also stamps `dragonscale.__version__`.

Usage:
    python scripts/ci/version.py read
    python scripts/ci/version.py nightly            # X.Y.(Z+1).devYYYYmmddHHMMSS
    python scripts/ci/version.py bump patch|minor|major
    python scripts/ci/version.py write <version>
"""

import argparse
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
SRC_INIT = ROOT / "src" / "dragonscale" / "__init__.py"
# Anchored to the start of a line, so a dependency pin containing "version = " cannot match
PATTERN = re.compile(r'(?m)^(version\s*=\s*")([^"]+)(")')
INIT_PATTERN = re.compile(r'(?m)^(__version__\s*=\s*")([^"]+)(")')


def read() -> str:
    match = PATTERN.search(PYPROJECT.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit("no version in pyproject.toml")
    return match.group(2)


def write(version: str) -> None:
    for path, pattern in ((PYPROJECT, PATTERN), (SRC_INIT, INIT_PATTERN)):
        text, count = pattern.subn(rf"\g<1>{version}\g<3>", path.read_text(encoding="utf-8"), 1)
        if count != 1:
            raise SystemExit(f"no version in {path.name}")
        path.write_text(text, encoding="utf-8")


def base(version: str) -> tuple[int, int, int]:
    core = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not core:
        raise SystemExit(f"cannot parse version {version!r}")
    major, minor, patch = (int(g) for g in core.groups())
    return major, minor, patch


def bump(version: str, part: str) -> str:
    major, minor, patch = base(version)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def nightly(version: str) -> str:
    """The PEP 440 dev release of the next patch: X.Y.(Z+1).devYYYYmmddHHMMSS, no local segment,
    so PyPI accepts it and `pip install --pre` finds it."""
    major, minor, patch = base(version)
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    return f"{major}.{minor}.{patch + 1}.dev{stamp}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("read", "nightly", "bump", "write"))
    parser.add_argument("value", nargs="?")
    args = parser.parse_args()
    current = read()
    if args.action == "read":
        print(current)
    elif args.action == "nightly":
        print(nightly(current))
    elif args.action == "bump":
        if args.value not in (None, "patch", "minor", "major"):
            raise SystemExit(f"bump {args.value!r} not in patch / minor / major")
        print(bump(current, args.value or "patch"))
    else:
        if not args.value:
            raise SystemExit("write needs a version")
        write(args.value)
        print(args.value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
