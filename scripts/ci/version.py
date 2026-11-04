"""Read, stamp and bump the version: `[project] version` in pyproject.toml is the only source.

kogine run scripts/ci/version.py                                  # read
kogine run scripts/ci/version.py --set ACTION=nightly             # X.Y.(Z+1).devYYYYmmddHHMMSS
kogine run scripts/ci/version.py --set ACTION=bump --set VALUE=minor  # patch | minor | major
kogine run scripts/ci/version.py --set ACTION=write --set VALUE=<version>
"""

import re
from datetime import UTC, datetime
from pathlib import Path

ACTION = "read"  # "read" | "nightly" | "bump" | "write"
VALUE = ""  # bump: "patch" | "minor" | "major"; write: the version

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
PATTERN = re.compile(r'(?m)^(version\s*=\s*")([^"]+)(")')
SRC_INIT = Path(__file__).resolve().parents[2] / "src" / "dragonscale" / "__init__.py"
INIT_PATTERN = re.compile(r'(?m)^(__version__\s*=\s*")([^"]+)(")')


def read() -> str:
    match = PATTERN.search(PYPROJECT.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit("no version in pyproject.toml")
    return match.group(2)


def write(version: str) -> None:
    """Stamp `version` into pyproject.toml and `dragonscale.__version__`."""
    for path, pattern in ((PYPROJECT, PATTERN), (SRC_INIT, INIT_PATTERN)):
        text, count = pattern.subn(rf"\g<1>{version}\g<3>", path.read_text(encoding="utf-8"), 1)
        if count != 1:
            raise SystemExit(f"no version in {path.name}")
        path.write_text(text, encoding="utf-8")


def parts(version: str) -> tuple[int, int, int]:
    core = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not core:
        raise SystemExit(f"cannot parse version {version!r}")
    major, minor, patch = (int(g) for g in core.groups())
    return major, minor, patch


def bump(version: str, part: str) -> str:
    major, minor, patch = parts(version)
    match part:
        case "major":
            return f"{major + 1}.0.0"
        case "minor":
            return f"{major}.{minor + 1}.0"
        case _:
            return f"{major}.{minor}.{patch + 1}"


def nightly(version: str) -> str:
    major, minor, patch = parts(version)
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    return f"{major}.{minor}.{patch + 1}.dev{stamp}"


def main() -> None:
    current = read()
    match ACTION:
        case "read":
            print(current)
        case "nightly":
            print(nightly(current))
        case "bump":
            print(bump(current, VALUE or "patch"))
        case "write":
            if not VALUE:
                raise SystemExit("write needs VALUE")
            write(VALUE)
            print(VALUE)
        case _:
            raise SystemExit(f"ACTION {ACTION!r} not in read / nightly / bump / write")


if __name__ == "__main__":
    main()
