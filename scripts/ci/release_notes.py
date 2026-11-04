"""Release notes of a version: its Change.md section, else the commits since the previous tag.

kogine run scripts/ci/release_notes.py --set VERSION=0.1.0
"""

import re
import subprocess
from pathlib import Path

VERSION = ""  # the version whose notes to print

ROOT = Path(__file__).resolve().parents[2]
CHANGELOG = ROOT / "Change.md"


def section(version: str) -> str | None:
    """The body under `## ... update to <version>`, up to the next `## ` heading."""
    if not CHANGELOG.exists():
        return None
    pattern = re.compile(rf"(?ms)^##\s+[^\n]*update to {re.escape(version)}\s*$(.*?)(?=^##\s|\Z)")
    match = pattern.search(CHANGELOG.read_text(encoding="utf-8"))
    return match.group(1).strip() if match else None


def commits() -> str:
    """One line per commit since the previous tag."""
    run = lambda *cmd: subprocess.run(  # noqa: E731
        cmd, capture_output=True, text=True, check=False, cwd=ROOT
    ).stdout
    tags = run("git", "tag", "--sort=-creatordate").split()
    span = f"{tags[1]}..HEAD" if len(tags) > 1 else "HEAD"
    return run("git", "log", "--no-merges", "--pretty=* %s (%h)", span).strip() or "* No changes."


def main() -> None:
    if not VERSION:
        raise SystemExit("VERSION is needed")
    body = section(VERSION) or f"### Changes\n\n{commits()}"
    print(f"## DragonScale {VERSION}\n\n{body}")


if __name__ == "__main__":
    main()
