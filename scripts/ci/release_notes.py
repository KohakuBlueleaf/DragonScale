"""Release notes of a version: its Change.md section (`## <date> update to <version>`), else the
commits since the previous tag, so a release is never published with empty notes.

Usage:
    python scripts/ci/release_notes.py --version 0.1.0
"""

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHANGELOG = ROOT / "Change.md"


def changelog_section(version: str) -> str | None:
    """The body under `## ... update to <version>`, up to the next `## ` heading."""
    if not CHANGELOG.exists():
        return None
    pattern = re.compile(rf"(?ms)^##\s+[^\n]*update to {re.escape(version)}\s*$(.*?)(?=^##\s|\Z)")
    match = pattern.search(CHANGELOG.read_text(encoding="utf-8"))
    return match.group(1).strip() if match else None


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False, cwd=ROOT
    ).stdout


def commits_since_last_tag() -> str:
    tags = git("tag", "--sort=-creatordate").split()
    span = f"{tags[1]}..HEAD" if len(tags) > 1 else "HEAD"
    return git("log", "--no-merges", "--pretty=* %s (%h)", span).strip() or "* No changes."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    body = changelog_section(args.version)
    if body is None:
        body = f"### Changes\n\n{commits_since_last_tag()}"
    print(f"## DragonScale {args.version}\n\n{body}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
