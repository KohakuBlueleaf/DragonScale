"""Problems are collected first and raised once, all together."""


class ProblemsError(ValueError):
    """Every problem found in one validation, one per line."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        lines = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"{len(self.problems)} problem(s):\n{lines}")


def raise_problems(problems: list[str]) -> None:
    """Raise `ProblemsError` listing every problem; nothing when there are none."""
    if problems:
        raise ProblemsError(problems)
