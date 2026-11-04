"""Helpers shared by the measurement scripts: output folders, JSONL rows, text tables, figures,
run bookkeeping, weight statistics, and a byte corpus from the Python standard library."""

import json
import sysconfig
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
DATA = ROOT / "data"
ASSETS = ROOT / "docs" / "assets"


def run_dir(name: str) -> Path:
    path = RUNS / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def append_row(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row) + "\n")


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def table(rows: list[dict], columns: list[str]) -> str:
    """A fixed-width text table; floats in 4 significant digits."""

    def cell(value) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        return "-" if value is None else str(value)

    cells = [[cell(row.get(c)) for c in columns] for row in rows]
    widths = [max(len(c), *(len(r[i]) for r in cells)) for i, c in enumerate(columns)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(columns, widths, strict=True))]
    lines.append("  ".join("-" * w for w in widths))
    lines += ["  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)) for r in cells]
    return "\n".join(lines)


def figure_path(name: str) -> Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    return ASSETS / f"{name}.png"


def finished(path: Path) -> bool:
    """Whether a run's JSONL output ends with its `done` row."""
    if not path.exists():
        return False
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return bool(lines) and '"done"' in lines[-1]


def as_floats(row: dict) -> dict:
    """Every tensor value of `row` as a float, in one device transfer."""
    keys = [k for k, v in row.items() if isinstance(v, torch.Tensor)]
    values = torch.stack([row[k].float().reshape(()) for k in keys]).tolist() if keys else []
    return {**row, **dict(zip(keys, values, strict=True))}


@torch.no_grad()
def weight_stats(model: torch.nn.Module, session) -> dict:
    """Per linear / conv / embedding: raw row-norm mean and max; per norm with an affine: its
    weight's max magnitude; per patched layer: gain mean and max."""
    from torch import nn

    import dragonscale as ds
    from dragonscale.functional.rows import row_norms

    out = {}
    for path, module in model.named_modules():
        if isinstance(module, nn.Linear | nn.Embedding | nn.Conv2d):
            norms = row_norms(module.weight.float(), ds.RowSpec())
            out[f"{path}/raw_norm_mean"] = norms.mean()
            out[f"{path}/raw_norm_max"] = norms.max()
        elif isinstance(module, nn.RMSNorm | nn.LayerNorm) and module.weight is not None:
            out[f"{path}/weight_max"] = module.weight.float().abs().max()
    for p in session.untied_layers() if session is not None else []:
        if p.gain is not None:
            gain = p.gain.detach().float().abs()
            out[f"{p.plan.path}/gain_mean"] = gain.mean()
            out[f"{p.plan.path}/gain_max"] = gain.max()
    return out


def stdlib_bytes(limit: int) -> torch.Tensor:
    """The first `limit` bytes of the standard library's .py files, sorted by path."""
    root = Path(sysconfig.get_paths()["stdlib"])
    chunks, total = [], 0
    for path in sorted(root.rglob("*.py")):
        chunks.append(path.read_bytes())
        total += len(chunks[-1])
        if total >= limit:
            break
    return torch.frombuffer(bytearray(b"".join(chunks)[:limit]), dtype=torch.uint8)
