"""Figures and result tables from `runs/`: one figure per experiment into `docs/assets/`, and the
README tables printed as Markdown.

    kogine run measure/plots.py
    kogine run measure/plots.py --set ONLY=a

A run `failed` when it diverged (its output says so, or its held-out loss is not finite at the
end) or its final held-out loss is above `FAILED_LOSS`: runs that train end near 0.9 to 1.2, runs
whose stream blew up stall near 2.4. Failed runs are drawn at the top of the axis with an x.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RUNS, figure_path

ONLY = ""  # "" = every figure; otherwise one of a b c d e jit
FAILED_LOSS = 2.0  # byte LM: a final held-out loss above this is a failed run
DPI = 150

PRESETS = ("stock", "matrices", "arch")
COLORS = {"stock": "#c0392b", "matrices": "#2980b9", "arch": "#27ae60"}


# ---- reading ----


def read_run(path: Path) -> dict:
    """The knobs, the log rows and the end state of one run."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    head = rows[0]
    logs = [r for r in rows if "held_loss" in r]
    done = rows[-1].get("done", "running")
    finite = [r["held_loss"] for r in logs if math.isfinite(r["held_loss"])]
    end = logs[-1]["held_loss"] if logs else math.nan
    diverged = done == "diverged" or not math.isfinite(end)
    return {
        "failed": diverged or end > FAILED_LOSS,
        "knobs": head.get("knobs", {}),
        "logs": logs,
        "done": done,
        "diverged": diverged,
        "best": min(finite) if finite else math.nan,
        "final": end,
    }


def runs_of(experiment: str, group: str = "bytelm") -> list[dict]:
    return [read_run(p) for p in sorted((RUNS / group / experiment).glob("*.jsonl"))]


def peak(run: dict, key: str) -> float:
    """The largest finite value of `key` over the run; inf when it overflowed."""
    values = [r[key] for r in run["logs"] if key in r and r[key] is not None]
    if not values:
        return math.nan
    return math.inf if any(not math.isfinite(v) for v in values) else max(values)


# ---- drawing ----


def draw(ax, xs, ys, flags, preset, style="-", label=None, top=None):
    """A line through the finite points; diverged or overflowed points as an x at `top`."""
    finite = [
        (x, y) for x, y, bad in zip(xs, ys, flags, strict=True) if not bad and math.isfinite(y)
    ]
    if finite:
        ax.plot(
            *zip(*finite, strict=True), style, color=COLORS[preset], marker="o", ms=4, label=label
        )
    bad = [x for x, y, f in zip(xs, ys, flags, strict=True) if f or not math.isfinite(y)]
    if bad and top is not None:
        ax.plot(bad, [top] * len(bad), "x", color=COLORS[preset], ms=8, mew=2)


def save(fig, name: str) -> None:
    path = figure_path(name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def by(runs: list[dict], *keys: str) -> dict:
    out = defaultdict(list)
    for run in runs:
        out[tuple(run["knobs"].get(k) for k in keys)].append(run)
    return out


# ---- A: learning rate x precision ----


def figure_a() -> None:
    runs = runs_of("a_lr_precision")
    groups = by(runs, "PRESET", "PRECISION")
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    for (preset, precision), group in sorted(groups.items()):
        group.sort(key=lambda r: r["knobs"]["LR"])
        lrs = [r["knobs"]["LR"] for r in group]
        flags = [r["failed"] for r in group]
        style = "-" if precision == "fp16" else "--"
        label = f"{preset} {precision}"
        draw(left, lrs, [r["best"] for r in group], flags, preset, style, label, top=2.6)
        stream = [peak(r, "max/stream_max") for r in group]
        draw(right, lrs, stream, flags, preset, style, label, top=3e8)
    left.set(xscale="log", xlabel="learning rate", ylabel="best held-out loss (nats per byte)")
    left.set_ylim(0.9, 2.7)
    right.set(xscale="log", yscale="log", xlabel="learning rate", ylabel="peak residual-stream |x|")
    left.set_title("held-out loss (x: failed)")
    right.set_title("peak residual stream over training")
    left.legend(fontsize=7, ncol=2)
    for ax in (left, right):
        ax.grid(alpha=0.3, which="both")
    save(fig, "a_lr_precision")

    print("\n### A: learning rate x precision\n")
    print("| preset | precision | best loss (lr) | largest lr that trains |")
    print("| --- | --- | --- | --- |")
    for (preset, precision), group in sorted(
        groups.items(), key=lambda kv: (PRESETS.index(kv[0][0]), kv[0][1])
    ):
        ok = [r for r in group if not r["failed"]]
        best = min(ok, key=lambda r: r["best"]) if ok else None
        largest = max((r["knobs"]["LR"] for r in ok), default=None)
        cell = f"{best['best']:.3f} ({best['knobs']['LR']:g})" if best else "-"
        print(f"| {preset} | {precision} | {cell} | {largest:g} |")


# ---- B: growth over time ----


def figure_b() -> None:
    runs = runs_of("b_long")
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    for run in sorted(runs, key=lambda r: (r["knobs"]["LR"], PRESETS.index(r["knobs"]["PRESET"]))):
        preset, lr = run["knobs"]["PRESET"], run["knobs"]["LR"]
        steps = [r["step"] for r in run["logs"] if r["step"] > 0]
        rows = [r for r in run["logs"] if r["step"] > 0]
        style = "-" if lr < 5e-3 else "--"
        label = f"{preset} lr {lr:g}" + (" (failed)" if run["failed"] else "")
        left.plot(
            steps, [r["max/stream_max"] for r in rows], style, color=COLORS[preset], label=label
        )
        raw = [
            max(v for k, v in r.items() if k.startswith("blocks.") and k.endswith("/raw_norm_max"))
            for r in rows
        ]
        right.plot(steps, raw, style, color=COLORS[preset], label=label)
    left.set(xscale="log", yscale="log", xlabel="step", ylabel="peak residual-stream |x|")
    right.set(
        xscale="log", yscale="log", xlabel="step", ylabel="largest raw row norm in the blocks"
    )
    left.set_title("residual stream over 100k steps")
    right.set_title("raw weight rows over 100k steps")
    left.legend(fontsize=7)
    for ax in (left, right):
        ax.grid(alpha=0.3, which="both")
    save(fig, "b_long")


# ---- C / E: depth and width ----


def sweep_figure(experiment: str, knob: str, label: str, name: str) -> None:
    runs = runs_of(experiment)
    lrs = sorted({r["knobs"]["LR"] for r in runs})
    fig, axes = plt.subplots(1, len(lrs), figsize=(4 * len(lrs), 3.6), sharey=True)
    for ax, lr in zip(axes, lrs, strict=True):
        for preset in PRESETS:
            group = sorted(
                (r for r in runs if r["knobs"]["PRESET"] == preset and r["knobs"]["LR"] == lr),
                key=lambda r: r["knobs"][knob],
            )
            xs = [r["knobs"][knob] for r in group]
            draw(
                ax,
                xs,
                [peak(r, "max/stream_max") for r in group],
                [r["failed"] for r in group],
                preset,
                "-",
                preset,
                top=3e5,
            )
        ax.set(xscale="log", yscale="log", xlabel=label, title=f"lr {lr:g}")
        ax.set_xticks(sorted({r["knobs"][knob] for r in runs}))
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("peak residual-stream |x| (x: failed)")
    axes[0].legend(fontsize=8)
    save(fig, name)

    print(f"\n### {name}: best held-out loss (x = failed)\n")
    values = sorted({r["knobs"][knob] for r in runs})
    print("| preset | lr | " + " | ".join(f"{knob.lower()} {v}" for v in values) + " |")
    print("| --- | --- | " + " | ".join("---" for _ in values) + " |")
    for preset in PRESETS:
        for lr in lrs:
            cells = []
            for v in values:
                match = [
                    r
                    for r in runs
                    if r["knobs"]["PRESET"] == preset
                    and r["knobs"]["LR"] == lr
                    and r["knobs"][knob] == v
                ]
                cells.append(
                    "-" if not match else ("x" if match[0]["failed"] else f"{match[0]['best']:.3f}")
                )
            print(f"| {preset} | {lr:g} | " + " | ".join(cells) + " |")


# ---- D: weight decay ----


def table_d() -> None:
    runs = runs_of("d_weight_decay")
    print("\n### D: weight decay on stock (best held-out loss, peak stream; x = diverged)\n")
    lrs = sorted({r["knobs"]["LR"] for r in runs})
    print("| weight decay | " + " | ".join(f"lr {lr:g}" for lr in lrs) + " |")
    print("| --- | " + " | ".join("---" for _ in lrs) + " |")
    for wd in sorted({r["knobs"]["WEIGHT_DECAY"] for r in runs}):
        cells = []
        for lr in lrs:
            run = next(
                (r for r in runs if r["knobs"]["LR"] == lr and r["knobs"]["WEIGHT_DECAY"] == wd),
                None,
            )
            if run is None:
                cells.append("-")
            elif run["failed"]:
                cells.append("x")
            else:
                cells.append(f"{run['best']:.3f}, {peak(run, 'max/stream_max'):.3g}")
        print(f"| {wd:g} | " + " | ".join(cells) + " |")


# ---- JiT ----


def figure_jit() -> None:
    runs = runs_of("jit_tokens", group="jit")
    lrs = sorted({r["knobs"]["LR"] for r in runs})
    keys = (
        ("held_loss", "held-out velocity loss", "linear"),
        ("max/stream_max", "peak residual-stream |x|", "log"),
        ("max/logit_max", "largest attention logit", "log"),
    )
    fig, axes = plt.subplots(
        len(keys), len(lrs), figsize=(4 * len(lrs), 3 * len(keys)), squeeze=False
    )
    for col, lr in enumerate(lrs):
        for run in (r for r in runs if r["knobs"]["LR"] == lr):
            preset, precision = run["knobs"]["PRESET"], run["knobs"]["PRECISION"]
            rows = [r for r in run["logs"] if r["step"] > 0]
            style = "-" if precision == "fp16" else "--"
            for row, (key, _, _) in enumerate(keys):
                axes[row][col].plot(
                    [r["step"] for r in rows],
                    [r.get(key, math.nan) for r in rows],
                    style,
                    color=COLORS[preset],
                    label=f"{preset} {precision}",
                )
        axes[0][col].set_title(f"lr {lr:g}")
        for row, (_, label, scale) in enumerate(keys):
            axes[row][col].set(yscale=scale, xlabel="step")
            axes[row][col].grid(alpha=0.3, which="both")
            if col == 0:
                axes[row][col].set_ylabel(label)
    axes[0][0].legend(fontsize=7)
    save(fig, "jit")


def main() -> None:
    figures = {
        "a": figure_a,
        "b": figure_b,
        "c": lambda: sweep_figure("c_depth", "DEPTH", "depth", "c_depth"),
        "d": table_d,
        "e": lambda: sweep_figure("e_width", "DIM", "width", "e_width"),
        "jit": figure_jit,
    }
    for name, make in figures.items():
        if ONLY in ("", name):
            make()


if __name__ == "__main__":
    main()
