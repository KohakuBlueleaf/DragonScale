"""Figures and result tables from `runs/`: one figure per experiment into `docs/assets/`, and the
README tables printed as Markdown.

    kogine run measure/plots.py
    kogine run measure/plots.py --set ONLY=pareto

A run `failed` when it diverged (its output says so, or its held-out loss is not finite at the
end) or its best held-out loss stayed above `FAILED_LOSS`: runs that train reach 0.9 to 1.2, runs
whose stream blew up stall near 2.4. Failed runs are drawn as an x in the strip above a panel.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RUNS, figure_path
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

ONLY = ""  # "" = every figure; otherwise one of a b depth width d pareto jit
FAILED_LOSS = 2.0  # byte LM: a run whose best held-out loss stays above this failed
DPI = 160
HEIGHT = 4.4  # inches; every figure is one row of panels, wider than tall

PRESETS = ("stock", "matrices", "arch")
COLORS = {"stock": "#D55E00", "matrices": "#0072B2", "arch": "#009E73"}  # Okabe-Ito
PRECISION_STYLE = {"fp16": "-", "fp32": "--"}
LR_STYLES = ("-", "--", ":")
BYTE_LM = "byte-level LM on the Python standard library"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titlepad": 18,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
    }
)


# ---- reading ----


def read_run(path: Path) -> dict:
    """The knobs, the log rows and the end state of one run."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    logs = [r for r in rows if "held_loss" in r]
    done = rows[-1].get("done", "running")
    finite = [r["held_loss"] for r in logs if math.isfinite(r["held_loss"])]
    end = logs[-1]["held_loss"] if logs else math.nan
    diverged = done == "diverged" or not math.isfinite(end)
    best = min(finite) if finite else math.nan
    return {
        "knobs": rows[0].get("knobs", {}),
        "logs": logs,
        "done": done,
        "diverged": diverged,
        "failed": diverged or not best < FAILED_LOSS,
        "best": best,
        "final": end,
    }


def runs_of(experiment: str, group: str = "bytelm") -> list[dict]:
    return [read_run(p) for p in sorted((RUNS / group / experiment).glob("*.jsonl"))]


def peak(run: dict, key: str = "max/stream_max") -> float:
    """The largest value of `key` over the run; inf when it overflowed."""
    values = [r[key] for r in run["logs"] if r.get(key) is not None]
    if not values:
        return math.nan
    return math.inf if any(not math.isfinite(v) for v in values) else max(values)


def knob(run: dict, name: str):
    return run["knobs"].get(name)


# ---- style ----


def plain(value, _=None) -> str:
    """1, 20, 300, 1024, 5k, 50k, 6M: k and M only for whole thousands and millions."""
    for size, suffix in ((1e6, "M"), (1e3, "k")):
        if abs(value) >= size and value % size == 0:
            return f"{value / size:g}{suffix}"
    return f"{value:g}"


def log_axis(axis, dense: bool = False) -> None:
    """Plain labels on a log axis; `dense` adds 2 and 5 between the powers of ten."""
    axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5) if dense else (1,)))
    axis.set_major_formatter(FuncFormatter(plain))
    axis.set_minor_formatter(NullFormatter())


def better(ax, direction: str) -> None:
    """Mark the way better values lie: "down" on the y axis, "down-left" on both. The arrow goes
    into the axis label when the figure is saved, so it never covers data."""
    ax.better = direction


def label_better(ax) -> None:
    """Append the arrow to the axis labels; the y label is rotated, so its arrow points down."""
    direction = getattr(ax, "better", None)
    if direction is None:
        return
    ax.set_ylabel(f"{ax.get_ylabel()}   \u2190 better")
    if direction == "down-left":
        ax.set_xlabel(f"\u2190 better   {ax.get_xlabel()}")


def failed_strip(ax, xs: list, presets: list) -> None:
    """Failed runs as x marks in a strip just above the panel."""
    for x, preset in zip(xs, presets, strict=True):
        ax.plot(
            x,
            1.04,
            "x",
            color=COLORS[preset],
            ms=7,
            mew=2,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )


def figure(panels: int, width: float):
    fig, axes = plt.subplots(1, panels, figsize=(width, HEIGHT), layout="constrained")
    return fig, list(axes) if panels > 1 else [axes]


def legend_handles(extra: list | None = None) -> list:
    return [Line2D([], [], color=COLORS[p], lw=2, label=p) for p in PRESETS] + (extra or [])


def preset_legend(ax, extra: list | None = None, **kwargs) -> None:
    ax.legend(handles=legend_handles(extra), frameon=False, **kwargs)


def top_legend(fig, extra: list | None = None) -> None:
    """One legend for the whole figure, in a row above the panels."""
    handles = legend_handles(extra)
    fig.top_legend = fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=len(handles),
        frameon=False,
    )


def headline(fig, text: str) -> None:
    """The figure's one-line description: model, task, length and what varies. It becomes the
    title of the top legend when there is one, so the two never overlap."""
    fig.headline = text


def footnote(fig, text: str) -> None:
    fig.text(0.5, -0.04, text, ha="center", fontsize=8, color="0.35")


def save(fig, name: str) -> None:
    for ax in fig.axes:
        label_better(ax)
    text = getattr(fig, "headline", None)
    legend = getattr(fig, "top_legend", None)
    if text and legend is not None:
        legend.set_title(text, prop={"size": 11, "weight": "bold"})
    elif text:
        fig.suptitle(text, fontsize=11, fontweight="bold")
    path = figure_path(name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def line(ax, runs: list[dict], x_of, y_of, style="-") -> None:
    """One preset's runs as a line over x; failed runs in the strip."""
    if not runs:
        return
    runs = sorted(runs, key=x_of)
    ok = [(x_of(r), y_of(r)) for r in runs if not r["failed"] and math.isfinite(y_of(r))]
    preset = knob(runs[0], "PRESET")
    if ok:
        ax.plot(*zip(*ok, strict=True), style, color=COLORS[preset], marker="o", ms=3.5, lw=1.6)
    bad = [r for r in runs if r["failed"]]
    failed_strip(ax, [x_of(r) for r in bad], [knob(r, "PRESET") for r in bad])


# ---- A: learning rate x precision ----


def figure_a() -> None:
    runs = runs_of("a_lr_precision")
    groups = defaultdict(list)
    for run in runs:
        groups[(knob(run, "PRESET"), knob(run, "PRECISION"))].append(run)
    fig, (loss, stream) = figure(2, width=10.5)
    for (_, precision), group in groups.items():
        style = PRECISION_STYLE[precision]
        line(loss, group, lambda r: knob(r, "LR"), lambda r: r["best"], style)
        line(stream, group, lambda r: knob(r, "LR"), peak, style)
    for ax in (loss, stream):
        ax.set_xscale("log")
        log_axis(ax.xaxis)
        ax.set_xlabel("learning rate")
        better(ax, "down")
    loss.set(ylabel="best held-out loss (nats / byte)", title="loss vs learning rate")
    stream.set_yscale("log")
    log_axis(stream.yaxis)
    stream.set(ylabel="peak residual stream |x|", title="stream vs learning rate")
    loss.set_ylim(top=1.22)
    styles = [Line2D([], [], color="0.3", ls=s, label=p) for p, s in PRECISION_STYLE.items()]
    top_legend(fig, styles)
    headline(
        fig,
        f"Learning rate x precision: {BYTE_LM}, 8 blocks, dim 256, 5k steps",
    )
    footnote(fig, "x above a panel: the run failed (diverged, or never got below loss 2)")
    save(fig, "a_lr_precision")

    print("\n### A: learning rate x precision\n")
    print("| preset | precision | best loss (lr) | largest lr that trains |")
    print("| --- | --- | --- | --- |")
    for (preset, precision), group in sorted(
        groups.items(), key=lambda kv: (PRESETS.index(kv[0][0]), kv[0][1])
    ):
        ok = [r for r in group if not r["failed"]]
        best = min(ok, key=lambda r: r["best"]) if ok else None
        largest = max((knob(r, "LR") for r in ok), default=None)
        cell = f"{best['best']:.3f} ({knob(best, 'LR'):g})" if best else "-"
        print(f"| {preset} | {precision} | {cell} | {largest:g} |")


# ---- B: growth over time ----


def b_value(row: dict, key: str) -> float:
    if key == "raw":
        return max(
            v for k, v in row.items() if k.startswith("blocks.") and k.endswith("/raw_norm_max")
        )
    if key == "gain":
        gains = [v for k, v in row.items() if k.endswith((".out/gain_max", ".down/gain_max"))]
        return max(gains) if gains else math.nan
    return row[key]


def figure_b() -> None:
    runs = sorted(runs_of("b_long"), key=lambda r: knob(r, "LR"))
    lrs = sorted({knob(r, "LR") for r in runs})
    metrics = (
        ("max/stream_max", "peak residual stream |x|", "residual stream"),
        ("raw", "largest raw row length (blocks)", "raw weight rows"),
        ("gain", "largest writer gain (out, down)", "learned gains"),
    )
    fig, axes = figure(3, width=13)
    for run in runs:
        preset, lr = knob(run, "PRESET"), knob(run, "LR")
        rows = [r for r in run["logs"] if r["step"] > 0]
        for ax, (key, _, _) in zip(axes, metrics, strict=True):
            ys = [b_value(r, key) for r in rows]
            if not any(math.isfinite(y) for y in ys):
                continue
            style = LR_STYLES[lrs.index(lr)]
            ax.plot([r["step"] for r in rows], ys, style, color=COLORS[preset], lw=1.4)
    for ax, (_, label, title) in zip(axes, metrics, strict=True):
        ax.set(xscale="log", yscale="log", xlabel="step", ylabel=label, title=title)
        log_axis(ax.xaxis)
        log_axis(ax.yaxis)
    better(axes[0], "down")
    styles = [
        Line2D([], [], color="0.3", ls=LR_STYLES[i], label=f"lr {lr:g}") for i, lr in enumerate(lrs)
    ]
    top_legend(fig, styles)
    headline(
        fig,
        f"Growth over 100k steps: {BYTE_LM}, 8 blocks, dim 256, fp16",
    )
    footnote(
        fig,
        "stock at lr 0.01 diverged near step 1k. Raw rows: stock's weights are what the layer "
        "uses; the patched models' raw rows do not reach the output",
    )
    save(fig, "b_long")


# ---- depth and width ----


def figure_sweep(experiment: str, name: str, label: str) -> None:
    """Peak stream over depth or width, one panel per learning rate (2e-3, 1e-2)."""
    runs = runs_of(experiment)
    values = sorted({knob(r, name) for r in runs})
    fig, axes = figure(2, width=9.5)
    for ax, lr in zip(axes, (2e-3, 1e-2), strict=True):
        for preset in PRESETS:
            group = [r for r in runs if knob(r, "PRESET") == preset and knob(r, "LR") == lr]
            line(ax, group, lambda r: knob(r, name), peak)
        ax.set(xscale="log", yscale="log", xlabel=label, title=f"lr {lr:g}")
        ax.set_xticks(values)
        ax.xaxis.set_major_formatter(FuncFormatter(plain))
        ax.xaxis.set_minor_formatter(NullFormatter())
        log_axis(ax.yaxis, dense=True)
        better(ax, "down")
    axes[0].set_ylabel("peak residual stream |x|")
    top_legend(fig)
    shape = "dim 256" if label == "depth" else "8 blocks"
    headline(
        fig,
        f"Peak stream vs {label}: {BYTE_LM}, {shape}, 5k steps, fp16",
    )
    footnote(fig, "x above a panel: the run failed")
    save(fig, experiment.split("_", 1)[1])

    lrs = sorted({knob(r, "LR") for r in runs})
    print(f"\n### {label}: best held-out loss (x = failed)\n")
    print("| preset | lr | " + " | ".join(f"{label} {v}" for v in values) + " |")
    print("| --- | --- | " + " | ".join("---" for _ in values) + " |")
    for preset in PRESETS:
        for lr in lrs:
            cells = []
            for v in values:
                match = [
                    r
                    for r in runs
                    if knob(r, "PRESET") == preset and knob(r, "LR") == lr and knob(r, name) == v
                ]
                if not match:
                    cells.append("-")
                else:
                    cells.append("x" if match[0]["failed"] else f"{match[0]['best']:.3f}")
            print(f"| {preset} | {lr:g} | " + " | ".join(cells) + " |")


# ---- D: weight decay ----


def table_d() -> None:
    runs = runs_of("d_weight_decay")
    print("\n### D: weight decay on stock (best held-out loss, peak stream; x = failed)\n")
    lrs = sorted({knob(r, "LR") for r in runs})
    print("| weight decay | " + " | ".join(f"lr {lr:g}" for lr in lrs) + " |")
    print("| --- | " + " | ".join("---" for _ in lrs) + " |")
    for wd in sorted({knob(r, "WEIGHT_DECAY") for r in runs}):
        cells = []
        for lr in lrs:
            run = next(
                (r for r in runs if knob(r, "LR") == lr and knob(r, "WEIGHT_DECAY") == wd), None
            )
            if run is None:
                cells.append("-")
            elif run["failed"]:
                cells.append("x")
            else:
                cells.append(f"{run['best']:.3f}, {peak(run):.3g}")
        print(f"| {wd:g} | " + " | ".join(cells) + " |")


# ---- stream vs loss ----


def pareto_points() -> list[dict]:
    """Every finished 5k-step byte-LM run that trained: setup, learning rate, stream and loss."""
    points = []
    for experiment in ("a_lr_precision", "c_depth", "d_weight_decay", "e_width"):
        for run in runs_of(experiment):
            if run["failed"] or run["done"] != "finished" or not math.isfinite(peak(run)):
                continue
            setup = tuple(
                knob(run, k) for k in ("PRESET", "PRECISION", "DEPTH", "DIM", "WEIGHT_DECAY")
            )
            kind = "wd" if knob(run, "WEIGHT_DECAY") else knob(run, "PRECISION")
            points.append(
                {
                    "setup": setup,
                    "preset": knob(run, "PRESET"),
                    "kind": kind,
                    "lr": knob(run, "LR"),
                    "stream": peak(run),
                    "loss": run["final"],
                }
            )
    return points


def frontier(points: list[dict]) -> list[dict]:
    """The points no other point beats on both stream and loss, by stream."""
    out, best = [], math.inf
    for point in sorted(points, key=lambda p: (p["stream"], p["loss"])):
        if point["loss"] < best:
            out.append(point)
            best = point["loss"]
    return out


def figure_pareto(lines: str = "frontier") -> None:
    """Peak stream against final held-out loss; `lines` is "frontier" (each preset's Pareto
    frontier) or "setups" (each setup through its learning rates)."""
    marks = {"fp16": "o", "fp32": "s", "wd": "^"}
    points = pareto_points()
    fig, (ax,) = figure(1, width=8)
    faint = 0.35 if lines == "frontier" else 0.8
    for point in points:
        ax.scatter(
            point["stream"],
            point["loss"],
            s=20,
            marker=marks[point["kind"]],
            color=COLORS[point["preset"]],
            alpha=faint,
            edgecolors="none",
        )
    if lines == "frontier":
        for preset in PRESETS:
            edge = frontier([p for p in points if p["preset"] == preset])
            ax.step(
                [p["stream"] for p in edge],
                [p["loss"] for p in edge],
                where="post",
                color=COLORS[preset],
                lw=2.2,
            )
            ax.scatter(
                [p["stream"] for p in edge], [p["loss"] for p in edge], s=26, color=COLORS[preset]
            )
    else:
        setups = defaultdict(list)
        for point in points:
            setups[point["setup"]].append(point)
        for group in setups.values():
            group.sort(key=lambda p: p["lr"])
            style = ":" if group[0]["kind"] == "wd" else PRECISION_STYLE[group[0]["kind"]]
            ax.plot(
                [p["stream"] for p in group],
                [p["loss"] for p in group],
                style,
                color=COLORS[group[0]["preset"]],
                lw=1.0,
                alpha=0.7,
            )
    ax.set(
        xscale="log", xlabel="peak residual stream |x|", ylabel="final held-out loss (nats / byte)"
    )
    log_axis(ax.xaxis)
    better(ax, "down-left")
    names = {"fp16": "fp16", "fp32": "fp32", "wd": "stock + weight decay"}
    kinds = [Line2D([], [], color="0.3", marker=m, ls="", label=names[k]) for k, m in marks.items()]
    preset_legend(ax, kinds, loc="upper left")
    headline(fig, f"Stream vs loss: every finished 5k-step run of the {BYTE_LM}")
    what = (
        "lines: each preset's Pareto frontier (no run of that preset is better on both)"
        if lines == "frontier"
        else "lines: one setup (preset, precision, depth, width, weight decay) through its "
        "learning rates"
    )
    footnote(fig, f"{what}; failed runs left out")
    save(fig, "pareto" if lines == "frontier" else "pareto_setups")


# ---- JiT ----


def figure_jit() -> None:
    runs = [r for r in runs_of("jit_tokens", group="jit") if knob(r, "PRECISION") == "fp16"]
    lrs = sorted({knob(r, "LR") for r in runs})
    metrics = (
        ("held_loss", "held-out velocity loss", False),
        ("max/stream_max", "peak residual stream |x|", True),
        ("max/logit_max", "largest attention logit", False),
    )
    fig, axes = figure(3, width=13)
    for run in runs:
        preset, lr = knob(run, "PRESET"), knob(run, "LR")
        rows = [r for r in run["logs"] if r["step"] > 0]
        for ax, (key, _, _) in zip(axes, metrics, strict=True):
            ax.plot(
                [r["step"] for r in rows],
                [r.get(key, math.nan) for r in rows],
                LR_STYLES[lrs.index(lr)],
                color=COLORS[preset],
                lw=1.4,
            )
    for ax, (_, label, logy) in zip(axes, metrics, strict=True):
        ax.set(xlabel="step", ylabel=label)
        ax.xaxis.set_major_formatter(FuncFormatter(plain))
        if logy:
            ax.set_yscale("log")
            log_axis(ax.yaxis, dense=True)
        better(ax, "down")
    styles = [
        Line2D([], [], color="0.3", ls=LR_STYLES[i], label=f"lr {lr:g}") for i, lr in enumerate(lrs)
    ]
    top_legend(fig, styles)
    headline(fig, "JiT-B/16 pixel diffusion on Flowers-102: 256 px, batch 64, 10k steps, fp16")
    save(fig, "jit")


def main() -> None:
    figures = {
        "a": figure_a,
        "b": figure_b,
        "depth": lambda: figure_sweep("c_depth", "DEPTH", "depth"),
        "width": lambda: figure_sweep("e_width", "DIM", "width"),
        "d": table_d,
        "pareto": figure_pareto,
        "pareto_setups": lambda: figure_pareto("setups"),
        "jit": figure_jit,
    }
    for name, make in figures.items():
        if ONLY in ("", name):
            make()


if __name__ == "__main__":
    main()
