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
from matplotlib.patches import FancyBboxPatch
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
B_STEPS = 16_000  # B is compared up to the step every preset reached
JIT_LOSS_FROM = 2_000  # JiT loss is compared from here: the first steps' losses are outliers
KERNEL_BACKENDS = {"triton": "#009E73", "compile": "#0072B2", "eager": "#999999"}
JIT_VARIANTS = (  # (folder, file prefix, label, color, line style)
    ("jit_diag", "stock-tokens-lr", "stock", "#D55E00", "-"),
    ("jit_diag", "arch-tokens-lr", "arch with a final norm", "0.45", "--"),
    ("jit_diag_read_only", "arch-tokens-nofinal", "arch", "#009E73", "-"),
)
RED, GREEN, GREY = "#B03A2E", "#1E7B5F", "0.35"
TEASER_NAMES = {
    "stock": "plain pre-norm transformer",
    "matrices": "plain + DragonScale patch",
    "arch": "ours, pre-norm transformer",
}

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


def runs_of(experiment: str, group: str = "bytelm", partial: bool = False) -> list[dict]:
    """The runs of an experiment; a run that neither finished nor diverged is left out (a
    placeholder in tables) unless `partial`."""
    runs = [read_run(p) for p in sorted((RUNS / group / experiment).glob("*.jsonl"))]
    return [r for r in runs if partial or r["done"] != "running" or r["diverged"]]


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
    if direction == "right":
        ax.set_xlabel(f"{ax.get_xlabel()}   better \u2192")
        return
    if direction == "up":
        ax.set_ylabel(f"{ax.get_ylabel()}   better \u2192")
        return
    ax.set_ylabel(f"{ax.get_ylabel()}   \u2190 better")
    if direction == "down-left":
        ax.set_xlabel(f"\u2190 better   {ax.get_xlabel()}")


def failed_strip(ax, xs: list, presets: list) -> None:
    """Failed runs as x marks in a strip just above the panel, one row per preset."""
    for x, preset in zip(xs, presets, strict=True):
        ax.plot(
            x,
            1.03 + 0.045 * PRESETS.index(preset),
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
        size = getattr(fig, "headline_size", 11)
        legend.set_title(text, prop={"size": size, "weight": "bold"})
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
    runs = sorted(runs_of("b_long", partial=True), key=lambda r: knob(r, "LR"))
    lrs = sorted({knob(r, "LR") for r in runs})
    metrics = (
        ("max/stream_max", "peak residual stream |x|", "residual stream"),
        ("raw", "largest raw row length (blocks)", "raw weight rows"),
        ("gain", "largest writer gain (out, down)", "learned gains"),
    )
    fig, axes = figure(3, width=13)
    for run in runs:
        preset, lr = knob(run, "PRESET"), knob(run, "LR")
        rows = [r for r in run["logs"] if 0 < r["step"] <= B_STEPS]
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
        f"Growth over {B_STEPS // 1000}k steps: {BYTE_LM}, 8 blocks, dim 256, fp16",
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
        ("held_loss", f"held-out velocity loss, from step {JIT_LOSS_FROM:,}", False),
        ("max/stream_max", "peak residual stream |x|", True),
        ("max/logit_max", "largest attention logit", False),
    )
    fig, axes = figure(3, width=13)
    for run in runs:
        preset, lr = knob(run, "PRESET"), knob(run, "LR")
        for ax, (key, _, _) in zip(axes, metrics, strict=True):
            start = JIT_LOSS_FROM if key == "held_loss" else 1
            rows = [r for r in run["logs"] if r["step"] >= start]
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


def figure_final_norm() -> None:
    """JiT at lr 1e-3: the image tokens' stream per block at the last step, and its peak over
    training, for stock, arch with a final norm, and arch as proposed (input norms, read-only
    context tokens, no final norm)."""
    variants = JIT_VARIANTS
    fig, (blocks, steps) = figure(2, width=11)
    handles = []
    for folder, prefix, label, color, style in variants:
        paths = sorted((RUNS / "jit" / folder).glob(f"{prefix}*.jsonl"))
        if not paths:
            continue
        run = read_run(paths[0])
        logs = [r for r in run["logs"] if r["step"] > 0 and "blocks.0/image_rms" in r]
        if not logs:
            continue
        last = logs[-1]
        depth = sum(1 for k in last if k.startswith("blocks.") and k.endswith("/image_rms"))
        blocks.plot(
            range(depth),
            [last[f"blocks.{j}/image_rms"] for j in range(depth)],
            style,
            color=color,
            marker="o",
            ms=3.5,
            lw=1.6,
        )
        peak_by_step = [max(r[f"blocks.{j}/image_max"] for j in range(depth)) for r in logs]
        steps.plot([r["step"] for r in logs], peak_by_step, style, color=color, lw=1.6)
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                ls=style,
                lw=2,
                label=f"{label} (loss {run['best']:.3f}, step {last['step']})",
            )
        )
    blocks.set(
        yscale="log",
        xlabel="block",
        ylabel="image-token stream RMS, last step",
        title="stream through depth",
    )
    steps.set(
        yscale="log", xlabel="step", ylabel="largest image-token value", title="peak over training"
    )
    for ax in (blocks, steps):
        log_axis(ax.yaxis, dense=True)
        better(ax, "down")
    steps.xaxis.set_major_formatter(FuncFormatter(plain))
    fig.top_legend = fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
    headline(fig, "Where the stream grows: JiT-B/16 on Flowers-102, lr 1e-3, fp16, 2k steps")
    footnote(fig, "loss: best held-out velocity loss so far")
    save(fig, "final_norm")


# ---- kernels ----


KERNEL_SHAPES = (  # (benchmark case, panel title: the weight shape only)
    ("byte LM gate/up 768x256", "weight 768 x 256"),
    ("JiT gate/up 2048x768", "weight 2048 x 768"),
    ("LM up 11008x4096", "weight 11008 x 4096"),
)
GPU_NAMES = {"Tesla_V100-SXM2-16GB": "V100", "Tesla_PG503-216": "V100",
             "NVIDIA_GeForce_RTX_2080_Ti": "2080 Ti",
             "NVIDIA_GeForce_RTX_3090": "3090", "NVIDIA_GeForce_RTX_4090": "4090"}  # fmt: skip


def figure_kernels() -> None:
    """Unit-row forward + backward effective bandwidth per backend: the bytes the two ops must
    move over their time, one panel per weight shape, one group of bars per GPU (fp32 weights);
    the dashed line is the card's memory bandwidth."""
    files = sorted((RUNS / "speed").glob("*-fp32.jsonl"))
    if not files:
        return
    by_gpu = {}
    for path in files:
        gpu = GPU_NAMES.get(path.stem.removesuffix("-fp32"), path.stem)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        by_gpu[gpu] = {(r["case"], r["backend"]): r for r in rows}
    gpus = [g for g in ("V100", "2080 Ti", "3090", "4090") if g in by_gpu]
    fig, axes = figure(len(KERNEL_SHAPES), width=15)
    width = 0.8 / len(KERNEL_BACKENDS)
    for ax, (case, title) in zip(axes, KERNEL_SHAPES, strict=True):
        large = case == KERNEL_SHAPES[-1][0]
        for g, gpu in enumerate(gpus):
            for i, (backend, color) in enumerate(KERNEL_BACKENDS.items()):
                r = by_gpu[gpu].get((case, backend))
                if r is None:
                    continue
                moved = r["fwd_gb_s"] * r["fwd_ms"] + r["bwd_gb_s"] * r["bwd_ms"]
                rate = moved / (r["fwd_ms"] + r["bwd_ms"])
                x = g + (i - 1) * width
                ax.bar(x, rate, width, color=color)
                peak = r["fwd_gb_s"] / r["fwd_peak"] if r.get("fwd_peak") else None
                if large and peak:
                    ax.text(x, rate, f"{100 * rate / peak:.0f}%", ha="center", va="bottom",
                            fontsize=7)  # fmt: skip
            if peak:
                ax.plot([g - 0.45, g + 0.45], [peak, peak], "--", color="0.35", lw=1)
        ax.set_xticks(range(len(gpus)))
        ax.set_xticklabels(gpus)
        ax.set(ylabel="effective bandwidth, GB/s", title=title)
        ax.grid(axis="x", visible=False)
        better(ax, "up")
    handles = [Line2D([], [], color=c, lw=6, label=b) for b, c in KERNEL_BACKENDS.items()]
    handles.append(Line2D([], [], color="0.35", ls="--", lw=1, label="card's memory bandwidth"))
    fig.top_legend = fig.legend(
        handles=handles, loc="outside upper center", ncol=len(handles), frameon=False
    )
    headline(fig, "Unit-row op per backend, same weight shape, forward + backward, fp32 weights")
    footnote(
        fig,
        "labels on the large weight: share of the card's memory bandwidth. Small weights are "
        "bound by launch time; the grouped launch covers them",
    )
    save(fig, "kernels")


LAYER_CASES = (  # (overhead case, panel title)
    ("linear 256->768, 16384 tokens", "Linear 256 -> 768, 16k tokens"),
    ("linear 768->2048, 16384 tokens", "Linear 768 -> 2048, 16k tokens"),
    ("linear 4096->11008, 8192 tokens", "Linear 4096 -> 11008, 8k tokens"),
)


def figure_layer_speed() -> None:
    """A unit-row Linear's forward + backward speed relative to the plain Linear (fp32 weights,
    fp16 autocast), one panel per shape, one group of bars per GPU."""
    files = sorted((RUNS / "overhead").glob("*-fp16.jsonl"))
    if not files:
        return
    by_gpu = {}
    for path in files:
        gpu = GPU_NAMES.get(path.stem.removesuffix("-fp16"), path.stem)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        by_gpu[gpu] = {(r["case"], r["backend"]): r for r in rows}
    gpus = [g for g in ("V100", "2080 Ti", "3090", "4090") if g in by_gpu]
    fig, axes = figure(len(LAYER_CASES), width=15)
    width = 0.8 / len(KERNEL_BACKENDS)
    for ax, (case, title) in zip(axes, LAYER_CASES, strict=True):
        for g, gpu in enumerate(gpus):
            for i, (backend, color) in enumerate(KERNEL_BACKENDS.items()):
                r = by_gpu[gpu].get((case, backend))
                if r is None:
                    continue
                x = g + (i - 1) * width
                ax.bar(x, 100 * r["relative_speed"], width, color=color)
                ax.text(x, 100 * r["relative_speed"], f"{100 * r['relative_speed']:.0f}",
                        ha="center", va="bottom", fontsize=7)  # fmt: skip
        ax.axhline(100, color="0.35", ls="--", lw=1)
        ax.set_xticks(range(len(gpus)))
        ax.set_xticklabels(gpus)
        ax.set(ylim=(0, 115), ylabel="speed of the plain Linear, %", title=title)
        ax.grid(axis="x", visible=False)
        better(ax, "up")
    handles = [Line2D([], [], color=c, lw=6, label=b) for b, c in KERNEL_BACKENDS.items()]
    handles.append(Line2D([], [], color="0.35", ls="--", lw=1, label="plain Linear"))
    fig.top_legend = fig.legend(
        handles=handles, loc="outside upper center", ncol=len(handles), frameon=False
    )
    headline(
        fig, "Unit-row Linear against plain Linear: forward + backward, fp32 weights, fp16 autocast"
    )
    footnote(
        fig,
        "one layer on its own; in a whole model the grouped launch computes every layer's weight "
        "in a few launches",
    )
    save(fig, "layer_speed")


# ---- teaser ----


TEASER_FONT = 14


def _box(ax, x, y, text, fill, w=3.6, h=1.0, size=TEASER_FONT, color="black"):
    """A rounded box centred at (x, y) in data coordinates."""
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            fc=fill,
            ec="0.3",
            lw=1.2,
            zorder=3,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color, zorder=4)


def _arrow(ax, start, end, color="0.3", lw=1.6):
    ax.annotate(
        "",
        end,
        start,
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": lw, "shrinkA": 0, "shrinkB": 0},
        zorder=2,
    )


def _add(ax, x, y, text=r"$+$"):
    ax.add_patch(plt.Circle((x, y), 0.36, fc="white", ec="0.3", lw=1.4, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=TEASER_FONT + 2, zorder=4)


def _note(ax, x, y, text, color, size=TEASER_FONT):
    ax.text(x, y, text, ha="left", va="center", fontsize=size, color=color, zorder=4)


STREAM_X, BRANCH_X, GLYPH_X = 1.9, 5.5, 7.95
NARROW_W, WIDE_W, BOX_H = 3.1, 4.3, 1.6
ATTN_Y, MLP_Y, TOP_Y, HEAD_Y = 2.2, 6.9, 11.5, 13.1


def _block(ax, y0, lower, upper, lower_fill, upper_fill, lower_w, upper_w, add_text=r"$+$"):
    """One residual sublayer: from the stream at y0 through two boxes, back into the stream with
    an add at the upper box. Returns (lower y, upper y)."""
    y1 = y0 + 2.1
    _arrow(ax, (STREAM_X, y0), (BRANCH_X - lower_w / 2, y0))
    _box(ax, BRANCH_X, y0, lower, lower_fill, w=lower_w, h=BOX_H)
    _box(ax, BRANCH_X, y1, upper, upper_fill, w=upper_w, h=BOX_H)
    _arrow(ax, (BRANCH_X, y0 + BOX_H / 2), (BRANCH_X, y1 - BOX_H / 2))
    _arrow(ax, (BRANCH_X - upper_w / 2, y1), (STREAM_X + 0.36, y1))
    _add(ax, STREAM_X, y1, add_text)
    return y0, y1


def _rows(ax, y, lengths, color, gains=None, gain_color=None, label=None, x=None):
    """A weight as its rows: one bar per row, its length the row's norm; with `gains`, a gain
    column to the right."""
    n, h, w = len(lengths), 1.25, 1.35
    x0 = GLYPH_X if x is None else x
    for i, length in enumerate(lengths):
        yi = y + h / 2 - (i + 0.5) * h / n
        ax.add_patch(plt.Rectangle((x0, yi - 0.4 * h / n), length * w, 0.8 * h / n, fc=color,
                                   ec="none", zorder=3))  # fmt: skip
    ax.plot([x0, x0], [y - h / 2, y + h / 2], color="0.3", lw=1, zorder=4)
    if gains is not None:
        gx = x0 + w + 0.35
        for i, g in enumerate(gains):
            yi = y + h / 2 - (i + 0.5) * h / n
            ax.add_patch(plt.Rectangle((gx, yi - 0.4 * h / n), g * 0.9, 0.8 * h / n,
                                       fc=gain_color, ec="none", zorder=3))  # fmt: skip
        ax.plot([gx, gx], [y - h / 2, y + h / 2], color="0.3", lw=1, zorder=4)
        ax.text(gx + 0.45, y + h / 2 + 0.25, "$g$", ha="center", fontsize=TEASER_FONT,
                color=gain_color)  # fmt: skip
    if label:
        ax.text(x0, y - h / 2 - 0.15, label, ha="left", va="top", fontsize=TEASER_FONT - 1,
                color=color)  # fmt: skip


def _diagram_axes(ax, title, color, caption):
    ax.set_axis_off()
    ax.set(xlim=(0, 11), ylim=(-2.2, 15.4))
    ax.text(5.5, 15.2, title, ha="center", va="top", fontsize=TEASER_FONT + 5,
            fontweight="bold", color=color)  # fmt: skip
    ax.text(5.5, -1.75, caption, ha="center", va="center", fontsize=TEASER_FONT + 1,
            color=color, style="italic")  # fmt: skip


def _stream(ax, y_from, y_to, label, color, grow=False):
    """The residual stream: a band that widens upward when its scale grows."""
    if grow:
        ys = [y_from, y_to, y_to, y_from]
        xs = [STREAM_X - 0.05, STREAM_X - 0.32, STREAM_X + 0.32, STREAM_X + 0.05]
        ax.add_patch(plt.Polygon(list(zip(xs, ys, strict=True)), fc=color, ec="none", zorder=1))
    else:
        ax.plot([STREAM_X, STREAM_X], [y_from, y_to], color=color, lw=3.2, zorder=1,
                solid_capstyle="butt")  # fmt: skip
    ax.text(STREAM_X - 0.55, (ATTN_Y + MLP_Y) / 2 + 1.0, label, rotation=90, ha="right",
            va="center", fontsize=TEASER_FONT, color=color)  # fmt: skip


UNEVEN = [0.55, 1.0, 0.3, 0.85, 0.65]
EVEN = [0.7] * 5
SHRINKING = [0.4, 0.25, 0.35, 0.2, 0.3]
GAINS = [0.35, 0.8, 0.5, 0.2, 0.6]


def _teaser_stock(ax) -> None:
    fill, norm = "#F6DCCB", "#FBEDE4"
    _diagram_axes(ax, "pre-norm transformer", RED, "free scales drift with noise")
    _box(ax, STREAM_X, 0.3, "embedding", fill, w=3.3, h=0.9)
    _stream(ax, 0.75, HEAD_Y - 0.45, r"stream $x$,   $\Vert x\Vert\uparrow$", RED, grow=True)
    _, attn = _block(ax, ATTN_Y, r"RMSNorm$\cdot\gamma$", r"$W_{qkv}\to$ attn" "\n" r"$\to W_o$", norm,
                     fill, NARROW_W, WIDE_W)  # fmt: skip
    _, mlp = _block(ax, MLP_Y, r"RMSNorm$\cdot\gamma$", r"$W_{up}\to$ SiLU" "\n" r"$\to W_{down}$",
                    norm, fill, NARROW_W, WIDE_W)  # fmt: skip
    _box(ax, STREAM_X, HEAD_Y, r"head $W$", fill, w=2.4, h=0.9)
    _rows(ax, HEAD_Y, SHRINKING, RED, x=3.45)
    _rows(ax, attn, UNEVEN, RED, label=r"$\gamma c,W/c$;  $W_q c,W_k/c$")
    _rows(ax, mlp, UNEVEN, RED, label=r"$W_{up}\,c,\ W_{down}/c$")
    _note(ax, 4.4, HEAD_Y, r"$x\,c\uparrow,\ \ W_{head}/c\downarrow$", RED)
    _note(ax, 3.45, TOP_Y - 0.15, "a final norm makes it worse:\nthe stream's scale is then\nfully free",
          RED, TEASER_FONT - 2)  # fmt: skip


def _teaser_fixed(ax) -> None:
    fill, norm = "#DCE3EC", "#EDF1F6"
    _diagram_axes(ax, "nGPT / EDM2", GREY, "every scale fixed by construction")
    _box(ax, STREAM_X, 0.3, r"$\hat w$ embedding", fill, w=3.5, h=0.9)
    _stream(ax, 0.75, HEAD_Y - 0.45, r"stream $h$,   $\Vert h\Vert = 1$", GREY)
    attn, _ = _block(ax, ATTN_Y, "cosine attn:\n" r"$s_{qk}\,\hat q\cdot\hat k$,  $\hat W_{qkv},\hat W_o$",
                     "Norm", fill, norm, WIDE_W, NARROW_W, r"$\oplus$")  # fmt: skip
    mlp, _ = _block(ax, MLP_Y, r"$\hat W_{up}\to$ SiLU$\cdot s_u$" "\n" r"$\to\hat W_{down}$",
                    "Norm", fill, norm, WIDE_W, NARROW_W, r"$\oplus$")  # fmt: skip
    _box(ax, STREAM_X, HEAD_Y, r"head $s_z\,\hat W$", fill, w=2.8, h=0.9)
    _rows(ax, attn + 2.1, EVEN, GREY, label=r"$\hat w = w/\Vert w\Vert$, every step")
    _rows(ax, mlp + 2.1, EVEN, GREY, label=r"$\alpha, s_{qk}, s_u, s_v$ by hand")
    _note(ax, 2.5, TOP_Y - 0.1,
          r"$\oplus$  nGPT: $h\leftarrow\mathrm{Norm}(h+\alpha\,(y-h))$" "\n"
          r"$\oplus$  EDM2: $\frac{(1-t)\,x+t\,y}{\sqrt{(1-t)^2+t^2}}$", GREY)  # fmt: skip


def _teaser_ours(ax) -> None:
    fill, norm = "#CFE8E0", "#E6F3EE"
    unit = r"$\frac{W}{\Vert W\Vert_{row}}$"
    _diagram_axes(ax, "DragonScale", GREEN, "same output at step 0, on the model you have")
    _box(ax, STREAM_X, -0.6, "input", "white", w=2.2, h=0.7)
    _arrow(ax, (STREAM_X, -0.25), (STREAM_X, 0.15))
    _box(ax, STREAM_X, 0.6, "Norm", norm, w=2.2, h=0.85)
    _stream(ax, 1.03, HEAD_Y - 0.6, r"stream $x$", GREEN)
    _, attn = _block(ax, ATTN_Y, "Norm", "attn: every\n" r"$W\to g\odot$" + unit, norm, fill,
                     NARROW_W, WIDE_W)  # fmt: skip
    _, mlp = _block(ax, MLP_Y, "Norm", "MLP: every\n" r"$W\to g\odot$" + unit, norm, fill,
                    NARROW_W, WIDE_W)  # fmt: skip
    _box(ax, STREAM_X, HEAD_Y, "head " + unit, fill, w=2.8, h=1.2)
    _rows(ax, attn, EVEN, GREEN, GAINS, "#0B5D44", label="unit rows, one gain per row")
    _rows(ax, mlp, EVEN, GREEN, GAINS, "#0B5D44", label="the loss moves $g$")
    _note(ax, GLYPH_X - 0.2, MLP_Y, "norms:\nno affine", GREEN)
    _note(ax, 3.85, TOP_Y, "no final norm:" "\n" "the head reads" "\n" "the stream", GREEN)


def _teaser_jit(ax) -> None:
    for folder, prefix, label, color, style in JIT_VARIANTS:
        paths = sorted((RUNS / "jit" / folder).glob(f"{prefix}*.jsonl"))
        if not paths or label.startswith("arch with"):
            continue
        logs = [
            r for r in read_run(paths[0])["logs"] if r["step"] > 0 and "blocks.0/image_max" in r
        ]
        depth = sum(1 for k in logs[-1] if k.startswith("blocks.") and k.endswith("/image_max"))
        peaks = [max(r[f"blocks.{j}/image_max"] for j in range(depth)) for r in logs]
        ax.plot([r["step"] for r in logs], peaks, style, color=color, lw=2)
        name = TEASER_NAMES["stock" if label == "stock" else "arch"]
        if label != "stock":  # in the empty band between the two curves
            ax.text(0.98, 0.3, f"{name}: {peaks[-1]:,.0f}", transform=ax.transAxes, ha="right",
                    color=color, fontsize=TEASER_FONT, fontweight="bold")  # fmt: skip
            continue
        ax.annotate(
            f"{name}: {peaks[-1]:,.0f}",
            (logs[-1]["step"], peaks[-1]),
            xytext=(-6, 8),
            textcoords="offset points",
            ha="right",
            color=color,
            fontsize=TEASER_FONT,
            fontweight="bold",
        )
    ax.set(
        yscale="log",
        xlabel="step",
        ylabel="largest stream value",
        title="JiT-B/16 diffusion, lr 1e-3, fp16",
    )
    log_axis(ax.yaxis, dense=True)
    ax.xaxis.set_major_formatter(FuncFormatter(plain))
    better(ax, "down")


def _teaser_lr(ax) -> None:
    runs = [r for r in runs_of("a_lr_precision") if knob(r, "PRECISION") == "fp16"]
    for preset in PRESETS:
        line(ax, [r for r in runs if knob(r, "PRESET") == preset], lambda r: knob(r, "LR"),
             lambda r: r["best"])  # fmt: skip
    ax.set(
        xscale="log",
        ylim=(0.9, 1.2),
        xlabel="learning rate",
        ylabel="best held-out loss",
    )
    ax.set_title("byte-level LM, 5k steps, fp16", pad=34)
    log_axis(ax.xaxis)
    ax.xaxis.set_major_formatter(FuncFormatter(plain))
    better(ax, "down")


def figure_teaser() -> None:
    with plt.rc_context({"font.size": TEASER_FONT, "axes.titlesize": TEASER_FONT + 1,
                         "axes.labelsize": TEASER_FONT, "xtick.labelsize": TEASER_FONT - 2,
                         "ytick.labelsize": TEASER_FONT - 2, "legend.fontsize": TEASER_FONT}):  # fmt: skip
        fig = plt.figure(figsize=(22, 9), layout="constrained")
        grid = fig.add_gridspec(2, 4)
        _teaser_stock(fig.add_subplot(grid[:, 0]))
        _teaser_fixed(fig.add_subplot(grid[:, 1]))
        _teaser_ours(fig.add_subplot(grid[:, 2]))
        _teaser_jit(fig.add_subplot(grid[0, 3]))
        _teaser_lr(fig.add_subplot(grid[1, 3]))
        handles = [Line2D([], [], color=COLORS[p], lw=3, label=TEASER_NAMES[p]) for p in PRESETS]
        fig.top_legend = fig.legend(
            handles=handles, loc="outside upper center", ncol=len(handles), frameon=False
        )
        headline(fig, "DragonScale: no scale moves where the loss cannot see it")
        fig.headline_size = TEASER_FONT + 8
        save(fig, "teaser")


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
        "final_norm": figure_final_norm,
        "kernels": figure_kernels,
        "layer_speed": figure_layer_speed,
        "teaser": figure_teaser,
    }
    for name, make in figures.items():
        if ONLY in ("", name):
            make()


if __name__ == "__main__":
    main()
