"""Draw the README figures from the aggregated toy-model results in `data/`.

Every run: a small transformer language model (2 blocks, width 64, byte vocabulary) trained with
Adam for 3000 steps. Each CSV row is one cell (model, learning rate, ...): the median over the
seeds that trained, and the lowest and highest seed (`low`, `high`). `ok` is 0 where fewer than 3
seeds trained; the value there is the median of the finite seeds and is drawn as an open marker.

Writes `<name>.png` (light) and `<name>_dark.png` (dark) next to this file.

    python docs/figures/plot.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# Categorical slots, light and dark steps (validated for color-vision deficiency)
SLOTS = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c"],
}
THEMES = {
    "light": {"surface": "#ffffff", "ink": "#1f1f1e", "muted": "#52514e", "grid": "#e4e3df"},
    "dark": {"surface": "#0d1117", "ink": "#f0efec", "muted": "#c3c2b7", "grid": "#33332f"},
}
MARKERS = ["o", "s", "^", "D", "v", "P"]
MODEL = {"patched": 0, "standard": 1}  # slot per model, the same in every figure


def rows(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8") as file:
        out = []
        for row in csv.DictReader(file):
            out.append({k: _num(v) for k, v in row.items()})
        return out


def _num(text: str):
    try:
        return float(text)
    except ValueError:
        return text


def style(theme: str) -> dict:
    t = THEMES[theme]
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.linewidth": 0.6, "axes.edgecolor": t["muted"],
        "axes.labelcolor": t["ink"], "xtick.color": t["muted"], "ytick.color": t["muted"],
        "text.color": t["ink"], "axes.facecolor": t["surface"], "figure.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.5, "lines.linewidth": 2,
        "lines.markersize": 5, "legend.frameon": False, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })  # fmt: skip
    return t


def series(ax, cells: list[dict], color: str, marker: str, label: str, ls: str = "-") -> None:
    """Median line with markers (open where fewer than 3 seeds trained) and the seed range."""
    cells = [c for c in cells if c.get("median") != ""]
    lr = [c["lr"] for c in cells]
    ax.fill_between(lr, [c["low"] for c in cells], [c["high"] for c in cells], color=color,
                    alpha=0.18, lw=0)  # fmt: skip
    ax.plot(lr, [c["median"] for c in cells], color=color, ls=ls, label=label)
    for c in cells:
        face = color if c["ok"] else "none"
        ax.plot(c["lr"], c["median"], marker=marker, color=color, mfc=face, mec=color, ls="none")


def lr_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")


def fig_a(theme: str):
    t, c = style(theme), SLOTS[theme]
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    stream = rows("a_stream.csv")
    for model in ("standard", "patched"):
        cells = [r for r in stream if r["model"] == model]
        series(ax, cells, c[MODEL[model]], MARKERS[MODEL[model]], model)
    bad = [
        r["lr"]
        for r in rows("a_fp16_amp_nonfinite.csv")
        if r["model"] == "standard" and r["nonfinite"]
    ]
    ys = [r["median"] for r in stream if r["model"] == "standard" and r["lr"] in bad]
    ax.plot(bad, ys, ls="none", marker="x", ms=9, mew=1.6, color=t["ink"],
            label="standard under fp16 AMP: every seed non-finite")  # fmt: skip
    lr_axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel("hidden-state RMS at the end")
    ax.legend(loc="upper left")
    return fig


def fig_b(theme: str):
    t, c = style(theme), SLOTS[theme]
    ladder = rows("b_ladder.csv")
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ys = list(range(len(ladder)))[::-1]
    for y, r in zip(ys, ladder, strict=True):
        color = c[MODEL["standard"]] if r["model"] == "standard" else c[MODEL["patched"]]
        ax.plot(
            [r["low"], r["high"]], [y, y], color=color, lw=3, alpha=0.35, solid_capstyle="round"
        )
        ax.plot(r["growth_per_decade"], y, marker="o", color=color, ms=7)
        count = int(r["free_scales"])
        text = f"{count} free scale{'s' if count != 1 else ''}"
        ax.annotate(text, (r["high"], y), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=8, color=t["muted"])  # fmt: skip
    ax.set_yticks(ys, [r["model"] for r in ladder])
    ax.set_xscale("log")
    ax.set_xlim(0.8, 3e5)
    ax.set_ylim(-0.6, len(ladder) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("growth of the hidden-state RMS per 10x learning rate")
    return fig


def fig_c(theme: str):
    t, c = style(theme), SLOTS[theme]
    data = rows("c_angle.csv")
    runs = list(dict.fromkeys((r["lr"], r["init_scale"]) for r in data))
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for i, (lr, s) in enumerate(runs):
        cells = [r for r in data if r["lr"] == lr and r["init_scale"] == s]
        steps = [r["step"] for r in cells]
        ax.plot(steps, [r["median"] for r in cells], color=c[i], marker=MARKERS[i], markevery=6,
                label=f"lr {lr:g}, init scale {s:g}")  # fmt: skip
    ax.axhline((0.1 / 1.9) ** 0.5, color=t["ink"], ls="--", lw=1.2,
               label="sqrt((1 - β1) / (1 + β1)) = 0.229")  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step t")
    ax.set_ylabel("row turn per step (rad) × sqrt(t)")
    ax.legend(loc="lower right", fontsize=7)
    return fig


def fig_d(theme: str):
    style(theme)
    c = SLOTS[theme]
    data = rows("d_cosine.csv")
    arms = {
        "standard": (c[1], "s"),
        "unit rows with gains": (c[2], "^"),
        "unit rows, no gains": (c[0], "o"),
    }
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for name, (color, marker) in arms.items():
        series(
            left,
            [r for r in data if r["model"] == name and r["quantity"] == "stream"],
            color,
            marker,
            name,
        )
        series(
            right,
            [r for r in data if r["model"] == name and r["quantity"] == "held_loss"],
            color,
            marker,
            name,
        )
    for ax in (left, right):
        lr_axis(ax)
    left.set_yscale("log")
    left.set_ylabel("hidden-state RMS at the end")
    right.set_ylabel("held-out loss")
    right.set_ylim(1.4, 2.6)
    left.legend(loc="upper left")
    return fig


def fig_e(theme: str):
    style(theme)
    c = SLOTS[theme]
    data = rows("e_fp16_amp.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    top = 2.6
    for model in ("standard", "patched"):
        for precision, ls in (("fp32", "-"), ("fp16 AMP", "--")):
            cells = [r for r in data if r["model"] == model and r["precision"] == precision]
            color, marker = c[MODEL[model]], MARKERS[MODEL[model]]
            series(
                ax,
                [r for r in cells if r["nonfinite"] < r["seeds"]],
                color,
                marker,
                f"{model}, {precision}",
                ls,
            )
            lost = [r["lr"] for r in cells if r["nonfinite"] == r["seeds"]]
            if lost:
                ax.plot(lost, [top * 0.985] * len(lost), ls="none", marker="x", ms=8, mew=1.6, color=color,
                        label=f"{model}, {precision}: every seed non-finite")  # fmt: skip
    lr_axis(ax)
    ax.set_ylim(1.4, top)
    ax.set_ylabel("held-out loss")
    ax.legend(loc="upper left", fontsize=7)
    return fig


def fig_f(theme: str):
    style(theme)
    c = SLOTS[theme]
    sources = rows("f_sources.csv")
    shapes = rows("f_shapes.csv")
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(8.2, 3.2), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.75}
    )
    for j, (source, ls) in enumerate(
        (("deterministic", "-"), ("low entropy", "--"), ("high entropy", ":"))
    ):
        for model in ("standard", "patched"):
            cells = [r for r in sources if r["source"] == source and r["model"] == model]
            series(left, cells, c[MODEL[model]], MARKERS[j], f"{model}, {source}", ls)
    lr_axis(left)
    left.set_yscale("log")
    left.set_ylabel("hidden-state RMS at the end")
    left.legend(fontsize=6.5, loc="upper left")
    names = list(dict.fromkeys(f"width {int(r['width'])}, depth {int(r['depth'])}" for r in shapes))
    ys = list(range(len(names)))[::-1]
    for model in ("standard", "patched"):
        color = c[MODEL[model]]
        cells = [r for r in shapes if r["model"] == model]
        for y, r in zip(ys, cells, strict=True):
            right.plot(
                [r["low"], r["high"]], [y, y], color=color, lw=3, alpha=0.35, solid_capstyle="round"
            )
        right.plot([r["growth_per_decade"] for r in cells], ys, ls="none", marker=MARKERS[MODEL[model]],
                   color=color, label=model)  # fmt: skip
    right.set_yticks(ys, names)
    right.set_xscale("log")
    right.grid(axis="y", visible=False)
    right.set_xlabel("growth per 10x learning rate")
    right.legend(loc="center")
    return fig


FIGURES = {"a_stream": fig_a, "b_ladder": fig_b, "c_angle": fig_c, "d_cosine": fig_d,
           "e_fp16_amp": fig_e, "f_data_and_shape": fig_f}  # fmt: skip


def main() -> None:
    for name, draw in FIGURES.items():
        for theme, suffix in (("light", ""), ("dark", "_dark")):
            fig = draw(theme)
            fig.savefig(HERE / f"{name}{suffix}.png", dpi=160)
            plt.close(fig)
            print(HERE / f"{name}{suffix}.png")


if __name__ == "__main__":
    main()
