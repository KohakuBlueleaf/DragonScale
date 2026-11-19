"""Draw the kernel figures from the benchmark results in `data/` (`measure/ops/speed.py` and
`measure/ops/overhead.py`, one GPU):

- `kernels.png`: the unit-row op's forward + backward per path, effective bandwidth from the GPU
  time of its kernels, on three large weights, fp32 / bf16 / fp16 weights;
- `layer_speed.png`: how much slower a patched layer's training step is than the plain layer's,
  bf16 AMP, per setting.

Writes light and dark PNGs next to this file.

    python docs/figures/kernels.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from plot import HERE, SLOTS, THEMES, rows, style  # noqa: E402

LARGE = ("linear 4096x4096", "linear 11008x4096", "embedding 50304x768")
PATHS = {  # path -> (label, slot or None for muted)
    "triton": ("Triton kernels (default)", 0),
    "compile": ("torch.compile backend", 2),
    "eager": ("plain torch", None),
    "inductor": ("inside a compiled model", 3),
}
SETTINGS = {
    "eager, triton": ("Triton kernels (default)", 0),
    "eager, grouped": ("Triton, grouped launch", 4),
    "eager, compile": ("torch.compile backend", 2),
    "eager, eager": ("plain torch", None),
    "torch.compile": ("torch.compile of the model", 3),
}


def color(theme, slot):
    return THEMES[theme]["muted"] if slot is None else SLOTS[theme][slot]


def speed(theme: str):
    t = style(theme)
    data = rows("kernel_speed.csv")
    gpu = data[0]["gpu"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True, layout="constrained")
    width = 0.8 / len(PATHS)
    for ax, dtype in zip(axes, ("fp32", "bf16", "fp16"), strict=True):
        for i, (path, (label, slot)) in enumerate(PATHS.items()):
            xs, ys = [], []
            for j, shape in enumerate(LARGE):
                r = next(
                    (
                        r
                        for r in data
                        if r["shape"] == shape and r["dtype"] == dtype and r["path"] == path
                    ),
                    None,
                )
                if r:
                    xs.append(j + (i - 1.5) * width)
                    ys.append(r["gb_s"] / 1e3)
            ax.bar(xs, ys, width, color=color(theme, slot), label=label)
        ax.set_xticks(range(len(LARGE)), [s.replace(" ", "\n") for s in LARGE])
        ax.set_title(f"{dtype} weights", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("effective bandwidth, TB/s")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    fig.text(0.5, -0.02, f"forward + backward, GPU time of the kernels, {gpu}", ha="center", fontsize=8,
             color=t["muted"])  # fmt: skip
    return fig


def overhead(theme: str):
    t = style(theme)
    data = [r for r in rows("kernel_overhead.csv") if r["precision"] == "bf16 AMP"]
    gpu = data[0]["gpu"]
    cases = list(dict.fromkeys(r["case"] for r in data))
    fig, ax = plt.subplots(figsize=(11, 3.8), layout="constrained")
    width = 0.8 / len(SETTINGS)
    for i, (setting, (label, slot)) in enumerate(SETTINGS.items()):
        xs, ys = [], []
        for j, case in enumerate(cases):
            r = next((r for r in data if r["case"] == case and r["setting"] == setting), None)
            if r:
                xs.append(j + (i - 2) * width)
                ys.append(100 * r["overhead"])
        ax.bar(xs, ys, width, color=color(theme, slot), label=label)
    ax.axhline(0, color=t["ink"], lw=1)
    ax.set_xticks(range(len(cases)), [c.replace(", ", ",\n") for c in cases], fontsize=7.5)
    ax.set_ylabel("slower than the plain layer, %")
    ax.set_ylim(-5, None)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=3)
    fig.text(0.5, -0.03, f"forward + backward, bf16 AMP, fp32 weights, {gpu}", ha="center", fontsize=8,
             color=t["muted"])  # fmt: skip
    return fig


def main() -> None:
    for name, draw in (("kernels", speed), ("layer_speed", overhead)):
        for theme, suffix in (("light", ""), ("dark", "_dark")):
            fig = draw(theme)
            fig.savefig(HERE / f"{name}{suffix}.png", dpi=150)
            plt.close(fig)
            print(HERE / f"{name}{suffix}.png")


if __name__ == "__main__":
    main()
