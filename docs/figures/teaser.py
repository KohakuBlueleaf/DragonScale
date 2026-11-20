"""Draw the README teaser: three architecture columns (a pre-norm transformer and its free scales;
nGPT / EDM2, every scale fixed by construction; DragonScale, unit rows and a gain only where the
loss sees the scale) and two result panels from `data/` (figures A and B of the README).

Writes `teaser.png` (light) and `teaser_dark.png` (dark) next to this file.

    python docs/figures/teaser.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from plot import HERE, MARKERS, MODEL, SLOTS, THEMES, rows, series  # noqa: E402

FONT = 14
# Per theme: column ink and box fills (stock, fixed, ours), and the gain color
COLUMNS = {
    "light": {
        "stock": ("#c4501f", "#f8dccd", "#fcefe7"),
        "fixed": ("#52514e", "#dfe3e8", "#eef0f3"),
        "ours": ("#1f63b5", "#d3e3f6", "#e9f1fb"),
        "gain": "#0f3f7a",
        "box_edge": "#5a5955",
        "plain": "#ffffff",
    },
    "dark": {
        "stock": ("#f08a5d", "#4a2a1c", "#33221a"),
        "fixed": ("#c3c2b7", "#2c3036", "#22252a"),
        "ours": ("#6fa8ee", "#1d3352", "#172538"),
        "gain": "#b9d4f6",
        "box_edge": "#a09f98",
        "plain": "#0d1117",
    },
}
STREAM_X, BRANCH_X, GLYPH_X = 1.9, 5.5, 7.95
NARROW_W, WIDE_W, BOX_H = 3.1, 4.4, 1.6
ATTN_Y, MLP_Y, TOP_Y, HEAD_Y = 2.2, 6.9, 11.5, 13.1
UNEVEN = [0.55, 1.0, 0.3, 0.85, 0.65]
EVEN = [0.7] * 5
SHRINKING = [0.4, 0.25, 0.35, 0.2, 0.3]
GAINS = [0.35, 0.8, 0.5, 0.2, 0.6]
UNIT = r"$\hat W$"
TITLE = (
    "Free scales in a pre-norm transformer, two ways to remove them, and what removing them does"
)


class Canvas:
    """One diagram column, drawn in the colors of one theme."""

    def __init__(self, ax, theme: str, column: str, title: str, caption: str):
        self.ax, self.c = ax, COLUMNS[theme]
        self.ink, self.fill, self.norm = self.c[column]
        self.text = THEMES[theme]["ink"]
        self.edge = self.c["box_edge"]
        ax.set_axis_off()
        ax.set(xlim=(0, 11), ylim=(-2.2, 15.4))
        ax.text(5.5, 15.2, title, ha="center", va="top", fontsize=FONT + 5, fontweight="bold",
                color=self.ink)  # fmt: skip
        ax.text(5.5, -1.75, caption, ha="center", va="center", fontsize=FONT + 1, color=self.ink,
                style="italic")  # fmt: skip

    def box(self, x, y, text, fill, w=3.6, h=1.0, size=FONT):
        self.ax.add_patch(
            FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                           fc=fill, ec=self.edge, lw=1.2, zorder=3)  # fmt: skip
        )
        self.ax.text(x, y, text, ha="center", va="center", fontsize=size, color=self.text, zorder=4)

    def arrow(self, start, end):
        self.ax.annotate("", end, start, zorder=2, arrowprops={"arrowstyle": "-|>", "color": self.edge,
                         "lw": 1.6, "shrinkA": 0, "shrinkB": 0})  # fmt: skip

    def add(self, x, y, text=r"$+$"):
        self.ax.add_patch(
            plt.Circle((x, y), 0.36, fc=self.c["plain"], ec=self.edge, lw=1.4, zorder=3)
        )
        self.ax.text(
            x, y, text, ha="center", va="center", fontsize=FONT + 2, color=self.text, zorder=4
        )

    def note(self, x, y, text, size=FONT):
        self.ax.text(x, y, text, ha="left", va="center", fontsize=size, color=self.ink, zorder=4)

    def block(self, y0, lower, upper, lower_fill, upper_fill, lower_w, upper_w, add_text=r"$+$"):
        y1 = y0 + 2.1
        self.arrow((STREAM_X, y0), (BRANCH_X - lower_w / 2, y0))
        self.box(BRANCH_X, y0, lower, lower_fill, w=lower_w, h=BOX_H)
        self.box(BRANCH_X, y1, upper, upper_fill, w=upper_w, h=BOX_H)
        self.arrow((BRANCH_X, y0 + BOX_H / 2), (BRANCH_X, y1 - BOX_H / 2))
        self.arrow((BRANCH_X - upper_w / 2, y1), (STREAM_X + 0.36, y1))
        self.add(STREAM_X, y1, add_text)
        return y0, y1

    def rows(self, y, lengths, gains=None, label=None, x=None):
        """A weight as its rows (bar length = row length); with `gains`, a gain column."""
        n, h, w = len(lengths), 1.25, 1.35
        x0 = GLYPH_X if x is None else x
        for i, length in enumerate(lengths):
            yi = y + h / 2 - (i + 0.5) * h / n
            self.ax.add_patch(plt.Rectangle((x0, yi - 0.4 * h / n), length * w, 0.8 * h / n, fc=self.ink,
                                            ec="none", zorder=3))  # fmt: skip
        self.ax.plot([x0, x0], [y - h / 2, y + h / 2], color=self.edge, lw=1, zorder=4)
        if gains is not None:
            gx = x0 + w + 0.35
            for i, g in enumerate(gains):
                yi = y + h / 2 - (i + 0.5) * h / n
                self.ax.add_patch(plt.Rectangle((gx, yi - 0.4 * h / n), g * 0.9, 0.8 * h / n,
                                                fc=self.c["gain"], ec="none", zorder=3))  # fmt: skip
            self.ax.plot([gx, gx], [y - h / 2, y + h / 2], color=self.edge, lw=1, zorder=4)
            self.ax.text(gx + 0.45, y + h / 2 + 0.25, "$g$", ha="center", fontsize=FONT,
                         color=self.c["gain"])  # fmt: skip
        if label:
            self.ax.text(
                x0, y - h / 2 - 0.15, label, ha="left", va="top", fontsize=FONT - 1, color=self.ink
            )

    def stream(self, y_from, y_to, label, grow=False):
        if grow:
            ys = [y_from, y_to, y_to, y_from]
            xs = [STREAM_X - 0.05, STREAM_X - 0.32, STREAM_X + 0.32, STREAM_X + 0.05]
            self.ax.add_patch(
                plt.Polygon(list(zip(xs, ys, strict=True)), fc=self.ink, ec="none", zorder=1)
            )
        else:
            self.ax.plot([STREAM_X, STREAM_X], [y_from, y_to], color=self.ink, lw=3.2, zorder=1,
                         solid_capstyle="butt")  # fmt: skip
        self.ax.text(STREAM_X - 0.55, (ATTN_Y + MLP_Y) / 2 + 1.0, label, rotation=90, ha="right",
                     va="center", fontsize=FONT, color=self.ink)  # fmt: skip


def stock(ax, theme):
    d = Canvas(ax, theme, "stock", "pre-norm transformer", "free scales drift with optimizer noise")
    d.box(STREAM_X, 0.3, "embedding", d.fill, w=3.3, h=0.9)
    d.stream(0.75, HEAD_Y - 0.45, r"stream $x$,   $\Vert x\Vert\uparrow$", grow=True)
    _, attn = d.block(ATTN_Y, r"RMSNorm$\cdot\gamma$", r"$W_{qkv}\to$ attn" "\n" r"$\to W_o$", d.norm,
                      d.fill, NARROW_W, WIDE_W)  # fmt: skip
    _, mlp = d.block(MLP_Y, r"RMSNorm$\cdot\gamma$", r"$W_{up}\to$ SiLU" "\n" r"$\to W_{down}$", d.norm,
                     d.fill, NARROW_W, WIDE_W)  # fmt: skip
    d.box(STREAM_X, HEAD_Y, r"head $W$", d.fill, w=2.4, h=0.9)
    d.rows(HEAD_Y, SHRINKING, x=3.45)
    d.rows(attn, UNEVEN, label=r"$\gamma c,W/c$;  $W_q c,W_k/c$")
    d.rows(mlp, UNEVEN, label=r"$W_{up}\,c,\ W_{down}/c$")
    d.note(4.4, HEAD_Y, r"$x\,c\uparrow,\ \ W_{head}/c\downarrow$")
    d.note(
        3.45,
        TOP_Y - 0.15,
        "with a final norm, the\nhead no longer pins\nthe stream at all",
        FONT - 2,
    )


def fixed(ax, theme):
    d = Canvas(ax, theme, "fixed", "nGPT / EDM2", "every scale fixed by construction")
    d.box(STREAM_X, 0.3, r"$\hat w$ embedding", d.fill, w=3.5, h=0.9)
    d.stream(0.75, HEAD_Y - 0.45, r"stream $h$,   $\Vert h\Vert = 1$")
    attn, _ = d.block(ATTN_Y, "cosine attn:\n" r"$s_{qk}\,\hat q\cdot\hat k$,  $\hat W_{qkv},\hat W_o$", "Norm",
                      d.fill, d.norm, WIDE_W, NARROW_W, r"$\oplus$")  # fmt: skip
    mlp, _ = d.block(MLP_Y, r"$\hat W_{up}\to$ SiLU$\cdot s_u$" "\n" r"$\to\hat W_{down}$", "Norm", d.fill,
                     d.norm, WIDE_W, NARROW_W, r"$\oplus$")  # fmt: skip
    d.box(STREAM_X, HEAD_Y, r"head $s_z\,\hat W$", d.fill, w=2.8, h=0.9)
    d.rows(attn + 2.1, EVEN, label=r"$\hat w = w/\Vert w\Vert$, every step")
    d.rows(mlp + 2.1, EVEN, label=r"$\alpha, s_{qk}, s_u, s_v$ by hand")
    d.note(2.5, TOP_Y - 0.1, r"$\oplus$  nGPT: $h\leftarrow\mathrm{Norm}(h+\alpha\,(y-h))$" "\n"
           r"$\oplus$  EDM2: $\frac{(1-t)\,x+t\,y}{\sqrt{(1-t)^2+t^2}}$")  # fmt: skip


def ours(ax, theme):
    d = Canvas(
        ax, theme, "ours", "DragonScale", "unit rows; a gain only where the loss sees the scale"
    )
    d.box(STREAM_X, 0.3, UNIT + " embedding", d.fill, w=3.5, h=0.9)
    d.stream(0.75, HEAD_Y - 0.6, r"stream $x$")
    _, attn = d.block(ATTN_Y, "Norm", r"$\hat W_{qkv}$, QK-norm" "\n" r"$\to g\odot\hat W_o$", d.norm, d.fill,
                      NARROW_W, WIDE_W)  # fmt: skip
    _, mlp = d.block(MLP_Y, "Norm", r"$g\odot\hat W_{gate},\ \hat W_{up}$" "\n" r"$\to g\odot\hat W_{down}$",
                     d.norm, d.fill, NARROW_W, WIDE_W)  # fmt: skip
    d.box(STREAM_X, HEAD_Y, "head " + UNIT, d.fill, w=2.8, h=0.9)
    d.rows(attn, EVEN, GAINS, label="unit rows, one gain per row")
    d.rows(mlp, EVEN, GAINS, label="the loss moves $g$")
    d.note(GLYPH_X - 0.2, MLP_Y, "norms:\nno affine")
    d.note(3.85, TOP_Y, "no gain on the\nembedding, q, k\nand the head", FONT - 1)
    d.note(6.0, HEAD_Y + 0.2, r"$\hat W = W/\Vert W\Vert_{row}$", FONT - 1)


def stream_panel(ax, theme):
    t, c = THEMES[theme], SLOTS[theme]
    data = rows("a_stream.csv")
    for model in ("standard", "patched"):
        series(
            ax,
            [r for r in data if r["model"] == model],
            c[MODEL[model]],
            MARKERS[MODEL[model]],
            model,
        )
    bad = [
        r["lr"]
        for r in rows("a_fp16_amp_nonfinite.csv")
        if r["model"] == "standard" and r["nonfinite"]
    ]
    ys = [r["median"] for r in data if r["model"] == "standard" and r["lr"] in bad]
    ax.plot(
        bad, ys, ls="none", marker="x", ms=10, mew=1.8, color=t["ink"], label="fp16 AMP: non-finite"
    )
    ax.set(xscale="log", yscale="log", xlabel="learning rate", ylabel="hidden-state RMS")
    ax.set_title("small LM, Adam, 5 seeds", fontsize=FONT)
    ax.legend(loc="upper left", fontsize=FONT - 3)


def ladder_panel(ax, theme):
    t, c = THEMES[theme], SLOTS[theme]
    data = rows("b_ladder.csv")
    ys = list(range(len(data)))[::-1]
    for y, r in zip(ys, data, strict=True):
        color = c[MODEL["standard"]] if r["model"] == "standard" else c[MODEL["patched"]]
        ax.plot(
            [r["low"], r["high"]], [y, y], color=color, lw=4, alpha=0.35, solid_capstyle="round"
        )
        ax.plot(r["growth_per_decade"], y, marker="o", color=color, ms=8)
        count = int(r["free_scales"])
        ax.annotate(f"{count} free scale{'s' if count != 1 else ''}", (r["high"], y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=FONT - 2, color=t["muted"])  # fmt: skip
    ax.set_yticks(ys, [r["model"] for r in data])
    ax.set(xscale="log", xlim=(0.8, 3e5), ylim=(-0.6, len(data) - 0.4))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("growth per 10x learning rate")
    ax.set_title("fewer free scales, less growth", fontsize=FONT)


def draw(theme: str):
    from plot import style

    style(theme)
    sizes = {"font.size": FONT, "axes.labelsize": FONT, "xtick.labelsize": FONT - 2,
             "ytick.labelsize": FONT - 2}  # fmt: skip
    with plt.rc_context(sizes):
        fig = plt.figure(figsize=(23, 9), layout="constrained")
        grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.05])
        stock(fig.add_subplot(grid[:, 0]), theme)
        fixed(fig.add_subplot(grid[:, 1]), theme)
        ours(fig.add_subplot(grid[:, 2]), theme)
        stream_panel(fig.add_subplot(grid[0, 3]), theme)
        ladder_panel(fig.add_subplot(grid[1, 3]), theme)
        c = SLOTS[theme]
        handles = [Line2D([], [], color=c[MODEL[m]], lw=3, label=label) for m, label in
                   (("standard", "standard pre-norm transformer"), ("patched", "with DragonScale"))]  # fmt: skip
        legend = fig.legend(
            handles=handles, loc="outside upper center", ncol=2, frameon=False, fontsize=FONT + 1
        )
        legend.set_title(TITLE, prop={"size": FONT + 7, "weight": "bold"})
    return fig


def main() -> None:
    for theme, suffix in (("light", ""), ("dark", "_dark")):
        fig = draw(theme)
        fig.savefig(HERE / f"teaser{suffix}.png", dpi=110)
        plt.close(fig)
        print(HERE / f"teaser{suffix}.png")


if __name__ == "__main__":
    main()
