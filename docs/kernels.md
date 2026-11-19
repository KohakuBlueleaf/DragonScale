# Kernels

The unit-row op computes `gain * r * w / ||w||_row` and its backward. Row lengths are summed in
fp32 whatever the weight's dtype, and the sum of squares is floored at 1e-24 before the square
root, so a zero row stays finite.

| backend | used when | what runs |
| --- | --- | --- |
| `triton` | CUDA with Triton installed (the default there) | hand-written kernels, and a grouped launch for many weights |
| `compile` | CUDA without the Triton kernels | `torch.compile` of the plain-torch op |
| `eager` | always (the default on the CPU) | plain torch with fp32 sums |

The first use of the op logs the backend chosen and why the others were passed over.
`ds.apply(model, plan, backend=...)` or `session.set_backend(...)` pins one. Under
`torch.compile` of a model, the op is traced as plain torch ops and fused with its neighbours.
The analysis always traces the plain-torch op.

## Benchmarks

The README shows the summary: speed of the op on three large weights, step time of a patched
layer, and the worst error per dtype. The per-shape numbers, measured on one NVIDIA B300
(PyTorch 2.15, Triton 3.9), are in `docs/figures/data/kernel_speed.csv`, `kernel_overhead.csv` and
`kernel_precision.csv`. To measure on your GPU:

```bash
python measure/ops/precision.py --out runs/bench   # error against fp64, every path
python measure/ops/speed.py --out runs/bench       # wall time and GPU time per path
python measure/ops/overhead.py --out runs/bench    # step time of a patched layer
```

Each script writes a `.jsonl` file and a markdown table into `--out`.

On a fast GPU the op is bound by the host's launch time for every shape measured, so `speed.py`
also reports the kernels' GPU time from the profiler; the README's bandwidth plot uses it.
