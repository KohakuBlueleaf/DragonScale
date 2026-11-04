# DragonScale

A PyTorch library that keeps weight and activation sizes from drifting during training.

Some scales in a network can change without changing what the loss sees: in a transformer MLP you
can multiply `up` by 2 and divide `down` by 2 and get the same output. The loss gives no gradient
along such a direction, but every optimizer step still moves the weights a little along it, and
those moves add up. Over training the residual stream, the attention logits and the weight norms
grow, until fp16 overflows or training diverges. DragonScale finds these directions in any model,
gives the rows of the weights behind them a fixed length, and keeps one learned gain per row in
their place.

<!-- FIGURE: docs/assets/teaser.png, held-out loss and peak residual-stream value vs learning
rate, stock / matrices / arch (experiment A) -->

## Install

```bash
uv pip install -e .                  # extras: triton, tilelang, hf, measure, dev
```

Needs Python 3.11+ and torch 2.9+.

## Quick start

```python
import torch
import dragonscale as ds

model = MyModel()

# Which scales does the loss not see, which of them are risky, what should the model change?
report = ds.analysis.free_scales(model, example_input)
print(report.describe())

# Unit rows and a gain on every Linear / Conv / Embedding; the output is unchanged.
session = ds.apply(model, ds.analysis.suggest(model, example_input, level="matrices"))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)   # gains are ordinary parameters
# ... train as usual ...

session.bake()      # back to stock modules with the trained weights, same output
```

`ds.apply` works on a model you already have: an existing checkpoint is converted with the same
output at step 0. Training from scratch works the same way.

## Why scales drift

Give every parameter tensor $\theta_i$ a log-scale $s_i$ and look at the model as a function of
$\theta_i e^{s_i}$. A **free direction** is a vector $s$ along which the output does not change:

```math
f(\theta_1 e^{s_1}, \dots, \theta_n e^{s_n}) = f(\theta_1, \dots, \theta_n).
```

Examples: `up` $\times c$ with `down` $\times 1/c$; `q` $\times c$ with `k` $\times 1/c$; a norm's
gain $\times c$ with every layer that reads it $\times 1/c$; and any weight followed by a norm on
its own.

The loss is constant along a free direction, so its gradient has no component there. For a weight
$w$ whose scale is free, $\langle \nabla_w L, w\rangle = 0$, and one step $w' = w - \eta g$ gives

```math
\lVert w' \rVert^2 = \lVert w \rVert^2 + \eta^2 \lVert g \rVert^2 \ \ge\ \lVert w \rVert^2 .
```

The norm can only grow. With Adam every entry moves about $\eta$ per step whatever its gradient,
so a row of $n$ entries grows like a random walk, and a matrix row moves about $\sqrt{n}$ times
faster than a single gain. A free direction is **dangerous** when it moves activations and has a
matrix in it: that is where the drift is fast and the network sees it.

## What DragonScale changes

A patched layer uses unit rows times a gain in place of its weight:

```math
W_{\text{eff}} = \mathrm{diag}(g)\, r\, \frac{W}{\lVert W \rVert_{\text{row}}}
```

with $r$ the row radius (1 for linears and convs, $\sqrt{\text{dim}}$ for embeddings, so their rows
have unit RMS). The raw row length $\lVert W \rVert_{\text{row}}$ no longer reaches the output, so
the scale that drifted now lives in $g$: one number per row, moved by a gradient that is aligned
with the loss instead of by noise. The gradient reaching $W$ is tangent to its rows,

```math
\frac{\partial L}{\partial W} = \frac{g\, r}{\lVert W \rVert}\,\bigl(I - \hat w \hat w^\top\bigr)\, \frac{\partial L}{\partial W_{\text{eff}}},
```

so the raw rows still grow, slowly, and their growth shrinks the effective step of the direction:
a per-row $1/\sqrt{t}$ schedule.

## Analysis

`ds.analysis.free_scales(model, *inputs)` runs one forward pass, follows the scale of every
parameter through every op, and solves for the free directions. For a one-block pre-norm LM:

```text
7 free direction(s), 7 dangerous, 6 constraints over 13 groups
  dangerous: +1*embed.weight +1*pos.weight +1*blocks.0.out.weight +1*blocks.0.down.weight
  dangerous: +1*blocks.0.norm1.weight -2*blocks.0.k.weight -1*blocks.0.out.weight
  dangerous: +1*blocks.0.q.weight -1*blocks.0.k.weight
  dangerous: +1*blocks.0.v.weight -1*blocks.0.out.weight
  dangerous: +1*blocks.0.norm2.weight -1*blocks.0.gate.weight -1*blocks.0.down.weight
  dangerous: +1*blocks.0.up.weight -1*blocks.0.down.weight
  dangerous: +1*norm.weight -1*head.weight
architecture:
  attention calls [0]: q and k trade scale freely; normalize q and k (QK-norm) with a gain on one side only
  embed.weight, pos.weight, blocks.0.out.weight, blocks.0.down.weight scale the residual stream together; give the model input a fixed scale (an embedding: mode 'unit')
```

`gate` and `down` do not trade freely: SiLU pins the scale of `gate`.

`ds.analysis.suggest(model, *inputs, level=...)` turns the report into a plan:

| level | plan |
| --- | --- |
| `report` | nothing; read the report and its architecture notes |
| `minimal` | unit rows and a gain only on the layers in a dangerous direction |
| `matrices` | unit rows and a gain on every Linear / Conv / Embedding |
| `absorb` | each norm's affine is moved into the layers that read it and frozen; each layer whose row scale can move into the next layer (`v` into `out`, `up` into `down`) gets unit rows without a gain; the rest get unit rows and a gain |

Every level keeps the model's output. `absorb` plans carry the moves; run them before applying:

```python
plan = ds.analysis.suggest(model, x, level="absorb")
session = ds.apply(model, ds.analysis.absorb_plan(model, plan, x))
```

`ds.analysis.absorb(model, source, *inputs, into=None)` makes one move by hand; it finds the
consumers when `into` is None, refuses a move that is not free, and checks that the output did not
change. `ds.analysis.scale_jacobian` gives the spectrum of the same scale map numerically.

The architecture notes are what a weight patch cannot do; they belong in the model code.

## Patching options

Plans are data: `ds.make_plan(model, rules)` matches `ds.Rule(match, **options)` against module
paths or types, first match wins. `ds.apply` accepts a plan, a rule list or a preset name
(`unit_gain`, `unit`, `linear_unit_gain`, `conv_unit_gain`, `parametrize_unit_gain`).

| option | values | meaning |
| --- | --- | --- |
| `mode` | `unit_gain`, `unit`, `plain` | unit rows times a gain, unit rows alone, untouched |
| `gain` | `channel`, `scalar` | one gain per row, or one for the layer |
| `init` | `preserve`, `ones`, `reset`, `absorb` | gain = row length (same output), gain = 1, re-initialize then preserve, rows already moved by `absorb_plan` |
| `install` | `forward`, `parametrize`, `bake` | replace the module's forward, a torch parametrization, write the unit-row weight once |
| `method` | `normalize`, `rescale` | divide by the row length in every forward, or use the raw rows and let the optimizer rescale them after each step |

Supported layers: Linear, Conv1d/2d/3d, ConvTranspose1d/2d/3d, Embedding and transformers'
`Conv1D` (GPT-2); `ds.layer_types.register_layer_type` adds more. Tied weights (an LM head shared
with the embedding) are patched as one layer with one gain.

`ds.nn` has the same layers as modules (`UnitRowLinear`, `UnitRowConv2d`, `UnitRowEmbedding`, ...)
for new models, and `ds.functional.unit_rows(w, spec, gain)` does it for a bare weight.

## Optimizer

`method="normalize"` needs nothing from the optimizer. `method="rescale"` needs the gradient
decorator, which works with any torch optimizer through its step hooks:

```python
decorator = ds.optim.attach(optimizer, session)
```

Before each step it removes the part of each gradient that would change a row's length; after the
step it removes the same part from the momentum (Adam's `exp_avg`, SGD's and Muon's
`momentum_buffer`) and rescales each row back to its radius. Second moments are left alone.

## Checkpoints

| call | result |
| --- | --- |
| `session.state_dict("raw")` | raw weights and gains, under the module's own keys plus `<path>.gain` |
| `session.state_dict("baked")` | the unit-row weights under the stock keys; loads into the stock model |
| `session.bake()` | the model becomes stock modules with the unit-row weights, same output |
| `session.canonicalize()` | the patched modules become `ds.nn` modules in place (`native` state dict) |
| `session.remove()` | everything undone; the weights keep what training wrote |

`load_state_dict(state, format)` reads all three formats. Patched models work with
`copy.deepcopy`, `torch.save` of the whole model (`forward` install) and FSDP2.

`ds.monitor.Monitor(session)` reads each patched layer's output size, gains and raw row lengths
during training.

## Measured

All runs are in [`measure/`](measure) and run through
[kogine](https://github.com/KohakuBlueleaf/KohakuEngine): a script's settings are module-level
knobs, a config file sets them, and every number below comes from one config.

Three presets throughout:

| preset | what it is |
| --- | --- |
| `stock` | the model as written |
| `matrices` | `suggest(level="matrices")` on the stock model |
| `arch` | `suggest(level="absorb")` plus the architecture notes done in the model code: norms without affine, QK-norm with a gain on q only, embeddings with a fixed row scale |

### Byte-level LM

A pre-norm transformer (dim 256, 8 blocks, SwiGLU) on the bytes of the Python standard library,
AdamW, constant learning rate, no weight decay, 5k steps unless stated.

<!-- RESULT A: learning rate x precision (fp16 AMP and fp32), V100. Table: largest stable
learning rate and best held-out loss per preset and precision. Figure: held-out loss and peak
stream value vs learning rate. -->

<!-- RESULT B: 100k steps at lr 2e-3 and 1e-2. Figure: peak stream value and raw row length vs
step (log-log). -->

<!-- RESULT C: depth 4 / 8 / 16 / 32. Figure: peak stream value vs depth per preset. -->

<!-- RESULT D: weight decay 0.01 / 0.1 on stock. Table: does it prevent the divergence, and at
what loss. -->

<!-- RESULT E: width 128 / 256 / 512 / 1024. Figure: peak stream value and attention logits vs
width per preset. -->

### JiT-B/16 on Flowers-102

Pixel-space diffusion with the official JiT recipe (x-prediction, velocity loss, adaLN-Zero) at
256 px, batch 64, 10k steps.

<!-- RESULT JiT: per preset and learning rate, the held-out velocity loss and the peak stream,
adaLN gate and attention-logit values over training; stock in fp16 and fp32. -->

### Overhead

<!-- RESULT overhead: time and memory of a patched layer against the stock layer, per backend. -->

Reproduce, for example:

```bash
kogine run measure/bytelm/train.py --set BUILD_CORPUS=true
kogine workflow parallel measure/bytelm/train.py --config measure/bytelm/configs/a_stock.py --workers 2
```

## Repository

| path | contents |
| --- | --- |
| `src/dragonscale/` | the library: patching, plans, analysis, optimizer hooks, unit-row modules |
| `src/dragonscale_kernels/` | the unit-row ops per backend (eager now; Triton and TileLang next) |
| `src/dragonscale_common/` | helpers shared by both, without torch: logger, name registry, dotted-path import, problem lists |
| `measure/` | experiments and op measurements, one folder each, run with kogine |
| `tests/` | patch / remove / bake / canonicalize behaviour and the analysis on hand-checked models |

## License

Apache-2.0.
