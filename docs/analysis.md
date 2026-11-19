# Analysis

`ds.analysis.free_scales(model, *inputs, **kwargs)` runs one forward pass and gives every
parameter tensor `i` a log-scale `s_i`. Each op adds linear constraints on the `s_i` (a matmul
adds degrees, a sum needs equal degrees on both sides, a norm removes the degree of its input).
A **free scale** is a vector `s` that satisfies every constraint: scaling each tensor by
`exp(s_i)` leaves the output unchanged. Parameters with `requires_grad=False` and buffers are
constants.

With `A` the constraints and `N` a basis of the null space of `A`, and `D` the degrees of every
tensor the forward computes from the input:

| class | count |
| --- | --- |
| change a hidden state | `rank(D N)` |
| change none | `dim ker A - rank(D N)` |
| weakly pinned | `dim ker A0 - dim ker A`, where `A0` drops the anchors: constraints that only come from adding a constant or a fixed-scale tensor (a buffer position table, `x + 1`); an `eps` below 1e-3 is not an anchor |
| ops without a rule | op calls the trace has no rule for |

The counts do not depend on the basis printed. `report.summary()` gives them as a dict,
`report.describe()` prints them with one readable basis per class and the notes.

```python
import torch
from torch import nn

import dragonscale as ds


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 16)
        self.up, self.down = nn.Linear(16, 64), nn.Linear(64, 16)
        self.norm = nn.RMSNorm(16, eps=1e-6)
        self.head = nn.Linear(16, 32)

    def forward(self, t):
        h = self.embed(t)
        h = h + self.down(torch.relu(self.up(h)))
        return self.head(self.norm(h))


torch.manual_seed(0)
net, t = Net(), torch.randint(0, 32, (4, 8))
report = ds.analysis.free_scales(net, t)
print(report.describe())
```

```text
3 free scale(s): 3 change a hidden state, 0 change none; 0 weakly pinned; 0 op call(s) without a rule
  +1 embed.weight +1 up.bias +1 down.bias
  +1 up.weight +1 up.bias -1 down.weight
  +1 norm.weight -1 head.weight
```

The first is the residual stream: the norm in front of the head hides its size from the loss.
The second is `up` against `down` through the ReLU, the third the norm's weight against the
head.

## Plans

`ds.analysis.suggest(model, *inputs, level=...)` turns the analysis into a plan for `ds.apply`.

| level | plan | output at the start |
| --- | --- | --- |
| `report` | nothing | unchanged |
| `minimal` | unit rows and a gain on the layers in a free scale that changes a hidden state | unchanged |
| `matrices` | unit rows and a gain on every supported layer | unchanged |
| `absorb` | as `matrices`, plus moves: each norm's affine into the layers that read it (then frozen), and each pair scale into the next layer, whose source keeps unit rows without a gain | unchanged |
| `selective` (default) | `absorb`, then the gains the loss cannot see removed | changes in the layers whose gain is removed |

Plans with moves are run through `ds.analysis.absorb_plan(model, plan, *inputs)` before
`ds.apply`; it makes the moves, checks that the output did not change, and returns the plan
without moves.

```python
plan = ds.analysis.suggest(net, t)
print(plan.notes[0])
session = ds.apply(net, ds.analysis.absorb_plan(net, plan, t))
print(ds.analysis.free_scales(net, t).summary())
```

```text
unit rows without a gain change the output at the start of: down, embed
{'change_hidden_state': 0, 'change_none': 4, 'weakly_pinned': 0, 'ops_without_rule': 0}
```

The plan removes the gains of `embed` and `down`, the only stream writer, so the stream keeps the
embedding's fixed scale. In a model whose stream writers keep their gains (attention outputs,
MLP outputs), a norm in front of the output leaves the stream's scale weakly pinned, and
`plan.notes` says that the model code should remove that norm.

## Options of `selective`

- `gains="selective" | "all" | "none"`: the rule above, every gain kept, or every gain removed but
  the output layer's.
- `output_gain=<float>`: the output layer keeps a gain that starts at this value.
- `roles={"embed" | "output" | "qk" | "plain": regex}`: module paths that take a role whatever the
  analysis finds. `embed`, `output` and `qk` get unit rows without a gain (`output` keeps one when
  `output_gain` is set); `plain` layers are left unpatched. Use roles for what one forward pass
  cannot show: parts that should stay as they are (a projection head, a frozen tower), or an
  input or output path through an op the trace has no rule for.
- `allow_unknown=True`: go on when the trace meets an op without a rule (the counts are then
  incomplete).

## Notes

`report.notes` and `plan.notes` list what a weight patch cannot do: a norm in front of the
output, q and k without a QK-norm, an input with a free scale, layers whose output the plan
changes, free scales that remain.
