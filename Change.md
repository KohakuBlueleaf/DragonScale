# Change Log

## 2026-11-19 update to 0.2.0

A clean break: no aliases for the removed or renamed names.

### Removed

- `ds.optim.attach` (the gradient decorator), `method="rescale"`, `Rule(method=...)`,
  `LayerPlan.method`, `reset_raw_norms`, `FIRST_MOMENTS`, `ds.functional.gain_scaled`, and the kernel
  ops used only by them (`PROJECT`, `RESCALE`, `SCALE_ROWS`, `OUT_SCALE`, `IN_SCALE`). The method
  changes the model only; any torch optimizer works unchanged.
- The tilelang backend and the `tilelang` extra; the unused autotune table (`tuning.py`).
- The `measure` extra and its `kohaku-engine` dependency; `measure/bytelm`, `measure/jit`,
  `measure/plots.py` and their figures.

### Renamed and moved

- `dragonscale_kernels` -> `dragonscale.kernels`; `dragonscale_common` is gone (`dragonscale.log`,
  `dragonscale.problems`). The name registry is removed; `PRESETS` is a dict.
- Report: `Direction.moves_activations`, `.dangerous`, `.fast_ends`, `FreeScales.dangerous` and
  `.directions` are replaced by `FreeScales.changing`, `.unchanging`, `.weak`, `.unknown` and
  `summary()`. "Dangerous" is now "changes a hidden state"; the counts are subspace dimensions and
  do not depend on the basis.

### Behaviour changes

- `ds.analysis.suggest` defaults to the new `level="selective"` (was `matrices`). It removes the
  gains the loss cannot see, which resets the rows of those layers: the output at the start
  changes there (`ds.apply` logs the layers). `level="absorb"` keeps the output.
- `suggest` raises when the trace meets an op without a rule (`allow_unknown=True` to go on);
  `free_scales` warns and `describe()` starts with a warning line.
- New degree rules: `norm`, `linalg_vector_norm`, in-place ops, `maximum` / `minimum`, comparisons,
  `addcmul` / `addcdiv`, `logsumexp`, `var_mean` / `std_mean`, scatter and index ops,
  `repeat_interleave`, the SDPA entry points, the unit-row op itself. Counts change for models that
  use them.
- Exactly zero tensors (`zeros`, `zeros_like`) fit any scale in `cat`, `add` and `where`: writers
  that update only part of a sequence (`cat([zeros, update])`) are no longer pinned.
- Parameters with `requires_grad=False` are constants in the analysis, like buffers (all of them
  count when none requires a gradient). Before, a norm affine frozen by `absorb_plan` still took
  part in free scales: on two `nn.TransformerEncoderLayer(norm_first=True)` between an embedding
  and a head, `selective` left 2 free scales that change a hidden state (each `in_proj` gain
  against its frozen `norm1`) and patched `linear2` without a gain. Now: 0 such free scales, and
  `linear2` keeps its gain.
- `nn.MultiheadAttention` is a supported layer (`in_proj_weight`, one row per output row); its
  `out_proj` and the linears of `nn.TransformerEncoderLayer`, which their parents read outside
  their own forward, are patched through `parametrize` automatically, and a `forward` install
  there is refused.
- The first forward in each mode checks that no op reads the raw weight of a patched layer outside
  the unit-row computation, and raises if one does.
- The analysis traces with plain torch ops whatever the kernel backend.
- Eval mode recomputes the unit rows on every forward; caching happens only inside
  `with session.cache_weights():`; `session.invalidate()` drops the cache.
- Tied layers whose rules disagree on mode, gain, init or spec are refused.
- `find_consumers` returns no consumers for a layer whose scale is free on its own; the `absorb`
  level keeps such a layer's gain.
- Unit rows: the sum of squares is floored before the square root in every path, so a zero row
  stays finite in the forward and backward.
- The kernel backend chosen at first use is logged once per op and device type.

### Added

- `level="selective"` with `gains`, `output_gain` and `roles`; `Plan.notes`.
- `ds.optim.param_groups(session, weight_decay, **overrides)`.
- `ds.nn.QKNorm`, `ds.fold_layerscale`.
- `Rule(init=<number>)`: a gain that starts at that value.
- `Monitor` readings `raw_norm_mean`, `gain_min`, `angle_median`, `angle_max` (`angles=True`).
- Tests for DDP and FSDP2 on two processes, the README and docs code blocks, and the plans of four
  small model families; benchmarks of speed, precision and step overhead in `measure/ops`.

## 2026-11-05 update to 0.1.0

First release.

- Unit-row versions of Linear, Conv1d/2d/3d, ConvTranspose1d/2d/3d and Embedding, each with a
  learned gain; `ds.functional.unit_rows` for a bare weight.
- `ds.apply(model, ...)` patches an existing model with the same output at step 0. Supported
  module types: the layers above and transformers' `Conv1D` (GPT-2). Tied weights, such as an LM
  head shared with the embedding, are patched as one layer with one gain.
- `session.remove()` restores the original model exactly; `session.bake()` and the `baked`
  checkpoint turn it into a stock model that computes the same output; `session.canonicalize()`
  turns it into `dragonscale.nn` modules (the `native` checkpoint).
- `method="rescale"` with the gradient decorator (`ds.optim.attach`) for any torch optimizer,
  Muon included: keeps each weight row at a fixed length and removes the part of the momentum that
  would change it.
- `ds.analysis.free_scales`: lists the scales the loss cannot see, marks the dangerous ones and
  the weakly pinned ones, and adds architecture notes. `suggest` builds a plan at four levels:
  `report`, `minimal`, `matrices`, `absorb`; `absorb_plan` moves a norm's or layer's scale into the
  next layer without changing the output.
- Kernels: a Triton backend for the row ops (sm70 and newer) at 85-96% of memory bandwidth on large
  weights, the `torch.compile` and eager backends, and the fastest available picked by default.
- Patched models compile with `torch.compile` into one graph; in eager mode the unit-row weights
  of all eligible layers are computed together in a few launches (`ds.apply(..., group=True)`).
- `ds.monitor.Monitor`: reads each patched layer's output size, gain and weight-row lengths.
- Patched models work with `copy.deepcopy`, whole-model `torch.save` (for the `forward`
  install), activation checkpointing and FSDP2.
