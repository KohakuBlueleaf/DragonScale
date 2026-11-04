# Change Log

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
