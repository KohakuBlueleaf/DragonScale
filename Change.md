# Change Log

## Unreleased

- Unit-row versions of Linear, Conv1d/2d/3d, ConvTranspose1d/2d/3d and Embedding, each with a
  learned gain; `ds.functional.unit_rows` for a bare weight.
- `ds.apply(model, ...)` patches an existing model. Supported module types: the layers above and
  transformers' `Conv1D` (GPT-2). Tied weights, such as an LM head shared with the embedding, are
  patched as one.
- `session.remove()` restores the original model exactly; `session.bake()` and the `baked`
  checkpoint turn it into a stock model that computes the same output; `session.canonicalize()`
  turns it into `dragonscale.nn` modules (the `native` checkpoint).
- Optimizer hooks for any torch optimizer, Muon included: keep each weight row at a fixed length
  and remove the part of the momentum that would change it.
- `ds.analysis`: lists the scales the loss cannot see, marks the risky ones, and adds
  architecture notes (free q/k scale, unanchored residual stream). `suggest` builds a plan at four
  levels: `report`, `minimal`, `matrices`, `absorb`. `absorb` moves a norm's or layer's scale
  into the next layer without changing the output.
- `ds.monitor.Monitor`: reads each patched layer's output size, gain and weight-row lengths.
- Patched models work with `copy.deepcopy`, whole-model `torch.save` (for the `forward`
  install), and FSDP2.
