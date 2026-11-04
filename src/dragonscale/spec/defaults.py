"""Named row specs: rows, columns, a whole matrix, blocks, and the layer-type defaults."""

from dragonscale.spec.rows import RowSpec


def rows(radius: float | str = 1.0) -> RowSpec:
    """Output rows of a Linear / Conv weight (`[out, in, *k]`)."""
    return RowSpec((0,), None, radius)


def columns(radius: float | str = 1.0) -> RowSpec:
    """Input columns of a Linear weight; per input channel over (out, spatial) for a Conv."""
    return RowSpec((1,), None, radius)


def matrix(radius: float | str = 1.0) -> RowSpec:
    """The whole weight as one Frobenius row."""
    return RowSpec((), None, radius)


def blocks(shape: tuple[int, ...], count: int, radius: float | str = 1.0) -> RowSpec:
    """`count` equal blocks along axis 0 (per-head blocks of a fused projection)."""
    if shape[0] % count:
        raise ValueError(f"axis 0 of {tuple(shape)} does not split into {count} blocks")
    return RowSpec((0,), (count, shape[0] // count, *shape[1:]), radius)


def conv_transpose(shape: tuple[int, ...], groups: int, radius: float | str = 1.0) -> RowSpec:
    """Output kernels of a ConvTranspose weight `[in, out / groups, *k]`."""
    channels_in, per_group, *kernel = shape
    return RowSpec((0, 2), (groups, channels_in // groups, per_group, *kernel), radius)


def embedding(radius: float | str = "sqrt_fan_in") -> RowSpec:
    """Token rows of an Embedding, unit RMS by default."""
    return RowSpec((0,), None, radius)
