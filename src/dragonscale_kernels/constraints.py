"""`requires`: builds the `supports(arch)` function a backend registers with an op."""

from collections.abc import Callable

from dragonscale_kernels.hardware import device_type_of, family_of, sm_of


def requires(
    device_types: tuple[str, ...] = ("cuda",),
    min_sm: int = 0,
    max_sm: int = 10_000,
    families: tuple[str, ...] = (),
) -> Callable[[str], str | None]:
    """A `supports` function: the reason an arch tag is rejected, else None."""

    def supports(arch: str) -> str | None:
        device_type = device_type_of(arch)
        if device_type not in device_types:
            return f"device type {device_type} not in {device_types}"
        if device_type == "cuda" and not min_sm <= sm_of(arch) <= max_sm:
            return f"{arch} outside sm{min_sm}..sm{max_sm}"
        if families and family_of(arch) not in families:
            return f"{family_of(arch)} not in {families}"
        return None

    return supports
