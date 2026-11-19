"""Arch tags: `smXX` for a CUDA device, `xpu` for Intel, `cpu` otherwise; and kernel families."""

import torch

FAMILIES = {
    70: "volta",
    75: "turing",
    80: "ampere",
    86: "ampere",
    87: "ampere",
    89: "ada",
    90: "hopper",
    100: "blackwell-datacenter",
    103: "blackwell-datacenter",
    120: "blackwell-consumer",
    121: "blackwell-consumer",
}


def arch_of(device: torch.device | str) -> str:
    """The arch tag of `device`."""
    device = torch.device(device)
    match device.type:
        case "cuda":
            major, minor = torch.cuda.get_device_capability(device)
            return f"sm{major * 10 + minor}"
        case "xpu":
            return "xpu"
        case _:
            return "cpu"


def device_type_of(arch: str) -> str:
    """The device type of an arch tag: `cuda`, `xpu` or `cpu`."""
    return "cuda" if arch.startswith("sm") else arch


def sm_of(arch: str) -> int:
    """The shader model of a CUDA arch tag, 0 otherwise."""
    return int(arch.removeprefix("sm")) if arch.startswith("sm") else 0


def family_of(arch: str) -> str:
    """The kernel family of an arch tag; an unknown card is its own family."""
    if not arch.startswith("sm"):
        return arch
    return FAMILIES.get(sm_of(arch), arch)
