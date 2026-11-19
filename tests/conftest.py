"""Test configuration: kernel backends loaded once, a fixed seed per test."""

import pytest
import torch

from dragonscale.kernels.registry import load_backends


def pytest_configure(config):
    load_backends()


@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(0)
