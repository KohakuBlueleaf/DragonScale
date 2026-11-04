"""Test configuration: kernel tuning off against a temporary cache, a fixed seed per test."""

import os
import shutil
import tempfile

import pytest
import torch

from dragonscale_kernels import tuning
from dragonscale_kernels.registry import load_backends

SESSION_CACHE: list[str] = []


def pytest_configure(config):
    root = tempfile.mkdtemp(prefix="dragonscale-test-cache-")
    SESSION_CACHE.append(root)
    os.environ[tuning.CACHE_ENV] = root
    tuning.set_mode("off")
    load_backends()


def pytest_unconfigure(config):
    for root in SESSION_CACHE:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(0)
