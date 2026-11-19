"""A patched model under DistributedDataParallel and FSDP2 (`fully_shard`), two CPU processes on
the gloo backend: after a few AdamW steps on different data per rank, the weights equal those of
one process that trains on both ranks' data."""

import copy
import os
import socket
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

import dragonscale as ds

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not dist.is_available(), reason="gloo process group on Linux / macOS"
)
STEPS, LR = 3, 1e-2


def _model() -> nn.Module:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Embedding(32, 16), nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 32))
    ds.apply(model, "unit_gain", group=False)
    return model


def _data(rank: int) -> torch.Tensor:
    return torch.randint(0, 32, (4, 8), generator=torch.Generator().manual_seed(rank))


def _loss(model, tokens):
    return nn.functional.cross_entropy(model(tokens).flatten(0, 1), tokens.flatten())


def _train(model, batches):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    for _ in range(STEPS):
        optimizer.zero_grad()
        for tokens in batches:
            (_loss(model, tokens) / len(batches)).backward()
        optimizer.step()


def _worker(rank: int, world: int, port: int, kind: str, out: str) -> None:
    # CPU processes: the CUDA device of the host, if any, stays out of both groups
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), CUDA_VISIBLE_DEVICES="")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        model = _model()
        if kind == "ddp":
            wrapped = nn.parallel.DistributedDataParallel(model)
            _train(wrapped, [_data(rank)])
            state = model.state_dict()
        else:
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.fsdp import fully_shard

            mesh = init_device_mesh("cpu", (world,))

            for layer in model:
                if any(True for _ in layer.parameters()):
                    fully_shard(layer, mesh=mesh)
            fully_shard(model, mesh=mesh)
            _train(model, [_data(rank)])
            state = {
                k: v.full_tensor() if hasattr(v, "full_tensor") else v
                for k, v in model.state_dict().items()
            }
        if rank == 0:
            torch.save({k: v.detach().clone() for k, v in state.items()}, out)
    finally:
        dist.destroy_process_group()


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("kind", ["ddp", "fsdp2"])
def test_distributed_training_matches_one_process(kind, tmp_path):
    out = str(tmp_path / "state.pt")
    mp.spawn(_worker, args=(2, _port(), kind, out), nprocs=2, join=True)
    got = torch.load(out)
    reference = _model()
    _train(reference, [_data(0), _data(1)])
    want = reference.state_dict()
    assert set(got) == set(want)
    for key, value in want.items():
        torch.testing.assert_close(got[key], value, rtol=1e-5, atol=1e-6, msg=key)


def test_deepcopy_and_whole_model_save(tmp_path):
    model = _model()
    x = _data(0)
    twin = copy.deepcopy(model)
    torch.save(model, tmp_path / "model.pt")
    loaded = torch.load(tmp_path / "model.pt", weights_only=False)
    for other in (twin, loaded):
        torch.testing.assert_close(other(x), model(x))
