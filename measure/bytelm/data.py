"""The byte corpus (the standard library's .py files) and random windows of it on the device."""

import torch
from common import DATA, stdlib_bytes

CORPUS = DATA / "stdlib.bin"
HELD_FRACTION = 0.05
EVAL_BATCHES = 8


def build_corpus() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    data = stdlib_bytes(1 << 40)
    CORPUS.write_bytes(data.numpy().tobytes())
    print(f"{CORPUS}: {len(data):,} bytes")


class Batches:
    """Random training windows, and `EVAL_BATCHES` fixed held-out windows."""

    def __init__(self, batch: int, context: int, seed: int, device: torch.device):
        data = torch.frombuffer(bytearray(CORPUS.read_bytes()), dtype=torch.uint8)
        split = int(len(data) * (1 - HELD_FRACTION))
        self.batch, self.context = batch, context
        self.train = data[:split].to(device)
        self.generator = torch.Generator(device).manual_seed(seed)
        held = data[split:].to(device)
        fixed = torch.Generator(device).manual_seed(1234)
        self.held = [self._window(held, fixed) for _ in range(EVAL_BATCHES)]

    def _window(self, data, generator):
        starts = torch.randint(
            len(data) - self.context - 1, (self.batch, 1), generator=generator, device=data.device
        )
        window = data[starts + torch.arange(self.context + 1, device=data.device)].long()
        return window[:, :-1], window[:, 1:]

    def next(self):
        return self._window(self.train, self.generator)
