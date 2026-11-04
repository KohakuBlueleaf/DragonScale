"""Flowers-102 as one uint8 tensor: every image resized (shorter side) and center-cropped to a
square, from the Parquet files of `dpdl-benchmark/oxford_flowers102` on the Hugging Face Hub.

The train set is the `train` and `test` splits (7169 images), the held-out set is
`validation` (1020).
"""

import io
from pathlib import Path

import numpy as np
import torch
from common import DATA

RAW = DATA / "flowers102" / "raw"
SPLITS = {"train": ("train", "test"), "held": ("validation",)}


def cache_path(size: int) -> Path:
    return DATA / "flowers102" / f"images{size}.pt"


def build_cache(size: int) -> None:
    """Write `{split: (images [n, 3, size, size] uint8, labels [n] int64)}` to `cache_path`."""
    import pyarrow.parquet as pq
    from PIL import Image

    out = {}
    for name, sources in SPLITS.items():
        images, labels = [], []
        for path in sorted(p for s in sources for p in RAW.glob(f"{s}-*.parquet")):
            table = pq.read_table(path, columns=["image", "label"]).to_pydict()
            for image, label in zip(table["image"], table["label"], strict=True):
                picture = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
                scale = size / min(picture.size)
                width, height = (round(side * scale) for side in picture.size)
                picture = picture.resize((width, height), Image.BICUBIC)
                left, top = (width - size) // 2, (height - size) // 2
                picture = picture.crop((left, top, left + size, top + size))
                images.append(torch.from_numpy(np.asarray(picture).copy()).permute(2, 0, 1))
                labels.append(label)
        out[name] = (torch.stack(images), torch.tensor(labels))
        print(f"{name}: {len(labels)} images")
    torch.save(out, cache_path(size))
    print(f"{cache_path(size)}")


class Images:
    """The cached images on the device: random flipped training batches in [-1, 1], and a fixed
    held-out batch with fixed noise and noise levels."""

    def __init__(self, size: int, batch: int, held: int, seed: int, device: torch.device):
        cache = torch.load(cache_path(size))
        self.train, self.train_labels = (t.to(device) for t in cache["train"])
        self.batch = batch
        self.generator = torch.Generator(device).manual_seed(seed)
        fixed = torch.Generator(device).manual_seed(1234)
        images, labels = (t[:held].to(device) for t in cache["held"])
        self.held_x = self.scale(images)
        self.held_y = labels
        self.held_noise = torch.randn(self.held_x.shape, generator=fixed, device=device)
        self.held_t = (torch.arange(held, device=device) + 0.5) / held

    @staticmethod
    def scale(images: torch.Tensor) -> torch.Tensor:
        return images.float() / 127.5 - 1

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(self.train)
        index = torch.randint(n, (self.batch,), generator=self.generator, device=self.train.device)
        x = self.scale(self.train[index])
        flip = torch.rand(self.batch, generator=self.generator, device=x.device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        return x, self.train_labels[index]
