"""One JiT training run on Flowers-102: train / held-out velocity loss and the hidden-state
statistics, as JSONL. Model and presets: `jit/model.py`; images: `jit/data.py`.

The model conditions by a token, not adaLN-Zero, unless `CONDITIONING = "adaln"`. Training follows
the official `denoiser.py`: noise level t = sigmoid(N(P_MEAN, P_STD)),
z = t x + (1 - t) noise, the network predicts x, the loss is on the velocity
v = (x - z) / max(1 - t, T_EPS); 10% of the labels are dropped to the null class.

    kogine run measure/jit/train.py --set BUILD_CACHE=true
    kogine run measure/jit/train.py --set PRESET=matrices --set BENCH=20
    kogine workflow parallel measure/jit/train.py --config measure/jit/configs/<file>.py \
        --workers 1

The output is `runs/jit/<EXPERIMENT>/<name>.jsonl` (`OUT` overrides it): the knobs and
environment, one row per log step, then `{"done": ...}`. A finished output is skipped. A run
stops after 3 log steps in a row with a non-finite held-out loss (`diverged`).
"""

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from common import RUNS, as_floats, finished, weight_stats

from jit.data import Images, build_cache
from jit.model import CHUNKS, Shape, build

# ---- what to run ----
PRESET = "stock"  # "stock" | "matrices" | "arch"
LR = 2e-4
STEPS = 20000
WARMUP = 500
BATCH = 128
BETA2 = 0.95
PRECISION = "fp16"  # "fp16" (autocast + loss scaler) | "fp32" (TF32 matmuls where available)
SEED = 0

# ---- denoiser (official defaults) ----
P_MEAN = -0.8
P_STD = 0.8
T_EPS = 0.05
NOISE = 1.0
LABEL_DROP = 0.1

# ---- model and data (JiT-B/16 at 256 px) ----
CONDITIONING = "tokens"  # "tokens" (time + class token, no zero init) | "adaln" (adaLN-Zero)
FINAL_NORM = True  # False: the output layer reads the residual stream directly
IMAGE = 256
PATCH = 16
DEPTH = 12
DIM = 768
HEADS = 12
HELD = 64  # held-out images, each at a fixed noise level spread over (0, 1)

# ---- execution ----
DEVICE = "cuda:0"  # nvidia-smi numbering
ATTENTION = "auto"  # SDPA backend: "efficient" | "flash" | "math" | "auto"
LOG_EVERY = 250
EXPERIMENT = "scratch"  # output folder under runs/jit/
OUT = ""  # "" = runs/jit/<EXPERIMENT>/<name>.jsonl
BENCH = 0  # > 0: time this many steps, print one line, write nothing
BUILD_CACHE = False  # write data/flowers102/images<IMAGE>.pt and stop

KNOBS = (
    "PRESET LR STEPS WARMUP BATCH BETA2 PRECISION SEED P_MEAN P_STD T_EPS NOISE LABEL_DROP "
    "CONDITIONING FINAL_NORM IMAGE PATCH DEPTH DIM HEADS HELD ATTENTION LOG_EVERY"
).split()
NULL_CLASS = 102


def velocity_loss(model, x, y, t, noise) -> torch.Tensor:
    """Mean squared velocity error of the x prediction at noise level `t` (one per image)."""
    t = t.view(-1, 1, 1, 1)
    z = t * x + (1 - t) * noise * NOISE
    gap = (1 - t).clamp_min(T_EPS)
    x_pred = model(z, t.flatten(), y)
    return ((x - x_pred) / gap).float().square().mean()


@torch.no_grad()
def evaluate(model, images: Images, autocast) -> dict:
    """Held-out velocity loss; block and final-layer magnitudes on the same batch."""
    for block in model.blocks:
        block.stats = {}
    with autocast():
        loss = velocity_loss(model, images.held_x, images.held_y, images.held_t, images.held_noise)
    row = {"held_loss": loss}
    blocks = [block.stats for block in model.blocks]
    for block in model.blocks:
        block.stats = None
    for key in blocks[0]:
        for j, stats in enumerate(blocks):
            row[f"blocks.{j}/{key}"] = stats[key]
        row[f"max/{key}"] = torch.stack([s[key] for s in blocks]).max()
    return row


def environment(device: torch.device) -> dict:
    return {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "autocast": "float16" if PRECISION == "fp16" else None,
        "chunks": list(CHUNKS),
    }


def run_name() -> str:
    final = "" if FINAL_NORM else "-nofinal"
    return f"{PRESET}-{CONDITIONING}{final}-lr{LR:g}-{PRECISION}-b{BATCH}-s{STEPS}"


def main() -> None:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if BUILD_CACHE:
        build_cache(IMAGE)
        return
    out = Path(OUT) if OUT else RUNS / "jit" / EXPERIMENT / f"{run_name()}.jsonl"
    if not BENCH and finished(out):
        print(f"skip, finished: {out}")
        return

    device = torch.device(DEVICE)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(SEED)
    images = Images(IMAGE, BATCH, HELD, SEED, device)
    shape = Shape(image=IMAGE, patch=PATCH, depth=DEPTH, dim=DIM, heads=HEADS)
    sample = (images.held_x[:2], images.held_t[:2], images.held_y[:2])
    model, session = build(
        PRESET, shape, sample, device, adaln=CONDITIONING == "adaln", final_norm=FINAL_NORM
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, BETA2), weight_decay=0)
    fp16 = PRECISION == "fp16"
    scaler = torch.amp.GradScaler(enabled=fp16)
    generator = torch.Generator(device).manual_seed(SEED + 1)

    def autocast():
        return torch.autocast(device.type, torch.float16, enabled=fp16)

    def train_step(step: int) -> torch.Tensor:
        for group in optimizer.param_groups:
            group["lr"] = LR * min(1.0, (step + 1) / WARMUP)
        x, y = images.next()
        drop = torch.rand(BATCH, generator=generator, device=device) < LABEL_DROP
        y = torch.where(drop, torch.full_like(y, NULL_CLASS), y)
        t = torch.sigmoid(torch.randn(BATCH, generator=generator, device=device) * P_STD + P_MEAN)
        noise = torch.randn(x.shape, generator=generator, device=device)
        with autocast():
            loss = velocity_loss(model, x, y, t, noise)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.detach()

    backends = {
        "efficient": [SDPBackend.EFFICIENT_ATTENTION],
        "flash": [SDPBackend.FLASH_ATTENTION],
        "math": [SDPBackend.MATH],
        "auto": [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
    }[ATTENTION]
    knobs = {name: globals()[name] for name in KNOBS}
    with sdpa_kernel(backends):
        if BENCH:
            for step in range(5):
                train_step(step)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            for step in range(BENCH):
                train_step(step)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - start
            row = {
                **knobs,
                **environment(device),
                "ms_per_step": 1000 * seconds / BENCH,
                "images_per_s": BENCH * BATCH / seconds,
                "peak_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                "parameters": sum(p.numel() for p in model.parameters()),
            }
            print(json.dumps(row))
            return

        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as file:

            def write(row: dict) -> None:
                file.write(json.dumps(row) + "\n")
                file.flush()

            write({"knobs": knobs, "environment": environment(device)})
            bad, started, train_losses = 0, time.time(), []
            for step in range(STEPS + 1):
                if step % LOG_EVERY == 0:
                    model.eval()
                    row = {"step": step, **evaluate(model, images, autocast)}
                    model.train()
                    row.update(weight_stats(model, session))
                    if train_losses:
                        row["train_loss"] = torch.stack(train_losses).mean()
                    train_losses = []
                    row = as_floats(row)
                    row["loss_scale"] = scaler.get_scale() if fp16 else None
                    row["seconds"] = time.time() - started
                    write(row)
                    bad = bad + 1 if not math.isfinite(row["held_loss"]) else 0
                    if bad >= 3:
                        write({"done": "diverged", "step": step})
                        return
                if step == STEPS:
                    break
                train_losses.append(train_step(step))
            write({"done": "finished", "step": STEPS, "seconds": time.time() - started})


if __name__ == "__main__":
    main()
