"""One training run of the byte-level LM: train / held-out loss and the hidden-state statistics,
as JSONL. Model and presets: `bytelm/model.py`.

    kogine run measure/bytelm/train.py --set BUILD_CORPUS=true
    kogine run measure/bytelm/train.py --set PRESET=matrices --set LR=1e-2
    kogine run measure/bytelm/train.py --set PRESET=arch --set BENCH=50
    kogine workflow parallel measure/bytelm/train.py --config measure/bytelm/configs/<file>.py \
        --workers 2

The output is `runs/bytelm/<EXPERIMENT>/<name>.jsonl` (`OUT` overrides it): the knobs and
environment, one row per log step, then `{"done": ...}`. A finished output is skipped. A run
stops after 3 log steps in a row with a non-finite or > 10 held-out loss (`diverged`).
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
import torch.nn.functional as F
from common import RUNS, as_floats, finished, weight_stats
from torch import nn

from bytelm.data import Batches, build_corpus
from bytelm.model import Shape, build

# ---- what to run ----
PRESET = "stock"  # "stock" | "matrices" | "arch"
LR = 3e-3
STEPS = 5000
WARMUP = 100
WEIGHT_DECAY = 0.0
BETA2 = 0.95
PRECISION = "fp16"  # "fp16" (autocast + loss scaler) | "fp32" (TF32 matmuls where available)
SEED = 0

# ---- model and data ----
DEPTH = 8
DIM = 256
HEADS = 4
CONTEXT = 256
BATCH = 64

# ---- execution ----
DEVICE = "cuda:0"  # nvidia-smi numbering
ATTENTION = "efficient"  # SDPA backend: "efficient" | "flash" | "math" | "auto"
COMPILE = False
LOG_EVERY = 100
EXPERIMENT = "scratch"  # output folder under runs/bytelm/
OUT = ""  # "" = runs/bytelm/<EXPERIMENT>/<name>.jsonl
BENCH = 0  # > 0: time this many steps, print one line, write nothing
BUILD_CORPUS = False  # write data/stdlib.bin and stop

KNOBS = (
    "PRESET LR STEPS WARMUP WEIGHT_DECAY BETA2 PRECISION SEED DEPTH DIM HEADS CONTEXT BATCH "
    "ATTENTION COMPILE LOG_EVERY"
).split()


@torch.no_grad()
def evaluate(model: nn.Module, batches: Batches, autocast) -> dict:
    """Held-out loss over the fixed batches; block magnitudes on the first one."""
    losses, row, blocks = [], {}, []
    for i, (tokens, targets) in enumerate(batches.held):
        for block in model.blocks:
            block.stats = {} if i == 0 else None
        with autocast():
            logits = model(tokens)
        losses.append(F.cross_entropy(logits.float().flatten(0, 1), targets.flatten()))
        if i == 0:
            blocks = [block.stats for block in model.blocks]
            row["logits_max"] = logits.float().abs().max()
    for block in model.blocks:
        block.stats = None
    row["held_loss"] = torch.stack(losses).mean()
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
    }


def run_name() -> str:
    wd = f"-wd{WEIGHT_DECAY:g}" if WEIGHT_DECAY else ""
    return f"{PRESET}-lr{LR:g}-{PRECISION}-d{DEPTH}-w{DIM}-s{STEPS}{wd}"


def main() -> None:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if BUILD_CORPUS:
        build_corpus()
        return
    out = Path(OUT) if OUT else RUNS / "bytelm" / EXPERIMENT / f"{run_name()}.jsonl"
    if not BENCH and finished(out):
        print(f"skip, finished: {out}")
        return

    device = torch.device(DEVICE)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(SEED)
    batches = Batches(BATCH, CONTEXT, SEED, device)
    shape = Shape(DEPTH, DIM, HEADS, CONTEXT)
    model, session = build(PRESET, shape, batches.held[0][0], device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, BETA2), weight_decay=WEIGHT_DECAY
    )
    fp16 = PRECISION == "fp16"
    scaler = torch.amp.GradScaler(enabled=fp16)

    def autocast():
        return torch.autocast(device.type, torch.float16, enabled=fp16)

    forward = torch.compile(model) if COMPILE else model

    def train_step(step: int) -> torch.Tensor:
        for group in optimizer.param_groups:
            group["lr"] = LR * min(1.0, (step + 1) / WARMUP)
        tokens, targets = batches.next()
        with autocast():
            logits = forward(tokens)
        loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
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
            for step in range(10):
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
                "tokens_per_s": BENCH * BATCH * CONTEXT / seconds,
                "peak_mib": torch.cuda.max_memory_allocated(device) / 2**20,
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
                    row = {"step": step, **evaluate(model, batches, autocast)}
                    row.update(weight_stats(model, session))
                    if train_losses:
                        row["train_loss"] = torch.stack(train_losses).mean()
                    train_losses = []
                    row = as_floats(row)
                    row["loss_scale"] = scaler.get_scale() if fp16 else None
                    row["seconds"] = time.time() - started
                    write(row)
                    held = row["held_loss"]
                    bad = bad + 1 if not math.isfinite(held) or held > 10 else 0
                    if bad >= 3:
                        write({"done": "diverged", "step": step})
                        return
                if step == STEPS:
                    break
                train_losses.append(train_step(step))
            write({"done": "finished", "step": STEPS, "seconds": time.time() - started})


if __name__ == "__main__":
    main()
