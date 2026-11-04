"""JiT-B/16 on Flowers-102, preset `matrices`, fp16, 10k steps at batch 128.
RTX 3090, GPU 1. Drives measure/jit/train.py.

    kogine workflow parallel measure/jit/train.py \
        --config measure/jit/configs/jit_matrices.py --workers 1
"""

DEVICE = "cuda:1"
EXPERIMENT = "jit_lr"
PRESET = "matrices"
PRECISION = "fp16"
STEPS = 10_000
_sweep = {"LR": [2e-4, 1e-3, 3e-3]}
