"""JiT-B/16 on Flowers-102, preset `stock`, fp16, 10k steps at batch 128.
RTX 3090, GPU 0. Drives measure/jit/train.py.

    kogine workflow parallel measure/jit/train.py \
        --config measure/jit/configs/jit_stock.py --workers 1
"""

DEVICE = "cuda:0"
EXPERIMENT = "jit_lr"
PRESET = "stock"
PRECISION = "fp16"
STEPS = 10_000
_sweep = {"LR": [2e-4, 1e-3, 3e-3]}
