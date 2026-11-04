"""JiT-B/16 on Flowers-102, preset `stock`, fp32, 10k steps at batch 64.
RTX 3090, GPU 5. Drives measure/jit/train.py.

    kogine workflow parallel measure/jit/train.py \
        --config measure/jit/configs/jit_stock_fp32.py --workers 1
"""

DEVICE = "cuda:5"
EXPERIMENT = "jit_lr"
PRESET = "stock"
PRECISION = "fp32"
STEPS = 10_000
BATCH = 64  # 128 does not fit a 24 GB card in eager mode
_sweep = {"LR": [1e-3, 3e-3]}
