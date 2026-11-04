"""JiT-B/16 (token conditioning) on Flowers-102, preset `arch`, fp16, 10k steps at batch 64.
RTX 3090, GPU 4. Drives measure/jit/train.py.

    kogine workflow parallel measure/jit/train.py \
        --config measure/jit/configs/jit_arch.py --workers 1
"""

DEVICE = "cuda:4"
EXPERIMENT = "jit_tokens"
PRESET = "arch"
PRECISION = "fp16"
STEPS = 10_000
BATCH = 64  # 128 does not fit a 24 GB card in eager mode
_sweep = {"LR": [2e-4, 1e-3, 3e-3]}
