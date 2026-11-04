"""E: width 128 (head dim 64), fp16. RTX 3090, GPU 0.
Drives measure/bytelm/train.py.

    kogine workflow parallel measure/bytelm/train.py \
        --config measure/bytelm/configs/e_width128.py --workers 2
"""

DEVICE = "cuda:0"
EXPERIMENT = "e_width"
DIM = 128
HEADS = 2
_sweep = {"PRESET": ["stock", "matrices", "arch"], "LR": [2e-3, 1e-2, 5e-2]}
