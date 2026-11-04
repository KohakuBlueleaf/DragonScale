"""E: width 1024 (head dim 64), fp16. RTX 3090, GPU 5.
Drives measure/bytelm/train.py.

    kogine workflow parallel measure/bytelm/train.py \
        --config measure/bytelm/configs/e_width1024.py --workers 2
"""

DEVICE = "cuda:5"
EXPERIMENT = "e_width"
DIM = 1024
HEADS = 16
_sweep = {"PRESET": ["stock", "matrices", "arch"], "LR": [2e-3, 1e-2, 5e-2]}
