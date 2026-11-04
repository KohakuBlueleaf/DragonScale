"""E: width 512 (head dim 64), fp16. RTX 3090, GPU 4.
Drives measure/bytelm/train.py.

    kogine workflow parallel measure/bytelm/train.py \
        --config measure/bytelm/configs/e_width512.py --workers 2
"""

DEVICE = "cuda:4"
EXPERIMENT = "e_width"
DIM = 512
HEADS = 8
_sweep = {"PRESET": ["stock", "matrices", "arch"], "LR": [2e-3, 1e-2, 5e-2]}
