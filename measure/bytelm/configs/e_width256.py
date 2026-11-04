"""E: width 256 (head dim 64), fp16. RTX 3090, GPU 1.
Drives measure/bytelm/train.py.

    kogine workflow parallel measure/bytelm/train.py \
        --config measure/bytelm/configs/e_width256.py --workers 2
"""

DEVICE = "cuda:1"
EXPERIMENT = "e_width"
DIM = 256
HEADS = 4
_sweep = {"PRESET": ["stock", "matrices", "arch"], "LR": [2e-3, 1e-2, 5e-2]}
