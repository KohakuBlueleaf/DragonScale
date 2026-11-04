"""B: growth over 16k steps at lr 1e-2, fp16. RTX 2080 Ti, GPU 3.
Drives measure/bytelm/train.py.

    kogine workflow parallel measure/bytelm/train.py \
        --config measure/bytelm/configs/b_long_lr1e-2.py --workers 3
"""

DEVICE = "cuda:3"
EXPERIMENT = "b_long"
LR = 1e-2
STEPS = 16_000
_sweep = {"PRESET": ["stock", "matrices", "arch"]}
