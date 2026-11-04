"""B: growth over 100k steps at lr 2e-3, fp16. RTX 2080 Ti, GPU 2.
Drives measure/bytelm/train.py.

    kogine workflow parallel measure/bytelm/train.py \
        --config measure/bytelm/configs/b_long_lr2e-3.py --workers 3
"""

DEVICE = "cuda:2"
EXPERIMENT = "b_long"
LR = 2e-3
STEPS = 100_000
_sweep = {"PRESET": ["stock", "matrices", "arch"]}
