"""A, extended: preset `arch` above lr 1e-1. V100 32GB GPU 0-2 ran the rest of A; this runs on
the V100 16GB, GPU 7. Drives measure/bytelm/train.py.

    kogine workflow parallel measure/bytelm/train.py \
        --config measure/bytelm/configs/a_arch_high.py --workers 2
"""

DEVICE = "cuda:7"
EXPERIMENT = "a_lr_precision"
PRESET = "arch"
_sweep = {"LR": [2e-1, 5e-1, 1.0], "PRECISION": ["fp16", "fp32"]}
