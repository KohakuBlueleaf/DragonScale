"""C: depth 16, fp16. V100 16GB, GPU 6.
Drives measure/bytelm/train.py.

    kogine workflow parallel measure/bytelm/train.py \
        --config measure/bytelm/configs/c_depth16.py --workers 2
"""

DEVICE = "cuda:6"
EXPERIMENT = "c_depth"
DEPTH = 16
_sweep = {"PRESET": ["stock", "matrices", "arch"], "LR": [2e-3, 1e-2, 5e-2]}
