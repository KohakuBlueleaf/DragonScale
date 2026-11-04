"""C: depth 32, fp16. V100 16GB, GPU 7.
Drives measure/bytelm/train.py.

    kogine workflow parallel measure/bytelm/train.py \
        --config measure/bytelm/configs/c_depth32.py --workers 2
"""

DEVICE = "cuda:7"
EXPERIMENT = "c_depth"
DEPTH = 32
_sweep = {"PRESET": ["stock", "matrices", "arch"], "LR": [2e-3, 1e-2, 5e-2]}
