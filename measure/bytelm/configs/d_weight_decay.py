"""D: weight decay on `stock`, fp16. V100 32GB, GPU 3.
Drives measure/bytelm/train.py.

    kogine workflow parallel measure/bytelm/train.py \
        --config measure/bytelm/configs/d_weight_decay.py --workers 2
"""

DEVICE = "cuda:3"
EXPERIMENT = "d_weight_decay"
PRESET = "stock"
_sweep = {"WEIGHT_DECAY": [0.01, 0.1], "LR": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]}
