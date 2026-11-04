"""A: learning rate x precision, preset `stock`. V100 32GB, GPU 0.
Drives measure/bytelm/train.py.

    kogine workflow parallel measure/bytelm/train.py \
        --config measure/bytelm/configs/a_stock.py --workers 2
"""

DEVICE = "cuda:0"
EXPERIMENT = "a_lr_precision"
PRESET = "stock"
_sweep = {
    "LR": [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1],
    "PRECISION": ["fp16", "fp32"],
}
