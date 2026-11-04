"""Diagnostic: stock, arch, and arch without the final norm, lr 1e-3, 2k steps, fp16, with the
stream split by token group. RTX 2080 Ti, GPU 2. Drives measure/jit/train.py.

    kogine run measure/jit/train.py --config measure/jit/configs/diag_final_norm.py
"""

DEVICE = "cuda:2"
EXPERIMENT = "jit_diag"
LR = 1e-3
STEPS = 2000
BATCH = 64
LOG_EVERY = 250
_sweep = {"__mode__": "zip", "PRESET": ["stock", "arch", "arch"], "FINAL_NORM": [True, True, False]}
