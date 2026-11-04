"""Diagnostic: arch with input norms, read-only context tokens and no final norm, lr 1e-3, 2k
steps, fp16. RTX 2080 Ti, GPU 2. Drives measure/jit/train.py.

    kogine run measure/jit/train.py --config measure/jit/configs/diag_arch.py
"""

DEVICE = "cuda:2"
EXPERIMENT = "jit_diag_read_only"
PRESET = "arch"
FINAL_NORM = False
LR = 1e-3
STEPS = 2000
BATCH = 64
LOG_EVERY = 250
