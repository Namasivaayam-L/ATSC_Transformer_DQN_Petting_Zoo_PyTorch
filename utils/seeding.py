"""Determinism harness — call at the start of every entry point.

Seeds Python's `random`, NumPy, PyTorch (CPU + CUDA), and threads. When feasible,
turns on `torch.use_deterministic_algorithms` so the only randomness left is from
the env (which is also seeded via SUMO's `--seed` flag, plumbed in evaluate.py).

Usage:
    from utils.seeding import seed_everything
    seed_everything(cfg.seed)
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed every relevant RNG.

    Args:
        seed: integer seed.
        deterministic: if True, also turn on
            `torch.use_deterministic_algorithms(True)`. Slower but fully
            reproducible on CPU. Default False (cheaper).
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        # cuDNN determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Some kernels in the env are non-deterministic by design; this is a
    # conscious trade-off documented in the limitations section.
