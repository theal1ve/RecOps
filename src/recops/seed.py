"""Global seeding for reproducible runs."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch.

    Note: PYTHONHASHSEED only affects subprocesses started after this call.
    Libraries with their own RNG (CatBoost, implicit) get the seed via config.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
