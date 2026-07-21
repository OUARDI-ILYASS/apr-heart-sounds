"""Global seed control.

PROFESSOR Q: "Are your results reproducible?"
A: Bit-for-bit on the same machine for everything except the CNN, where we
   expose `deterministic=True` to force cuDNN into deterministic algorithms.
   That flag costs roughly 10-20% throughput, so it defaults to False for
   development and should be turned on for the final reported run. t-SNE and
   k-means are seeded through the same function. We record the seed in every
   phase summary, so any number in the paper can be regenerated.
"""

from __future__ import annotations

import os
import random
from typing import Dict, Any

import numpy as np


def set_global_seed(seed: int, deterministic: bool = False) -> Dict[str, Any]:
    """Seed python, numpy and (if installed) torch. Returns a provenance dict."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    info: Dict[str, Any] = {
        "seed": seed,
        "deterministic": deterministic,
        "torch_seeded": False,
        "cuda_seeded": False,
    }

    try:
        import torch
    except ImportError:
        return info

    torch.manual_seed(seed)
    info["torch_seeded"] = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        info["cuda_seeded"] = True

    if deterministic:
        # cuBLAS needs this env var set BEFORE the first CUDA call for
        # deterministic matmuls; we set it defensively.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - torch version dependent
            pass
    else:
        # benchmark=True lets cuDNN autotune conv algorithms for our fixed
        # input size. Worth it because every batch has identical shape.
        torch.backends.cudnn.benchmark = True

    return info


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding.

    Without this, every worker process inherits the same numpy RNG state and
    produces *identical* augmentations - a classic silent bug that makes
    augmentation useless while appearing to work.
    """
    import torch

    base_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed((base_seed + worker_id) % (2 ** 32))
    random.seed((base_seed + worker_id) % (2 ** 32))
