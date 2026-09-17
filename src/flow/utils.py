"""Utilities for reproducible experiments."""

import random

import numpy as np
import torch


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker for reproducibility.

    Use as `worker_init_fn` in torch.utils.data.DataLoader when num_workers > 0.

    Args:
        worker_id: Worker ID (passed automatically by DataLoader).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed: int = 0) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
