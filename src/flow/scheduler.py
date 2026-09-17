"""Scheduler classes for affine probability paths."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Scheduler(ABC):
    def __init__(self, **kwargs): ...

    @abstractmethod
    def __call__(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Return alpha_t and beta_t

        Args:
            t: Times in [0, 1], shape (...)

        Returns:
            alpha_t: alpha_t, shape (...)
            beta_t: beta_t, shape (...)
        """
        pass

    @abstractmethod
    def derivative(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Returns the time-derivative of alpha_t and beta_t

        Args:
            t: Time in [0, 1], shape (...)

        Returns:
            d_alpha_t: Time derivative of alpha_t, shape (...)
            d_beta_t: Time derivative of beta_t, shape (...)
        """
        pass


class CondOTScheduler(Scheduler):
    """CondOT (linear) scheduler.

    alpha_t = t, beta_t = 1 - t
    """

    def __call__(self, t: Tensor) -> tuple[Tensor, Tensor]:
        return t, 1.0 - t

    def derivative(self, t: Tensor) -> tuple[Tensor, Tensor]:
        return torch.ones_like(t), -1 * torch.ones_like(t)


class PolynomialScheduler(Scheduler):
    """Polynomial scheduler."""

    def __init__(self, n: float | int, **kwargs):
        if n <= 0:
            raise ValueError(f"n must be positive. Got n={n}")
        self.n = n

    def __call__(self, t: Tensor) -> tuple[Tensor, Tensor]:
        return t**self.n, 1 - t**self.n

    def derivative(self, t: Tensor) -> tuple[Tensor, Tensor]:
        d_alpha_t = self.n * (t ** (self.n - 1))
        d_beta_t = -1 * self.n * (t ** (self.n - 1))
        return d_alpha_t, d_beta_t


class CosineScheduler(Scheduler):
    pass


class VPScheduler(Scheduler):
    pass


class LinearVPScheduler(Scheduler):
    pass


SCHEDULER_REGISTRY = {
    "condot": CondOTScheduler,
    "polynomial": PolynomialScheduler,
}
"""Mapping from scheduler name to class."""
