"""Diffusion coefficient classes for stochastic solvers."""

from abc import ABC, abstractmethod

import torch


class DiffusionCoefficient(ABC):
    def __init__(self, sigma: float):
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative. Got sigma={sigma}")
        self.sigma = sigma

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Return the diffusion coefficient at time t.

        Args:
            t: Time in [0, 1], shape (...)

        Returns:
            Diffusion coefficient at time t, shape (...)
        """
        pass


class ConstantDiffusionCoefficient(DiffusionCoefficient):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(t, self.sigma)


class ParabolicBridgeDiffusionCoefficient(DiffusionCoefficient):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma * t * (1 - t)


class SqrtParabolicBridgeDiffusionCoefficient(DiffusionCoefficient):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma * torch.sqrt(t * (1 - t))


DIFFUSION_COEFFICIENT_REGISTRY = {
    "constant": ConstantDiffusionCoefficient,
    "parabolic_bridge": ParabolicBridgeDiffusionCoefficient,
    "sqrt_parabolic_bridge": SqrtParabolicBridgeDiffusionCoefficient,
}
"""Mapping from diffusion coefficient type strings to their corresponding classes."""
