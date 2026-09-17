from abc import ABC, abstractmethod

import torch


class Distribution(ABC):
    """Abstract base class for distribution samplers."""

    @abstractmethod
    def sample(self, shape: tuple) -> torch.Tensor:
        """Draw a sample from the distribution.

        Args:
            shape: Shape of the sample to draw.

        Returns:
            Tensor of the given shape sampled from the distribution.
        """


class GaussianDistribution(Distribution):
    """Gaussian distribution sampler."""

    def __init__(self, mean: float, std: float, device: torch.device):
        """Initialise Gaussian distribution sampler.

        Args:
            mean: Mean of the Gaussian distribution.
            std: Standard deviation of the Gaussian distribution.
            device: Device to place samples on.
        """
        self.mean = mean
        self.std = std
        self.device = device

    def sample(self, shape: tuple) -> torch.Tensor:
        return torch.normal(self.mean, self.std, size=shape, device=self.device)
