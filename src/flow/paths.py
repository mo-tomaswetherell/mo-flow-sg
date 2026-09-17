"""Probability paths."""

from torch import Tensor

from flow.scheduler import Scheduler


class AffineProbabilityPath:
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Returns samples from the path.

        X_t = alpha_t * X_1 + beta_t * X_0

        Args:
            x_0: Source points, shape (batch_size, ...)
            x_1: Target points, shape (batch_size, ...)
            t: Times between [0, 1], shape (batch_size, 1).

        Returns:
            x_t: Samples from path, shape (batch_size, ...)
        """
        t = self._expand_t(t, x_0)
        alpha_t, beta_t = self.scheduler(t)
        x_t = alpha_t * x_1 + beta_t * x_0
        return x_t

    def compute_vector_field(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Returns the conditional vector/velocity field at time t and position x_t.

        dX_t/dt = (d_alpha_t - d_beta_t/beta_t * alpha_t) * X_1 + d_beta_t/beta_t * X_t

        Equivalently, in terms of X_0 and X_1: dX_t/dt = d_alpha_t * X_1 + d_beta_t * X_0

        Args:
            x_0: Source points, shape (batch_size, ...)
            x_1: Target points, shape (batch_size, ...)
            t: Times between [0, 1], shape (batch_size, 1).

        Returns:
            v_t: Velocity values at time t and position x_t, shape (batch_size, ...)
        """
        t = self._expand_t(t, x_0)
        d_alpha_t, d_beta_t = self.scheduler.derivative(t)
        v_t = d_alpha_t * x_1 + d_beta_t * x_0
        return v_t

    def _expand_t(self, t: Tensor, x: Tensor) -> Tensor:
        if t.ndim != 2 or t.shape[1] != 1:
            raise ValueError(f"t must have shape (B, 1), got {t.shape}")
        # Expand to (batch_size, 1, ..., 1)
        return t.view(t.shape[0], *([1] * (x.ndim - 1)))


class CondOTPath(AffineProbabilityPath):
    """Special case of the affine probability path, where alpha=t and beta=1-t.

    The expression for the conditional vector field simplifies in this case.
    """

    def compute_vector_field(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Returns the conditional vector field at time t and position x_t.

        dX_t/dt = d_alpha_t * X_1 + d_beta_t * X_0 = X_1 - X_0

        Equivalently, in terms of X_0 and X_1: dX_t/dt = X_1 - X_0
        """
        return x_1 - x_0
