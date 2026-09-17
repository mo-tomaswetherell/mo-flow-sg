"""Numerical ODE and SDE solvers."""

from abc import ABC, abstractmethod

import torch

from flow.scheduler import Scheduler
from flow.diffusion_coefficient import DiffusionCoefficient


class Solver(ABC):
    def __init__(
        self,
        n_steps: int,
        alpha: float,
        model: torch.nn.Module,
        device: torch.device,
        guidance_scale: float = 1.0,
    ):
        """Initialise.

        Args:
            n_steps: Number of steps.
            alpha: Exponent for the time-stepping scheme. If alpha=1, then the time steps
                are uniformly spaced between 0 and 1 (constant step size). If alpha>1, then the
                step size is smaller near t=1 (the data/target side) and larger near t=0 (the
                noise/source side).
            model: Neural network model representing the velocity/vector field.
            device: Device
            guidance_scale: Guidance scale for classifier-free guidance. Should only be used if
                the model was trained with classifier-free guidance (i.e. with conditioning
                dropout). The default value of 1.0 corresponds to "vanilla guidance" (i.e. not
                using classifer-free guidance).

        Raises:
            ValueError: If guidance_scale is negative.
        """
        if guidance_scale < 0:
            raise ValueError("guidance_scale should be non-negative.")

        self.n_steps = n_steps
        self.alpha = alpha
        self.model = model
        self.device = device
        self.guidance_scale = guidance_scale
        self.time_steps = (
            1 - (1 - torch.tensor(range(self.n_steps + 1)) / self.n_steps) ** self.alpha
        )  # shape (n_steps + 1, )
        self.step_sizes = self.time_steps[1:] - self.time_steps[:-1]  # shape (n_steps, )

    def simulate(
        self, x_0: torch.Tensor, predictors: torch.Tensor, static: torch.Tensor | None
    ) -> torch.Tensor:
        """Simulate the ODE or SDE from t=0 to t=1.

        Args:
            x_0: Tensor of shape (batch_size, num_target_variables, height, width)
            predictors: Predictor variables,
                shape (batch_size, num_predictor_variables, height, width)
            static: Static fields (e.g., orography), shape (batch_size, num_static_variables, height, width).
                If static fields are not used, this should be None.

        Returns:
            x_1: Tensor of shape (batch_size, num_target_variables, height, width)
        """
        x_current = x_0.to(self.device)

        for idx, step_size in enumerate(self.step_sizes):
            t_current = self.time_steps[idx].item()
            t_current = torch.full(
                (x_current.shape[0], 1), t_current, device=self.device
            )  # shape (batch_size, 1)
            x_current = self.step(x_current, t_current, predictors, static, step_size.item())

        return x_current

    def _get_velocity(
        self,
        x_current: torch.Tensor,
        t_current: torch.Tensor,
        predictors: torch.Tensor,
        static: torch.Tensor | None,
    ) -> torch.Tensor:
        """Returns the velocity/vector field with optional classifier-free guidance.

        If using classifier-free guidance, we assume that the model was trained with conditioning
        dropout, and that the dropped predictors are represented by a zero tensor.
        Note that we don't replace the static fields (e.g., orography) with zeroes, as these
        fields are constant for each sample (i.e. they don't provide sample/time-dependent information).
        """
        conditioning = predictors if static is None else torch.cat([predictors, static], dim=1)

        velocity = self.model(x_current, t_current, conditioning)

        if self.guidance_scale != 1.0:
            # Using classifier-free guidance

            conditioning = (
                torch.zeros_like(predictors)
                if static is None
                else torch.cat([torch.zeros_like(predictors), static], dim=1)
            )
            velocity_unconditioned = self.model(x_current, t_current, conditioning)
            velocity = (
                1 - self.guidance_scale
            ) * velocity_unconditioned + self.guidance_scale * velocity

        return velocity

    @abstractmethod
    def step(
        self,
        x_current: torch.Tensor,
        t_current: torch.Tensor,
        predictors: torch.Tensor,
        static: torch.Tensor | None,
        step_size: float,
    ) -> torch.Tensor:
        """Take a single step of the ODE solver.

        Args:
            x_current: Current state, shape (batch_size, num_target_variables, height, width)
            t_current: Current time between 0 and 1, shape (batch_size, 1)
            predictors: Predictor variables,
                shape (batch_size, num_predictor_variables, height, width)
            static: Static fields (e.g., orography), shape (batch_size, num_static_variables, height, width).
                If static fields are not used, this should be None.
            step_size: Step size for this step.

        Returns:
            x_next: Next state, shape (batch_size, num_target_variables, height, width
        """
        pass


class EulerSolver(Solver):
    def step(
        self,
        x_current: torch.Tensor,
        t_current: torch.Tensor,
        predictors: torch.Tensor,
        static: torch.Tensor | None,
        step_size: float,
    ) -> torch.Tensor:
        x_next = x_current + step_size * self._get_velocity(
            x_current, t_current, predictors, static
        )
        return x_next


class HeunSolver(Solver):
    def step(
        self,
        x_current: torch.Tensor,
        t_current: torch.Tensor,
        predictors: torch.Tensor,
        static: torch.Tensor | None,
        step_size: float,
    ) -> torch.Tensor:
        dx = self._get_velocity(x_current, t_current, predictors, static)
        x_next_estimate = x_current + step_size * dx
        t_next = t_current + step_size
        x_next = x_current + (step_size / 2) * (
            dx + self._get_velocity(x_next_estimate, t_next, predictors, static)
        )
        return x_next


class EulerMaruyamaSolver(Solver):
    def __init__(
        self,
        n_steps: int,
        alpha: float,
        model: torch.nn.Module,
        device: torch.device,
        scheduler: Scheduler,
        diffusion_coefficient: DiffusionCoefficient,
        guidance_scale: float = 1.0,
    ):
        super().__init__(n_steps, alpha, model, device, guidance_scale)
        self.scheduler = scheduler
        self.diffusion_coefficient = diffusion_coefficient

    def _get_score(
        self,
        velocity: torch.Tensor,
        x_current: torch.Tensor,
        t_current: torch.Tensor,
    ):
        """Returns the score.

        Computes the score from the velocity field, using the conversion formula for
        Gaussian Probability Paths (Proposition 1, https://arxiv.org/pdf/2506.02070):

        score(x, t) = (alpha_t*velocity(x, t) - d_alpha_t/dt * x)/(beta_t*[d_alpha_t/dt*beta_t - d_beta_t/dt*alpha_t])

        At t=0 this conversion formula is degenerate (evaluates to 0/0) for schedulers with
        d_alpha_0/dt = 0 (e.g. polynomial schedulers with n > 1). At t=0 the marginal is the
        source distribution N(0, beta_0^2 * I), whose score is known in closed form as
        -x/beta_0^2, so we use this analytic expression there instead.

        Args:
            velocity: Velocity field, shape (batch_size, num_target_variables, height, width)
            x_current: Current state, shape (batch_size, num_target_variables, height, width)
            t_current: Current time between 0 and 1, shape (batch_size, 1)

        Returns:
            score: Score, shape (batch_size, num_target_variables, height, width)
        """
        t = t_current.view(t_current.shape[0], *([1] * (x_current.ndim - 1)))
        alpha, beta = self.scheduler(t)
        d_alpha, d_beta = self.scheduler.derivative(t)
        score = (alpha * velocity - d_alpha * x_current) / (
            beta * (d_alpha * beta - d_beta * alpha)
        )

        # Replace the potentially degenerate t=0 values with the analytic source-distribution score.
        beta_0 = self.scheduler(torch.zeros_like(t))[1]
        analytic_score = -x_current / beta_0**2
        score = torch.where(t == 0, analytic_score, score)

        return score

    def step(
        self,
        x_current: torch.Tensor,
        t_current: torch.Tensor,
        predictors: torch.Tensor,
        static: torch.Tensor | None,
        step_size: float,
    ) -> torch.Tensor:
        velocity = self._get_velocity(x_current, t_current, predictors, static)

        # On the final step, take a deterministic probability-flow ODE (Euler) step using only the
        # velocity (avoids adding noise at the final step).
        is_final_step = t_current[0].item() + step_size >= 1.0
        if is_final_step:
            return x_current + step_size * velocity

        score = self._get_score(velocity, x_current, t_current)

        t = t_current.view(t_current.shape[0], *([1] * (x_current.ndim - 1)))
        sigma_t = self.diffusion_coefficient(t)

        drift = velocity + 0.5 * sigma_t**2 * score
        noise = torch.randn_like(x_current)
        x_next = x_current + step_size * drift + sigma_t * (step_size**0.5) * noise

        return x_next
