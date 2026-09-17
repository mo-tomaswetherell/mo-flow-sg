"""Training and validation loops."""

import logging

import mlflow
import torch
from torch.amp import autocast

from flow.distributions import Distribution
from flow.paths import AffineProbabilityPath


logger = logging.getLogger(__name__)


def train_loop(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    coupled: bool,
    source_sampler: Distribution | None,
    probability_path: AffineProbabilityPath,
    gradient_accumulation_steps: int = 1,
    conditioning_dropout: float | int = 0,
    use_amp: bool = True,
    experiment_name: str | None = None,
    ema_model: torch.nn.Module | None = None,
):
    """Training loop for a single epoch.

    Runs the training loop for a single epoch. Metrics are logged to MLflow.

    Args:
        dataloader: DataLoader for the training data.
        model: The model to train.
        loss_fn: The loss function.
        optimizer: The optimizer.
        device: The device to run the training on.
        epoch: The current epoch.
        coupled: Whether the source distribution is coupled with the target distribution (e.g.,
            low resolution precip. and high resolution precip.). If false, x_0 is sampled from a
            source distribution (e.g., Gaussian(0, 1)) via source_sampler.
        source_sampler: Distribution used to sample x_0 from the source/initital distribution.
            If coupled is false, then this is required. Otherwise, it should be None.
        probability_path: Probability path to sample x_t and the conditional vector field.
        gradient_accumulation_steps: Number of batches to accumulate gradients over before updating
            weights. Effective batch size = batch_size * gradient_accumulation_steps.
        conditioning_dropout: Probability to drop the conditioning. Used for classifier-free
            guidance. Default is 0 (no dropout), which should be used for vanilla guidance.
        use_amp: Whether to use automatic mixed precision. Uses bfloat16 dtype, which
            doesn't require loss scaling.
        experiment_name: Name of experiment. If provided, used in the metric name logged to
            MLflow (useful if multiple experiments are being run in the same run).
        ema_model: If not None, the EMA version of the model.

    Raises:
        ValueError: If coupled is True and source_sampler is not None
        ValueError: If coupled is False and source_sampler is None
        ValueError: If conditioning_dropout is not between 0 and 1 (inclusive)
    """
    if coupled:
        if source_sampler is not None:
            raise ValueError(
                "source_sampler is not required (and should be None) when coupled is True."
            )
    else:
        if source_sampler is None:
            raise ValueError("source_sampler is required when coupled is False.")

    if conditioning_dropout < 0 or conditioning_dropout > 1:
        raise ValueError("conditioning_dropout should be between 0 and 1.")

    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.train()

    for batch_idx, batch in enumerate(dataloader):
        if coupled:
            predictors, x_1, x_0, static = batch
        else:
            predictors, x_1, _, static = batch
            x_0 = source_sampler.sample(x_1.shape)

        predictors = predictors.to(device)
        x_1 = x_1.to(device)
        x_0 = x_0.to(device)
        batch_size = x_1.shape[0]

        # Classifer-free guidance: randomly drop each sample's conditioning with probability conditioning_dropout
        if conditioning_dropout > 0:
            mask_shape = (batch_size,) + (1,) * (
                predictors.dim() - 1
            )  # shape (batch_size, 1, 1, 1)
            # The mask is used to set the conditioning of samples to a zero-tensor with probability
            # conditioning_dropout. We use a zero-tensor to represent 'no conditioning' as - if the
            # predictors are normalised to have zero mean - this is equivalent to providing the
            # model with average atmospheric conditions.
            # Note that we don't apply the mask to the static fields (e.g., orography), as these
            # fields are constant for each sample (i.e. they don't provide sample/time-dependent information).
            mask = torch.rand(mask_shape, device=device) < conditioning_dropout
            predictors = predictors * ~mask

        # If static fields are used, concatenate them with the predictor variables
        conditioning = (
            torch.cat([predictors, static.to(device)], dim=1) if static.shape[1] > 0 else predictors
        )

        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            # TODO: Make sampling of t configurable, instead of always Unif(0, 1)
            t = torch.rand(batch_size, 1, device=device)
            x_t = probability_path.sample(x_0, x_1, t)
            v_t = probability_path.compute_vector_field(x_0, x_1, t)

            v_t_theta = model(x_t, t, conditioning)

            loss = loss_fn(v_t_theta, v_t)

            # Scale loss for gradient accumulation
            current_loss_val = loss.item()  # for logging
            loss = loss / gradient_accumulation_steps

        loss.backward()

        # Update weights after accumulating gradients for gradient_accumulation_steps batches
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
            optimizer.step()
            optimizer.zero_grad()
            if ema_model is not None:
                ema_model.update_parameters(model)

        if batch_idx % 100 == 0:
            current = (batch_idx * dataloader.batch_size) + batch_size
            logger.info(f"loss: {current_loss_val:>7f}  [{current:>5d}/{num_samples:>5d}]")
            batch_number = (epoch * num_batches) + batch_idx
            metric_name = "Training loss" + (f" ({experiment_name})" if experiment_name else "")
            mlflow.log_metric(metric_name, current_loss_val, step=batch_number)

            # Also log GPU memory usage.
            if device.type == "cuda":
                peak_bytes = torch.cuda.max_memory_allocated(device=device)
                reserved_bytes = torch.cuda.memory_reserved(device=device)
                mlflow.log_metric("GPU_Peak_Allocated_GB", peak_bytes / (1024**3))
                mlflow.log_metric("GPU_Reserved_GB", reserved_bytes / (1024**3))
                torch.cuda.reset_peak_memory_stats(device=device)


def val_loop(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    epoch: int,
    coupled: bool,
    source_sampler: Distribution | None,
    probability_path: AffineProbabilityPath,
    use_amp: bool = True,
    experiment_name: str | None = None,
):
    """Compute loss over the validation dataset."""
    if coupled:
        if source_sampler is not None:
            raise ValueError(
                "source_sampler is not required (and should be None) when coupled is True."
            )
    else:
        if source_sampler is None:
            raise ValueError("source_sampler is required when coupled is False.")

    model.eval()
    num_batches = len(dataloader)
    val_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            if coupled:
                predictors, x_1, x_0, static = batch
            else:
                predictors, x_1, _, static = batch
                x_0 = source_sampler.sample(x_1.shape)

            predictors = predictors.to(device)
            x_1 = x_1.to(device)
            x_0 = x_0.to(device)
            batch_size = x_0.shape[0]

            conditioning = (
                torch.cat([predictors, static.to(device)], dim=1)
                if static.shape[1] > 0
                else predictors
            )

            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                t = torch.rand(batch_size, 1, device=device)
                x_t = probability_path.sample(x_0, x_1, t)
                v_t = probability_path.compute_vector_field(x_0, x_1, t)

                v_t_theta = model(x_t, t, conditioning)

                loss = loss_fn(v_t_theta, v_t)

            loss = loss.item()
            val_loss += loss

    val_loss /= num_batches

    logger.info(f"\nValidation loss: {val_loss:>7f}\n")
    metric_name = "Validation loss" + (f" ({experiment_name})" if experiment_name else "")
    mlflow.log_metric(metric_name, val_loss, step=epoch + 1)

    return val_loss
