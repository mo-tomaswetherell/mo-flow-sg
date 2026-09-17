"""Train flow matching model using CCRS V3 data."""

import argparse
import logging
from copy import deepcopy

import torch
import xarray as xr
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from flow.mlflow import setup_mlflow
from flow.utils import set_seed
from flow.transforms import Transform, fit_transforms, save_transforms
from flow.dataset import V3Dataset
from flow.scheduler import SCHEDULER_REGISTRY
from flow.distributions import Distribution, GaussianDistribution
from flow.paths import AffineProbabilityPath, CondOTPath
from flow.networks import ADM
from flow.train import train_loop, val_loop
from flow.config import load_config
from flow.config.schema import (
    FlowMatchingConfig,
    CoupledSourceConfig,
    GaussianSourceConfig,
    SchedulerConfig,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_WORKERS = 32
"""Number of workers for data loading. Must be less than number of CPU cores."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train network.")

    parser.add_argument(
        "--config_filename",
        type=str,
        help="Path to the configuration file relative to 'src/flow/config/'.",
    )
    parser.add_argument(
        "--predictors_path",
        type=str,
        help="Path to directory containing zarr-format predictor variables.",
    )
    parser.add_argument(
        "--targets_path",
        type=str,
        help="Path to directory containing zarr-format target variables.",
    )
    parser.add_argument("--outputs_path", type=str, help="Path to directory to write outputs to.")
    parser.add_argument(
        "--static_path",
        type=str,
        help="Path to netcdf file containing static variables",
        required=False,
    )
    parser.add_argument("--compile", action="store_true", help="Whether to compile the model.")
    parser.add_argument(
        "--use_amp", action="store_true", help="Whether to use automatic mixed precision."
    )

    args = parser.parse_args()
    return args


def _parse_years(year_list: list[str | int]) -> list[int]:
    """Parses a list of years/ranges into a list of integers."""
    parsed_years: list[int] = []
    for item in year_list:
        if isinstance(item, str) and "-" in item:
            start, end = map(int, item.split("-"))
            parsed_years.extend(range(start, end + 1))
        else:
            parsed_years.append(int(item))
    return sorted(list(set(parsed_years)))


def load_dataset(path: str, variables: list[str], years: list[int]) -> xr.Dataset:
    """Lazily loads dataset for the given variables and years."""
    ds_list: list[xr.DataArray] = []
    for var_name in variables:
        var_path = f"{path}/{var_name}.zarr"
        ds = xr.open_zarr(var_path, consolidated=True, chunks="auto")
        ds_list.append(ds[var_name])

    ds = xr.merge(ds_list)
    ds = ds.sel(time=ds.time.dt.year.isin(years))
    return ds


@setup_mlflow
def main(
    config: FlowMatchingConfig,
    predictors_path: str,
    targets_path: str,
    outputs_path: str,
    static_path: str | None = None,
    compile: bool = True,
    use_amp: bool = True,
):
    train_years = _parse_years(config.training.years)
    val_years = _parse_years(config.validation.years)

    # Fit transforms on the training dataset.

    logger.info("Fitting transforms...")
    predictors_ds = load_dataset(predictors_path, list(config.predictors.keys()), train_years)
    predictor_transforms = fit_transforms(config, predictors_ds, groups=["predictors"])

    targets_ds = load_dataset(targets_path, list(config.targets.keys()), train_years)
    target_transforms = fit_transforms(config, targets_ds, groups=["targets"])

    transforms: dict[str, dict[str, Transform]] = {
        "predictors": predictor_transforms["predictors"],
        "targets": target_transforms["targets"],
    }
    if isinstance(config.source, CoupledSourceConfig):
        source_transforms = fit_transforms(config, predictors_ds, groups=["source"])
        transforms["source"] = source_transforms["source"]
    save_transforms(transforms, f"{outputs_path}/transforms.json")
    logger.info("Saved transforms")

    # Initialise datasets and dataloaders

    training_dataset = V3Dataset(
        predictors_path=predictors_path,
        targets_path=targets_path,
        config=config,
        years=train_years,
        transforms=transforms,
        static_path=static_path,
    )
    validation_dataset = V3Dataset(
        predictors_path=predictors_path,
        targets_path=targets_path,
        config=config,
        years=val_years,
        transforms=transforms,
        static_path=static_path,
    )

    batch_size = config.training.batch_size
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    # Configure probability path

    scheduler_config: SchedulerConfig = config.scheduler
    scheduler_type = scheduler_config.type
    scheduler = SCHEDULER_REGISTRY[scheduler_type](**scheduler_config.model_dump())

    if scheduler_type == "condot":
        probability_path = CondOTPath(scheduler)
    else:
        probability_path = AffineProbabilityPath(scheduler)

    coupled: bool = isinstance(config.source, CoupledSourceConfig)

    source_sampler: Distribution | None = None
    if coupled:
        source_sampler = None
    elif isinstance(config.source, GaussianSourceConfig):
        source_sampler = GaussianDistribution(
            mean=config.source.mean, std=config.source.std, device=DEVICE
        )
    else:
        raise ValueError(f"Unsupported source type: {type(config.source)}")

    # Initialise model

    static_variables = config.static or []
    if static_variables:
        logger.info(f"Using static variables: {static_variables}")
    else:
        logger.info("Not using static variables")
    if config.pos_emb:
        logger.info("Using sinusoidal positional embeddings (4 channels)")
    num_conditioning_variables = (
        len(config.predictors) + len(static_variables) + (4 if config.pos_emb else 0)
    )

    model = ADM(
        num_conditioning_variables=num_conditioning_variables,
        num_target_variables=len(config.targets),
        **config.model.model_dump(),
    ).to(DEVICE)

    if compile:
        logger.info("Compiling model...")
        model = torch.compile(
            model,
            mode="default",
            fullgraph=True,
        )

    # Configure and run training loop

    learning_rate = config.training.learning_rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    epochs = config.training.epochs

    ema_decay = config.training.ema_decay
    if ema_decay is not None:
        logger.info(f"Using EMA with decay={ema_decay}")
        ema_model = AveragedModel(
            model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True
        )
    else:
        ema_model = None

    # Model used for validation and checkpointing: the EMA weights when EMA is enabled,
    # otherwise the online weights.
    eval_model = ema_model if ema_model is not None else model

    best_val_loss = torch.inf
    best_model_state = None
    patience = config.training.patience
    epochs_without_improvement = 0

    gradient_accumulation_steps = config.training.gradient_accumulation_steps

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(
            dataloader=training_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=DEVICE,
            epoch=epoch,
            coupled=coupled,
            source_sampler=source_sampler,
            probability_path=probability_path,
            gradient_accumulation_steps=gradient_accumulation_steps,
            conditioning_dropout=config.training.conditioning_dropout,
            use_amp=use_amp,
            ema_model=ema_model,
        )

        val_loss = val_loop(
            dataloader=validation_dataloader,
            model=eval_model,
            loss_fn=loss_fn,
            device=DEVICE,
            epoch=epoch,
            coupled=coupled,
            source_sampler=source_sampler,
            probability_path=probability_path,
            use_amp=use_amp,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_module = eval_model.module if ema_model is not None else model
            best_model_state = deepcopy(checkpoint_module.state_dict())
            epochs_without_improvement = 0
        else:
            if patience is not None:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(
                        f"No improvement in validation loss for {patience} epochs. Early stopping."
                    )
                    break

    model_path = f"{outputs_path}/model.pth"
    logger.info(f"Training complete. Saving model to {model_path}")
    torch.save(best_model_state, model_path)


if __name__ == "__main__":
    args = parse_args()

    config: FlowMatchingConfig = load_config(args.config_filename)
    set_seed(config.seed)

    main(
        config=config,
        predictors_path=args.predictors_path,
        targets_path=args.targets_path,
        outputs_path=args.outputs_path,
        static_path=args.static_path,
        compile=args.compile,
        use_amp=args.use_amp,
    )
