"""Inference on the V3 dataset."""

import argparse
import logging

import numpy as np
import torch
import xarray as xr

from flow.transforms import Transform, load_transforms
from flow.datasets.v3 import V3Dataset
from flow.networks import ADM
from flow.distributions import Distribution, GaussianDistribution
from flow.scheduler import SCHEDULER_REGISTRY
from flow.solvers import EulerSolver, HeunSolver, EulerMaruyamaSolver
from flow.diffusion_coefficient import DIFFUSION_COEFFICIENT_REGISTRY
from flow.config import load_config
from flow.config.schema import FlowMatchingConfig, CoupledSourceConfig, GaussianSourceConfig
from flow.mlflow import setup_mlflow
from flow.utils import seed_worker, set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_WORKERS = 8
"""Number of workers for data loading. Must be less than number of CPU cores."""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create predictions.")

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
    parser.add_argument(
        "--static_path",
        type=str,
        help="Path to netcdf file containing static variables",
        required=False,
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Data split to use. Expected to be 'test' or 'validation'.",
    )
    parser.add_argument(
        "--solver_type",
        type=str,
        help="Name of numerical solver to use. Expected to be 'euler', 'heun' or 'euler_maruyama'.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        help=(
            "Number of samples to generate. If using a coupled source, this must be 1 (as the "
            "model is deterministic in this case)."
        ),
        default=1,
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        help="Number of time steps to simulate.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Exponent for the time-stepping scheme.",
    )
    parser.add_argument(
        "--diffusion_coefficient_type",
        type=str,
        help=(
            "Type of diffusion coefficient to use. Expected to be 'none', 'constant', "
            "'parabolic_bridge', or 'sqrt_parabolic_bridge'. Only used when using stochastic solvers "
            "(solver_type=euler_maruyama)."
        ),
        default="none",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        help=(
            "Value of sigma for the diffusion coefficient - see diffusion_coefficient.py. "
            "Not used if diffusion_coefficient_type is 'none'.",
        ),
        default=1.0,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to trained model file.",
    )
    parser.add_argument(
        "--transforms_path",
        type=str,
        help="Path to transforms JSON file. Fit during training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for data loading.",
    )
    parser.add_argument(
        "--input_descriptor",
        type=str,
        default="v3-wmc-128x128cutout",
        help="Descriptor for the input predictor/conditioning data source (e.g., 'v3').",
    )
    parser.add_argument(
        "--outputs_path",
        type=str,
        required=True,
        help="Path to directory to save the model predictions.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Whether to compile the model. Should be used if the model was compiled during training.",
    )

    args = parser.parse_args()

    if args.solver_type not in ("euler", "heun", "euler_maruyama"):
        raise ValueError(
            f"Expected solver to be 'euler', 'heun' or 'euler_maruyama', but got {args.solver_type}."
        )

    if args.solver_type == "euler_maruyama" and args.diffusion_coefficient_type == "none":
        raise ValueError(
            "If using the Euler-Maruyama solver, a diffusion coefficient type must be specified. "
            "Expected to be 'constant', 'parabolic_bridge', or 'sqrt_parabolic_bridge'."
        )

    if args.diffusion_coefficient_type not in (
        "none",
        "constant",
        "parabolic_bridge",
        "sqrt_parabolic_bridge",
    ):
        raise ValueError(
            f"Expected diffusion_coefficient_type to be 'none', 'constant', 'parabolic_bridge' or 'sqrt_parabolic_bridge', but got {args.diffusion_coefficient_type}."
        )

    if args.split not in ("test", "validation"):
        raise ValueError(f"Expected split to be 'test' or 'validation', but got {args.split}.")

    return args


def _parse_years(year_list: list[str | int]) -> list[int]:
    """Parses a list of years/ranges into a list of integers."""
    parsed_years = []
    for item in year_list:
        if isinstance(item, str) and "-" in item:
            start, end = map(int, item.split("-"))
            parsed_years.extend(range(start, end + 1))
        else:
            parsed_years.append(int(item))
    return sorted(list(set(parsed_years)))


@setup_mlflow
def main(
    config: FlowMatchingConfig,
    predictors_path: str,
    targets_path: str,
    split: str,
    solver_type: str,
    n_samples: int,
    n_steps: int,
    alpha: float,
    diffusion_coefficient_type: str,
    sigma: float,
    model_path: str,
    transforms_path: str,
    batch_size: int,
    input_descriptor: str,
    outputs_path: str,
    static_path: str | None = None,
    compile: bool = True,
):
    if solver_type == "euler_maruyama" and not isinstance(config.source, GaussianSourceConfig):
        raise ValueError(
            "If using the Euler-Maruyama solver, the source must be Gaussian. "
            f"Got source type: {type(config.source)}"
        )

    target_var_names: list[str] = list(config.targets.keys())  # list of target variable names
    logger.info(f"Configuration:\n{config}")

    transforms: dict[str, Transform] = load_transforms(transforms_path)

    logger.info(f"Initialising inference dataset with split '{split}'...")
    inference_dataset = V3Dataset(
        predictors_path=predictors_path,
        targets_path=targets_path,
        config=config,
        years=_parse_years(getattr(config, split).years),
        transforms=transforms,
        static_path=static_path,
    )

    inference_dataloader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
    )

    # Initialise model

    static_variables = config.static or []
    if static_variables:
        logger.info(f"Using static variables: {static_variables}")
    if config.pos_emb:
        logger.info("Using sinusoidal positional embeddings (4 channels)")
    num_conditioning_variables = (
        len(config.predictors) + len(static_variables) + (4 if config.pos_emb else 0)
    )

    model = ADM(
        num_conditioning_variables=num_conditioning_variables,
        num_target_variables=len(target_var_names),
        **config.model.model_dump(),
    ).to(DEVICE)

    if compile:
        model = torch.compile(
            model,
            mode="default",
            fullgraph=True,
        )

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    logger.info(f"Loaded model from {model_path}")

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

    if solver_type == "euler":
        solver = EulerSolver(n_steps=n_steps, alpha=alpha, model=model, device=DEVICE)
    elif solver_type == "heun":
        solver = HeunSolver(n_steps=n_steps, alpha=alpha, model=model, device=DEVICE)
    elif solver_type == "euler_maruyama":
        scheduler = SCHEDULER_REGISTRY[config.scheduler.type](**config.scheduler.model_dump())
        diffusion_coefficient = DIFFUSION_COEFFICIENT_REGISTRY[diffusion_coefficient_type](
            sigma=sigma
        )
        solver = EulerMaruyamaSolver(
            n_steps=n_steps,
            alpha=alpha,
            model=model,
            device=DEVICE,
            scheduler=scheduler,
            diffusion_coefficient=diffusion_coefficient,
        )
    else:
        raise ValueError(f"Unsupported solver type: {solver_type}")

    if coupled and n_samples != 1:
        logger.warning(
            "When using a coupled source the model is deterministic. Setting n_samples to 1."
        )
        n_samples = 1

    prediction_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []

    for batch in inference_dataloader:
        logger.info(f"Generating {n_samples} samples for batch of size {batch[0].shape[0]}")
        if coupled:
            predictors, targets, x_0, static = batch
        else:
            predictors, targets, _, static = batch

        target_batches.append(targets.cpu())

        samples: list[
            torch.Tensor
        ] = []  # list of samples of shape (batch_size, num_target_variables, lat, lon)
        for _ in range(n_samples):
            if not coupled:
                # Resample source for each sample
                x_0 = source_sampler.sample(
                    (
                        predictors.shape[0],
                        len(config.targets),
                        predictors.shape[2],
                        predictors.shape[3],
                    )
                )  # shape (batch_size, num_target_variables, lat, lon)

            predictors = predictors.to(DEVICE)
            x_0 = x_0.to(DEVICE)
            static = static.to(DEVICE) if static.shape[1] > 0 else None

            model.eval()
            with torch.no_grad():
                x_1 = solver.simulate(
                    x_0=x_0, predictors=predictors, static=static
                )  # shape (batch_size, num_target_variables, lat, lon)
                samples.append(x_1.cpu())

        samples_tensor = torch.stack(
            samples, dim=0
        )  # shape (n_samples, batch_size, num_target_variables, lat, lon)
        prediction_batches.append(samples_tensor)

    predictions = torch.cat(
        prediction_batches, dim=1
    ).numpy()  # shape (n_samples, time, num_target_variables, lat, lon)

    targets_tensor = torch.cat(
        target_batches, dim=0
    ).numpy()  # shape (time, num_target_variables, lat, lon)

    data_vars: dict[str, xr.DataArray] = {}
    for idx, target in enumerate(target_var_names):
        # Inverse any transforms applied
        logger.info(f"Applying inverse transform to target variable '{target}'")
        target_transform = transforms["targets"][target]
        predictions[:, :, idx, :, :] = target_transform.inverse_transform(
            predictions[:, :, idx, :, :]
        )
        targets_tensor[:, idx, :, :] = target_transform.inverse_transform(
            targets_tensor[:, idx, :, :]
        )

        # Propogate relevant attributes from the target variable
        target_attrs = inference_dataset.target_attrs[target]
        attrs = {
            key: target_attrs[key] for key in ["units", "standard_name"] if key in target_attrs
        }

        data_vars[f"pred_{target}"] = (
            ("sample", "input", "time", "grid_latitude", "grid_longitude"),
            np.expand_dims(
                predictions[:, :, idx, :, :], axis=1
            ),  # expand_dims adds 'dummy' input axis
            attrs,
        )

        data_vars[f"target_{target}"] = (
            ("input", "time", "grid_latitude", "grid_longitude"),
            np.expand_dims(targets_tensor[:, idx, :, :], axis=0),
            attrs,
        )

    logger.info("Forming xarray Dataset and saving to netCDF")

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "sample": np.arange(n_samples),
            "input": [input_descriptor],
            "time": inference_dataset.time,
            "grid_latitude": inference_dataset.target_lat,
            "grid_longitude": inference_dataset.target_lon,
        },
        attrs={
            "solver_type": solver_type,
            "n_steps": n_steps,
            "alpha": alpha,
            **(
                {"diffusion_coefficient_type": diffusion_coefficient_type, "sigma": sigma}
                if solver_type == "euler_maruyama"
                else {}
            ),
        },
    )

    output_file = f"{outputs_path}/predictions.nc"
    ds.to_netcdf(output_file, format="NETCDF4", engine="netcdf4")
    logger.info(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    args = parse_args()

    config: FlowMatchingConfig = load_config(args.config_filename)
    set_seed(config.seed)

    main(
        config=config,
        predictors_path=args.predictors_path,
        targets_path=args.targets_path,
        split=args.split,
        solver_type=args.solver_type,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        alpha=args.alpha,
        diffusion_coefficient_type=args.diffusion_coefficient_type,
        sigma=args.sigma,
        model_path=args.model_path,
        transforms_path=args.transforms_path,
        batch_size=args.batch_size,
        input_descriptor=args.input_descriptor,
        outputs_path=args.outputs_path,
        static_path=args.static_path,
        compile=args.compile,
    )
