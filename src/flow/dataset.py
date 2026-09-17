"""Dataset module for the CCRS V3 dataset.

See https://www.mss-int.sg/v3-climate-projections/learn-about-v3/v3-explained
"""

import logging

import torch
import numpy as np
import xarray as xr

from flow.transforms import Transform
from flow.datasets import BaseDataset
from flow.config.schema import FlowMatchingConfig, CoupledSourceConfig


logger = logging.getLogger(__name__)


class V3Dataset(BaseDataset):
    """V3 Dataset class."""

    SPATIAL_DIMS = ("lat", "lon")

    def __init__(
        self,
        predictors_path: str,
        targets_path: str,
        config: FlowMatchingConfig,
        years: list[int],
        transforms: dict[str, dict[str, Transform]],
        static_path: str | None = None,
    ):
        """
        Initialise.

        Args:
            predictors_path: Path to the base directory containing the predictor zarr stores.
            targets_path: Path to the base directory containing the target zarr stores.
            config: Flow matching configuration.
            years: List of years to include in the dataset.
            transforms: Mapping from group name to dictionary of variable name to Transform.
            static_path: Path to netcdf file containing static variables.

        Raises:
            ValueError
                If predictor and targets time indicies are not aligned.
        """
        self.predictors_path = predictors_path
        self.targets_path = targets_path
        self.config = config
        self.years = years
        self.transforms = transforms

        self.predictors: list[str] = list(config.predictors.keys())
        self.targets: list[str] = list(config.targets.keys())
        if isinstance(config.source, CoupledSourceConfig):
            self.source = list(config.source.variables.keys())
            if len(self.source) != len(self.targets):
                raise ValueError(
                    f"Number of coupled variables ({len(self.source)}) must equal the number "
                    f"of target variables ({len(self.targets)})."
                )
        else:
            self.source = []

        # Temporarily open the targets dataset to read the metadata.
        # Note: We don't store this dataset on 'self' to avoid serialising/pickling open file
        # handles when using multiple workers (num_workers > 0 in DataLoader).
        with self._create_dataset(self.targets, targets_path, years) as temp_targets_ds:
            self.n_time = temp_targets_ds.sizes["time"]
            self.n_lat = temp_targets_ds.sizes["lat"]
            self.n_lon = temp_targets_ds.sizes["lon"]
            self.time = temp_targets_ds["time"].values
            self.target_lat = temp_targets_ds["lat"].values
            self.target_lon = temp_targets_ds["lon"].values
            self.target_attrs: dict[str, dict] = {
                target: dict(temp_targets_ds[target].attrs) for target in self.targets
            }

        # Load the static data, if required.
        self.static_variables: list[str] = config.static or []
        self.static_data: xr.Dataset | None = None
        if self.static_variables:
            if static_path is None:
                raise ValueError("static_path must be provided to use static variables.")
            with xr.open_dataset(static_path) as static_ds:
                self.static_data = static_ds[self.static_variables].compute()

        # Generate sinusoidal positional embeddings if requested, and attach them to
        # static_data so they are stacked alongside any other static fields.
        if config.pos_emb:
            if self.static_data is not None:
                emb_lat = self.static_data["lat"].values
                emb_lon = self.static_data["lon"].values
            else:
                emb_lat = self.target_lat
                emb_lon = self.target_lon
                self.static_data = xr.Dataset(coords={"lat": emb_lat, "lon": emb_lon})
            pos_emb_arrays = self._generate_pos_emb(emb_lat, emb_lon)
            for name, arr in pos_emb_arrays.items():
                self.static_data[name] = (("lat", "lon"), arr)

        # Placeholders for worker-specific datasets (initialised in __getitem__)
        self.ds_predictors = None
        self.ds_targets = None
        self.ds_source = None

    def __len__(self) -> int:
        return self.n_time

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            predictors: Tensor of predictor variables, shape (no. predictors, lat, lon)
            targets: Tensor of target variables, shape (no. targets, lat, lon)
            source: If config source type is "coupled", this is a tensor of coupled source
                variables, shape (no. coupled variables, lat, lon). Otherwise, returns an empty
                placeholder tensor of the same shape as targets.
            static: Tensor of static variables, shape (no. static variables, lat, lon), or an empty
                placeholder tensor of shape (0, lat, lon) if no static variables are used.
        """
        self._init_worker_datasets()

        return self._get_sample(idx)

    def _generate_pos_emb(self, lat: np.ndarray, lon: np.ndarray) -> dict[str, np.ndarray]:
        """Generate 4-channel sinusoidal positional embeddings.

        Each lat/lon coordinate is scaled to [0, π] across the full domain and encoded as a
        (sin, cos) pair. A half-cycle (rather than full-cycle) scaling is used because the
        domain is regional and non-cyclic: with a full cycle the two ends of each axis would
        have identical (sin, cos) values, encoding opposite edges of the domain as the same
        position. With a half cycle, cos varies monotonically from 1 to -1 across the axis,
        so every grid point is uniquely identified by its (sin, cos) pair.

        The embeddings are computed using the exact lat/lon arrays they will be associated
        with, so they remain correctly oriented regardless of whether the coordinate axis runs
        north-to-south or south-to-north (and similarly for lon).

        Args:
            lat: 1D array of latitude values for the grid.
            lon: 1D array of longitude values for the grid.

        Returns:
            Dict mapping variable name to a 2D numpy array of shape (n_lat, n_lon).
        """
        lat_scaled = (lat - lat.min()) / (lat.max() - lat.min()) * np.pi
        lon_scaled = (lon - lon.min()) / (lon.max() - lon.min()) * np.pi

        shape = (len(lat), len(lon))
        return {
            "pos_sin_lat": np.broadcast_to(np.sin(lat_scaled)[:, None], shape).copy(),
            "pos_cos_lat": np.broadcast_to(np.cos(lat_scaled)[:, None], shape).copy(),
            "pos_sin_lon": np.broadcast_to(np.sin(lon_scaled)[None, :], shape).copy(),
            "pos_cos_lon": np.broadcast_to(np.cos(lon_scaled)[None, :], shape).copy(),
        }

    def _init_worker_datasets(self):
        """Helper to initialise datasets once per worker."""
        if self.ds_predictors is None:
            # Creating file handle for the datasets. This only happens once per worker.
            self.ds_predictors = self._create_dataset(
                self.predictors, self.predictors_path, self.years
            )
            self.ds_targets = self._create_dataset(self.targets, self.targets_path, self.years)
            if self.source:
                self.ds_source = self._create_dataset(self.source, self.predictors_path, self.years)

    def _create_dataset(self, variables: list[str], base_path: str, years: list[int]) -> xr.Dataset:
        """Return an xarray dataset containing the specified variables.

        Args:
            variables: List of variable names to include in the dataset.
            base_path: Path to the base directory containing the variable zarr files.
            years: List of years to include in the dataset.

        Returns:
            ds: Dataset containing the specified variables, concatenated along the "variable"
                dimension, and filtered to the specified years.
        """
        ds_vars_list: list[xr.DataArray] = []
        for var in variables:
            path = f"{base_path}/{var}.zarr"
            ds = xr.open_zarr(path, consolidated=True, chunks=None)
            ds_vars_list.append(ds[var])
        ds = xr.merge(ds_vars_list)
        ds = ds.sel(time=ds.time.dt.year.isin(years))
        return ds

    def _get_sample(
        self, time_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a sample from the dataset for the given coordinates.

        If using static variables, they will be concatenated with the predictors.

        Args:
            time_idx: Time index of the sample to retrieve.

        Returns:
            predictors: Tensor of predictor variables, shape (no. predictors, lat, lon)
            targets: Tensor of target variables, shape (no. targets, lat, lon)
            source: If using a coupled source, this is a tensor of coupled source
                variables, shape (no. coupled variables, lat, lon). Otherwise, returns an empty
                placeholder tensor of the same shape as targets.
            static: Tensor of static variables, shape (no. static variables, lat, lon), or an empty
                placeholder tensor of shape (0, lat, lon) if no static variables are used.
        """
        targets = self.ds_targets.isel(time=time_idx)
        predictors = self.ds_predictors.isel(time=time_idx)

        targets = targets.compute()
        predictors = predictors.compute()

        # Interpolate predictors to the same spatial grid as the targets, using linear interpolation.
        predictors = predictors.interp(
            lat=targets.lat,
            lon=targets.lon,
            method="linear",
        )

        # Apply transforms and convert to torch tensors
        targets = torch.from_numpy(self._apply_transforms(targets, group="targets").values).float()
        predictors = torch.from_numpy(
            self._apply_transforms(predictors, group="predictors").values
        ).float()

        if self.static_data is not None:
            static = self.static_data.to_array(dim="variable").transpose(
                "variable", *self.SPATIAL_DIMS
            )
            static = torch.from_numpy(static.values).float()
        else:
            static = torch.empty((0, targets.shape[1], targets.shape[2]))

        if self.source:
            source = self.ds_source.isel(time=time_idx).compute()
            source = source.interp(
                lat=targets.lat,
                lon=targets.lon,
                method="linear",
            )
            source = torch.from_numpy(self._apply_transforms(source, group="source").values).float()
        else:
            source = torch.empty_like(targets)

        return predictors, targets, source, static

    def _apply_transforms(self, sample: xr.Dataset, group: str) -> xr.DataArray:
        """Apply the transforms to a sample of variables.

        Args:
            sample: Dataset containing the variables to transform.
            group: Name of the group of variables, one of "predictors", "targets" or "source".

        Returns:
            transformed_sample: Stacked data array (no. vars, lat, lon)
        """
        transformed_vars_list: list[xr.DataArray] = []
        for var in sample.data_vars:
            transformed_var = self.transforms[group][var].transform(sample[var])
            transformed_vars_list.append(transformed_var)

        transformed_sample = xr.concat(transformed_vars_list, dim="variable")
        transformed_sample = transformed_sample.transpose("variable", *self.SPATIAL_DIMS)
        return transformed_sample
