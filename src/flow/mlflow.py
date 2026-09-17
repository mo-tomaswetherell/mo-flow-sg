"""Utility functions for setting up MLflow experiment tracking."""

import os
import functools

import mlflow


DEFAULT_EXPERIMENT_NAME = "default_experiment"
"""Default experiment name to use if not specified in the environment."""


def setup_mlflow(func: callable) -> callable:
    """Decorator to setup MLflow tracking.

    Sets the MLflow experiment name, configures the tracking URI, and starts a new run before
    calling the decorated function.

    The tracking URI is resolved as follows:
    - If ``MLFLOW_TRACKING_URI`` is set, it is respected as-is (MLflow reads this automatically,
      so no explicit configuration is needed).
    - Otherwise, if ``MLFLOW_PORT`` is set, the tracking URI is set to a local server at
      ``http://127.0.0.1:{MLFLOW_PORT}``.
    - Otherwise, MLflow's default tracking location (a local ``./mlruns`` directory) is used.

    Reads the following optional environment variables:
    - EXPERIMENT: The name of the experiment to use (defaults to ``DEFAULT_EXPERIMENT_NAME``).
    - RUN_NAME: The name of the run to start.
    - MLFLOW_TRACKING_URI: The MLflow tracking URI, if set externally.
    - MLFLOW_PORT: The port on which a local MLflow server is running (e.g., 5000).

    Args:
        func: The function to decorate.

    Returns:
        wrapper: The decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        experiment_name = os.getenv("EXPERIMENT", DEFAULT_EXPERIMENT_NAME)
        mlflow.set_experiment(experiment_name)

        # Respect an externally-provided tracking URI; otherwise fall back to the local server
        # convention via MLFLOW_PORT; otherwise use MLflow's default location.
        if os.getenv("MLFLOW_TRACKING_URI") is None:
            mlflow_port = os.getenv("MLFLOW_PORT")
            if mlflow_port is not None:
                mlflow.set_tracking_uri(f"http://127.0.0.1:{mlflow_port}")

        run_name = os.getenv("RUN_NAME")
        with mlflow.start_run(run_name=run_name):
            return func(*args, **kwargs)

    return wrapper
