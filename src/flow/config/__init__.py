import logging
import importlib.resources as resources
import yaml

from .schema import FlowMatchingConfig


logger = logging.getLogger(__name__)


def load_config(rel_path: str) -> FlowMatchingConfig:
    """Return a FlowMatchingConfig object loaded from a configuration file.

    The configuration is validated against the FlowMatchingConfig schema.

    Args:
        rel_path: Path to config file relative to config directory.

    Returns:
        FlowMatchingConfig: Loaded configuration object.
    """
    config: dict = load_config_yaml(rel_path)
    return FlowMatchingConfig.model_validate(config)


def load_config_yaml(rel_path: str) -> dict:
    """Return the contents of a yaml file in the config directory as a dictionary.

    Args:
        rel_path: Path to yaml configuration file relative to config directory.

    Returns:
        dict: Loaded configuration as a dictionary.

    Raises:
        ValueError: If the config file does not end with .yaml or .yml.
        FileNotFoundError: If the config file is not found.
    """
    if not rel_path.lower().endswith((".yaml", ".yml")):
        raise ValueError(f"Config filename must end with .yaml or .yml; got {rel_path}")

    # Access the base package directory
    pkg_files = resources.files(__package__)

    # Traverse the path (e.g., 'gaussian/sm.yaml' -> pkg/gaussian/sm.yaml)
    file_path = pkg_files.joinpath(rel_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    logger.info(f"Loading config from {file_path}")

    with file_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
