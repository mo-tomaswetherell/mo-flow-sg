"""Pydantic models for the flow matching configuration."""

from pydantic import BaseModel, Field, model_validator
from typing import Annotated, Literal


class NormalisationSpec(BaseModel):
    name: Literal["normalisation"]
    dims: list[str]


class MinMaxSpec(BaseModel):
    name: Literal["minmax"]
    out_min: float
    out_max: float


class LogSpec(BaseModel):
    name: Literal["log"]
    eps: float


class Log1pSpec(BaseModel):
    name: Literal["log1p"]


class SqrtSpec(BaseModel):
    name: Literal["sqrt"]


class ClipOutputsSpec(BaseModel):
    name: Literal["clip_outputs"]
    min: float | None = None
    max: float | None = None


TransformSpec = Annotated[
    NormalisationSpec | MinMaxSpec | LogSpec | Log1pSpec | SqrtSpec | ClipOutputsSpec,
    Field(discriminator="name"),
]


class VariableConfig(BaseModel):
    transforms: list[TransformSpec]


class GaussianSourceConfig(BaseModel):
    type: Literal["gaussian"]
    mean: float
    std: float


class CoupledSourceConfig(BaseModel):
    type: Literal["coupled"]
    variables: dict[str, VariableConfig]


SourceConfig = Annotated[GaussianSourceConfig | CoupledSourceConfig, Field(discriminator="type")]


class CondOTSchedulerConfig(BaseModel):
    type: Literal["condot"]


class PolynomialSchedulerConfig(BaseModel):
    type: Literal["polynomial"]
    n: float | int


SchedulerConfig = Annotated[
    CondOTSchedulerConfig | PolynomialSchedulerConfig, Field(discriminator="type")
]


class TrainingConfig(BaseModel):
    years: list[str]
    """List of year range strings (e.g., "1961-1970") or individual year strings (e.g., "2032")."""

    learning_rate: float
    epochs: int
    batch_size: int

    gradient_accumulation_steps: int = 1
    """Number of steps to accumulate gradients for before performing an optimiser step. Useful for
    effectively increasing training batch size when limited by memory constraints."""

    patience: int | None = None
    """Number of epochs with no improvement on validation loss after which training will be
    stopped. If None, no early stopping is used."""

    conditioning_dropout: float = 0.0
    """Probability of dropping out conditioning variables during training. Used for classifier-free
    guidance training. Default is 0 (no dropout), which corresponds to vanilla guidance. Must be
    between 0 and 1."""

    ema_decay: float | None = None
    """If not None, the exponential moving average decay rate to use for tracking an EMA version
     of the model weights during training. Must be between 0 and 1, or None. Typically set to a
     value close to 1, e.g., 0.999."""

    @model_validator(mode="after")
    def check_conditioning_dropout(self) -> "TrainingConfig":
        if not (0 <= self.conditioning_dropout <= 1):
            raise ValueError("conditioning_dropout must be between 0 and 1")
        return self

    @model_validator(mode="after")
    def check_ema_decay(self) -> "TrainingConfig":
        if self.ema_decay is not None and not (0 <= self.ema_decay <= 1):
            raise ValueError("ema_decay must be between 0 and 1, or None")
        return self


class ValidationConfig(BaseModel):
    years: list[str]
    """List of year range strings (e.g., "1961-1970") or individual year strings (e.g., "2032")."""


class TestConfig(BaseModel):
    years: list[str]
    """List of year range strings (e.g., "1961-1970") or individual year strings (e.g., "2032")."""


class ModelConfig(BaseModel):
    img_resolution: int
    model_channels: int
    channel_mult: list[int]
    channel_mult_emb: int
    num_residual_blocks: int
    attention_resolutions: list[int]
    dropout: float


class FlowMatchingConfig(BaseModel):
    predictors: dict[str, VariableConfig]
    targets: dict[str, VariableConfig]
    source: SourceConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    validation: ValidationConfig
    test: TestConfig
    model: ModelConfig

    static: list[str] | None = None
    """List of static variable names."""

    pos_emb: bool = False
    """Whether to use positional embedding, which allows the network to condition on absolute
    spatial location."""

    seed: int = 0
    """Random seed for reproducibility."""

    @model_validator(mode="after")
    def check_num_coupled_variables_equals_num_targets(self) -> "FlowMatchingConfig":
        if isinstance(self.source, CoupledSourceConfig):
            if len(self.source.variables) != len(self.targets):
                raise ValueError(
                    f"Number of coupled variables ({len(self.source.variables)}) must equal the "
                    f"number of target variables ({len(self.targets)})."
                )
        return self
