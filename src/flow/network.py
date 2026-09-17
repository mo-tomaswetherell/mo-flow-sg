"""Module for the AMD U-Net architecture."""

import math

import torch
from torch.nn import Module
from torch import Tensor
from einops import rearrange


class FourierEmbedding(Module):
    """Fourier Embedding.

    FE(t) = [sin(2 * pi * t * w_1), cos(2 * pi * t * w_1), ..., sin(2 * pi * t * w_d/2), cos(2 * pi * t * w_d/2)] * sqrt(2),
    where w_i are the weights of the Fourier embedding and t is a scalar time in [0, 1].
    """

    def __init__(self, dim: int, learnable: bool = False):
        """Initialise.

        Args:
            dim: Dimension of the output embedding. Must be even.
            learnable: If True, the weights will be learnable parameters. If False, they
                will be fixed random values drawn from a normal distribution.
        """
        super().__init__()
        assert dim % 2 == 0
        half_dim = dim // 2
        self.weights = torch.nn.Parameter(torch.randn(half_dim), requires_grad=learnable)

    def forward(self, t: Tensor) -> Tensor:
        """Embed time tensor using fourier embedding.

        Args:
            t: Times between 0 and 1, shape (batch_size, 1)

        Returns:
            Fourier embedding, shape (batch_size, dim)
        """
        freqs = t * rearrange(self.weights, "half_dim -> 1 half_dim") * 2 * math.pi
        return torch.cat((freqs.sin(), freqs.cos()), dim=-1) * math.sqrt(2)


class Conv2d(torch.nn.Module):
    """Convolution layer, with optional downsampling or upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
        upsample: bool = False,
        downsample: bool = False,
        resample_filter: list[int] = [1, 1],
    ):
        """Inititalise block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size of convolution layer. If 0, no convolution is applied.
            bias: If True, add a bias term to the convolution layer.
            upsample: If True, upsample by a factor of 2.
            downsample: If True, downsample by a factor of 2.
            resample_filter: 1D kernel/filter for up/downsampling. Defaults to [1, 1], which is
                equivalent to average pooling.
        """
        assert not (upsample and downsample)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        self.downsample = downsample

        self.conv = (
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2 if kernel_size > 0 else 0,
                bias=bias,
            )
            if kernel_size > 0
            else None
        )

        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if upsample or downsample else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsampling / downsampling
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.upsample:
            x = torch.nn.functional.conv_transpose2d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=f_pad,
            )
        elif self.downsample:
            x = torch.nn.functional.conv2d(
                x,
                f.tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=f_pad,
            )

        if self.conv is not None:
            x = self.conv(x)

        return x


class Encoder(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        downsample: bool = False,
        adaptive_scale: bool = True,
        resample_filter: list[int] = [1, 1],
        attention: bool = False,
        channels_per_head: int = 64,
        dropout: float = 0.0,
    ):
        """Initialise encoder block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            emb_channels: Number of channels in the time embedding.
            downsample: If True, downsample by a factor of 2.
            adaptive_scale: If True, apply FiLM conditioning (scale and shift the input features).
                If False, apply additive conditioning (add the embedding to the input features).
            resample_filter: 1D kernel/filter for downsampling. Defaults to [1, 1], which is
                equivalent to average pooling.
            attention: If True, apply attention mechanism.
            channels_per_head: Number of channels per attention head, if attention is True.
            dropout: Dropout rate applied after the first convolution layer.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.adaptive_scale = adaptive_scale

        # Affine transformation applied to time embedding
        self.affine = torch.nn.Linear(
            in_features=emb_channels,
            out_features=out_channels * (2 if adaptive_scale else 1),
            bias=True,
        )

        layers_pre_modulation: list[torch.nn.Module] = [
            torch.nn.GroupNorm(num_groups=32, num_channels=in_channels),
            torch.nn.SiLU(),
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                bias=True,
                downsample=downsample,
                resample_filter=resample_filter,
            ),
            torch.nn.GroupNorm(num_groups=32, num_channels=out_channels),
        ]
        self.layers_pre_modulation = torch.nn.Sequential(*layers_pre_modulation)

        layers_post_modulation: list[torch.nn.Module] = [
            torch.nn.SiLU(),
            torch.nn.Dropout(p=dropout),
            Conv2d(out_channels, out_channels, kernel_size=3, bias=True, downsample=False),
        ]
        self.layers_post_modulation = torch.nn.Sequential(*layers_post_modulation)

        # Skip connection
        if out_channels != in_channels or downsample:
            # Use 1x1 convolution if we need to change the number of channels.
            # Otherwise (if we're just downsampling), there is no convolution applied.
            kernel = 1 if out_channels != in_channels else 0
            self.skip = Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                downsample=downsample,
                resample_filter=resample_filter,
            )
        else:
            # No need to modify the input if the number of channels is constant and no downsampling is applied.
            self.skip = torch.nn.Identity()

        # Multi-head self-attention
        self.attention = attention
        if self.attention:
            if out_channels % channels_per_head != 0:
                raise ValueError(
                    f"out_channels ({out_channels}) must be divisible by channels_per_head ({channels_per_head})"
                )
            self.num_heads = out_channels // channels_per_head
            self.norm2 = torch.nn.GroupNorm(num_groups=32, num_channels=out_channels)
            self.qkv = Conv2d(
                in_channels=out_channels, out_channels=out_channels * 3, kernel_size=1
            )
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (batch_size, in_channels, height, width)
            emb: Time embedding, shape (batch_size, emb_channels)

        Returns:
            x: Output tensor, shape (batch_size, out_channels, height', width'), where height' and
                width' depend on whether downsampling was applied (will be half the input size if downsample=True).
        """
        orig = x

        x = self.layers_pre_modulation(x)

        emb = rearrange(self.affine(emb), "bs c -> bs c 1 1")
        if self.adaptive_scale:
            # FiLM conditioning: scale and shift the input features
            scale, shift = torch.chunk(emb, 2, dim=1)
            x = x * (1 + scale) + shift  # identity if scale and shift are zero
        else:
            # Additive conditioning: add the embedding to the input features
            x = x + emb

        x = self.layers_post_modulation(x)
        x = self.skip(orig) + x

        if self.attention:
            q, k, v = (
                self.qkv(self.norm2(x))
                .reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1)
                .unbind(2)
            )

            scale = 1 / math.sqrt(k.size(1))
            w = torch.einsum("ncq,nck->nqk", q, k).mul_(scale)
            w = w.to(dtype=torch.float32).softmax(dim=2).to(dtype=q.dtype)
            a = torch.einsum("nqk,nck->ncq", w, v)

            x = self.proj(a.reshape(*x.shape)).add_(x)

        return x


class Decoder(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        upsample: bool = False,
        adaptive_scale: bool = True,
        resample_filter: list[int] = [1, 1],
        attention: bool = False,
        channels_per_head: int = 64,
        dropout: float = 0.0,
    ):
        """Initialise decoder block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            emb_channels: Number of channels in the time embedding.
            upsample: If True, upsample by a factor of 2.
            adaptive_scale: If True, apply FiLM conditioning (scale and shift the input features).
                If False, apply additive conditioning (add the embedding to the input features).
            resample_filter: 1D kernel/filter for upsampling.
            attention: If True, apply attention mechanism.
            channels_per_head: Number of channels per attention head, if attention is True.
            dropout: Dropout rate applied after the first convolution layer.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.adaptive_scale = adaptive_scale

        # Affine transformation applied to time embedding
        self.affine = torch.nn.Linear(
            in_features=emb_channels,
            out_features=out_channels * (2 if adaptive_scale else 1),
            bias=True,
        )

        layers_pre_modulation: list[torch.nn.Module] = [
            torch.nn.GroupNorm(num_groups=32, num_channels=in_channels),
            torch.nn.SiLU(),
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                bias=True,
                upsample=upsample,
                resample_filter=resample_filter,
            ),
            torch.nn.GroupNorm(num_groups=32, num_channels=out_channels),
        ]
        self.layers_pre_modulation = torch.nn.Sequential(*layers_pre_modulation)

        layers_post_modulation: list[torch.nn.Module] = [
            torch.nn.SiLU(),
            torch.nn.Dropout(p=dropout),
            Conv2d(out_channels, out_channels, kernel_size=3, bias=True, upsample=False),
        ]
        self.layers_post_modulation = torch.nn.Sequential(*layers_post_modulation)

        # Skip connection
        if out_channels != in_channels or upsample:
            # Use 1x1 convolution if we need to change the number of channels.
            # Otherwise (if we're just upsampling), there is no convolution applied.
            kernel = 1 if out_channels != in_channels else 0
            self.skip = Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                upsample=upsample,
                resample_filter=resample_filter,
            )
        else:
            # No need to modify the input if the number of channels is constant and no upsampling is applied.
            self.skip = torch.nn.Identity()

        # Multi-head self-attention
        self.attention = attention
        if self.attention:
            if out_channels % channels_per_head != 0:
                raise ValueError(
                    f"out_channels ({out_channels}) must be divisible by channels_per_head ({channels_per_head})"
                )
            self.num_heads = out_channels // channels_per_head
            self.norm2 = torch.nn.GroupNorm(num_groups=32, num_channels=out_channels)
            self.qkv = Conv2d(
                in_channels=out_channels, out_channels=out_channels * 3, kernel_size=1
            )
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (batch_size, in_channels, height, width)
            emb: Time embedding, shape (batch_size, emb_channels)

        Returns:
            x: Output tensor, shape (batch_size, out_channels, height', width'), where height' and
                width' depend on whether upsampling was applied (will be double the input size if upsample=True).
        """
        orig = x

        x = self.layers_pre_modulation(x)

        emb = rearrange(self.affine(emb), "bs c -> bs c 1 1")
        if self.adaptive_scale:
            # FiLM conditioning: scale and shift the input features
            scale, shift = torch.chunk(emb, 2, dim=1)
            x = x * (1 + scale) + shift  # identity if scale and shift are zero
        else:
            # Additive conditioning: add the embedding to the input features
            x = x + emb

        x = self.layers_post_modulation(x)
        x = self.skip(orig) + x

        if self.attention:
            q, k, v = (
                self.qkv(self.norm2(x))
                .reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1)
                .unbind(2)
            )

            scale = 1 / math.sqrt(k.size(1))
            w = torch.einsum("ncq,nck->nqk", q, k).mul_(scale)
            w = w.to(dtype=torch.float32).softmax(dim=2).to(dtype=q.dtype)
            a = torch.einsum("nqk,nck->ncq", w, v)

            x = self.proj(a.reshape(*x.shape)).add_(x)

        return x


class ADM(Module):
    def __init__(
        self,
        num_conditioning_variables: int,
        num_target_variables: int,
        img_resolution: int = 64,
        model_channels: int = 192,
        channel_mult: list[int] = [1, 2, 3, 4],
        channel_mult_emb: int = 4,
        num_residual_blocks: int = 3,
        attention_resolutions: list[int] = [16, 8],
        dropout: float = 0.1,
    ):
        """Initialise ADM U-Net.

        Args:
            num_conditioning_variables: Number of conditioning variables.
            img_resolution: Resolution at input and output.
            model_channels: Base mutiplier for the number of channels.
            channel_mult: Per-resolution multipliers for the number of channels.
            channel_mult_emb: Multiplier for the dimensionality of the embedding vector.
            num_residual_blocks: Number of residual blocks per resolution.
            attention_resolutions: List of resolutions with self-attention.
            dropout: Dropout rate.
        """
        super().__init__()

        # The conditioning variables are concatenated with x_t (which has num_target_variables
        # channels).
        self.num_variables = num_conditioning_variables + num_target_variables
        self.num_target_variables = num_target_variables

        # Embedding
        emb_channels = model_channels * channel_mult_emb
        embedding_layers: list[torch.nn.ModuleDict] = [
            FourierEmbedding(dim=model_channels),
            torch.nn.Linear(in_features=model_channels, out_features=emb_channels, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features=emb_channels, out_features=emb_channels, bias=True),
            torch.nn.SiLU(),
        ]
        self.embedding = torch.nn.Sequential(*embedding_layers)

        # Encoder
        self.enc = torch.nn.ModuleDict()
        for level, mult in enumerate(channel_mult):
            res = img_resolution // (2**level)
            if level == 0:
                in_ch = self.num_variables
                out_ch = model_channels * mult
                self.enc[f"{res}x{res}_conv"] = Conv2d(
                    in_channels=in_ch, out_channels=out_ch, kernel_size=3
                )
            else:
                self.enc[f"{res}x{res}_down"] = Encoder(
                    in_channels=out_ch,
                    out_channels=out_ch,
                    emb_channels=emb_channels,
                    downsample=True,
                    attention=False,
                    dropout=dropout,
                )

            for idx in range(num_residual_blocks):
                in_ch = out_ch
                out_ch = model_channels * mult
                self.enc[f"{res}x{res}_block{idx}"] = Encoder(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    emb_channels=emb_channels,
                    downsample=False,
                    attention=(res in attention_resolutions),
                    dropout=dropout,
                )
        skip_num_channels = [block.out_channels for block in self.enc.values()]

        # Decoder
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution // (2**level)
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_mid1"] = Decoder(
                    in_channels=out_ch,
                    out_channels=out_ch,
                    emb_channels=emb_channels,
                    attention=True,
                    dropout=dropout,
                )
                self.dec[f"{res}x{res}_mid2"] = Decoder(
                    in_channels=out_ch,
                    out_channels=out_ch,
                    emb_channels=emb_channels,
                    attention=False,
                    dropout=dropout,
                )
            else:
                self.dec[f"{res}x{res}_up"] = Decoder(
                    in_channels=out_ch,
                    out_channels=out_ch,
                    emb_channels=emb_channels,
                    upsample=True,
                    attention=False,
                    dropout=dropout,
                )

            for idx in range(num_residual_blocks + 1):
                in_ch = out_ch + skip_num_channels.pop()
                out_ch = model_channels * mult
                self.dec[f"{res}x{res}_block{idx}"] = Decoder(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    emb_channels=emb_channels,
                    upsample=False,
                    attention=(res in attention_resolutions),
                    dropout=dropout,
                )

        # Final output block
        self.out_block = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32, num_channels=out_ch),
            torch.nn.SiLU(),
            Conv2d(
                in_channels=out_ch, out_channels=self.num_target_variables, kernel_size=3, bias=True
            ),
        )

    def forward(self, x_t: Tensor, t: Tensor, conditioning: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x_t: Samples from probability path, shape (batch_size, num_target_variables, height, width)
            t: Times between [0, 1], shape (batch_size, 1)
            conditioning: Conditioning variables, shape (batch_size, num_conditioning_variables, height, width)

        Returns:
            x: Velocity estimates, shape (batch_size, num_target_variables, height, width)
        """
        emb = self.embedding(t)  # shape (bs, emb_channels)

        x = torch.concat((x_t, conditioning), dim=1)

        # Encoder
        skips: torch.Tensor = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, Encoder) else block(x)
            skips.append(x)

        # Decoder
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat((x, skips.pop()), dim=1)
            x = block(x, emb)

        x = self.out_block(x)
        return x
