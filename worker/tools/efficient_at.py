"""Minimal EfficientAT MN10 inference model.

Adapted from the MIT-licensed EfficientAT reference implementation.  This
runtime-only copy avoids adding torchvision and never downloads weights.
"""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _make_divisible(
    value: float,
    divisor: int,
    minimum: int | None = None,
) -> int:
    minimum = divisor if minimum is None else minimum
    rounded = max(minimum, int(value + divisor / 2) // divisor * divisor)
    return rounded + divisor if rounded < 0.9 * value else rounded


def _output_size(
    input_size: int,
    padding: int,
    dilation: int,
    kernel: int,
    stride: int,
) -> int:
    return math.floor(
        (input_size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    )


class _ConvNormActivation(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        norm_layer: Callable[[int], nn.Module],
        activation_layer: type[nn.Module] | None,
    ):
        padding = (kernel_size - 1) // 2 * dilation
        layers: list[nn.Module] = [
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias=False,
            ),
            norm_layer(output_channels),
        ]
        if activation_layer is not None:
            layers.append(activation_layer(inplace=True))
        super().__init__(*layers)


class _SqueezeExcitation(nn.Module):
    def __init__(self, input_dim: int, squeeze_dim: int, se_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, squeeze_dim)
        self.fc2 = nn.Linear(squeeze_dim, input_dim)
        self.se_dim = [1, 2, 3]
        self.se_dim.remove(se_dim)
        self.activation = nn.ReLU()
        self.scale_activation = nn.Sigmoid()

    def forward(self, value: Tensor) -> Tensor:
        scale = torch.mean(value, self.se_dim, keepdim=True)
        shape = scale.size()
        scale = self.fc1(scale.squeeze(2).squeeze(2))
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale).view(shape) * value


class _ConcurrentSEBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        frequency_dim: int,
        time_dim: int,
        *,
        se_dims: tuple[int, ...],
        reduction: int,
    ):
        super().__init__()
        dimensions = (channel_dim, frequency_dim, time_dim)
        self.conc_se_layers = nn.ModuleList(
            [
                _SqueezeExcitation(
                    dimensions[dimension - 1],
                    _make_divisible(
                        dimensions[dimension - 1] // reduction,
                        8,
                    ),
                    dimension,
                )
                for dimension in se_dims
            ]
        )

    def forward(self, value: Tensor) -> Tensor:
        return torch.stack(
            [layer(value) for layer in self.conc_se_layers],
            dim=0,
        ).amax(dim=0)


class _InvertedResidualConfig:
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        output_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = _make_divisible(input_channels * width_mult, 8)
        self.kernel = kernel
        self.expanded_channels = _make_divisible(
            expanded_channels * width_mult,
            8,
        )
        self.output_channels = _make_divisible(
            output_channels * width_mult,
            8,
        )
        self.use_se = use_se
        self.use_hardswish = activation == "HS"
        self.stride = stride
        self.dilation = dilation
        self.frequency_dim = 0
        self.time_dim = 0

    def output_size(self, input_size: int) -> int:
        padding = (self.kernel - 1) // 2 * self.dilation
        return _output_size(
            input_size,
            padding,
            self.dilation,
            self.kernel,
            self.stride,
        )


class _InvertedResidual(nn.Module):
    def __init__(
        self,
        config: _InvertedResidualConfig,
        *,
        norm_layer: Callable[[int], nn.Module],
    ):
        super().__init__()
        if config.stride not in {1, 2}:
            raise ValueError("EfficientAT stride must be 1 or 2")
        activation = nn.Hardswish if config.use_hardswish else nn.ReLU
        layers: list[nn.Module] = []
        if config.expanded_channels != config.input_channels:
            layers.append(
                _ConvNormActivation(
                    config.input_channels,
                    config.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation,
                )
            )
        layers.append(
            _ConvNormActivation(
                config.expanded_channels,
                config.expanded_channels,
                kernel_size=config.kernel,
                stride=(
                    1
                    if config.dilation > 1
                    else config.stride
                ),
                dilation=config.dilation,
                groups=config.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation,
            )
        )
        if config.use_se:
            layers.append(
                _ConcurrentSEBlock(
                    config.expanded_channels,
                    config.frequency_dim,
                    config.time_dim,
                    se_dims=(1,),
                    reduction=4,
                )
            )
        layers.append(
            _ConvNormActivation(
                config.expanded_channels,
                config.output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = (
            config.stride == 1
            and config.input_channels == config.output_channels
        )

    def forward(self, value: Tensor) -> Tensor:
        result = self.block(value)
        return result + value if self.use_residual else result


class EfficientATMN10(nn.Module):
    """The 4.88M parameter MN10 AudioSet classifier."""

    def __init__(self, *, classes: int = 527):
        super().__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        block = partial(_InvertedResidualConfig, width_mult=1.0)
        settings = [
            block(16, 3, 16, 16, False, "RE", 1, 1),
            block(16, 3, 64, 24, False, "RE", 2, 1),
            block(24, 3, 72, 24, False, "RE", 1, 1),
            block(24, 5, 72, 40, True, "RE", 2, 1),
            block(40, 5, 120, 40, True, "RE", 1, 1),
            block(40, 5, 120, 40, True, "RE", 1, 1),
            block(40, 3, 240, 80, False, "HS", 2, 1),
            block(80, 3, 200, 80, False, "HS", 1, 1),
            block(80, 3, 184, 80, False, "HS", 1, 1),
            block(80, 3, 184, 80, False, "HS", 1, 1),
            block(80, 3, 480, 112, True, "HS", 1, 1),
            block(112, 3, 672, 112, True, "HS", 1, 1),
            block(112, 5, 672, 160, True, "HS", 2, 1),
            block(160, 5, 960, 160, True, "HS", 1, 1),
            block(160, 5, 960, 160, True, "HS", 1, 1),
        ]
        frequency_dim = 128
        time_dim = 1000
        frequency_dim = _output_size(frequency_dim, 1, 1, 3, 2)
        time_dim = _output_size(time_dim, 1, 1, 3, 2)
        layers: list[nn.Module] = [
            _ConvNormActivation(
                1,
                settings[0].input_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        ]
        for config in settings:
            frequency_dim = config.output_size(frequency_dim)
            time_dim = config.output_size(time_dim)
            config.frequency_dim = frequency_dim
            config.time_dim = time_dim
            layers.append(_InvertedResidual(config, norm_layer=norm_layer))
        layers.append(
            _ConvNormActivation(
                settings[-1].output_channels,
                960,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(960, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, classes),
        )

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.features(value)
        features = F.adaptive_avg_pool2d(encoded, (1, 1)).flatten(1)
        return self.classifier(encoded), features


def load_efficientat_mn10(weights: str | Path) -> EfficientATMN10:
    model = EfficientATMN10()
    state = torch.load(
        Path(weights).expanduser().resolve(),
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    return model.eval()
