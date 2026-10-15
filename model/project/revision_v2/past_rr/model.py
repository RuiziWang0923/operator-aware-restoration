from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_to_multiple(
    image: torch.Tensor, multiple: int
) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = image.shape[-2:]
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    if not pad_height and not pad_width:
        return image, (0, 0)
    mode = "reflect"
    if pad_height >= height or pad_width >= width:
        mode = "replicate"
    return F.pad(image, (0, pad_width, 0, pad_height), mode=mode), (
        pad_height,
        pad_width,
    )


def crop_padding(image: torch.Tensor, padding: tuple[int, int]) -> torch.Tensor:
    pad_height, pad_width = padding
    height = image.shape[-2] - pad_height if pad_height else image.shape[-2]
    width = image.shape[-1] - pad_width if pad_width else image.shape[-1]
    return image[..., :height, :width]


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.epsilon = epsilon

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mean = value.mean(dim=1, keepdim=True)
        variance = (value - mean).square().mean(dim=1, keepdim=True)
        return (value - mean) * torch.rsqrt(variance + self.epsilon) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        expanded = channels * expansion
        if expanded % 2:
            raise ValueError("NAF expansion must produce an even channel count")
        gated = expanded // 2
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, expanded, 1)
        self.depthwise = nn.Conv2d(
            expanded, expanded, 3, padding=1, groups=expanded
        )
        self.gate1 = SimpleGate()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(gated, gated, 1)
        )
        self.conv2 = nn.Conv2d(gated, channels, 1)
        self.beta = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, expanded, 1)
        self.gate2 = SimpleGate()
        self.ffn2 = nn.Conv2d(gated, channels, 1)
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.depthwise(self.conv1(self.norm1(value)))
        value = self.gate1(value)
        value = value * self.channel_attention(value)
        value = residual + self.beta * self.conv2(value)
        residual = value
        value = self.ffn2(self.gate2(self.ffn1(self.norm2(value))))
        return residual + self.gamma * value


def naf_stack(channels: int, blocks: int) -> nn.Sequential:
    return nn.Sequential(*[NAFBlock(channels) for _ in range(blocks)])


class LocalComplexSourceAllocator(nn.Module):
    """Locally allocate a feature mixture with complementary complex masks.

    The source masks sum to one exactly in the rFFT domain. Hann-window
    overlap-add then reconstructs transmission and nuisance feature estimates.
    FFT operations are intentionally forced to FP32 for numerical stability.
    """

    def __init__(
        self,
        channels: int,
        *,
        groups: int = 8,
        window_size: int = 16,
        stride: int = 8,
        hidden_per_group: int = 12,
        complex_radius: float = 0.1,
    ) -> None:
        super().__init__()
        if channels % groups:
            raise ValueError("Allocator channels must be divisible by groups")
        if not 0 < stride <= window_size:
            raise ValueError("Allocator stride must be in (0, window_size]")
        if not 0.0 < complex_radius <= 0.25:
            raise ValueError("complex_radius must be in (0, 0.25]")
        self.channels = channels
        self.groups = groups
        self.channels_per_group = channels // groups
        self.window_size = window_size
        self.stride = stride
        self.complex_radius = complex_radius

        input_channels = groups * 3
        hidden_channels = groups * hidden_per_group
        self.mask_network = nn.Sequential(
            nn.Conv2d(
                input_channels,
                hidden_channels,
                3,
                padding=1,
                groups=groups,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                padding=1,
                groups=groups,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                groups * 3,
                1,
                groups=groups,
            ),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 2, 1),
            SimpleGate(),
            nn.Conv2d(channels, channels, 1),
        )
        self.residual_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), 0.1)
        )

        one_dimensional = torch.hann_window(window_size, periodic=False).clamp_min(
            1e-3
        )
        analysis_window = torch.outer(one_dimensional, one_dimensional).sqrt()
        self.register_buffer(
            "analysis_window",
            analysis_window[None, None],
            persistent=False,
        )

    def _overlap_add(
        self,
        patches: torch.Tensor,
        *,
        batch: int,
        channels: int,
        padded_height: int,
        padded_width: int,
        windows: int,
        padding: int,
        output_height: int,
        output_width: int,
    ) -> torch.Tensor:
        window = self.analysis_window.float()
        patches = patches.float() * window
        flattened = (
            patches.reshape(batch, windows, channels * self.window_size**2)
            .transpose(1, 2)
            .contiguous()
        )
        reconstructed = F.fold(
            flattened,
            output_size=(padded_height, padded_width),
            kernel_size=self.window_size,
            stride=self.stride,
        )
        weights = (
            window.square()
            .reshape(1, self.window_size**2, 1)
            .expand(batch, -1, windows)
        )
        denominator = F.fold(
            weights,
            output_size=(padded_height, padded_width),
            kernel_size=self.window_size,
            stride=self.stride,
        ).clamp_min(1e-6)
        reconstructed = reconstructed / denominator
        return reconstructed[
            ...,
            padding : padding + output_height,
            padding : padding + output_width,
        ]

    def forward(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch, channels, height, width = value.shape
        padding = self.window_size - self.stride
        pad_mode = "reflect"
        if padding >= height or padding >= width:
            pad_mode = "replicate"
        padded = F.pad(value, (padding, padding, padding, padding), mode=pad_mode)
        padded_height, padded_width = padded.shape[-2:]
        unfolded = F.unfold(
            padded,
            kernel_size=self.window_size,
            stride=self.stride,
        )
        windows = unfolded.shape[-1]
        patches = (
            unfolded.transpose(1, 2)
            .reshape(batch, windows, channels, self.window_size, self.window_size)
        )
        windowed = patches * self.analysis_window.to(dtype=value.dtype)

        with torch.autocast(device_type=value.device.type, enabled=False):
            spectrum = torch.fft.rfft2(windowed.float(), norm="ortho")
            frequency_width = spectrum.shape[-1]
            grouped = spectrum.reshape(
                batch,
                windows,
                self.groups,
                self.channels_per_group,
                self.window_size,
                frequency_width,
            )
            group_amplitude = grouped.abs().mean(dim=3)
            group_complex_mean = grouped.mean(dim=3)
            phase = group_complex_mean / group_complex_mean.abs().clamp_min(1e-6)
            phase_reliability = group_amplitude / (
                group_amplitude + group_amplitude.mean(
                    dim=(-2, -1), keepdim=True
                ).clamp_min(1e-6)
            )
            mask_input = torch.stack(
                (
                    torch.log1p(group_amplitude),
                    phase.real * phase_reliability,
                    phase.imag * phase_reliability,
                ),
                dim=3,
            ).reshape(
                batch * windows,
                self.groups * 3,
                self.window_size,
                frequency_width,
            )

        mask_parameters = self.mask_network(
            mask_input.to(dtype=value.dtype)
        ).reshape(
            batch,
            windows,
            self.groups,
            3,
            self.window_size,
            frequency_width,
        )
        with torch.autocast(device_type=value.device.type, enabled=False):
            mask_parameters = mask_parameters.float()
            allocation = torch.sigmoid(mask_parameters[:, :, :, 0])
            correction = torch.complex(
                self.complex_radius * torch.tanh(mask_parameters[:, :, :, 1]),
                self.complex_radius * torch.tanh(mask_parameters[:, :, :, 2]),
            )
            transmission_mask = torch.complex(
                allocation, torch.zeros_like(allocation)
            ) + correction
            nuisance_mask = torch.ones_like(transmission_mask) - transmission_mask
            transmission_spectrum = grouped * transmission_mask.unsqueeze(3)
            nuisance_spectrum = grouped * nuisance_mask.unsqueeze(3)
            transmission_patches = torch.fft.irfft2(
                transmission_spectrum.reshape_as(spectrum),
                s=(self.window_size, self.window_size),
                norm="ortho",
            )
            nuisance_patches = torch.fft.irfft2(
                nuisance_spectrum.reshape_as(spectrum),
                s=(self.window_size, self.window_size),
                norm="ortho",
            )

        transmission = self._overlap_add(
            transmission_patches,
            batch=batch,
            channels=channels,
            padded_height=padded_height,
            padded_width=padded_width,
            windows=windows,
            padding=padding,
            output_height=height,
            output_width=width,
        )
        nuisance = self._overlap_add(
            nuisance_patches,
            batch=batch,
            channels=channels,
            padded_height=padded_height,
            padded_width=padded_width,
            windows=windows,
            padding=padding,
            output_height=height,
            output_width=width,
        )
        transmission = transmission.to(dtype=value.dtype)
        nuisance = nuisance.to(dtype=value.dtype)
        fused = self.fuse(
            torch.cat((transmission, nuisance, transmission - nuisance), dim=1)
        )
        output = value + self.residual_scale * fused

        grid_height = (padded_height - self.window_size) // self.stride + 1
        grid_width = (padded_width - self.window_size) // self.stride + 1
        allocation_map = allocation.mean(dim=(2, 3, 4)).reshape(
            batch, 1, grid_height, grid_width
        )
        allocation_map = F.interpolate(
            allocation_map,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).to(dtype=value.dtype)
        entropy = -(
            allocation.clamp(1e-6, 1.0 - 1e-6) * allocation.clamp_min(1e-6).log()
            + (1.0 - allocation).clamp(1e-6, 1.0).log()
            * (1.0 - allocation)
        )
        diagnostics = {
            "allocation_mean": allocation.detach().mean(),
            "allocation_std": allocation.detach().std(unbiased=False),
            "allocation_entropy": entropy.detach().mean(),
            "complex_correction_abs_mean": correction.detach().abs().mean(),
            "mask_sum_error_max": (
                transmission_mask + nuisance_mask - 1.0
            ).detach().abs().max(),
            "reconstruction_mae": (
                transmission.float() + nuisance.float() - value.float()
            ).detach().abs().mean(),
            "transmission_rms": transmission.detach().float().square().mean().sqrt(),
            "nuisance_rms": nuisance.detach().float().square().mean().sqrt(),
        }
        return output, transmission, nuisance, {
            "map": allocation_map,
            **diagnostics,
        }


class PhaseCorrelationContext(nn.Module):
    """Build a differentiable phase-alignment context at low resolution."""

    def __init__(self, channels: int, heads: int = 8) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("Phase context channels must be divisible by heads")
        self.heads = heads
        self.transmission_projection = nn.Conv2d(channels, heads, 1)
        self.nuisance_projection = nn.Conv2d(channels, heads, 1)
        self.local_projection = nn.Sequential(
            nn.Conv2d(heads * 3, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
        )
        self.global_projection = nn.Sequential(
            nn.Linear(heads * 6, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    @staticmethod
    def _circular_statistics(probability: torch.Tensor) -> torch.Tensor:
        batch, heads, height, width = probability.shape
        y = torch.arange(height, device=probability.device, dtype=probability.dtype)
        x = torch.arange(width, device=probability.device, dtype=probability.dtype)
        y_angle = 2.0 * torch.pi * y / max(height, 1)
        x_angle = 2.0 * torch.pi * x / max(width, 1)
        marginal_y = probability.sum(dim=-1)
        marginal_x = probability.sum(dim=-2)
        y_sin = (marginal_y * y_angle.sin()).sum(dim=-1)
        y_cos = (marginal_y * y_angle.cos()).sum(dim=-1)
        x_sin = (marginal_x * x_angle.sin()).sum(dim=-1)
        x_cos = (marginal_x * x_angle.cos()).sum(dim=-1)
        peak = probability.flatten(-2).amax(dim=-1)
        entropy = -(
            probability.clamp_min(1e-8).log() * probability
        ).flatten(-2).sum(dim=-1)
        entropy = entropy / max(math.log(float(height * width)), 1.0)
        return torch.stack((y_sin, y_cos, x_sin, x_cos, peak, entropy), dim=-1)

    def forward(
        self,
        transmission: torch.Tensor,
        nuisance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        transmission_projection = self.transmission_projection(transmission).float()
        nuisance_projection = self.nuisance_projection(nuisance).float()
        transmission_projection = transmission_projection - transmission_projection.mean(
            dim=(-2, -1), keepdim=True
        )
        nuisance_projection = nuisance_projection - nuisance_projection.mean(
            dim=(-2, -1), keepdim=True
        )
        height, width = transmission_projection.shape[-2:]
        with torch.autocast(device_type=transmission.device.type, enabled=False):
            transmission_spectrum = torch.fft.rfft2(
                transmission_projection, norm="ortho"
            )
            nuisance_spectrum = torch.fft.rfft2(
                nuisance_projection, norm="ortho"
            )
            cross = transmission_spectrum * nuisance_spectrum.conj()
            normalized_cross = cross / cross.abs().clamp_min(1e-6)
            correlation = torch.fft.irfft2(
                normalized_cross,
                s=(height, width),
                norm="ortho",
            )
            aligned_nuisance = torch.fft.irfft2(
                nuisance_spectrum * normalized_cross,
                s=(height, width),
                norm="ortho",
            )
            probability = torch.softmax(
                correlation.flatten(-2) * 8.0, dim=-1
            ).reshape_as(correlation)
            statistics_by_head = self._circular_statistics(probability)
            statistics = statistics_by_head.reshape(
                transmission.shape[0], self.heads * 6
            )
        local = torch.cat(
            (
                transmission_projection,
                aligned_nuisance,
                (transmission_projection - aligned_nuisance).abs(),
            ),
            dim=1,
        )
        context = self.local_projection(
            local.to(dtype=transmission.dtype)
        )
        global_context = self.global_projection(statistics).to(
            dtype=transmission.dtype
        )[:, :, None, None]
        context = context + global_context
        correlation_map = correlation.mean(dim=1, keepdim=True).to(
            dtype=transmission.dtype
        )
        flat_peak = correlation.flatten(-2).argmax(dim=-1)
        shift_y = torch.div(flat_peak, width, rounding_mode="floor")
        shift_x = flat_peak % width
        shift_y = torch.where(shift_y > height // 2, shift_y - height, shift_y)
        shift_x = torch.where(shift_x > width // 2, shift_x - width, shift_x)
        diagnostics = {
            "phase_peak_mean": correlation.detach().flatten(-2).amax(dim=-1).mean(),
            "phase_entropy_mean": statistics_by_head.detach()[..., 5].mean(),
            "phase_shift_y_abs_mean": shift_y.detach().float().abs().mean(),
            "phase_shift_x_abs_mean": shift_x.detach().float().abs().mean(),
            "phase_shift_y_signed_mean": shift_y.detach().float().mean(),
            "phase_shift_x_signed_mean": shift_x.detach().float().mean(),
            "phase_shift_y_std": shift_y.detach().float().std(unbiased=False),
            "phase_shift_x_std": shift_x.detach().float().std(unbiased=False),
            "phase_context_rms": context.detach().float().square().mean().sqrt(),
        }
        return context, correlation_map, diagnostics


class PhaseAwareTransformerBlock(nn.Module):
    """Restormer-style content attention with bounded phase-context injection."""

    def __init__(
        self,
        channels: int,
        *,
        heads: int,
        ffn_expansion: float = 2.0,
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("Transformer channels must be divisible by heads")
        hidden = int(channels * ffn_expansion)
        self.channels = channels
        self.heads = heads
        self.norm1 = LayerNorm2d(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.qkv_depthwise = nn.Conv2d(
            channels * 3,
            channels * 3,
            3,
            padding=1,
            groups=channels * 3,
        )
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.project = nn.Conv2d(channels, channels, 1)
        self.attention_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), 0.1)
        )
        self.phase_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), 0.1)
        )

        self.norm2 = LayerNorm2d(channels)
        self.ffn_in = nn.Conv2d(channels, hidden * 2, 1)
        self.ffn_depthwise = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            3,
            padding=1,
            groups=hidden * 2,
        )
        self.ffn_out = nn.Conv2d(hidden, channels, 1)
        self.ffn_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), 0.1)
        )

    def forward(
        self, value: torch.Tensor, phase_context: torch.Tensor
    ) -> torch.Tensor:
        batch, channels, height, width = value.shape
        conditioned = value + self.phase_scale * phase_context
        qkv = self.qkv_depthwise(self.qkv(self.norm1(conditioned)))
        query, key, projected_value = qkv.chunk(3, dim=1)
        head_channels = channels // self.heads
        query = query.reshape(batch, self.heads, head_channels, height * width)
        key = key.reshape(batch, self.heads, head_channels, height * width)
        projected_value = projected_value.reshape(
            batch, self.heads, head_channels, height * width
        )
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        attention = (query @ key.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)
        attended = (attention @ projected_value).reshape(
            batch, channels, height, width
        )
        value = value + self.attention_scale * self.project(attended)
        first, second = self.ffn_depthwise(
            self.ffn_in(self.norm2(value))
        ).chunk(2, dim=1)
        feed_forward = self.ffn_out(F.gelu(first) * second)
        return value + self.ffn_scale * feed_forward


@dataclass(frozen=True)
class PASTRRConfig:
    width: int = 48
    encoder_blocks: tuple[int, int, int] = (1, 1, 4)
    bottleneck_naf_blocks: int = 2
    transformer_blocks: int = 10
    decoder_blocks: tuple[int, int, int] = (4, 1, 1)
    allocator_groups: int = 8
    allocator_window: int = 16
    allocator_stride: int = 8
    transformer_heads: int = 8
    output_residual_scale: float = 0.1


class PASTRR(nn.Module):
    """Complete B + M + N model; no router, critic, or frozen fallback."""

    def __init__(self, config: PASTRRConfig = PASTRRConfig()) -> None:
        super().__init__()
        if len(config.encoder_blocks) != 3 or len(config.decoder_blocks) != 3:
            raise ValueError("PAST-RR requires exactly three down/up stages")
        self.config = config
        width = config.width
        widths = (width, width * 2, width * 4, width * 8)
        if widths[2] % config.allocator_groups or widths[3] % config.allocator_groups:
            raise ValueError("Configured widths are incompatible with allocator groups")
        if widths[3] % config.transformer_heads:
            raise ValueError("Bottleneck width is incompatible with transformer heads")

        self.intro = nn.Conv2d(3, widths[0], 3, padding=1)
        self.encoders = nn.ModuleList(
            [
                naf_stack(widths[index], blocks)
                for index, blocks in enumerate(config.encoder_blocks)
            ]
        )
        self.downs = nn.ModuleList(
            [
                nn.Conv2d(widths[index], widths[index + 1], 2, stride=2)
                for index in range(3)
            ]
        )
        self.quarter_allocator = LocalComplexSourceAllocator(
            widths[2],
            groups=config.allocator_groups,
            window_size=config.allocator_window,
            stride=config.allocator_stride,
        )
        self.bottleneck_naf = naf_stack(
            widths[3], config.bottleneck_naf_blocks
        )
        self.eighth_allocator = LocalComplexSourceAllocator(
            widths[3],
            groups=config.allocator_groups,
            window_size=config.allocator_window,
            stride=config.allocator_stride,
        )
        self.phase_context = PhaseCorrelationContext(
            widths[3], heads=config.transformer_heads
        )
        self.transformers = nn.ModuleList(
            [
                PhaseAwareTransformerBlock(
                    widths[3], heads=config.transformer_heads
                )
                for _ in range(config.transformer_blocks)
            ]
        )
        self.ups = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(widths[index], widths[index - 1] * 4, 1),
                    nn.PixelShuffle(2),
                )
                for index in (3, 2, 1)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                naf_stack(widths[index], blocks)
                for index, blocks in zip((2, 1, 0), config.decoder_blocks)
            ]
        )
        self.transmission_head = nn.Conv2d(widths[0], 3, 3, padding=1)
        self.nuisance_head = nn.Conv2d(widths[0], 3, 3, padding=1)
        self.output_scale = nn.Parameter(
            torch.full((1, 3, 1, 1), config.output_residual_scale)
        )
        nn.init.normal_(self.transmission_head.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.nuisance_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.transmission_head.bias)
        nn.init.zeros_(self.nuisance_head.bias)

    def forward(self, image: torch.Tensor) -> dict[str, Any]:
        padded, padding = pad_to_multiple(image, 8)
        value = self.intro(padded)
        skip_zero = self.encoders[0](value)
        value = self.downs[0](skip_zero)
        skip_one = self.encoders[1](value)
        value = self.downs[1](skip_one)
        value = self.encoders[2](value)
        value, quarter_transmission, quarter_nuisance, quarter_diagnostics = (
            self.quarter_allocator(value)
        )
        skip_two = value
        value = self.downs[2](value)
        value = self.bottleneck_naf(value)
        value, eighth_transmission, eighth_nuisance, eighth_diagnostics = (
            self.eighth_allocator(value)
        )
        context, correlation_map, phase_diagnostics = self.phase_context(
            eighth_transmission, eighth_nuisance
        )
        for transformer in self.transformers:
            value = transformer(value, context)

        for up, decoder, skip in zip(
            self.ups,
            self.decoders,
            (skip_two, skip_one, skip_zero),
        ):
            value = decoder(up(value) + skip)

        correction = self.output_scale * self.transmission_head(value)
        transmission = torch.clamp(padded + correction, 0.0, 1.0)
        nuisance = self.nuisance_head(value)
        return {
            "transmission": crop_padding(transmission, padding),
            "nuisance": crop_padding(nuisance, padding),
            "correction": crop_padding(correction, padding),
            "allocation_quarter": crop_padding(
                F.interpolate(
                    quarter_diagnostics["map"],
                    size=padded.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ),
                padding,
            ),
            "allocation_eighth": crop_padding(
                F.interpolate(
                    eighth_diagnostics["map"],
                    size=padded.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ),
                padding,
            ),
            "phase_correlation": correlation_map,
            "diagnostics": {
                **{
                    f"quarter_{key}": value
                    for key, value in quarter_diagnostics.items()
                    if key != "map"
                },
                **{
                    f"eighth_{key}": value
                    for key, value in eighth_diagnostics.items()
                    if key != "map"
                },
                **phase_diagnostics,
            },
        }

    def specification(self) -> dict[str, Any]:
        return {
            "name": "PAST-RR",
            "full_hypothesis": "B+M+N",
            "config": asdict(self.config),
            "parameters": component_parameter_counts(self),
        }


def component_parameter_counts(model: PASTRR) -> dict[str, int]:
    modules = {
        "B_spatial": (
            model.intro,
            model.encoders,
            model.downs,
            model.bottleneck_naf,
            model.ups,
            model.decoders,
        ),
        "M_complex_allocation": (
            model.quarter_allocator,
            model.eighth_allocator,
        ),
        "N_phase_transformer": (
            model.phase_context,
            model.transformers,
        ),
        "output_heads": (
            model.transmission_head,
            model.nuisance_head,
        ),
    }
    counts = {
        name: sum(
            parameter.numel()
            for module in group
            for parameter in module.parameters()
        )
        for name, group in modules.items()
    }
    counts["output_scales"] = model.output_scale.numel()
    counts["total"] = sum(parameter.numel() for parameter in model.parameters())
    counts["trainable"] = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return counts


def build_past_rr(config: PASTRRConfig | None = None) -> PASTRR:
    return PASTRR(config or PASTRRConfig())


__all__ = [
    "LocalComplexSourceAllocator",
    "PASTRR",
    "PASTRRConfig",
    "PhaseAwareTransformerBlock",
    "PhaseCorrelationContext",
    "build_past_rr",
    "component_parameter_counts",
]
