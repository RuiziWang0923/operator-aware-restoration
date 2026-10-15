from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from benefitsirr.utils import file_sha256
from revision_v2.past_rr.model import (
    LayerNorm2d,
    NAFBlock,
    PASTRR,
    PASTRRConfig,
    crop_padding,
    naf_stack,
    pad_to_multiple,
)


EXPERT_NAMES = ("removal", "detail", "photometric")


@dataclass(frozen=True)
class ARISERRConfig:
    """Configuration for the trainable add-on around a frozen PAST-RR anchor."""

    evidence_channels: int = 18
    refiner_width: int = 64
    refiner_half_width: int = 128
    refiner_quarter_width: int = 192
    refiner_full_blocks: int = 2
    refiner_half_blocks: int = 2
    refiner_quarter_blocks: int = 3
    refiner_attention_blocks: int = 2
    refiner_attention_heads: int = 4
    removal_residual_cap: float = 0.12
    detail_residual_cap: float = 0.08
    photometric_residual_cap: float = 0.06
    assessor_width: int = 48
    assessor_blocks: int = 2
    assessor_attention_blocks: int = 2
    assessor_attention_heads: int = 4
    conservative_kappa: float = 0.50
    initial_temperature: float = 0.75
    minimum_temperature: float = 0.25
    maximum_temperature: float = 2.00


class ChannelContextBlock(nn.Module):
    """Linear-memory Restormer-style attention at a coarse spatial scale."""

    def __init__(self, channels: int, *, heads: int, expansion: float = 2.0) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("Attention channels must be divisible by heads")
        hidden = int(channels * expansion)
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
        self.attention_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
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
        self.ffn_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        query, key, projected = self.qkv_depthwise(
            self.qkv(self.norm1(value))
        ).chunk(3, dim=1)
        head_channels = channels // self.heads
        query = query.reshape(batch, self.heads, head_channels, height * width)
        key = key.reshape(batch, self.heads, head_channels, height * width)
        projected = projected.reshape(
            batch, self.heads, head_channels, height * width
        )
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        attention = (query @ key.transpose(-2, -1)) * self.temperature
        attended = (attention.softmax(dim=-1) @ projected).reshape(
            batch, channels, height, width
        )
        value = value + self.attention_scale * self.project(attended)
        first, second = self.ffn_depthwise(
            self.ffn_in(self.norm2(value))
        ).chunk(2, dim=1)
        return value + self.ffn_scale * self.ffn_out(F.gelu(first) * second)


def _zero_last_convolution(module: nn.Sequential) -> None:
    convolution = module[-1]
    if not isinstance(convolution, nn.Conv2d):
        raise TypeError("Expert head must end in a convolution")
    nn.init.zeros_(convolution.weight)
    nn.init.zeros_(convolution.bias)


class ResidualExpertRefiner(nn.Module):
    """Shared encoder-decoder with three deliberately specialized residual heads."""

    def __init__(self, config: ARISERRConfig) -> None:
        super().__init__()
        full = config.refiner_width
        half = config.refiner_half_width
        quarter = config.refiner_quarter_width
        if quarter % config.refiner_attention_heads:
            raise ValueError("Refiner quarter width must be divisible by heads")
        self.config = config
        self.stem = nn.Conv2d(config.evidence_channels, full, 3, padding=1)
        self.full_encoder = naf_stack(full, config.refiner_full_blocks)
        self.down_half = nn.Conv2d(full, half, 2, stride=2)
        self.half_encoder = naf_stack(half, config.refiner_half_blocks)
        self.down_quarter = nn.Conv2d(half, quarter, 2, stride=2)
        self.quarter_encoder = naf_stack(quarter, config.refiner_quarter_blocks)
        self.context = nn.ModuleList(
            ChannelContextBlock(
                quarter, heads=config.refiner_attention_heads
            )
            for _ in range(config.refiner_attention_blocks)
        )
        self.up_half = nn.Sequential(
            nn.Conv2d(quarter, half * 4, 1), nn.PixelShuffle(2)
        )
        self.half_decoder = naf_stack(half, config.refiner_half_blocks)
        self.up_full = nn.Sequential(
            nn.Conv2d(half, full * 4, 1), nn.PixelShuffle(2)
        )
        self.full_decoder = naf_stack(full, config.refiner_full_blocks)

        self.detail_adapter = nn.Conv2d(6, full, 3, padding=1)
        self.removal_head = nn.Sequential(NAFBlock(full), nn.Conv2d(full, 3, 3, padding=1))
        self.detail_head = nn.Sequential(NAFBlock(full), nn.Conv2d(full, 3, 3, padding=1))
        self.photometric_head = nn.Sequential(
            nn.Conv2d(full, full, 5, padding=2, groups=full),
            nn.GELU(),
            nn.Conv2d(full, 3, 3, padding=1),
        )
        for head in (self.removal_head, self.detail_head, self.photometric_head):
            _zero_last_convolution(head)

    @staticmethod
    def _highpass(value: torch.Tensor) -> torch.Tensor:
        return value - F.avg_pool2d(value, 5, stride=1, padding=2)

    def forward(
        self,
        evidence: torch.Tensor,
        *,
        image: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        padded, padding = pad_to_multiple(evidence, 4)
        pad_height, pad_width = padding
        image_padded = F.pad(
            image,
            (0, pad_width, 0, pad_height),
            mode="reflect" if image.shape[-2] > pad_height and image.shape[-1] > pad_width else "replicate",
        )
        anchor_padded = F.pad(
            anchor,
            (0, pad_width, 0, pad_height),
            mode="reflect" if anchor.shape[-2] > pad_height and anchor.shape[-1] > pad_width else "replicate",
        )

        full_skip = self.full_encoder(self.stem(padded))
        half_skip = self.half_encoder(self.down_half(full_skip))
        value = self.quarter_encoder(self.down_quarter(half_skip))
        for block in self.context:
            value = block(value)
        value = self.half_decoder(self.up_half(value) + half_skip)
        value = self.full_decoder(self.up_full(value) + full_skip)

        removal = self.config.removal_residual_cap * torch.tanh(
            self.removal_head(value)
        )
        detail_evidence = torch.cat(
            (self._highpass(image_padded), self._highpass(anchor_padded)), dim=1
        )
        detail = self.config.detail_residual_cap * torch.tanh(
            self.detail_head(value + self.detail_adapter(detail_evidence))
        )
        smooth_value = F.avg_pool2d(value, 9, stride=1, padding=4)
        photometric = self.config.photometric_residual_cap * torch.tanh(
            self.photometric_head(smooth_value)
        )
        return torch.stack(
            tuple(crop_padding(item, padding) for item in (removal, detail, photometric)),
            dim=1,
        )


class CounterfactualRiskAssessor(nn.Module):
    """Predict per-patch expert utility and uncertainty relative to identity."""

    def __init__(self, config: ARISERRConfig) -> None:
        super().__init__()
        width = config.assessor_width
        input_channels = config.evidence_channels + len(EXPERT_NAMES)
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, 3, stride=2, padding=1),
            nn.GELU(),
            NAFBlock(width),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.GELU(),
            NAFBlock(width * 2),
            nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1),
            nn.GELU(),
        )
        channels = width * 2
        self.blocks = nn.Sequential(*[NAFBlock(channels) for _ in range(config.assessor_blocks)])
        self.attention = nn.ModuleList(
            ChannelContextBlock(channels, heads=config.assessor_attention_heads)
            for _ in range(config.assessor_attention_blocks)
        )
        self.utility_head = nn.Conv2d(channels, len(EXPERT_NAMES), 1)
        self.log_scale_head = nn.Conv2d(channels, len(EXPERT_NAMES), 1)
        self.risk_head = nn.Conv2d(channels, 1, 1)
        self.identity_bias = nn.Parameter(torch.tensor(0.0))
        self.risk_coupling = nn.Parameter(torch.tensor(0.0))
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(config.initial_temperature))
        )
        nn.init.zeros_(self.utility_head.weight)
        nn.init.zeros_(self.utility_head.bias)
        nn.init.zeros_(self.log_scale_head.weight)
        nn.init.constant_(self.log_scale_head.bias, -1.5)
        nn.init.zeros_(self.risk_head.weight)
        nn.init.zeros_(self.risk_head.bias)

    def forward(
        self, evidence: torch.Tensor, residuals: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        magnitudes = residuals.detach().abs().mean(dim=2)
        value = self.stem(torch.cat((evidence, magnitudes), dim=1))
        value = self.blocks(value)
        for block in self.attention:
            value = block(value)
        utility_mean = self.utility_head(value)
        utility_scale = F.softplus(self.log_scale_head(value)) + 1e-4
        anchor_risk = self.risk_head(value)
        minimum = math.log(self.config.minimum_temperature)
        maximum = math.log(self.config.maximum_temperature)
        temperature = self.log_temperature.clamp(minimum, maximum).exp()
        conservative = (
            utility_mean
            - self.config.conservative_kappa * utility_scale
            + self.risk_coupling * torch.tanh(anchor_risk)
        ) / temperature
        identity = self.identity_bias.expand(
            conservative.shape[0], 1, *conservative.shape[-2:]
        )
        logits = torch.cat((identity, conservative), dim=1)
        weights = logits.softmax(dim=1)
        return {
            "weights": weights,
            "logits": logits,
            "utility_mean": utility_mean,
            "utility_scale": utility_scale,
            "anchor_risk": anchor_risk,
            "temperature": temperature,
        }


class ARISERR(nn.Module):
    """Frozen strong anchor plus an identity-safe spatial mixture of experts."""

    def __init__(
        self,
        anchor: PASTRR,
        config: ARISERRConfig = ARISERRConfig(),
        *,
        anchor_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.anchor = anchor
        self.refiner = ResidualExpertRefiner(config)
        self.assessor = CounterfactualRiskAssessor(config)
        self.anchor_metadata = anchor_metadata or {}
        self._freeze_anchor()

    def _freeze_anchor(self) -> None:
        self.anchor.eval()
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> ARISERR:
        super().train(mode)
        self.anchor.eval()
        return self

    def set_training_phase(self, phase: str) -> None:
        aliases = {
            "risk": (False, True),
            "enhancer": (True, False),
            "router": (False, True),
            "alternate_enhancer": (True, False),
            "alternate_router": (False, True),
            "evaluation": (False, False),
        }
        if phase not in aliases:
            raise ValueError(f"Unknown ARISE-RR phase: {phase}")
        refiner_enabled, assessor_enabled = aliases[phase]
        for parameter in self.refiner.parameters():
            parameter.requires_grad_(refiner_enabled)
        for parameter in self.assessor.parameters():
            parameter.requires_grad_(assessor_enabled)
        self._freeze_anchor()

    @staticmethod
    def _anchor_evidence(image: torch.Tensor, result: dict[str, Any]) -> torch.Tensor:
        anchor = result["transmission"].detach()
        nuisance = result["nuisance"].detach()
        correction = anchor - image
        quarter = result["allocation_quarter"].detach()
        eighth = result["allocation_eighth"].detach()
        phase = F.interpolate(
            result["phase_correlation"].detach(),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return torch.cat(
            (
                image,
                anchor,
                nuisance,
                correction,
                correction.abs(),
                quarter,
                eighth,
                phase,
            ),
            dim=1,
        )

    def forward(self, image: torch.Tensor) -> dict[str, Any]:
        with torch.no_grad():
            anchor_result = self.anchor(image)
        anchor = anchor_result["transmission"].detach()
        evidence = self._anchor_evidence(image, anchor_result)
        residuals = self.refiner(evidence, image=image, anchor=anchor)
        candidates = torch.clamp(anchor[:, None] + residuals, 0.0, 1.0)
        assessor = self.assessor(evidence, residuals)
        coarse_weights = assessor["weights"]
        weights = F.interpolate(
            coarse_weights,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        all_candidates = torch.cat((anchor[:, None], candidates), dim=1)
        transmission = (weights[:, :, None] * all_candidates).sum(dim=1)
        correction = transmission - image
        entropy = -(
            coarse_weights.clamp_min(1e-8).log() * coarse_weights
        ).sum(dim=1).mean()
        diagnostics = {
            **anchor_result["diagnostics"],
            "arise_identity_weight_mean": coarse_weights[:, 0].detach().mean(),
            "arise_removal_weight_mean": coarse_weights[:, 1].detach().mean(),
            "arise_detail_weight_mean": coarse_weights[:, 2].detach().mean(),
            "arise_photometric_weight_mean": coarse_weights[:, 3].detach().mean(),
            "arise_router_entropy_mean": entropy.detach(),
            "arise_utility_mean": assessor["utility_mean"].detach().mean(),
            "arise_uncertainty_mean": assessor["utility_scale"].detach().mean(),
            "arise_anchor_risk_mean": assessor["anchor_risk"].detach().mean(),
            "arise_temperature": assessor["temperature"].detach(),
            "arise_residual_rms": residuals.detach().float().square().mean().sqrt(),
        }
        return {
            "transmission": transmission,
            "anchor_transmission": anchor,
            "candidate_transmissions": candidates,
            "expert_residuals": residuals,
            "router_weights": coarse_weights,
            "router_logits": assessor["logits"],
            "utility_mean": assessor["utility_mean"],
            "utility_scale": assessor["utility_scale"],
            "anchor_risk": assessor["anchor_risk"],
            "nuisance": anchor_result["nuisance"].detach(),
            "correction": correction,
            "allocation_quarter": anchor_result["allocation_quarter"].detach(),
            "allocation_eighth": anchor_result["allocation_eighth"].detach(),
            "phase_correlation": anchor_result["phase_correlation"].detach(),
            "diagnostics": diagnostics,
        }

    @torch.no_grad()
    def evaluation_diagnostics(
        self,
        *,
        result: dict[str, Any],
        image: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        from .oracle import oracle_diagnostics

        return oracle_diagnostics(result=result, target=target)

    def addon_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "refiner": self.refiner.state_dict(),
            "assessor": self.assessor.state_dict(),
        }

    def load_addon_state_dict(
        self, state: dict[str, dict[str, torch.Tensor]]
    ) -> None:
        self.refiner.load_state_dict(state["refiner"], strict=True)
        self.assessor.load_state_dict(state["assessor"], strict=True)

    def specification(self) -> dict[str, Any]:
        return {
            "name": "ARISE-RR",
            "full_hypothesis": "frozen B+M+N anchor + identity + three residual experts + risk assessor",
            "mechanism": (
                "anchor-relative counterfactual utility routing with uncertainty-"
                "penalized spatial convex mixing and an exact identity action"
            ),
            "config": asdict(self.config),
            "anchor": self.anchor_metadata,
            "parameters": component_parameter_counts(self),
        }


def component_parameter_counts(model: ARISERR) -> dict[str, int]:
    anchor = sum(parameter.numel() for parameter in model.anchor.parameters())
    refiner = sum(parameter.numel() for parameter in model.refiner.parameters())
    assessor = sum(parameter.numel() for parameter in model.assessor.parameters())
    return {
        "frozen_anchor": anchor,
        "residual_experts_shared_refiner": refiner,
        "counterfactual_risk_assessor": assessor,
        "addon": refiner + assessor,
        "total": anchor + refiner + assessor,
        "trainable": refiner + assessor,
        "active_trainable_in_current_phase": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def config_from_dict(values: dict[str, Any]) -> ARISERRConfig:
    return ARISERRConfig(**values)


def build_arise_from_checkpoint(
    checkpoint_path: str | Path,
    config: ARISERRConfig | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> ARISERR:
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frozen anchor checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    specification = payload.get("model_specification", {})
    if specification.get("name") != "PAST-RR":
        raise RuntimeError("ARISE-RR requires an evaluated PAST-RR anchor checkpoint")
    anchor_values = dict(specification["config"])
    anchor_values["encoder_blocks"] = tuple(anchor_values["encoder_blocks"])
    anchor_values["decoder_blocks"] = tuple(anchor_values["decoder_blocks"])
    anchor = PASTRR(PASTRRConfig(**anchor_values))
    anchor.load_state_dict(payload["model"], strict=True)
    anchor_metadata = {
        "checkpoint": str(path),
        "sha256": file_sha256(path),
        "epoch": int(payload["epoch"]),
        "model_select_psnr": float(payload["model_select_psnr"]),
        "identity_sha256": payload.get("identity_sha256"),
        "specification": specification,
    }
    model = ARISERR(anchor, config or ARISERRConfig(), anchor_metadata=anchor_metadata)
    return model.to(map_location)


__all__ = [
    "ARISERR",
    "ARISERRConfig",
    "ChannelContextBlock",
    "CounterfactualRiskAssessor",
    "EXPERT_NAMES",
    "ResidualExpertRefiner",
    "build_arise_from_checkpoint",
    "component_parameter_counts",
    "config_from_dict",
]
