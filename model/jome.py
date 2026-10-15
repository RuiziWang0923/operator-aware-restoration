from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import file_sha256
from .past import (
    LayerNorm2d,
    NAFBlock,
    PASTRR,
    PASTRRConfig,
    crop_padding,
    naf_stack,
    pad_to_multiple,
)


@dataclass(frozen=True)
class JOMERRConfig:
    """Configuration for the joint-output homogeneous MoE add-on."""

    evidence_channels: int = 18
    experts: int = 4
    shared_width: int = 64
    shared_half_width: int = 128
    shared_quarter_width: int = 192
    shared_full_blocks: int = 2
    shared_half_blocks: int = 2
    shared_quarter_blocks: int = 3
    shared_attention_blocks: int = 2
    shared_attention_heads: int = 4
    expert_blocks: int = 2
    residual_cap: float = 0.10
    router_width: int = 40
    router_blocks: int = 2
    router_attention_blocks: int = 1
    router_attention_heads: int = 4
    dense_through_epoch: int = 8
    initial_temperature: float = 1.0
    final_temperature: float = 0.5
    target_epoch: int = 80


class ChannelContextBlock(nn.Module):
    """Linear-memory channel attention at a coarse spatial scale."""

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
            channels * 3, channels * 3, 3, padding=1, groups=channels * 3
        )
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.project = nn.Conv2d(channels, channels, 1)
        self.attention_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.norm2 = LayerNorm2d(channels)
        self.ffn_in = nn.Conv2d(channels, hidden * 2, 1)
        self.ffn_depthwise = nn.Conv2d(
            hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2
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
        raise TypeError("Expert branch must end in a convolution")
    nn.init.zeros_(convolution.weight)
    nn.init.zeros_(convolution.bias)


class HomogeneousExpertRefiner(nn.Module):
    """Shared encoder-decoder followed by exchangeable residual experts."""

    def __init__(self, config: JOMERRConfig) -> None:
        super().__init__()
        full = config.shared_width
        half = config.shared_half_width
        quarter = config.shared_quarter_width
        if quarter % config.shared_attention_heads:
            raise ValueError("Shared quarter width must be divisible by attention heads")
        if config.experts < 2:
            raise ValueError("Joint-output MoE requires at least two experts")
        self.config = config
        self.stem = nn.Conv2d(config.evidence_channels, full, 3, padding=1)
        self.full_encoder = naf_stack(full, config.shared_full_blocks)
        self.down_half = nn.Conv2d(full, half, 2, stride=2)
        self.half_encoder = naf_stack(half, config.shared_half_blocks)
        self.down_quarter = nn.Conv2d(half, quarter, 2, stride=2)
        self.quarter_encoder = naf_stack(quarter, config.shared_quarter_blocks)
        self.context = nn.ModuleList(
            ChannelContextBlock(quarter, heads=config.shared_attention_heads)
            for _ in range(config.shared_attention_blocks)
        )
        self.up_half = nn.Sequential(
            nn.Conv2d(quarter, half * 4, 1), nn.PixelShuffle(2)
        )
        self.half_decoder = naf_stack(half, config.shared_half_blocks)
        self.up_full = nn.Sequential(
            nn.Conv2d(half, full * 4, 1), nn.PixelShuffle(2)
        )
        self.full_decoder = naf_stack(full, config.shared_full_blocks)
        self.expert_branches = nn.ModuleList(
            nn.Sequential(
                *[NAFBlock(full) for _ in range(config.expert_blocks)],
                nn.Conv2d(full, 3, 3, padding=1),
            )
            for _ in range(config.experts)
        )
        for branch in self.expert_branches:
            _zero_last_convolution(branch)

    def forward(self, evidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padded, padding = pad_to_multiple(evidence, 4)
        full_skip = self.full_encoder(self.stem(padded))
        half_skip = self.half_encoder(self.down_half(full_skip))
        value = self.quarter_encoder(self.down_quarter(half_skip))
        for block in self.context:
            value = block(value)
        value = self.half_decoder(self.up_half(value) + half_skip)
        value = self.full_decoder(self.up_full(value) + full_skip)
        residuals = torch.stack(
            [
                self.config.residual_cap * torch.tanh(branch(value))
                for branch in self.expert_branches
            ],
            dim=1,
        )
        return crop_padding(residuals, padding), crop_padding(value, padding)


class SpatialOutcomeRouter(nn.Module):
    """Image-conditioned dense/soft-top-2 router with no semantic supervision."""

    def __init__(self, config: JOMERRConfig) -> None:
        super().__init__()
        width = config.router_width
        channels = width * 2
        if channels % config.router_attention_heads:
            raise ValueError("Router channels must be divisible by attention heads")
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv2d(config.shared_width + config.experts, width, 3, stride=2, padding=1),
            nn.GELU(),
            NAFBlock(width),
            nn.Conv2d(width, channels, 3, stride=2, padding=1),
            nn.GELU(),
            NAFBlock(channels),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[NAFBlock(channels) for _ in range(config.router_blocks)]
        )
        self.attention = nn.ModuleList(
            ChannelContextBlock(channels, heads=config.router_attention_heads)
            for _ in range(config.router_attention_blocks)
        )
        self.logit_head = nn.Conv2d(channels, config.experts, 1)
        nn.init.zeros_(self.logit_head.weight)
        nn.init.zeros_(self.logit_head.bias)
        self.register_buffer(
            "routing_epoch", torch.zeros((), dtype=torch.int64), persistent=True
        )

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("Routing epoch cannot be negative")
        self.routing_epoch.fill_(int(epoch))

    def _temperature(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        progress = min(
            max(float(self.routing_epoch.item()) / max(self.config.target_epoch, 1), 0.0),
            1.0,
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        value = self.config.final_temperature + cosine * (
            self.config.initial_temperature - self.config.final_temperature
        )
        return torch.tensor(value, dtype=dtype, device=device)

    def forward(
        self, features: torch.Tensor, residuals: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # Absolute magnitude has a finite subgradient at the exact-zero expert
        # initialization; sqrt(mean(square(.))) does not.
        magnitudes = residuals.float().abs().mean(dim=2).to(features.dtype)
        value = self.stem(torch.cat((features, magnitudes), dim=1))
        value = self.blocks(value)
        for block in self.attention:
            value = block(value)
        logits = self.logit_head(value)
        temperature = self._temperature(logits.dtype, logits.device)
        dense_weights = (logits / temperature).softmax(dim=1)
        top2_active = int(self.routing_epoch.item()) > self.config.dense_through_epoch
        if top2_active:
            indices = dense_weights.topk(2, dim=1).indices
            mask = torch.zeros_like(dense_weights).scatter_(1, indices, 1.0)
            weights = dense_weights * mask
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            weights = dense_weights
        return {
            "weights": weights,
            "dense_weights": dense_weights,
            "logits": logits,
            "temperature": temperature,
            "top2_active": torch.tensor(
                float(top2_active), dtype=logits.dtype, device=logits.device
            ),
        }


class JOMERR(nn.Module):
    """Frozen PAST-RR anchor plus a joint-output homogeneous spatial MoE."""

    def __init__(
        self,
        anchor: PASTRR,
        config: JOMERRConfig = JOMERRConfig(),
        *,
        anchor_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.anchor = anchor
        self.refiner = HomogeneousExpertRefiner(config)
        self.router = SpatialOutcomeRouter(config)
        self.anchor_metadata = anchor_metadata or {}
        self._freeze_anchor()

    def _freeze_anchor(self) -> None:
        self.anchor.eval()
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> JOMERR:
        super().train(mode)
        self.anchor.eval()
        return self

    def set_training_phase(self, phase: str) -> None:
        if phase not in {"joint_dense", "joint_top2", "evaluation"}:
            raise ValueError(f"Unknown JOME-RR phase: {phase}")
        enabled = phase != "evaluation"
        for module in (self.refiner, self.router):
            for parameter in module.parameters():
                parameter.requires_grad_(enabled)
        self._freeze_anchor()

    def set_routing_epoch(self, epoch: int) -> None:
        self.router.set_epoch(epoch)

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
        residuals, features = self.refiner(evidence)
        routing = self.router(features, residuals)
        coarse_weights = routing["weights"]
        weights = F.interpolate(
            coarse_weights,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused_residual = (weights[:, :, None] * residuals).sum(dim=1)
        transmission = torch.clamp(anchor + fused_residual, 0.0, 1.0)
        candidates = torch.clamp(anchor[:, None] + residuals, 0.0, 1.0)
        correction = transmission - image
        nuisance = anchor_result["nuisance"].detach() - fused_residual
        entropy = -(
            coarse_weights.clamp_min(1e-8).log() * coarse_weights
        ).sum(dim=1).mean()
        mean_weights = coarse_weights.mean(dim=(0, 2, 3))
        spatial_std = coarse_weights.flatten(2).std(dim=2, unbiased=False).mean()
        diagnostics: dict[str, torch.Tensor] = {
            **anchor_result["diagnostics"],
            **{
                f"jome_expert_{index}_weight_mean": coarse_weights[:, index].detach().mean()
                for index in range(self.config.experts)
            },
            **{
                f"jome_expert_{index}_top1_fraction": (
                    coarse_weights.argmax(dim=1) == index
                ).detach().float().mean()
                for index in range(self.config.experts)
            },
            "jome_router_entropy_mean": entropy.detach(),
            "jome_router_spatial_std_mean": spatial_std.detach(),
            "jome_load_imbalance": (
                self.config.experts * mean_weights.square().sum() - 1.0
            ).detach(),
            "jome_temperature": routing["temperature"].detach(),
            "jome_top2_active": routing["top2_active"].detach(),
            "jome_residual_rms": residuals.detach().float().square().mean().sqrt(),
            "jome_fused_residual_rms": fused_residual.detach().float().square().mean().sqrt(),
        }
        return {
            "transmission": transmission,
            "anchor_transmission": anchor,
            "candidate_transmissions": candidates,
            "expert_residuals": residuals,
            "fused_residual": fused_residual,
            "router_weights": coarse_weights,
            "router_dense_weights": routing["dense_weights"],
            "router_logits": routing["logits"],
            "nuisance": nuisance,
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
            "router": self.router.state_dict(),
        }

    def load_addon_state_dict(
        self, state: dict[str, dict[str, torch.Tensor]]
    ) -> None:
        self.refiner.load_state_dict(state["refiner"], strict=True)
        self.router.load_state_dict(state["router"], strict=True)

    def specification(self) -> dict[str, Any]:
        return {
            "name": "JOME-RR",
            "full_hypothesis": (
                "frozen B+M+N anchor + four exchangeable residual experts + "
                "end-result-driven spatial router"
            ),
            "mechanism": (
                "joint final-output optimization with no expert semantics, no "
                "per-expert targets, an explicit anchor skip, and soft top-2 routing"
            ),
            "config": asdict(self.config),
            "anchor": self.anchor_metadata,
            "parameters": component_parameter_counts(self),
        }


def component_parameter_counts(model: JOMERR) -> dict[str, int]:
    anchor = sum(parameter.numel() for parameter in model.anchor.parameters())
    refiner = sum(parameter.numel() for parameter in model.refiner.parameters())
    router = sum(parameter.numel() for parameter in model.router.parameters())
    return {
        "frozen_anchor": anchor,
        "shared_encoder_decoder_and_homogeneous_experts": refiner,
        "spatial_outcome_router": router,
        "addon": refiner + router,
        "total": anchor + refiner + router,
        "trainable": refiner + router,
        "active_trainable_in_current_phase": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def config_from_dict(values: dict[str, Any]) -> JOMERRConfig:
    return JOMERRConfig(**values)


def build_jome_from_checkpoint(
    checkpoint_path: str | Path,
    config: JOMERRConfig | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> JOMERR:
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frozen anchor checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    specification = payload.get("model_specification", {})
    if specification.get("name") != "PAST-RR":
        raise RuntimeError("JOME-RR requires an evaluated PAST-RR anchor checkpoint")
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
    model = JOMERR(anchor, config or JOMERRConfig(), anchor_metadata=anchor_metadata)
    return model.to(map_location)


__all__ = [
    "ChannelContextBlock",
    "HomogeneousExpertRefiner",
    "JOMERR",
    "JOMERRConfig",
    "SpatialOutcomeRouter",
    "build_jome_from_checkpoint",
    "component_parameter_counts",
    "config_from_dict",
]
