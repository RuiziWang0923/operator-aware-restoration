"""Historical local SSIM, extracted verbatim from t0_nafnet/epoch_evaluation.py.

The four-pixel crop and FP32 valid Gaussian filtering preserve the original
local implementation. Its documented difference from challenge SSIM remains.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _metric_view(tensor: torch.Tensor, border: int) -> torch.Tensor:
    if border < 0:
        raise ValueError("Border must be non-negative")
    if border and min(tensor.shape[-2:]) <= 2 * border:
        raise ValueError("Image too small for metric border")
    return tensor[..., border:-border, border:-border] if border else tensor


def _gaussian_kernel_1d(
    *, device: torch.device, dtype: torch.dtype, size: int = 11, sigma: float = 1.5
) -> torch.Tensor:
    coordinate = torch.arange(size, device=device, dtype=dtype) - size // 2
    kernel = torch.exp(-(coordinate.square()) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _gaussian_filter_rgb(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("SSIM expects a BCHW RGB tensor")
    kernel = _gaussian_kernel_1d(device=image.device, dtype=image.dtype)
    horizontal = kernel.view(1, 1, 1, -1).repeat(3, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(3, 1, 1, 1)
    filtered = F.conv2d(image, horizontal, groups=3)
    return F.conv2d(filtered, vertical, groups=3)


def ssim_value(
    prediction: torch.Tensor, target: torch.Tensor, *, border: int = 4
) -> float:
    """RGB SSIM matching the standard 11x11, sigma=1.5 implementation.

    Inputs are in [0, 1]. The official OpenRR evaluator crops four pixels before
    calling pyiqa with test_y_channel=False; Stage 3D validates this independent
    implementation against every released team output before it is reportable.
    """

    prediction = _metric_view(prediction, border).float()
    target = _metric_view(target, border).float()
    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(0)
    if target.ndim == 3:
        target = target.unsqueeze(0)
    if prediction.shape != target.shape:
        raise ValueError("SSIM inputs must have the same shape")
    if prediction.shape[-2] < 11 or prediction.shape[-1] < 11:
        raise ValueError("SSIM inputs are smaller than the 11x11 window")
    mu_prediction = _gaussian_filter_rgb(prediction)
    mu_target = _gaussian_filter_rgb(target)
    mu_prediction_sq = mu_prediction.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_prediction * mu_target
    variance_prediction = (
        _gaussian_filter_rgb(prediction.square()) - mu_prediction_sq
    ).clamp_min(0.0)
    variance_target = (
        _gaussian_filter_rgb(target.square()) - mu_target_sq
    ).clamp_min(0.0)
    covariance = _gaussian_filter_rgb(prediction * target) - mu_product
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2.0 * mu_product + c1) * (2.0 * covariance + c2)) / (
        (mu_prediction_sq + mu_target_sq + c1)
        * (variance_prediction + variance_target + c2)
    )
    return float(score.mean().item())
