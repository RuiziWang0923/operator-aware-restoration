from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def parameter_counts(module: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in module.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in module.parameters() if parameter.requires_grad
        ),
    }


def module_parameter_counts(modules: dict[str, torch.nn.Module]) -> dict[str, int]:
    return {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, module in modules.items()
    }


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(destination)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        record = _json_safe({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        })
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        atomic_json_dump(record, self.path.parent / "LATEST_TRAINING_RECORD.json")


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def git_revision(path: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def hardware_report() -> dict[str, Any]:
    package_versions = {}
    for package in ("numpy", "Pillow", "PyYAML", "tensorboard", "tqdm"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = None
    report: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "logical_cpu_count": os.cpu_count(),
        "torch_cpu_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "packages": package_versions,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        report.update(
            {
                "gpu": properties.name,
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "compiled_cuda_arches": torch.cuda.get_arch_list(),
                "vram_gib": properties.total_memory / (1024**3),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "cudnn": torch.backends.cudnn.version(),
            }
        )
    return report


def nvidia_smi_snapshot() -> dict[str, float]:
    """Return a best-effort telemetry sample without making training depend on it."""
    fields = [
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "clocks.current.sm",
    ]
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        values = [float(item.strip()) for item in result.stdout.splitlines()[0].split(",")]
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
        IndexError,
    ):
        return {}
    names = [
        "gpu_temperature_c",
        "gpu_power_draw_w",
        "gpu_power_limit_w",
        "gpu_utilization_percent",
        "gpu_memory_utilization_percent",
        "gpu_memory_used_mib",
        "gpu_memory_total_mib",
        "gpu_sm_clock_mhz",
    ]
    return dict(zip(names, values))


def file_sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def format_seconds(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def flatten_finite(values: Iterable[torch.Tensor]) -> torch.Tensor:
    flattened = [value.detach().float().reshape(-1).cpu() for value in values]
    if not flattened:
        return torch.empty(0)
    combined = torch.cat(flattened)
    return combined[torch.isfinite(combined)]
