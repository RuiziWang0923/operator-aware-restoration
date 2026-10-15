#!/usr/bin/env python3
"""Frozen native-resolution E1/E2/E3 evaluation; no training or tensor archival.

Package layout: project/{revision_v2,revision_v5,src}, checkpoints/{base,jome}.pt,
manifest.csv with id,split_role,input_path,target_path relative to package root.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


CROP = 4
EPS = 0.1
GRAM_ORDER = ["expert_0", "expert_1", "expert_2", "expert_3", "base", "input"]
REPLAY_VECTORS = {
    "uniform4": [0.25, 0.25, 0.25, 0.25, 0, 0],
    "uniform6": [1 / 6] * 6,
    "prespecified_1234": [0.1, 0.2, 0.3, 0.4, 0, 0],
}
REPLAY_TOLERANCE = 1e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value: torch.Tensor) -> float:
    return float(value.item())


def mse(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return (first.double() - second.double()).square().mean()


def metric(value: float) -> dict:
    # JSON null denotes an infinite PSNR for exact zero MSE (strict JSON).
    return {"mse": value, "psnr": -10 * math.log10(value) if value > 0 else None}


def geometry(output: torch.Tensor, image: torch.Tensor, target: torch.Tensor) -> dict:
    target_delta = target.double() - image.double()
    output_delta = output.double() - image.double()
    return {"a": scalar(target_delta.square().mean()),
            "b": scalar(output_delta.square().mean()),
            "d": scalar((output_delta * target_delta).mean())}


def gram_and_replay(candidates: torch.Tensor, target: torch.Tensor) -> tuple[list, dict]:
    """Float64 Gram and independent direct output-mixture replay, crop RGB mean."""
    values = candidates.double().reshape(6, -1)
    truth = target.double().reshape(1, -1)
    errors = values - truth
    gram = errors @ errors.T / errors.shape[1]
    gram = (gram + gram.T) / 2
    replay = {"tolerance": REPLAY_TOLERANCE, "tests": {}}
    largest = 0.0
    for name, vector in REPLAY_VECTORS.items():
        alpha = torch.tensor(vector, dtype=torch.float64, device=values.device)
        direct = ((alpha @ values) - truth[0]).square().mean()
        reconstructed = alpha @ gram @ alpha
        error = scalar((direct - reconstructed).abs())
        largest = max(largest, error)
        replay["tests"][name] = {"weights": vector, "direct_mse": scalar(direct),
                                 "gram_mse": scalar(reconstructed), "absolute_error": error}
    replay["max_absolute_error"] = largest
    replay["passed"] = largest <= REPLAY_TOLERANCE
    if not replay["passed"]:
        raise RuntimeError(f"Gram replay failed: {largest} > {REPLAY_TOLERANCE}")
    return gram.cpu().tolist(), replay


def numerical_record(result: dict, image: torch.Tensor, target: torch.Tensor) -> dict:
    """All diagnostic quantities use the same four-pixel RGB crop."""
    crop = lambda value: value[..., CROP:-CROP, CROP:-CROP]
    base = crop(result["anchor_transmission"]).float()
    residuals = crop(result["expert_residuals"]).float()
    native = crop(result["transmission"]).float()
    image, target = crop(image).float(), crop(target).float()
    # Interpolate in the original router dtype, matching deployed forward exactly.
    full_weights = F.interpolate(result["router_weights"],
                  size=result["transmission"].shape[-2:], mode="bilinear",
                  align_corners=False).float()
    weights = crop(full_weights)
    preclip = base[:, None] + residuals
    experts = preclip.clamp(0, 1)
    ur = (base + residuals.mean(dim=1)).clamp(0, 1)
    uo = experts.mean(dim=1)
    residual_mixture = (weights[:, :, None] * residuals).sum(dim=1)
    yr = (base + residual_mixture).clamp(0, 1)
    yo = (weights[:, :, None] * experts).sum(dim=1)
    # E2 uses clipped output candidates; native candidate equality is audited.
    deployed_experts = crop(result["candidate_transmissions"]).float()
    candidates = torch.cat((deployed_experts[0], base, image), dim=0)
    gram, replay = gram_and_replay(candidates, target)
    metrics = {name: metric(gram[index][index]) for index, name in enumerate(GRAM_ORDER)}
    for name, value in {"jome": native, "Yr": yr, "Ur": ur, "Uo": uo, "Yo": yo}.items():
        metrics[name] = metric(scalar(mse(value, target)))

    displacement = (experts.double() - preclip.double()).abs()
    mean_displacement = displacement.mean(dim=1)
    uniform_difference = uo.double() - ur.double()
    # Coordinatewise |mean C(Ak)-C(mean Ak)| <= mean |C(Ak)-Ak|.
    # Taking RMS gives a valid norm upper bound. Mean absolute displacement
    # alone bounds mean absolute error, not RMS.
    uniform_rms = scalar(uniform_difference.square().mean().sqrt())
    uniform_bound = scalar(mean_displacement.square().mean().sqrt())
    uniform_mae = scalar(uniform_difference.abs().mean())
    candidate_mae = scalar(displacement.mean())
    # Float32 mean/clamp operations introduce a tiny numerical residual.
    clipping_tolerance = 2e-7
    clipping = {
        "candidate_out_of_range_fraction": scalar(((preclip < 0) | (preclip > 1)).double().mean()),
        "candidate_out_of_range_fraction_by_expert": [scalar(((preclip[:, k] < 0) | (preclip[:, k] > 1)).double().mean()) for k in range(4)],
        "mean_candidate_clipping_absolute": candidate_mae,
        "mean_candidate_clipping_absolute_by_expert": [scalar(displacement[:, k].mean()) for k in range(4)],
        "uniform_rms_difference": uniform_rms,
        "uniform_rms_bound": uniform_bound,
        "uniform_mean_absolute_difference": uniform_mae,
        "uniform_mean_absolute_bound": candidate_mae,
        "bound_tolerance": clipping_tolerance,
        "norm_bound_passed": uniform_rms <= uniform_bound + clipping_tolerance,
        "mean_absolute_bound_passed": uniform_mae <= candidate_mae + clipping_tolerance,
        "candidate_reconstruction_max_error": scalar((experts - deployed_experts).abs().max()),
        "router_weight_min": scalar(full_weights.min()),
        "router_weight_max": scalar(full_weights.max()),
        "router_sum_max_absolute_error": scalar((full_weights.sum(dim=1) - 1).abs().max()),
        "effective_residual_max_absolute": scalar(residual_mixture.abs().max()),
        "native_fused_residual_max_absolute": scalar(crop(result["fused_residual"]).float().abs().max()),
        "native_vs_float32_residual_reconstruction_rms": scalar(mse(native, yr).sqrt()),
        "native_vs_float32_residual_reconstruction_max_absolute": scalar((native - yr).abs().max()),
        "native_vs_float32_residual_reconstruction_psnr_difference": (metrics["jome"]["psnr"] - metrics["Yr"]["psnr"]) if metrics["jome"]["psnr"] is not None and metrics["Yr"]["psnr"] is not None else None,
        "mse_interaction": metrics["Yr"]["mse"] - metrics["Ur"]["mse"] - metrics["Yo"]["mse"] + metrics["Uo"]["mse"],
    }
    if not clipping["norm_bound_passed"] or not clipping["mean_absolute_bound_passed"]:
        raise RuntimeError("Clipping bound numerical verification failed")

    base_input_distance = (base.double() - image.double()).abs()
    native_correction = (native.double() - base.double()).abs()
    bound = scalar((base_input_distance - EPS).clamp_min(0).square().mean())
    jome_input_mse = scalar(mse(native, image))
    envelope_excess = max(0.0, scalar(native_correction.max()) - EPS)
    # Preserve the requested strict eps=.1 bound and explicitly account for
    # BF16 cap/weight rounding with an empirically audited coordinate envelope.
    effective_eps = EPS + envelope_excess
    effective_bound = scalar((base_input_distance - effective_eps).clamp_min(0).square().mean())
    reachability = {
        "eps": EPS, "bound": bound, "mse_jome_input": jome_input_mse,
        "affected_fraction": scalar((base_input_distance > EPS).double().mean()),
        "strict_bound_passed": jome_input_mse + REPLAY_TOLERANCE >= bound,
        "max_coordinate_envelope_excess": envelope_excess,
        "coordinate_envelope_violated_fraction": scalar((native_correction > EPS).double().mean()),
        "effective_eps_for_roundoff": effective_eps,
        "roundoff_adjusted_bound": effective_bound,
        "roundoff_adjusted_bound_passed": jome_input_mse + REPLAY_TOLERANCE >= effective_bound,
        "bound_tolerance": REPLAY_TOLERANCE,
    }
    if not reachability["roundoff_adjusted_bound_passed"]:
        raise RuntimeError("Reachability lower bound numerical verification failed")
    finite_actions = ["input", "base", "jome"]
    oracle_action = min(finite_actions, key=lambda name: metrics[name]["mse"])
    oracle = {"actions": finite_actions, "selected_action": oracle_action,
              **metrics[oracle_action]}
    return {"metrics": metrics, "gram": gram,
            "geometry": {"base": geometry(base, image, target),
                         "jome": geometry(native, image, target)},
            "reachability": reachability, "clipping": clipping,
            "replay": replay, "finite_action_oracle": oracle}


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as source:
        array = np.array(source.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32).div_(255)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def save_metadata(path: Path, values: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def existing_records(path: Path) -> list[dict]:
    """Recover a truncated final line after interruption; reject interior damage."""
    if not path.exists():
        return []
    records, offset = [], 0
    with path.open("rb") as stream:
        lines = stream.readlines()
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines) - 1:
                raise RuntimeError("Malformed interior JSONL record")
            with path.open("r+b") as stream:
                stream.truncate(offset)
            break
        records.append(record)
        offset += len(line)
    # A valid final JSON object without newline must be separated before append.
    if records and path.read_bytes()[-1:] != b"\n":
        with path.open("ab") as stream:
            stream.write(b"\n")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=8, help="Total manifest rows in this run; 0 selects all")
    parser.add_argument("--max-seconds", type=float, default=0, help="Wall budget including load/hash; 0 unlimited; stop between images")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    if args.limit < 0 or args.max_seconds < 0:
        raise ValueError("limit and max-seconds must be nonnegative")
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_path, metadata_path = output / "per_image.jsonl", output / "metadata.json"
    if not args.resume and (data_path.exists() or metadata_path.exists()):
        raise FileExistsError("Output exists; use --resume or a new output directory")
    manifest = root / "manifest.csv"
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"id", "split_role", "input_path", "target_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest requires {sorted(required)} and at least one row")
    identities = [(row["split_role"], row["id"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate (split_role,id) in manifest")
    selected = rows[:args.limit] if args.limit else rows
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required")
    # No compile, gradients, training, or TF32 metric matmuls.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    sys.path[:0] = [str(root / "project"), str(root / "project" / "src")]
    from revision_v2.past_rr.model import PASTRR, PASTRRConfig
    from revision_v5.joint_output_moe.model import JOMERR, JOMERRConfig
    checkpoint_paths = {"base": root / "checkpoints" / "base.pt",
                        "jome": root / "checkpoints" / "jome.pt"}
    hashes = {name: sha256(path) for name, path in checkpoint_paths.items()}
    previous = json.loads(metadata_path.read_text()) if args.resume and metadata_path.exists() else {}
    if previous and (previous.get("checkpoint_sha256") != hashes or previous.get("manifest_sha256") != sha256(manifest)):
        raise RuntimeError("Resume checkpoint/manifest mismatch")
    prior_records = existing_records(data_path) if args.resume else []
    completed = {(record["split_role"], record["id"]) for record in prior_records}
    if len(completed) != len(prior_records):
        raise RuntimeError("Duplicate prior records")
    if not completed.issubset(set(identities)):
        raise RuntimeError("Prior records absent from manifest")
    bp = torch.load(checkpoint_paths["base"], map_location="cpu", weights_only=True)
    jp = torch.load(checkpoint_paths["jome"], map_location="cpu", weights_only=True)
    if int(bp["epoch"]) != 159 or int(jp["epoch"]) != 80:
        raise RuntimeError("Expected selected base epoch159 and selected JOME EMA epoch80")
    base_config = dict(bp["model_specification"]["config"])
    for key in ("encoder_blocks", "decoder_blocks"):
        base_config[key] = tuple(base_config[key])
    jome_config = dict(jp["model_specification"]["config"])
    if jome_config.get("experts", 4) != 4 or jome_config.get("residual_cap", EPS) != EPS:
        raise RuntimeError("Expected four experts with residual cap .1")
    base = PASTRR(PASTRRConfig(**base_config))
    base.load_state_dict(bp["model"], strict=True)
    model = JOMERR(base, JOMERRConfig(**jome_config))
    model.load_addon_state_dict(jp["addon"])
    model.set_routing_epoch(80)
    model.set_training_phase("evaluation")
    model.eval().requires_grad_(False).to(device)
    del bp, jp
    precision = "CUDA BF16 autocast; model parameters FP32" if device.type == "cuda" else "FP32"
    if previous and previous.get("precision") != precision:
        raise RuntimeError("Resume precision mismatch")
    metadata = {
        "schema_version": 1, "checkpoint_sha256": hashes,
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "manifest_sha256": sha256(manifest), "base_epoch": 159, "jome_epoch": 80,
        "base_config": base_config, "jome_config": jome_config,
        "torch_version": torch.__version__, "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "precision": precision, "gram_order": GRAM_ORDER, "replay_vectors": REPLAY_VECTORS,
        "eval_protocol": {"batch_size": 1, "native_shape": True, "border_crop": CROP,
            "rgb_range": [0, 1], "metrics": "raw RGB; float64 reductions; max_val=1",
            "routing_epoch": 80, "phase": "evaluation", "training": False,
            "torch_compile": False, "geometry": "a=MSE(T,I), b=MSE(Y,I), d=mean((Y-I)*(T-I)); MSE(Y,T)=a+b-2d",
            "reachability": "input identity target: mean(max(abs(B-I)-.1,0)^2); separate native BF16 coordinate-envelope audit",
            "clipping_bound": "RMS(Uo-Ur)<=RMS(mean_k abs(clip(B+rk)-(B+rk))); MAE<=mean candidate clipping absolute",
            "e1_arithmetic": "Matched Ur/Uo/Yr/Yo operators all FP32 on fixed BF16-forward residuals and native-dtype bilinearly interpolated weights; jome separately retains native deployment output",
            "e1_mse_interaction": "[MSE(Yr,T)-MSE(Ur,T)]-[MSE(Yo,T)-MSE(Uo,T)]",
            "gram": "6x6 symmetric float64 Gram of candidate errors; mean over cropped RGB coordinates",
            "psnr_zero_mse": "null means positive infinity", "jpeg": "omitted", "lpips": "omitted"},
        "retained": "Per-image Gram and scalar sufficient statistics; frozen checkpoints and inputs rerunnable. Raw residuals/native tensors NOT persisted.",
        "limitations": ["No globally fitted residual-space mixture; no training.",
            "Gram supports global output-space convex combinations of six saved candidate definitions only; spatial/residual fits require rerun.",
            "Checkpoint EMA identity relies on supplied selected addon checkpoint provenance.",
            "BF16 native residual-cap/weight rounding can weaken strict eps=.1 envelope; both strict and measured-envelope bounds reported.",
            "Time budget checked between images; one native forward may overrun budget."],
        "manifest_count": len(rows), "selected_count": len(selected), "completed_count": len(completed),
        "requested_limit": args.limit, "max_seconds": args.max_seconds,
        "sessions": list(previous.get("sessions", [])), "status": "running",
    }
    session = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "resume": args.resume, "initial_completed": len(completed)}
    metadata["sessions"].append(session)
    save_metadata(metadata_path, metadata)
    print(json.dumps({"event": "loaded", "seconds": time.perf_counter() - start,
                      "selected_count": len(selected), "completed_count": len(completed), "precision": precision}), flush=True)
    status = "complete"
    try:
        with data_path.open("a", buffering=1) as stream, torch.inference_mode():
            for row in selected:
                key = row["split_role"], row["id"]
                if key in completed:
                    continue
                if args.max_seconds and time.perf_counter() - start >= args.max_seconds:
                    status = "budget_stop"
                    break
                record_start = time.perf_counter()
                paths = [(root / row[name]).resolve() for name in ("input_path", "target_path")]
                if any(not path.is_relative_to(root) for path in paths):
                    raise ValueError("Manifest path escapes package root")
                image, target = (load_image(path, device) for path in paths)
                if image.shape != target.shape or min(image.shape[-2:]) <= 2 * CROP:
                    raise ValueError(f"Invalid paired shape for {row['id']}: {image.shape}, {target.shape}")
                synchronize(device)
                forward_start = time.perf_counter()
                amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
                with amp:
                    result = model(image)
                synchronize(device)
                forward_seconds = time.perf_counter() - forward_start
                record = {"id": row["id"], "split_role": row["split_role"],
                          "height": image.shape[-2], "width": image.shape[-1],
                          "forward_seconds": forward_seconds}
                if device.type == "mps":
                    # MPS has no float64 reductions. Copy only diagnostic inputs;
                    # CUDA retains all reductions on device.
                    needed = ("anchor_transmission", "expert_residuals", "transmission",
                              "router_weights", "candidate_transmissions", "fused_residual")
                    diagnostic_result = {key: result[key].float().cpu() for key in needed}
                    record.update(numerical_record(diagnostic_result, image.cpu(), target.cpu()))
                    del diagnostic_result
                else:
                    record.update(numerical_record(result, image, target))
                synchronize(device)
                record["record_seconds"] = time.perf_counter() - record_start
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                completed.add(key)
                metadata["completed_count"] = len(completed)
                metadata["elapsed_seconds_this_session"] = time.perf_counter() - start
                save_metadata(metadata_path, metadata)
                print(json.dumps({"event": "image", "id": row["id"], "split_role": row["split_role"],
                                  "forward_seconds": forward_seconds, "record_seconds": record["record_seconds"],
                                  "completed_count": len(completed)}), flush=True)
                del image, target, result, record
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        session["elapsed_seconds"] = time.perf_counter() - start
        session["new_completed"] = len(completed) - session["initial_completed"]
        session["status"] = status
        metadata["status"] = status
        metadata["completed_count"] = len(completed)
        metadata["completed_selected_count"] = sum((row["split_role"], row["id"]) in completed for row in selected)
        metadata["elapsed_seconds_all_sessions"] = sum(item.get("elapsed_seconds", 0) for item in metadata["sessions"])
        save_metadata(metadata_path, metadata)
        print(json.dumps({"event": "stopped", "status": status, "completed_count": len(completed),
                          "elapsed_seconds": session["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
