#!/usr/bin/env python3
"""Evaluate selected PAST/JOME/ARISE EMA models on external OpenRR pairs."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from benchmark_metrics import ssim_value

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from model.past import PASTRR, PASTRRConfig
from model.jome import JOMERR, JOMERRConfig
from model.arise import ARISERR, ARISERRConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_model(name: str, device: torch.device, protocol: str = 'historical',
               checkpoint_dir: Path | None = None) -> tuple[torch.nn.Module, dict]:
    """Strictly load packaged, selected EMA states without pickle fallback."""
    checkpoint_dir = checkpoint_dir or REPO / 'checkpoints'
    manifest = json.loads((REPO / 'model/MODEL_MANIFEST.json').read_text())
    entries = {entry['model']: entry for entry in manifest['models']}
    needed = ['past'] if name == 'past' else ['past', name]
    hashes = {}
    payloads = {}
    for method in needed:
        entry = entries[method]
        path = checkpoint_dir / Path(entry['checkpoint']).name
        hashes[method] = sha256(path)
        if hashes[method] != entry['sha256']:
            raise RuntimeError(f'Checkpoint hash mismatch: {path}')
        payloads[method] = torch.load(path, map_location='cpu', weights_only=True)
        if payloads[method]['epoch'] != entry['epoch']:
            raise RuntimeError(f'Selected checkpoint epoch mismatch: {method}')
    base_payload = payloads['past']
    config = dict(base_payload['model_specification']['config'])
    for key in ('encoder_blocks', 'decoder_blocks'):
        config[key] = tuple(config[key])
    base = PASTRR(PASTRRConfig(**config))
    base.load_state_dict(base_payload['model'], strict=True)
    if name == 'past':
        model = base
    else:
        payload = payloads[name]
        if payload['anchor']['sha256'] != hashes['past']:
            raise RuntimeError('Add-on anchor identity mismatch')
        config = dict(payload['model_specification']['config'])
        if name == 'jome':
            model = JOMERR(base, JOMERRConfig(**config))
            model.set_routing_epoch(payload['epoch'])
        else:
            model = ARISERR(base, ARISERRConfig(**config))
        model.load_addon_state_dict(payload['addon'])  # Each component loads strictly.
        model.set_training_phase('evaluation')
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != entries[name]['parameters']['total']:
        raise RuntimeError(f'Parameter-count mismatch for {name}: {count}')
    model.eval().requires_grad_(False).to(device)
    if protocol == 'historical':
        model.to(memory_format=torch.channels_last)
    return model, {'checkpoint_sha256': hashes, 'epoch': entries[name]['epoch'],
                   'parameters_total': count, 'config': config}


def load_rgb(path: Path, device: torch.device, protocol: str) -> torch.Tensor:
    with Image.open(path) as source:
        if protocol == 'historical':
            array = np.asarray(source.convert('RGB'), dtype=np.float32) / 255.0
            value = torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()
        else:
            array = np.array(source.convert('RGB'), dtype=np.uint8, copy=True)
            # Preserve the frozen evaluator's device-side normalization order.
            return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(
                device=device, dtype=torch.float32).div_(255)
    value = value.unsqueeze(0).to(device)
    return value.contiguous(memory_format=torch.channels_last) if protocol == 'historical' else value


def to_pil(value: torch.Tensor) -> Image.Image:
    array = value.detach().float().clamp(0, 1).mul(255).round().to(torch.uint8)
    return Image.fromarray(array.permute(1, 2, 0).cpu().numpy())


def jpeg100(value: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, bytes]:
    buffer = io.BytesIO()
    to_pil(value[0]).save(buffer, format='JPEG', quality=100, subsampling=0)
    encoded = buffer.getvalue()
    buffer.seek(0)
    with Image.open(buffer) as source:
        array = np.asarray(source.convert('RGB'), dtype=np.float32) / 255.0
    decoded = torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous().unsqueeze(0)
    return decoded.to(device), encoded


def metrics(prediction: torch.Tensor, target: torch.Tensor, protocol: str) -> dict:
    p = prediction[..., 4:-4, 4:-4].float()
    t = target[..., 4:-4, 4:-4].float()
    if protocol == 'controlled':
        mse = float((p.cpu().double() - t.cpu().double()).square().mean())
    else:
        mse = float((p - t).square().mean())
    return {'mse': mse, 'psnr': -10 * math.log10(mse) if mse > 0 else None,
            'ssim': ssim_value(prediction, target)}


def data_path(root: Path, relative: str) -> Path:
    # Accept either OpenRR-5k/ or its data/ directory without rearranging files.
    path = Path(relative)
    if not path.parts or path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Invalid manifest path: {relative}')
    full = root / path
    if path.parts[0] == 'data':
        direct = root / Path(*path.parts[1:])
        if direct.is_file() and not full.is_file():
            return direct
        if root.resolve().name == 'data' and not full.is_file():
            return direct
    return full


def sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True, help='OpenRR-5k/ or OpenRR-5k/data/')
    parser.add_argument('--model', choices=['past', 'jome', 'arise', 'all'], default='all')
    parser.add_argument('--split', choices=['official', 'selection', 'locked', 'all'], default='official')
    parser.add_argument('--manifest', type=Path, default=REPO / 'data/manifest.csv')
    parser.add_argument('--checkpoint-dir', type=Path, default=REPO / 'checkpoints')
    parser.add_argument('--output', type=Path, required=True, help='New directory; refuses to overwrite')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--protocol', choices=['historical', 'controlled'], default='historical')
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default=None,
                        help='Default CUDA BF16, otherwise FP32; override recorded in metadata')
    parser.add_argument('--limit', type=int, default=0, help='0 means all selected rows')
    parser.add_argument('--ids', nargs='+', help='Optional explicit IDs within the selected split')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save native HWC FP32 .npy, rounded RGB8 .png, and JPEG100 .jpg')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError('limit must be nonnegative')
    device = torch.device(args.device)
    precision = args.precision or ('bf16' if device.type == 'cuda' else 'fp32')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu or mps')
    if precision == 'bf16':
        if device.type != 'cuda':
            raise RuntimeError('BF16 forward requires a compatible CUDA GPU')
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError('BF16 forward requires a compatible CUDA GPU')
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = args.protocol == 'historical'
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = args.protocol == 'historical'
    torch.set_float32_matmul_precision('high' if args.protocol == 'historical' else 'highest')
    with args.manifest.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {'id', 'split_role', 'input_path', 'target_path'} <= set(rows[0]):
        raise ValueError('Manifest requires id,split_role,input_path,target_path')
    if len({(r['split_role'], r['id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate manifest identities')
    rows = [r for r in rows if args.split == 'all' or r['split_role'] == args.split]
    if args.ids:
        if len(set(args.ids)) != len(args.ids):
            raise ValueError('Requested IDs contain duplicates')
        missing = set(args.ids) - {r['id'] for r in rows}
        if missing:
            raise ValueError(f'IDs absent from selected split: {sorted(missing)}')
        rows = [r for r in rows if r['id'] in args.ids]
    rows = rows[:args.limit] if args.limit else rows
    if not rows:
        raise ValueError('No pairs selected')
    for row in rows:
        for key in ('input_path', 'target_path'):
            path = data_path(args.data_root, row[key])
            if not path.is_file():
                raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=False)
    models = ['past', 'jome', 'arise'] if args.model == 'all' else [args.model]
    metadata = {'schema_version': 1, 'torch_version': torch.__version__,
        'pillow_version': Image.__version__, 'device': str(device),
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device),
        'protocol': args.protocol, 'precision': precision, 'manifest_sha256': sha256(args.manifest),
        'evaluator_sha256': sha256(Path(__file__)), 'metric_helper_sha256': sha256(Path(__file__).with_name('benchmark_metrics.py')),
        'split': args.split, 'images_per_model': len(rows), 'crop': 4, 'native_dimensions': True,
        'psnr_zero_mse': 'null means positive infinity',
        'prediction_files': 'HWC float32 .npy is native output; PNG is rounded 8-bit; JPEG quality100 subsampling0' if args.save_predictions else None,
        'metric_scope': 'raw/JPEG100 RGB PSNR and local SSIM only; no perceptual metrics or full-pipeline timing parity',
        'status': 'running', 'models': {}}
    def save_metadata():
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    save_metadata()
    try:
        for name in models:
            model, model_info = load_model(name, device, args.protocol, args.checkpoint_dir)
            metadata['models'][name] = model_info
            output = args.output / name
            output.mkdir()
            scores = []
            with (output / 'per_image.jsonl').open('w') as stream, torch.inference_mode():
                for row in rows:
                    image = load_rgb(data_path(args.data_root, row['input_path']), device, args.protocol)
                    target = load_rgb(data_path(args.data_root, row['target_path']), device, args.protocol)
                    if image.shape != target.shape or min(image.shape[-2:]) < 19:
                        raise ValueError(f'Invalid pair shape for {row["id"]}')
                    sync(device)
                    start = time.perf_counter()
                    amp = torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else contextlib.nullcontext()
                    with amp:
                        result = model(image)
                    sync(device)
                    elapsed = time.perf_counter() - start
                    prediction = result['transmission'].float()
                    # MPS lacks float64; historical metrics retain native device,
                    # while controlled MSE copies only cropped tensors for doubles.
                    jpeg, encoded = jpeg100(prediction, device)
                    record = {'id': row['id'], 'split_role': row['split_role'],
                        'height': image.shape[-2], 'width': image.shape[-1], 'forward_seconds': elapsed,
                        'raw': metrics(prediction, target, args.protocol),
                        'jpeg100': metrics(jpeg, target, args.protocol)}
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                    stream.flush()
                    if args.save_predictions:
                        np.save(output / f'{row["id"]}.npy', prediction[0].permute(1, 2, 0).cpu().numpy(), allow_pickle=False)
                        to_pil(prediction[0]).save(output / f'{row["id"]}.png')
                        (output / f'{row["id"]}.jpg').write_bytes(encoded)
                    scores.append(record)
                    print(json.dumps({'model': name, 'id': row['id'], 'raw_psnr': record['raw']['psnr'],
                                      'jpeg100_psnr': record['jpeg100']['psnr']}), flush=True)
                    del image, target, prediction, jpeg, result
            summary = {'model': name, 'images': len(scores), 'by_split': {}}
            for split in sorted({r['split_role'] for r in scores}):
                subset = [r for r in scores if r['split_role'] == split]
                stats = {'images': len(subset)}
                for representation in ('raw', 'jpeg100'):
                    stats[representation] = {}
                    for metric in ('mse', 'psnr', 'ssim'):
                        values = [r[representation][metric] for r in subset]
                        stats[representation][metric + '_mean'] = None if any(v is None for v in values) else float(np.mean(values))
                summary['by_split'][split] = stats
            (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
            metadata['models'][name]['completed_images'] = len(scores)
            save_metadata()
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        metadata['status'] = 'complete'
    except BaseException as error:
        metadata['status'] = 'failed'
        metadata['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        save_metadata()


if __name__ == '__main__':
    main()
