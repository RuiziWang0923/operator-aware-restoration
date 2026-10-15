# From correction capacity to correction control: operator-aware evaluation of image restoration

Ruizi Wang · [ORCID 0000-0002-6646-9907](https://orcid.org/0000-0002-6646-9907) · [454762251@qq.com](mailto:454762251@qq.com)

This code and data repository accompanies the paper and provides PAST, JOME and ARISE implementations and evaluation weights, native-resolution inference, the paper's 800-image operator records, and statistical replay. Its reproducibility scope is evaluation and analysis; model training is not included.

| Path | Contents |
|---|---|
| `model/` | PAST backbone, JOME/ARISE add-ons and [model manifest](model/MODEL_MANIFEST.json) |
| `checkpoints/` | Evaluation-weight download links and checksums |
| `data/` | Per-image sufficient statistics, split manifest and attributed comparison summaries |
| `scripts/` | Benchmark inference, operator evaluation and numerical replay |
| [DATA.md](DATA.md) | Evidence dictionary, protocols, paper mapping and limitations |

## Replay the paper's results

With Python 3.11 or later:

```bash
python -m pip install -r requirements.txt
python scripts/reproduce.py
```

The command validates all 800 records and their split membership, fits mixtures using only the 300 selection images, and writes eight table CSVs, five quantitative plot PDFs and `results.json` under `outputs/`. It checks all 208 numeric table reference values in `data/reference_tables.json` at their stated rounding precision, including Table 3 interval endpoints. The conceptual framework table has no numerical replay. Records for the four purposefully selected examples discussed in Figure 4 are documented in `data/qualitative_examples.json`.

The replay reconstructs the paper's bootstrap protocol and mean-PSNR optimization from the retained records. The original mean-PSNR initialization seed/distribution and paper bootstrap implementation are unavailable; the replay matches the displayed coefficients and intervals. Mean-PSNR fitting is nonconvex and has no global-optimality guarantee. [DATA.md](DATA.md) explains the distinct original-analysis interval and uncertainty limits. The replay was tested with Python 3.14, NumPy 2.4.4, SciPy 1.17.1 and Matplotlib 3.10.9.

## Evaluate the models

| System | Evaluation checkpoint | Selected epoch | Backbone |
|---|---|---:|---|
| PAST | `checkpoints/base.pt` | 159 | Self-contained |
| JOME | `checkpoints/jome.pt` | 80 | `checkpoints/base.pt` |
| ARISE | `checkpoints/arise.pt` | 80 | `checkpoints/base.pt` |

The three checkpoints occupy approximately 107 MiB and are byte-identical to the evaluated EMA artifacts. Download them from the [paper release](https://github.com/RuiziWang0923/operator-aware-restoration/releases/tag/paper) with the hash-verifying command below. Configurations, seeds, parameter counts and SHA-256 identities are recorded in [MODEL_MANIFEST.json](model/MODEL_MANIFEST.json).

Obtain paired images from the [OpenRR-5k provider](https://huggingface.co/datasets/qiuzhangTiTi/OpenRR-5k). For the official split, extract `val_300_blended.zip` and `val_300_transmission.zip` into `data/val/input/` and `data/val/target/`, preserving original filenames and encoded bytes. For example, the first input is `data/val/input/val_0001.jpg`. Some `.jpg` targets contain PNG data; Pillow decodes them correctly. Selection and internal-holdout evaluation also requires the training pairs listed in `data/manifest.csv`.

```bash
python -m pip install -r requirements-inference.txt
python scripts/download_checkpoints.py
python scripts/evaluate_benchmark.py --data-root /path/to/OpenRR-5k/data --model all --output outputs/official
```

The data root can be the dataset's `data/` directory or its parent `OpenRR-5k/`. The default split is the 300 official validation pairs. Use `--split selection`, `--split locked` (internal holdout) or `--split all` for the other manifest-defined groups. `--model past`, `--model jome` and `--model arise` select one system; `--limit 2` provides a short pilot.

Outputs contain per-image and equal-image-mean raw/JPEG100 RGB PSNR and local SSIM, with execution and checkpoint metadata. Inference preserves native dimensions; metrics crop four pixels. CUDA uses BF16 forward with FP32 parameters; install a compatible PyTorch build for the GPU. The recorded CUDA evaluation used PyTorch 2.11.0+cu130 on an A100.

The default `--protocol historical` uses channels-last layout, TF32, FP32 PSNR reductions, and quality-100 JPEG with chroma subsampling disabled. `--protocol controlled` uses contiguous layout, device-side RGB normalization, disabled TF32 matmul and float64 MSE reductions. CPU/MPS use FP32 forward by default and can produce different scores. Add `--save-predictions` to retain native float32 `.npy` predictions, RGB8 PNGs and the scored JPEG bytes. Use the `.npy` files to remeasure raw pixels. Evaluation output directories must be new.

Full-benchmark revalidation of the direct evaluator remains incomplete. Verification covers CPU operation, bitwise FP32 equivalence with the original implementations on a synthetic input, and an earlier run of one official image on an A100 for all three models; its largest absolute historical JPEG PSNR difference was 0.002065 dB. The controlled normalization protocol has not been rechecked on GPU. A separate five-image PAST/JOME export with the original operator evaluator matched recorded raw PSNR within 2.14e-13 dB. These checks establish their stated scope, not full historical score parity. Historical LPIPS, DISTS, NIQE and full-pipeline timings are retained statistics, not recomputed by this command.

## Recompute operator records

After obtaining the manifest-listed image pairs, run:

```bash
python scripts/evaluate_operators.py --data-root /path/to/OpenRR-5k/data --output outputs/operators --device cuda --limit 0
```

This evaluates the supplied PAST/JOME checkpoints and writes matched-operator, Gram-matrix, reachability and correction-geometry records for all 800 images. It uses BF16 network forward, FP32 matched operators, float64 reductions and a four-pixel crop. `--limit 2` provides a pilot. Raw residual tensors and spatial router maps are not included in the evidence records; analyses requiring those tensors must rerun inference. To analyze newly generated records:

```bash
python scripts/analyze_operators.py --input outputs/operators/per_image.jsonl --output outputs/operator-analysis
```

## License and citation

Copyright (c) 2026 Ruizi Wang. Original authored code, weights, documentation and results are available under [CC BY-NC-SA 4.0](LICENSE). Third-party material retains its own terms; see [NOTICE](NOTICE). Citation metadata is provided in [CITATION.cff](CITATION.cff).
