# Evidence dictionary and paper mapping

The evidence supports direct model evaluation and statistical replay of the paper. Training code and a training-reproduction claim are outside this release.

| Input | Grain and contents | Paper use |
|---|---|---|
| `data/per_image.jsonl` | 800 unique `(split_role,id)` records: selection 300, official 300, internal holdout (`locked`) 200 | Tables 3–5 and 7–9; Figures 1–3 and 5–6 for raw PAST/JOME |
| `data/manifest.csv` | The same 800 IDs/roles and relative `input_path,target_path` | Authoritative image membership for inference and analysis |
| `data/released_geometry.csv` | 2,100 records: 300 official images × seven released systems | Released-system portions of Table 9 and Figures 5–6 |
| `data/run_metrics.csv` | Model × partition summaries, with selected epoch, seed, parameters and recorded metrics | Table 2 and historical local context in Table 6 |
| `data/capacity_context.csv` | 14 systems, parameter/data scope, score provenance and source links | Table 6; published aggregates remain attributed context |
| `data/metadata.json` | Configuration, checkpoints, historical evaluator identity and protocol for the completed 800-image pass | Raw protocol and evidence provenance |
| `data/qualitative_examples.json` | Four selected Figure 4 examples, source hashes, metrics and display protocol | Figure 4 source identification and interpretation |
| `model/MODEL_MANIFEST.json` | PAST/JOME/ARISE configurations, selected epochs, seeds, parameter counts and checkpoint hashes | Direct evaluation and model identity |
| `data/reference_tables.json` | 208 numeric reference values with table/row/column identifiers and rounding precision | Independent checks of the eight quantitative table replays |
| `data/provenance.json` | Included file paths and SHA-256 identities | Integrity manifest; historical execution hashes remain separately identified |

The replay produces eight table CSVs and five quantitative plot PDFs from the retained records. Its numeric checks use `data/reference_tables.json`. The conceptual framework table has no numerical replay. The four illustrative examples discussed in Figure 4 are `val_0001`, `val_0002`, `val_0003` and `val_0248`; the corresponding records retain source hashes, scores and display-conversion metadata. They were selected for illustration rather than as a representative sample or prevalence estimate. The PAST/JOME scores use floating-point predictions before quantization, while the KLETech-CEVI scores use original released encodings.

Obtain the original image/reference pairs from the [OpenRR-5k provider](https://huggingface.co/datasets/qiuzhangTiTi/OpenRR-5k), preserving filenames and bytes as described in [README.md](README.md). Dataset image paths in the manifest and qualitative metadata refer to this external dataset. Original comparison-system output files are identified by provider/system, filename and hash; the release contains their recorded measurements and source identifiers, while the original image files are obtained from the providers.

## Per-image records

`data/per_image.jsonl` retains these fields:

- `id,split_role,height,width`: image identity, declared role and native dimensions.
- `metrics[method].mse,psnr`: cropped raw RGB MSE and unit-peak PSNR in dB. Methods are `input`, `base` (PAST), native `jome`, four individually clipped `expert_0`–`expert_3`, and matched `Ur,Uo,Yr,Yo`. Residual operators mix before clipping; output operators mix clipped candidates. `U` uses uniform weights and `Y` adaptive spatial weights.
- `gram`: symmetric 6×6 mean error-product matrix, ordered `[expert_0,expert_1,expert_2,expert_3,base,input]`. Its 21 distinct entries replay a global output mixture through `w.T @ gram @ w`; its diagonal equals candidate MSE. It cannot replay residual clipping or spatially varying weights.
- `geometry[base|jome].a,b,d`: `a=MSE(T,I)`, `b=MSE(Y,I)` and `d=mean((T-I)*(Y-I))`. Output MSE is `a+b−2d`; attenuation at strength λ gives `a+λ²b−2λd`. Reference-optimal strength is `clip(d/b,0,1)`, and the equal-loss reference margin is `abs(b−2d)/(2*sqrt(b))`.
- `reachability`: nominal `eps=0.1`, `bound=mean(max(abs(B-I)−eps,0)^2)`, affected RGB-coordinate fraction, actual JOME/input MSE, strict/roundoff-adjusted envelope diagnostics and numerical tolerance. A positive bound concerns exact input recovery within the declared action set; it does not establish target-relative harm.
- `clipping,replay`: clipping-order discrepancy/bound checks, router/residual arithmetic diagnostics, and direct-versus-Gram replay errors for three prespecified weights.
- `finite_action_oracle`: reference-conditioned minimum-MSE choice among input/PAST/native JOME; not a deployable selector.
- `forward_seconds,record_seconds`: single-execution timings including evaluator diagnostics, not repeated latency benchmarks.

Released geometry stores `required_edit_mse=a`, `correction_mse=b` and `alignment_cosine=d/sqrt(a*b)`, plus saved/recomputed output PSNR, norm ratio, loss change and energy-identity error. The replay recovers `d` from alignment and norms. Released encodings, historical model evaluations and controlled raw PAST/JOME outputs remain distinct evidence sources.

All analyses weight images equally. Mean PSNR is the mean of `−10 log10(image MSE)`, rather than PSNR of mean MSE. MSE tables multiply values by `10^4`; geometry margins multiply unit-range RMS values by 255. Near identity means input/reference PSNR `>=40` dB. Figure 5 strata are `<25`, `[25,30)`, `[30,35)`, `[35,40)` and `>=40` dB, containing 65, 114, 82, 31 and eight official images.

## Calibration and uncertainty

Mixtures are fitted only on selection data. The mean-PSNR reconstruction uses the paper's barycentre, vertices and 32 deterministic interior starts with SLSQP and analytic gradients. The original initialization seed and start distribution are unavailable. Reconstructed coefficients match the paper at displayed precision; the nonconvex objective has no global-optimality guarantee.

Intervals use 5,000 paired image resamples with fixed fitted weights. The original operator analysis gives an official matched-residual-gain interval upper endpoint of 0.015774 dB, whereas Table 3 displays 0.0150 dB. The replay implements the paper's documented common-resampling protocol: reset the random state to `20261013 + role_index` for each contrast, with selection 0, official 1 and internal holdout 2. This reproduces all displayed Table 3 endpoints and the textual calibrated F4P endpoints. The original paper bootstrap implementation is unavailable; `results.json` retains the original-analysis intervals separately.

Intervals omit training-seed and calibration uncertainty. Split roles are retrospective, and scene identifiers are unavailable. BF16 residual/weight rounding can weaken the strict nominal `eps=0.1` envelope; records retain both strict and measured-envelope diagnostics. Scalar/Gram records support the declared global output mixtures. Arbitrary spatial or residual-operator analyses require rerunning inference.

## Direct-evaluation outputs and validation

`evaluate_benchmark.py` writes one folder per model with `per_image.jsonl` (`raw` and `jpeg100`, each containing `mse,psnr,ssim`) and `summary.json` (equal-image means by split). Output metadata records the actual device, precision, software, protocol, completed counts and checkpoint/evaluator hashes. Historical raw MSE uses FP32 reductions; controlled raw MSE uses float64. Zero-MSE PSNR is JSON `null`, meaning positive infinity. Optional native predictions are HWC float32 `.npy`; PNG/JPEG are rounded RGB8 representations. LPIPS, DISTS, NIQE and historical full-pipeline latency are retained evidence and are not recomputed by this evaluator.

Verification covers CPU benchmark/operator operation checks, strict state loading and bitwise FP32 forward equivalence with the original implementations on a synthetic input, plus an earlier one-image A100 run of all three models. The full benchmark has not been rerun with the direct evaluator, and the current controlled device-normalization protocol has not been rechecked on GPU. A separate five-image PAST/JOME export with the original operator evaluator validates those exported raw scores only. Hardware, precision, PyTorch, CUDA and Pillow differences can alter results. The released-output audit found a maximum absolute difference of 0.000228 between measured team-mean SSIM and published challenge values. Exact published-score parity is not established. Detailed verification quantities and checkpoint identities are recorded in `model/MODEL_MANIFEST.json`.
