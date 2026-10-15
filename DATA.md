# Evidence dictionary and manuscript mapping

| Input | Grain and fields | Use |
|---|---|---|
| `data/per_image.jsonl` | 800 records, one unique `(split_role,id)`; `selection` 300, `official` 300, `locked` 200 (internal holdout) | Tables 3–5, 7–9; Figures 1–3, 5–6 for current raw base/JOME |
| `data/manifest.csv` | Same 800 IDs/roles plus relative `input_path,target_path`; obtain image pairs from the dataset provider | Authoritative membership; optional inference staging |
| `data/released_geometry.csv` | 2,100 records: 300 official images × seven released systems | Released parts of Table 9 and Figures 5–6 |
| `data/run_metrics.csv` | Historical run × partition summaries | Table 2; local historical parameter/JPEG context |
| `data/capacity_context.csv` | 14 system rows with reported parameter count, paired-data scope, score provenance and source URL | Table 6; published aggregate values are retained context, not independently regenerated |
| `data/metadata.json` | One protocol/configuration/checkpoint identity record for the executed 800-image pass | Precision, crop, device, epochs, candidate definitions, timing and limits |
| `data/qualitative_examples.json` | Four purposefully selected Figure 4 examples; IDs, original image hashes, raw/released metrics, checkpoint identities, display protocol and selection reasons | Figure 4 provenance; not a representative sample or prevalence estimate |
| `model/MODEL_MANIFEST.json` | PAST, JOME and ARISE evaluation weights; hashes, configs, validation-selected epochs, shared backbone and source run mapping | Direct PAST/JOME/ARISE inference and model provenance |
| `data/provenance.json` | One entry per copied repository input; relative original source and SHA-256 | Origin tracking without personal machine paths |

The eight `.tex` table files are editable manuscript inputs. The inline framework table in `sections/03_framework.tex` has no data source. Figure 4 is supplied artwork. PAST/JOME panels are 8-bit PNG display copies of native predictions; PSNR labels use the floating-point predictions before quantization. KLETech-CEVI panels and labels use original released encodings. The full-frame examples are val_0001, val_0002, val_0003, and val_0248. Model implementations, EMA evaluation checkpoints and evaluators support direct PAST/JOME/ARISE inference. Obtain the original input/reference pixels from the [OpenRR-5k provider](https://huggingface.co/datasets/qiuzhangTiTi/OpenRR-5k), following the paths and byte-preservation instructions in [README.md](README.md). Historical ablation rows are replayed from their recorded summary statistics.

`per_image.jsonl` fields:

- `id,split_role,height,width`: image identity, declared role and native dimensions.
- `metrics[method].mse,psnr`: cropped raw RGB image MSE and unit-peak PSNR in dB. Methods are `input`, `base`, native `jome`, four individually clipped `expert_0`–`expert_3`, and matched `Ur,Uo,Yr,Yo`. Residual operators mix before clipping; output operators mix clipped candidates. `U` uses uniform weights; `Y` uses adaptive spatial weights.
- `gram`: symmetric 6×6 mean error-product matrix, order `[expert_0,expert_1,expert_2,expert_3,base,input]`. Its 21 distinct entries replay a global output mixture through `w.T @ gram @ w`. Its diagonal equals candidate MSE. It cannot replay residual clipping or spatially varying weights.
- `geometry[base|jome].a,b,d`: `a=MSE(T,I)`, `b=MSE(Y,I)`, `d=mean((T-I)*(Y-I))`; output MSE is `a+b-2d`. Attenuation MSE at strength λ is `a+λ²b−2λd`. Reference-optimal strength is `clip(d/b,0,1)`; equal-loss reference margin is `abs(b-2d)/(2*sqrt(b))`.
- `reachability`: nominal `eps=0.1`, `bound=mean(max(abs(B-I)-eps,0)^2)`, affected RGB-coordinate fraction, actual JOME/input MSE, strict/roundoff-adjusted envelope diagnostics and numerical tolerance. Bound positivity describes exact input recovery in the declared action set, not target-relative harm.
- `clipping,replay`: clipping-order discrepancy/bound checks, router/residual arithmetic diagnostics, and direct-versus-Gram replay errors for three prespecified weights.
- `finite_action_oracle`: reference-conditioned minimum-MSE choice among input/base/native JOME; not a deployable selector.
- `forward_seconds,record_seconds`: single execution timings including evaluator diagnostics; not repeated latency benchmarks.

Released geometry retains `required_edit_mse=a`, `correction_mse=b`, and `alignment_cosine=d/sqrt(a*b)`, plus recomputed/saved output PSNR, norm ratio, loss change, and energy-identity error. The replay recovers `d` from alignment and norms. These outputs preserve their released encodings and are not merged with historical or current raw JOME predictions.

All analyses weight images equally. Mean PSNR is the mean of `−10 log10(image MSE)`, not PSNR of mean MSE. Reported MSE tables multiply by `10^4`; geometry margins multiply RMS unit-range values by 255. Near identity means input PSNR `>=40` dB against the reference. Figure 5 strata are `<25`, `[25,30)`, `[30,35)`, `[35,40)`, and `>=40` dB and contain 65, 114, 82, 31, and eight official images.

Historical `run_metrics.csv` stores model/run identity, partition, epoch/seed, parameter counts, raw/JPEG metrics, input score, pipeline timing, anchor score and paired gain/loss summaries. Only its official rows for ARISE/M/N/Asymmetric/Dual/JOME enter Table 2. Public capacity counts and published scores cannot be reconstructed without provider evidence; their retained source attribution is part of the table's scope.

Direct evaluation writes separate model folders with `per_image.jsonl` (`raw` and `jpeg100` each containing `mse,psnr,ssim`) and `summary.json` (equal-image means by split). `metadata.json` identifies the actual protocol, precision, device/software, checkpoint/evaluator hashes, and completed image counts. Raw MSE uses FP32 under the historical protocol and float64 under controlled protocol. Raw PSNR zero-MSE cases are JSON `null`, meaning positive infinity. Optional native predictions are HWC float32 `.npy`; PNG and JPEG files are rounded RGB8 representations. Only PSNR/SSIM are recomputed by this command; retained historical perceptual metrics are not claimed to be regenerated.

Verification of direct inference is limited to CPU operation checks and a one-image A100 run of all three models; the full benchmark has not been rerun with `evaluate_benchmark.py`. Its final controlled device-normalization change has not been rechecked on GPU because of a Colab connection failure. The separate frozen-evaluator five-image PAST/JOME export validates those exported raw scores only. Exact numerical comparisons and checkpoint identities are recorded in `model/MODEL_MANIFEST.json`.
