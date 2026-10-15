#!/usr/bin/env python3
"""Selection-only convex calibration and frozen-model E1/E2/E3 analysis.

Usage: python analyze_priority.py --input output/per_image.jsonl --output output
No inference or training is performed. All MSE averages give each image equal weight.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

CANDIDATES = [f"expert_{k}" for k in range(4)] + ["base", "input"]
REQUIRED = ["input", "base", "jome", "Ur", "Uo", "Yr", "Yo"] + CANDIDATES[:4]
ROLES = ("selection", "official", "locked")
RNG_SEED = 20261013  # analysis resampling only; never a restoration model seed
FLOOR = 1e-15
THRESHOLDS = (0, 0.05, 0.1, 0.25, 0.5, 1.0)
METHOD_LABELS = {
    "input": "Original input", "base": "Base restoration", "jome": "Native JOME deployment",
    "Ur": "Uniform residual mix, then clip", "Uo": "Uniform mix of clipped expert outputs",
    "Yr": "Adaptive residual mix, then clip", "Yo": "Adaptive mix of clipped expert outputs",
    "fixed4": "Selection-fitted mixture of four clipped experts",
    "fixed6": "Selection-fitted mixture of experts, base, and input",
    "forced_expert": "Selection-chosen forced expert",
    "oracle_input_base_jome": "Reference oracle: input, base, and native JOME",
    "oracle4": "Reference capacity oracle: four clipped experts",
    "oracle6": "Reference capacity oracle: clipped experts, base, and input",
    **{f"expert_{k}": f"Individually clipped expert {k}" for k in range(4)},
}


def scalar(x):
    try:
        value = float(x)
        return value if np.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def finite_summary(values):
    a = np.asarray([v for v in values if scalar(v) is not None], dtype=float)
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max()),
            "q05": float(np.quantile(a, .05)), "q95": float(np.quantile(a, .95))}


def psnr(mse):
    # Unit RGB range; no pixel-weighted aggregation and no PSNR-of-mean-MSE proxy.
    return -10.0 * np.log10(np.maximum(np.asarray(mse, dtype=float), FLOOR))


def bootstrap_mean(values, rng, repeats):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return [None, None]
    means = []
    for start in range(0, repeats, 256):
        sample = rng.integers(0, len(a), size=(min(256, repeats-start), len(a)))
        means.extend(a[sample].mean(axis=1))
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def contrast(a, b, rng, repeats):
    """a-b PSNR is positive for improvement; a-b MSE is negative for improvement."""
    gain = psnr(a) - psnr(b)
    delta = np.asarray(a) - np.asarray(b)
    return gain_summary(gain, delta, rng, repeats)


def gain_summary(gain, delta_mse, rng, repeats):
    gain, delta_mse = np.asarray(gain), np.asarray(delta_mse)
    n = len(gain)
    result = {"n": n, "mean_psnr_gain_db": float(gain.mean()),
              "psnr_gain_ci95": bootstrap_mean(gain, rng, repeats),
              "mean_mse_delta": float(delta_mse.mean()),
              "mse_delta_ci95": bootstrap_mean(delta_mse, rng, repeats),
              "wins": int((gain > 1e-10).sum()), "losses": int((gain < -1e-10).sum()),
              "ties": int((np.abs(gain) <= 1e-10).sum()),
              "positive_gain_db_per_image": float(np.maximum(gain, 0).mean()),
              "negative_burden_db_per_image": float(np.maximum(-gain, 0).mean()),
              "loss_gt_0_1_db_count": int((gain < -.1).sum()),
              "loss_gt_0_1_db_fraction": float((gain < -.1).mean()),
              "q05_psnr_gain_db": float(np.quantile(gain, .05)),
              "median_psnr_gain_db": float(np.median(gain)),
              "thresholds_db": {str(t): {"gain_gt_fraction": float((gain > t).mean()),
                                         "loss_gt_fraction": float((gain < -t).mean())}
                                for t in THRESHOLDS}}
    return result


def read_records(path, expected, expected_ids=None):
    records, invalid, errors = [], [], []
    seen = set()
    raw_counts, ids_by_role = Counter(), defaultdict(set)
    checks = defaultdict(list)
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as exc:
            invalid.append({"line": lineno, "reason": f"invalid JSON: {exc}"})
            continue
        ident, role = str(r.get("id", "")), r.get("split_role")
        raw_counts[str(role)] += 1
        reasons = []
        if not ident or role not in ROLES:
            reasons.append("missing id or unknown split_role")
        key = (role, ident)
        if key in seen:
            reasons.append("duplicate (split_role,id)")
        seen.add(key)
        ids_by_role[role].add(ident)
        metrics = r.get("metrics", {})
        for name in REQUIRED:
            m = scalar(metrics.get(name, {}).get("mse"))
            if m is None or m < 0:
                reasons.append(f"missing/nonfinite/negative {name}.mse")
        try:
            g = np.asarray(r.get("gram"), dtype=float)
            if g.shape != (6, 6) or not np.isfinite(g).all():
                raise ValueError("Gram must be finite 6x6")
            symmetry = float(np.max(np.abs(g-g.T)))
            eigmin = float(np.linalg.eigvalsh((g+g.T)/2).min())
            diagerr = max(abs(g[j, j]-float(metrics[name]["mse"])) for j, name in enumerate(CANDIDATES))
            checks["gram_symmetry_absolute_error"].append(symmetry)
            checks["gram_min_eigenvalue"].append(eigmin)
            checks["gram_diagonal_mse_absolute_error"].append(diagerr)
            tol = 1e-8 + 1e-5 * max(float(np.max(np.abs(g))), FLOOR)
            if symmetry > tol or eigmin < -tol or diagerr > tol:
                reasons.append("Gram symmetry/PSD/metric diagonal verification failed")
            r["_gram"] = (g+g.T)/2
        except (ValueError, TypeError, KeyError) as exc:
            reasons.append(str(exc))
        if reasons:
            invalid.append({"line": lineno, "id": ident, "split_role": role, "reason": "; ".join(reasons)})
        else:
            records.append(r)
    overlap = {}
    for j, left in enumerate(ROLES):
        for right in ROLES[j+1:]:
            overlap[f"{left}/{right}"] = sorted(ids_by_role[left] & ids_by_role[right])
    valid_counts = Counter(r["split_role"] for r in records)
    mismatches = {role: {"expected": n, "observed": raw_counts.get(role, 0)}
                  for role, n in expected.items() if n is not None and raw_counts.get(role, 0) != n}
    if mismatches:
        errors.append("Expected split counts do not match; this is an incomplete/partial analysis.")
    if invalid:
        errors.append("Invalid or duplicate records excluded from all analyses; inspect integrity.invalid_records.")
    if any(overlap.values()):
        errors.append("IDs overlap between split roles; calibration/evaluation separation is compromised.")
    missing_ids, unexpected_ids = {}, {}
    if expected_ids is not None:
        for role in ROLES:
            missing_ids[role] = sorted(expected_ids.get(role, set())-ids_by_role[role])
            unexpected_ids[role] = sorted(ids_by_role[role]-expected_ids.get(role, set()))
        if any(missing_ids.values()) or any(unexpected_ids.values()):
            errors.append("Per-image IDs do not match the authoritative manifest; inspect missing/unexpected IDs.")
    return records, {"raw_counts": dict(raw_counts), "valid_counts": dict(valid_counts),
                     "expected_counts": expected, "count_mismatches": mismatches,
                     "missing_manifest_ids": missing_ids, "unexpected_manifest_ids": unexpected_ids,
                     "invalid_records": invalid, "cross_role_id_overlap": overlap,
                     "checks": {k: finite_summary(v) for k, v in checks.items()},
                     "warnings": errors, "passed": not errors}


def fit_simplex(grams, n_candidates):
    if not len(grams):
        return {"success": False, "reason": "No valid selection records; no fitted mixture produced."}
    mean_g = np.mean(grams[:, :n_candidates, :n_candidates], axis=0)
    scale = max(float(np.max(np.abs(mean_g))), FLOOR)
    g = mean_g / scale
    objective = lambda w: float(w @ g @ w)
    result = minimize(objective, np.full(n_candidates, 1/n_candidates), jac=lambda w: 2*g@w,
                      method="SLSQP", bounds=[(0, 1)]*n_candidates,
                      constraints={"type": "eq", "fun": lambda w: w.sum()-1,
                                   "jac": lambda w: np.ones_like(w)},
                      options={"ftol": 1e-14, "maxiter": 2000})
    w = np.asarray(result.x)
    gradient = 2*g@w
    # Convex Frank-Wolfe dual gap verifies optimality independently of solver status.
    gap = float(w @ gradient - gradient.min())
    feasible = abs(float(w.sum())-1) <= 1e-8 and w.min() >= -1e-9
    verified = bool(result.success and feasible and gap <= 1e-7)
    return {"success": verified, "solver_success": bool(result.success), "message": str(result.message),
            "iterations": int(result.nit), "weights": w.tolist(),
            "candidate_names": CANDIDATES[:n_candidates], "selection_n": int(len(grams)),
            "objective_mean_image_mse": float(w@mean_g@w),
            "scaled_dual_gap": gap, "objective_dual_gap_mse": gap*scale,
            "simplex_sum_error": abs(float(w.sum())-1), "minimum_weight": float(w.min()),
            "uniform_selection_mean_image_mse": float(mean_g.sum()/n_candidates**2),
            "best_forced_selection_mean_image_mse": float(np.diag(mean_g).min())}


def numeric_leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from numeric_leaves(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(obj, (list, tuple)):
        for key, value in enumerate(obj):
            yield from numeric_leaves(value, f"{prefix}.{key}")
    elif scalar(obj) is not None:
        yield prefix, float(obj)


def diagnostic_summary(group, name):
    values = defaultdict(list)
    for r in group:
        for key, value in numeric_leaves(r.get(name, {})):
            values[key].append(value)
    return {k: finite_summary(v) for k, v in sorted(values.items())}


def verification_flags(group):
    replay_bad, clipping_bad, clipping_tolerances = [], [], []
    for r in group:
        replay, clip = r.get("replay", {}), r.get("clipping", {})
        err, tolerance = scalar(replay.get("max_absolute_error")), scalar(replay.get("tolerance"))
        if replay.get("passed") is False or (err is not None and tolerance is not None and err > tolerance):
            replay_bad.append(r["id"])
        observed, bound = scalar(clip.get("uniform_rms_difference")), scalar(clip.get("uniform_rms_bound"))
        bound_tolerance = scalar(clip.get("bound_tolerance"))
        bound_tolerance = 2e-7 if bound_tolerance is None else bound_tolerance
        clipping_tolerances.append(bound_tolerance)
        if observed is not None and bound is not None and observed > bound+bound_tolerance:
            clipping_bad.append(r["id"])
    return {"gram_replay_failed_count": len(replay_bad), "gram_replay_failed_ids": replay_bad,
            "uniform_clipping_bound_violation_count": len(clipping_bad),
            "uniform_clipping_bound_violation_ids": clipping_bad,
            "uniform_clipping_bound_absolute_tolerance": "per-record clipping.bound_tolerance; default 2e-7",
            "uniform_clipping_bound_tolerance_distribution": finite_summary(clipping_tolerances)}


def geometry_analysis(group):
    result = {}
    for name in ("base", "jome"):
        rows = []
        for r in group:
            geo = r.get("geometry", {}).get(name, {})
            vals = [scalar(geo.get(k)) for k in ("a", "b", "d")]
            if any(v is None for v in vals):
                continue
            a, b, d = vals
            actual = float(r["metrics"][name]["mse"])
            reconstructed = a+b-2*d
            rows.append((a, b, d, abs(actual-reconstructed), actual-reconstructed))
        if rows:
            x = np.asarray(rows)
            result[name] = {"n": len(rows), "a": finite_summary(x[:, 0]), "b": finite_summary(x[:, 1]),
                            "d": finite_summary(x[:, 2]), "mse_identity_abs_error": finite_summary(x[:, 3]),
                            "mse_identity_violations": int((x[:, 3] > 1e-8+1e-5*np.asarray(
                                [r["metrics"][name]["mse"] for r in group if all(scalar(r.get("geometry", {}).get(name, {}).get(k)) is not None for k in ("a", "b", "d"))])).sum())}
    return result


def reachability_analysis(group):
    rows, strict_violations, adjusted_violations = [], [], []
    for r in group:
        reach = r.get("reachability", {})
        vals = [scalar(reach.get(k)) for k in ("bound", "mse_jome_input", "affected_fraction")]
        if all(v is not None for v in vals):
            rows.append(vals)
            tolerance = scalar(reach.get("bound_tolerance"))
            tolerance = 1e-10 if tolerance is None else tolerance
            if vals[1] < vals[0]-tolerance:
                strict_violations.append(r["id"])
            adjusted = scalar(reach.get("roundoff_adjusted_bound"))
            if adjusted is not None and vals[1] < adjusted-tolerance:
                adjusted_violations.append(r["id"])
    if not rows:
        return {"n": 0, "status": "No reachability records"}
    bound, actual, fraction = np.asarray(rows).T
    slack = actual-bound
    return {"n": len(rows), "epsilon": .1, "positive_bound_images": int((bound > 0).sum()),
            "positive_bound_fraction": float((bound > 0).mean()),
            "bound": finite_summary(bound), "mse_jome_input": finite_summary(actual),
            "affected_rgb_fraction": finite_summary(fraction), "inequality_slack": finite_summary(slack),
            "inequality_violations": len(strict_violations), "strict_violation_ids": strict_violations,
            "roundoff_adjusted_inequality_violations": len(adjusted_violations),
            "roundoff_adjusted_violation_ids": adjusted_violations,
            "tolerance": "per-record bound_tolerance; default absolute 1e-10",
            "precision_diagnostics": diagnostic_summary(group, "reachability"),
            "interpretation": "Strict epsilon=0.1 bound is the stated proposition. An adjusted bound uses the measured native output envelope and is a separate numerical diagnostic, not evidence for the exact epsilon=0.1 envelope."}


def analyze(records, integrity, repeats):
    selection = [r for r in records if r["split_role"] == "selection"]
    grams_selection = np.asarray([r["_gram"] for r in selection])
    fits = {"four_expert": fit_simplex(grams_selection, 4), "six_candidate_extension": fit_simplex(grams_selection, 6)}
    best_expert = int(np.argmin(np.mean(grams_selection[:, :4, :4], axis=0).diagonal())) if selection else None
    fits["forced_expert"] = {"candidate": f"expert_{best_expert}" if best_expert is not None else None,
                              "selection_rule": "minimum arithmetic mean image MSE on selection only", "selection_n": len(selection)}
    rng, splits, tables = np.random.default_rng(RNG_SEED), {}, []
    for role in ROLES:
        group = [r for r in records if r["split_role"] == role]
        if not group:
            splits[role] = {"n": 0, "status": "No valid records"}
            continue
        mse = {name: np.asarray([r["metrics"][name]["mse"] for r in group]) for name in REQUIRED}
        grams = np.asarray([r["_gram"] for r in group])
        for fit_name, fit in fits.items():
            if fit_name == "forced_expert" or not fit.get("success"):
                continue
            w = np.zeros(6)
            w[:len(fit["weights"])] = fit["weights"]
            mse["fixed4" if fit_name == "four_expert" else "fixed6"] = np.maximum(np.einsum("i,nij,j->n", w, grams, w), 0)
        if best_expert is not None:
            mse["forced_expert"] = mse[f"expert_{best_expert}"]
        mse["oracle4"] = np.min(np.column_stack([mse[c] for c in CANDIDATES[:4]]), axis=1)
        mse["oracle6"] = np.min(np.column_stack([mse[c] for c in CANDIDATES]), axis=1)
        mse["oracle_input_base_jome"] = np.min(np.column_stack([mse[c] for c in ("input", "base", "jome")]), axis=1)
        methods = {name: {"mean_image_mse": float(value.mean()), "mean_psnr_db": float(psnr(value).mean())}
                   for name, value in mse.items()}
        comparisons = {}
        pairs = [("E1 matched residual weighting", "Yr", "Ur"), ("E1 matched output weighting", "Yo", "Uo"),
                 ("E1 adaptive clipping order", "Yr", "Yo"), ("E1 uniform clipping order", "Ur", "Uo"),
                 ("E1 residual adaptive versus uniform output", "Yr", "Uo"),
                 ("Native deployment versus matched FP32 operator", "jome", "Yr"),
                 ("E2 four adaptive versus fixed", "Yo", "fixed4"),
                 ("E2 four fixed versus uniform", "fixed4", "Uo"),
                 ("E2 four fixed versus forced", "fixed4", "forced_expert"),
                 ("E2 four adaptive versus forced", "Yo", "forced_expert"),
                 ("E2 six extension versus four fixed", "fixed6", "fixed4"),
                 ("E3 oracle {input,base,native JOME} versus deployed", "oracle_input_base_jome", "jome"),
                 ("E3 oracle {input,base,native JOME} versus input", "oracle_input_base_jome", "input"),
                 ("Capacity oracle {four clipped experts} versus output adaptive", "oracle4", "Yo"),
                 ("Capacity oracle {four clipped experts,base,input} versus deployed", "oracle6", "jome")]
        for label, left, right in pairs:
            if left in mse and right in mse:
                comparisons[label] = contrast(mse[left], mse[right], rng, repeats)
                comparisons[label].update({"left": left, "right": right})
        gain_interaction = (psnr(mse["Yr"])-psnr(mse["Ur"]))-(psnr(mse["Yo"])-psnr(mse["Uo"]))
        mse_interaction = (mse["Yr"]-mse["Ur"])-(mse["Yo"]-mse["Uo"])
        comparisons["E1 weighting-by-clipping interaction"] = gain_summary(gain_interaction, mse_interaction, rng, repeats)
        for name in ("base", "jome", "Yr", "Yo", "Ur", "Uo", "fixed4", "fixed6", "forced_expert"):
            if name in mse:
                for reference in ("input", "base"):
                    if name != reference:
                        comparisons[f"Preservation {name} versus {reference}"] = contrast(mse[name], mse[reference], rng, repeats)
        for label, comp in comparisons.items():
            row = {"split_role": role, "comparison": label}
            row.update({k: v for k, v in comp.items() if not isinstance(v, (dict, list))})
            row.update({"psnr_ci95_low": comp["psnr_gain_ci95"][0], "psnr_ci95_high": comp["psnr_gain_ci95"][1],
                        "mse_ci95_low": comp["mse_delta_ci95"][0], "mse_ci95_high": comp["mse_delta_ci95"][1]})
            tables.append(row)
        splits[role] = {"n": len(group), "methods": methods, "comparisons": comparisons,
                        "reachability": reachability_analysis(group), "geometry": geometry_analysis(group),
                        "clipping": diagnostic_summary(group, "clipping"), "replay": diagnostic_summary(group, "replay"),
                        "artifact_verification": verification_flags(group),
                        "forward_seconds": finite_summary([r.get("forward_seconds") for r in group]),
                        "record_seconds": finite_summary([r.get("record_seconds") for r in group]),
                        "finite_action_oracle": {"definition": "Per-image minimum reference MSE among actual candidates; unavailable target-dependent action at deployment.",
                                                 "primary_E3_candidates": ["input", "base", "jome"],
                                                 "primary_E3_method": "oracle_input_base_jome",
                                                 "four_expert_capacity_candidates": CANDIDATES[:4],
                                                 "six_candidate_capacity_candidates": CANDIDATES},
                        "paired_loss_transitions_vs_input": {"matched_residual_weighting": loss_transitions(mse["Yr"], mse["Ur"], mse["input"]),
                                                              "matched_output_weighting": loss_transitions(mse["Yo"], mse["Uo"], mse["input"])}}
    return {"analysis_version": 1, "integrity": integrity, "fitted_mixtures": fits,
            "bootstrap": {"seed": RNG_SEED, "repeats": repeats, "interval": "paired image percentile 95%",
                          "scope": "exploratory conditional intervals; fitted weights held fixed; no calibration uncertainty or model-seed expansion"},
            "methods": {"mse_aggregation": "arithmetic mean image MSE, equal image weights", "psnr": "mean of -10 log10(per-image MSE), unit RGB peak; numerical MSE floor 1e-15",
                        "gram_candidate_order": CANDIDATES, "fit": "SLSQP simplex constrained convex quadratic, selection only; ftol=1e-14, maxiter=2000; independent convex dual-gap check <=1e-7 on scale-normalized objective",
                        "geometry_identity": "MSE(output,T)=a+b-2d; a=MSE(I,T), b=MSE(output,I), d=mean((T-I)*(output-I))",
                        "primary_E1_comparison": "Yr versus Ur; Yo versus Uo is second matched operator contrast",
                        "E2_scope": "Only global mixtures of individually clipped output candidates; output Gram does not optimize residual-then-clip mixtures.",
                        "operator_precision": "E1 Ur/Uo/Yr/Yo are matched FP32 arithmetic on fixed BF16-forward residuals and native-dtype interpolated router weights. jome is the native deployed arithmetic output; parity is reported separately.",
                        "E3_reachability": "mean(max(abs(B-I)-0.1,0)^2) <= MSE(JOME,I)"},
            "cautions": ["The 300-image historical model-selection partition is reused to calibrate global mixtures and select the forced expert; selection results are development performance.",
                         "Official and locked partitions are existing retrospective evidence; E4 independent-dataset evaluation has not been done.",
                         "Scene IDs are unavailable. Image-level bootstrap intervals are exploratory and may be too narrow when images share scenes; no multiplicity correction.",
                         "Weights are fitted for mean image MSE, not mean PSNR. They remain unchanged on official and locked splits.",
                         "The six-candidate extension adds base and input and changes the action set; gains cannot be attributed solely to weighting.",
                         "Native tensors are not retained in this compact record. Gram replay supports constant output-space mixtures only; spatial weights and residual clipping require richer artifacts.",
                         "Device, precision, and operator arithmetic can cause small differences from historical archived PSNR; interpret native results under the recorded evaluation precision and inspect any parity diagnostics.",
                         "Finite-action oracles use reference targets and are descriptive capacity bounds, not deployable selectors.",
                         "Timing is a single recorded execution per image, including record construction separately; it is not a repeated latency benchmark."],
            "splits": splits}, tables


def loss_transitions(left, right, input_mse):
    l = psnr(left)-psnr(input_mse) < -.1
    r = psnr(right)-psnr(input_mse) < -.1
    return {"loss_threshold_db": .1, "rescued_loss_count": int((r & ~l).sum()),
            "introduced_loss_count": int((~r & l).sum()), "both_loss_count": int((r & l).sum()),
            "neither_loss_count": int((~r & ~l).sum())}


def render_report(summary):
    esc = lambda x: html.escape(str(x))
    fmt = lambda x: "—" if x is None else f"{x:.6g}" if isinstance(x, (int, float)) else esc(x)
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Frozen-model E1–E3 priority analysis</title>',
             '<style>body{font:16px/1.55 system-ui,sans-serif;color:#173044;max-width:1200px;margin:32px auto;padding:0 24px}h1,h2,h3{line-height:1.25}table{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}th,td{padding:8px;text-align:right;border-bottom:1px solid #d8e1e8}th:first-child,td:first-child{text-align:left}thead{background:#eaf1f5}.scroll{overflow:auto}.caution{background:#fff5dc;padding:16px;border-left:4px solid #b57c13}code{font-size:13px}pre{overflow:auto;background:#eff4f7;padding:12px}a{color:#12608b}</style>',
             '<h1>Frozen-model E1–E3 priority analysis</h1><p>Matched aggregation, selection-calibrated convex mixtures, and empirical input reachability. <a href="summary.json">Full summary JSON</a> · <a href="tables.csv">Paired comparison CSV</a></p>']
    parts.append('<h2>Split overview</h2><p>Each image has equal weight. Gains are paired mean PSNR differences in dB; positive values favor the first named method. Intervals resample images with fitted weights held fixed.</p><div class="scroll"><table><thead><tr><th>Split</th><th>Images</th><th>Native JOME PSNR</th><th>Base PSNR</th><th>Adaptive − uniform residual mix<br>gain [95% CI]</th><th>Fixed four-expert − adaptive output mix<br>gain [95% CI]</th><th>Positive input-reachability bound</th></tr></thead><tbody>')
    for role in ('official', 'locked', 'selection'):
        split = summary['splits'].get(role, {'n': 0})
        methods, comps = split.get('methods', {}), split.get('comparisons', {})
        matched = comps.get('E1 matched residual weighting')
        adaptive_fixed = comps.get('E2 four adaptive versus fixed')
        def gain_cell(comp, reverse=False):
            if comp is None:
                return '—'
            gain, ci = comp['mean_psnr_gain_db'], comp['psnr_gain_ci95']
            if reverse:
                gain, ci = -gain, [-ci[1], -ci[0]]
            return f'{gain:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]'
        reach = split.get('reachability', {})
        bound_cell = (f'{reach["positive_bound_images"]}/{reach["n"]} ({100*reach["positive_bound_fraction"]:.1f}%)'
                      if reach.get('n') else '—')
        role_label = {'official': 'Official · retrospective', 'locked': 'Locked · retrospective',
                      'selection': 'Selection · reused development'}[role]
        parts.append(f'<tr><td>{esc(role_label)}</td><td>{split["n"]}</td><td>{fmt(methods.get("jome", {}).get("mean_psnr_db"))}</td><td>{fmt(methods.get("base", {}).get("mean_psnr_db"))}</td><td>{gain_cell(matched)}</td><td>{gain_cell(adaptive_fixed, True)}</td><td>{bound_cell}</td></tr>')
    parts.append('</tbody></table></div><p>The residual comparison holds clipping order fixed. The output comparison holds the four clipped experts fixed. A positive reachability bound concerns the exact amplitude limit; numerical envelope checks appear below.</p>')
    notes = ['Positive reachability counts identify images where the original input lies outside the exact amplitude-limited base neighborhood. They do not establish that an image was damaged or that the reference prefers the input.',
             'The calibrated fixed mixture minimizes mean image MSE. Mean PSNR averages a logarithmic score and can rank the adaptive and fixed methods differently.']
    clipping_effects = [abs(v['mean_psnr_gain_db']) for split in summary['splits'].values()
                        for label,v in split.get('comparisons', {}).items()
                        if label in ('E1 adaptive clipping order', 'E1 uniform clipping order')]
    if clipping_effects and max(clipping_effects) < 1e-4:
        notes.append(f'The largest absolute mean clipping-order PSNR effect across the recorded splits is {max(clipping_effects):.3g} dB, a practically tiny aggregate effect under this frozen execution.')
    fit4 = summary['fitted_mixtures'].get('four_expert', {})
    if fit4.get('success') and max(fit4['weights']) > .999:
        expert = int(np.argmax(fit4['weights']))
        notes.append(f'The four-expert MSE-optimal calibration places essentially all weight on expert_{expert} (expert {expert+1} in one-based numbering); this is a selection-calibrated fixed action, not a newly trained expert.')
    parts.append('<h2>Reading the measurements</h2><ul>'+''.join(f'<li>{esc(note)}</li>' for note in notes)+'</ul>')
    if summary.get('archive_parity', {}).get('contents', {}).get('splits'):
        parity = summary['archive_parity']['contents']['splits']
        parts.append('<h2>Parity with historical archived outputs</h2><p>These are measurements from a new runtime execution of the supplied checkpoint identities. They are not a bitwise reproduction of the historical execution. The table compares native JOME PSNR with archived PSNR on matched image IDs.</p><table><thead><tr><th>Split</th><th>Images</th><th>Mean absolute Δ PSNR (dB)</th><th>Maximum absolute Δ PSNR (dB)</th></tr></thead><tbody>')
        for role in ('official', 'locked', 'selection'):
            if role in parity:
                row = parity[role]
                parts.append(f'<tr><td>{esc(role)}</td><td>{row.get("n", "—")}</td><td>{fmt(row.get("jome", {}).get("mean_absolute_delta_db"))}</td><td>{fmt(row.get("jome", {}).get("max_absolute_delta_db"))}</td></tr>')
        parts.append('</tbody></table><p>Source: <code>'+esc(summary['archive_parity']['path'])+'</code>.</p>')
    parts.append('<details><summary><strong>Operator legend</strong></summary><table><thead><tr><th>Record name</th><th>Operator or action</th></tr></thead><tbody>'+''.join(f'<tr><td><code>{esc(k)}</code></td><td>{esc(v)}</td></tr>' for k,v in METHOD_LABELS.items())+'</tbody></table><p>The four matched aggregation operators use FP32 arithmetic on frozen forward-pass residuals and interpolated router weights. Native JOME preserves deployed arithmetic. Fixed mixtures are global output mixtures calibrated only on selection data.</p></details>')
    if summary.get('analysis_invocation'):
        parts.append('<details><summary><strong>Reproduction command and evaluation metadata</strong></summary><pre>'+esc(summary['analysis_invocation'])+'</pre><pre>'+esc(json.dumps(summary.get('evaluation_metadata', {'status': 'No metadata.json found beside the records or analysis outputs.'}), indent=2))+'</pre></details>')
    parts.extend([
             '<div class="caution"><strong>Interpretation limits</strong><ul>'+''.join(f'<li>{esc(x)}</li>' for x in summary['cautions'])+'</ul></div>',
             '<h2>Data integrity</h2><pre>'+esc(json.dumps(summary['integrity'], indent=2))+'</pre>',
             '<h2>Methods and fixed weights</h2><ul>'+''.join(f'<li><strong>{esc(k)}:</strong> {esc(v)}</li>' for k,v in summary['methods'].items())+'</ul>',
             '<p>'+esc(json.dumps(summary['bootstrap']))+'</p><pre>'+esc(json.dumps(summary['fitted_mixtures'], indent=2))+'</pre>'])
    for role in ('official', 'locked', 'selection'):
        split = summary['splits'][role]
        parts.append(f'<h2>{esc(role)} · {split["n"]} images</h2>')
        if not split['n']:
            parts.append('<p>No valid records.</p>')
            continue
        parts.append('<h3>Method averages</h3><table><thead><tr><th>Method</th><th>Mean image MSE</th><th>Mean PSNR (dB)</th></tr></thead><tbody>')
        parts.extend(f'<tr><td>{esc(METHOD_LABELS.get(k, k))} <code>({esc(k)})</code></td><td>{fmt(v["mean_image_mse"])}</td><td>{fmt(v["mean_psnr_db"])}</td></tr>' for k,v in split['methods'].items())
        parts.append('</tbody></table><h3>Paired comparisons</h3><p>Positive PSNR gain favors the left method; negative MSE change favors the left method. Negative burden averages max(−gain,0) over every image.</p><div class="scroll"><table><thead><tr><th>Comparison</th><th>Δ PSNR</th><th>95% image CI</th><th>Δ MSE</th><th>Wins / losses</th><th>Loss &gt;0.1 dB</th><th>Negative burden (dB)</th></tr></thead><tbody>')
        for label, v in split['comparisons'].items():
            ci = ' to '.join(fmt(x) for x in v['psnr_gain_ci95'])
            parts.append(f'<tr><td>{esc(label)}</td><td>{fmt(v["mean_psnr_gain_db"])}</td><td>{ci}</td><td>{fmt(v["mean_mse_delta"])}</td><td>{v["wins"]} / {v["losses"]}</td><td>{v["loss_gt_0_1_db_count"]} ({100*v["loss_gt_0_1_db_fraction"]:.1f}%)</td><td>{fmt(v["negative_burden_db_per_image"])}</td></tr>')
        parts.append('</tbody></table></div>')
        for name in ('reachability', 'geometry', 'clipping', 'replay', 'artifact_verification', 'paired_loss_transitions_vs_input', 'forward_seconds', 'record_seconds'):
            parts.append(f'<details><summary><strong>{esc(name)}</strong></summary><pre>{esc(json.dumps(split[name], indent=2))}</pre></details>')
    return '\n'.join(parts) + '</html>\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap-repeats', type=int, default=5000)
    parser.add_argument('--manifest', type=Path, help='Authoritative CSV with id,split_role; otherwise discover beside input or its parent.')
    parser.add_argument('--expected-selection', type=int)
    parser.add_argument('--expected-official', type=int)
    parser.add_argument('--expected-locked', type=int)
    args = parser.parse_args()
    if args.bootstrap_repeats < 100:
        parser.error('--bootstrap-repeats must be at least 100')
    if not args.input.is_file():
        parser.error(f'Input missing: {args.input}. Run evaluation first; this script performs no inference.')
    manifest = args.manifest
    if manifest is None:
        manifest = next((p for p in (args.input.parent/'manifest.csv', args.input.parent.parent/'manifest.csv') if p.is_file()), None)
    expected_ids, manifest_counts = None, {}
    if manifest is not None:
        if not manifest.is_file():
            parser.error(f'Manifest missing: {manifest}')
        expected_ids = defaultdict(set)
        with manifest.open(newline='') as stream:
            for row in csv.DictReader(stream):
                if row.get('split_role') not in ROLES or not row.get('id'):
                    parser.error('Manifest requires nonempty id and recognized split_role on every row.')
                if row['id'] in expected_ids[row['split_role']]:
                    parser.error(f'Duplicate manifest id: {row["id"]}')
                expected_ids[row['split_role']].add(row['id'])
        manifest_counts = {role: len(expected_ids[role]) for role in ROLES}
    defaults = {'selection': 300, 'official': 300, 'locked': 200}
    expected = {role: getattr(args, f'expected_{role}') if getattr(args, f'expected_{role}') is not None else manifest_counts.get(role, defaults[role]) for role in ROLES}
    records, integrity = read_records(args.input, expected, expected_ids)
    integrity['manifest_path'] = str(manifest.resolve()) if manifest is not None else None
    summary, rows = analyze(records, integrity, args.bootstrap_repeats)
    summary['source'] = {'path': str(args.input.resolve()), 'sha256': hashlib.sha256(args.input.read_bytes()).hexdigest()}
    summary['analysis_invocation'] = shlex.join([sys.executable, *sys.argv])
    metadata_path = next((p for p in (args.input.parent/'metadata.json', args.output/'metadata.json') if p.is_file()), None)
    if metadata_path is not None:
        metadata_bytes = metadata_path.read_bytes()
        summary['evaluation_metadata'] = {'path': str(metadata_path.resolve()),
                                          'sha256': hashlib.sha256(metadata_bytes).hexdigest()}
        try:
            summary['evaluation_metadata']['contents'] = json.loads(metadata_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            summary['evaluation_metadata']['read_error'] = str(exc)
    parity_path = next((p for p in (args.input.parent/'archive_parity.json', args.output/'archive_parity.json') if p.is_file()), None)
    if parity_path is not None:
        parity_bytes = parity_path.read_bytes()
        summary['archive_parity'] = {'path': str(parity_path.resolve()),
                                    'sha256': hashlib.sha256(parity_bytes).hexdigest()}
        try:
            summary['archive_parity']['contents'] = json.loads(parity_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            summary['archive_parity']['read_error'] = str(exc)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['split_role','comparison']
    with (args.output/'tables.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output/'report.html').write_text(render_report(summary))
    print(json.dumps({'valid_records': len(records), 'integrity_passed': integrity['passed'],
                      'fit4_verified': summary['fitted_mixtures']['four_expert']['success'],
                      'fit6_verified': summary['fitted_mixtures']['six_candidate_extension']['success'],
                      'report': str((args.output/'report.html').resolve())}))
    if not integrity['passed']:
        print('WARNING: partial/integrity-limited analysis; inspect report and summary.json.', file=sys.stderr)


if __name__ == '__main__':
    main()
