#!/usr/bin/env python3
"""Replay retained evidence; write eight numeric tables and five quantitative plots.

No network inference is performed. Mean-PSNR fitting is an independently
implemented offline reconstruction; it is nonconvex and has no global guarantee.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from analyze_operators import (CANDIDATES, RNG_SEED, ROLES, analyze,
                              bootstrap_mean, contrast, fit_simplex, psnr,
                              read_records)

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, data):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def verify_reference_tables(tables):
    """Compare every reference value at its declared rounding precision."""
    reference = json.loads((ROOT / "data/reference_tables.json").read_text())
    if reference["schema_version"] != 1 or reference["numeric_values"] != 208:
        raise ValueError("Expected reference schema 1 with 208 numeric values")
    if set(reference["tables"]) != set(tables):
        raise ValueError("Calculated and reference table sets differ")
    checks = 0
    for name, specification in reference["tables"].items():
        row_key = specification["row_key"]
        actual = {str(row[row_key]): row for row in tables[name]}
        expected = specification["rows"]
        if (len(actual) != len(tables[name]) or len(expected) != len(actual)
                or set(actual) != {row["id"] for row in expected}):
            raise ValueError(f"Calculated and reference row identities differ: {name}")
        columns, decimals = specification["columns"], specification["decimals"]
        if len(columns) != len(decimals):
            raise ValueError(f"Reference column/precision mismatch: {name}")
        for row in expected:
            if len(row["values"]) != len(columns):
                raise ValueError(f"Reference value count mismatch: {name}/{row['id']}")
            for column, value, precision in zip(columns, row["values"], decimals):
                observed = float(actual[row["id"]][column])
                if not abs(observed - value) <= .5001 * 10 ** (-precision):
                    raise ValueError((name, row["id"], column, observed, value, precision))
                checks += 1
    if checks != 208:
        raise ValueError(f"Expected 208 numeric comparisons, got {checks}")
    return checks


def psnr_fit(grams, n):
    """Deterministic multistart SLSQP of mean log-MSE on selection only."""
    g = grams[:, :n, :n]
    scale = np.max(np.mean(g, axis=0))
    g = g / scale

    def fun(w):
        m = np.maximum(np.einsum("i,nij,j->n", w, g, w), 1e-15)
        return float(np.log(m).mean())

    def jac(w):
        m = np.maximum(np.einsum("i,nij,j->n", w, g, w), 1e-15)
        return np.mean(2 * np.einsum("nij,j->ni", g, w) / m[:, None], axis=0)

    starts = [np.ones(n) / n, *np.eye(n)]
    rng = np.random.default_rng(RNG_SEED)
    starts.extend(rng.dirichlet(np.ones(n), size=32))
    fits = [minimize(fun, w, jac=jac, method="SLSQP", bounds=[(0, 1)] * n,
                     constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                                  "jac": lambda w: np.ones(n)},
                     options={"ftol": 1e-13, "maxiter": 2000}) for w in starts]
    valid = [x for x in fits if x.success and abs(x.x.sum()-1) < 1e-8]
    if not valid:
        raise RuntimeError("No feasible successful PSNR calibration")
    best = min(valid, key=lambda x: x.fun)
    return {"weights": best.x.tolist(), "selection_n": len(g),
            "objective": "mean PSNR", "successful_starts": len(valid),
            "attempted_starts": len(starts), "solver": "SLSQP multistart",
            "global_optimality_claimed": False}


def geometry_rows(records, released):
    result = []
    for r in records:
        if r["split_role"] != "official":
            continue
        for method in ("base", "jome"):
            g = r["geometry"][method]
            result.append({"id": r["id"], "model": method, **g})
    for r in released:
        a, b = float(r["required_edit_mse"]), float(r["correction_mse"])
        # d recovered from independently retained alignment and edit norms.
        d = float(r["alignment_cosine"]) * np.sqrt(a*b)
        result.append({"id": r["id"], "model": r["model"], "a": a, "b": b, "d": d})
    return result


def make_plots(out, groups, arrays, fits, geo, contrasts):
    os.environ.setdefault("MPLCONFIGDIR", str(out/".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})

    def save(fig, number):
        fig.tight_layout()
        fig.savefig(out/f"Fig{number}_replayed.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    labels = ["Yr − Ur", "Yo − F4 MSE", "Yo − F4 PSNR", "Yo − F6 MSE", "Yo − F6 PSNR"]
    for offset, role in [(-.12, "official"), (.12, "locked")]:
        c = [x for x in contrasts if x["split_role"] == role]
        means = np.array([x["mean_psnr_gain_db"] for x in c])
        lo = np.array([x["ci_low"] for x in c]); hi = np.array([x["ci_high"] for x in c])
        ax.errorbar(means, np.arange(5)+offset, xerr=[means-lo, hi-means], fmt="o", capsize=3, label=role)
    ax.set(yticks=np.arange(5), yticklabels=labels, xlabel="Mean paired PSNR gain (dB)")
    ax.axvline(0, color=".6", lw=.7); ax.legend(); save(fig, 1)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    threshold = np.linspace(0, 1, 101)
    arr = arrays["official"]
    for m in ("Ur", "Yr", "F4M", "F4P"):
        gains = psnr(arr[m])-psnr(arr["base"])
        ax.plot(threshold, [(gains < -x).sum() for x in threshold], label=m)
    ax.set(xlabel="Base-relative loss threshold (dB)", ylabel="Number of images"); ax.legend(); save(fig, 2)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = psnr(arr["jome"])-psnr(arr["base"])
    y = psnr(arr["jome"])-psnr(arr["input"])
    near = psnr(arr["input"]) >= 40
    ax.scatter(x[~near], y[~near], s=10, alpha=.6)
    ax.scatter(x[near], y[near], marker="x", color="darkred", s=35)
    ax.axvline(0, color=".5", lw=.7); ax.axhline(0, color=".5", lw=.7)
    ax.set(xlabel="JOME gain over base (dB)", ylabel="JOME gain over input (dB)"); save(fig, 3)

    bins = [-np.inf, 25, 30, 35, 40, np.inf]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for m in ("base", "jome", "1_RRay", "6_KLETech-CEVI"):
        g = [r for r in geo if r["model"] == m]
        if not g:
            continue
        inp = psnr([r["a"] for r in g]); gain = psnr([r["a"]+r["b"]-2*r["d"] for r in g])-inp
        vals = [gain[(inp >= low)&(inp < high)].mean() for low, high in zip(bins[:-1],bins[1:])]
        ax.plot(range(5), vals, "o-", label=m)
    ax.set(xticks=range(5), xticklabels=["<25", "25–30", "30–35", "35–40", "≥40"],
           xlabel="Reference-defined input PSNR (dB)", ylabel="Mean gain over input (dB)")
    ax.legend(); save(fig, 5)

    fig, ax = plt.subplots(figsize=(6, 4))
    for m in ("jome", "1_RRay", "6_KLETech-CEVI"):
        g = [r for r in geo if r["model"] == m]
        if not g:
            continue
        a,b,d = np.array([[r[k] for k in ("a", "b", "d")] for r in g]).T
        near = psnr(a)>=40
        lam = np.linspace(0, 1, 101)
        gain = psnr(a[None, :]+lam[:, None]**2*b-2*lam[:, None]*d)-psnr(a)
        line, = ax.plot(gain.mean(axis=1), gain[:, near].mean(axis=1), label=m)
        for t in (.1, .25, 1):
            j = int(round(t*100)); ax.plot(gain[j].mean(), gain[j,near].mean(), "o", color=line.get_color(), ms=4)
    ax.set(xlabel="Complete-set mean gain (dB)", ylabel="Near-identity mean gain (dB)")
    ax.legend(); save(fig, 6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest = rows(ROOT/"data/manifest.csv")
    expected_ids = {role: {r["id"] for r in manifest if r["split_role"] == role} for role in ROLES}
    records, integrity = read_records(ROOT/"data/per_image.jsonl", {"selection":300,"official":300,"locked":200}, expected_ids)
    if not integrity["passed"]:
        raise ValueError(json.dumps(integrity))
    summary, _ = analyze(records, integrity, 5000)
    groups = {role: [r for r in records if r["split_role"] == role] for role in ROLES}
    selection_g = np.array([r["_gram"] for r in groups["selection"]])
    fits = {"F4M": fit_simplex(selection_g,4), "F6M": fit_simplex(selection_g,6),
            "F4P": psnr_fit(selection_g,4), "F6P": psnr_fit(selection_g,6)}
    if not all(fits[m]["success"] for m in ("F4M","F6M")):
        raise RuntimeError("MSE calibration failed its independent convex optimality check")
    arrays = {}
    for role, group in groups.items():
        arr = {m: np.array([r["metrics"][m]["mse"] for r in group]) for m in ("input","base","jome","Ur","Yr","Uo","Yo")}
        grams = np.array([r["_gram"] for r in group])
        for m, fit in fits.items():
            w = np.zeros(6);w[:len(fit["weights"])] = fit["weights"]
            arr[m] = np.einsum("i,nij,j->n",w,grams,w)
        arrays[role]=arr
    tables = {}
    historical=rows(ROOT/"data/run_metrics.csv")
    base=next(r for r in historical if r["model"]=="PAST" and r["split"]=="official")
    wanted={"ARISE":"ARISE","M":"Spectral (M)","N":"Phase context (N)","Asymmetric":"Phase-guided","Dual":"Dual-scale","JOME":"JOME"}
    tables["fixed_backbone"]=[{"method":wanted[r["model"]], "parameters_m":float(r["parameters_total"])/1e6,
        "raw_gain_db":float(r["gain_mean"]), "jpeg_gain_db":float(r["jpeg100_psnr"])-float(base["jpeg100_psnr"]),
        "losses_gt_0_1_db":int(float(r["gain_harm_0_1"])), "ms_image":1000*float(r["inference_seconds"])}
        for r in historical if r["split"]=="official" and r["model"] in wanted]
    fixed_order=["ARISE","Spectral (M)","Phase context (N)","Phase-guided","Dual-scale","JOME"]
    tables["fixed_backbone"].sort(key=lambda r:fixed_order.index(r["method"]))
    tables["operators"]=[];tables["quadrants"]=[];tables["reachability"]=[]
    for role_index,role in enumerate(ROLES):
        s=summary["splits"][role];a=arrays[role];c=s["comparisons"]["matched residual weighting"]
        ci=bootstrap_mean(psnr(a["Yr"])-psnr(a["Ur"]),np.random.default_rng(RNG_SEED+role_index),5000)
        tables["operators"].append({"split":role,"n":s["n"],"Ur_minus_B":float((psnr(a["Ur"])-psnr(a["base"])).mean()),
            "Yr_minus_B":float((psnr(a["Yr"])-psnr(a["base"])).mean()),"Yr_minus_Ur":c["mean_psnr_gain_db"],"ci_low":ci[0],"ci_high":ci[1]})
        b=psnr(a["jome"])-psnr(a["base"]);i=psnr(a["jome"])-psnr(a["input"])
        tables["quadrants"].append({"split":role,"both":int(((b>0)&(i>0)).sum()),"base_only":int(((b>0)&(i<0)).sum()),"input_only":int(((b<0)&(i>0)).sum()),"neither":int(((b<0)&(i<0)).sum())})
        bounds=np.array([r["reachability"]["bound"] for r in groups[role]])
        tables["reachability"].append({"split":role,"n":s["n"],"active":int((bounds>0).sum()),"mean_bound_x1e4":float(bounds.mean()*1e4),
            "rgb_percent":float(np.mean([r["reachability"]["affected_fraction"] for r in groups[role]])*100),
            "input_loss_n":int((i<0).sum()),"active_input_loss":int(((i<0)&(bounds>0)).sum())})
    tables["weights"]=[{"mixture":m,**{name:float(w) for name,w in zip(CANDIDATES,[*fit["weights"],*([0]*(6-len(fit["weights"])))])}} for m,fit in fits.items()]
    tables["calibrated_scores"]=[{"method":m,**{f"{role}_{metric}":float(psnr(arrays[role][m]).mean()) if metric=="psnr_db" else float(arrays[role][m].mean()*1e4) for role in ("official","locked") for metric in ("psnr_db","mse_x1e4")}} for m in ("base","Ur","Yr","Uo","Yo","F4M","F4P","F6M","F6P")]
    tables["capacity"]=rows(ROOT/"data/capacity_context.csv")
    geo=geometry_rows(records,rows(ROOT/"data/released_geometry.csv"))
    tables["geometry"]=[]
    for m in dict.fromkeys(r["model"] for r in geo):
        g=[r for r in geo if r["model"]==m and psnr(r["a"])>=40]
        a,b,d=np.array([[r[k] for k in ("a","b","d")] for r in g]).T
        q=b-2*d;gain=psnr(a+q)-psnr(a)
        tables["geometry"].append({"model":m,"n":len(g),"mean_gain_db":float(gain.mean()),"losses":int((gain<0).sum()),"positive_alignment":int((d>0).sum()),
            "median_lambda_star":float(np.median(np.clip(d/b,0,1))),"median_eta_x255":float(np.median(np.abs(q)/(2*np.sqrt(b)))*255)})
    contrasts=[]
    for role in ("official","locked"):
        for left,right in [("Yr","Ur"),("Yo","F4M"),("Yo","F4P"),("Yo","F6M"),("Yo","F6P")]:
            rng=np.random.default_rng(RNG_SEED+ROLES.index(role))
            c=contrast(arrays[role][left],arrays[role][right],rng,5000)
            contrasts.append({"split_role":role,"comparison":f"{left}-{right}","mean_psnr_gain_db":c["mean_psnr_gain_db"],"ci_low":c["psnr_gain_ci95"][0],"ci_high":c["psnr_gain_ci95"][1]})
    # Check the independently reconstructed PSNR fits against reference precision.
    assert np.allclose(fits["F4P"]["weights"],[.0779,.4345,.4876,0],atol=6e-5)
    assert np.allclose(fits["F6P"]["weights"],[.0182,.5819,.3739,0,0,.0261],atol=6e-5)
    checked_values=verify_reference_tables(tables)
    for t,data in tables.items():write_csv(out/f"{t}.csv",data)
    make_plots(out,groups,arrays,fits,geo,contrasts)
    result={"integrity":integrity,"reference_numeric_values_checked":checked_values,"calibration":fits,"tables":tables,"contrasts":contrasts,
        "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"data").iterdir() if p.is_file()},
        "coverage":"eight numeric reference tables and quantitative data for Figures 1,2,3,5,6; figure artwork is not included",
        "recorded_matched_operator_intervals":{role:summary["splits"][role]["comparisons"]["matched residual weighting"]["psnr_gain_ci95"] for role in ROLES},
        "interval_note":"Recorded analysis intervals differ from the later manuscript (official upper endpoint 0.015774 versus displayed 0.0150). The documented manuscript protocol resets the RNG to seed 20261013 plus role index (selection=0, official=1, locked=2) for every contrast; 5000 paired resamples. This implementation of the documented recipe reproduces all displayed Table 3 intervals and the textual F4P intervals. The retained sufficient statistics support an independent reconstruction of the reported values."}
    (out/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"records":len(records),"tables":len(tables),"quantitative_figures":5,"reference_values_checked":checked_values,"integrity_passed":True,"psnr_weights_match_reference":True}))


if __name__=="__main__":main()
