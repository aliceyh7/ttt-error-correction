#!/usr/bin/env python3
"""Layer-bootstrap CI for Spearman rho between CV(gamma^2) and the live
error-correction fraction (paper Fig. 3 right / Sec. 4.2).

Reads the JSON written by run_ablation_ladder.py or run_delta_probe.py
(key `delta_term_per_layer`, one entry per layer with `delta_fraction_mean`
and `gamma_cv2`).

    python3 bootstrap_spearman.py results/ladder_linear760m/ablation_results_linear760m.json
    python3 bootstrap_spearman.py results/probe_linear-1.3b.json --out results/spearman_1.3b.json
"""
import argparse
import json
import numpy as np
from scipy.stats import rankdata


def spearman(x, y):
    rx = rankdata(x, method="average").astype(float)
    ry = rankdata(y, method="average").astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def boot_ci(x, y, n_boot=10_000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(x); rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(idx)) < 3:
            continue
        rhos.append(spearman(x[idx], y[idx]))
    rhos = np.array(rhos)
    if len(rhos) == 0:
        return spearman(x, y), float("nan"), float("nan"), 0
    lo, hi = np.percentile(rhos, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return spearman(x, y), float(lo), float(hi), int(len(rhos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()
    d = json.load(open(args.json_path))
    per = d["delta_term_per_layer"]
    layers = sorted(per, key=int)
    cv = np.array([per[l]["gamma_cv2"] for l in layers])
    df = np.array([per[l]["delta_fraction_mean"] for l in layers])
    out = {"source": args.json_path, "n_layers": len(layers)}
    r, lo, hi, nb = boot_ci(cv, df, args.n_boot)
    out["all_layers"] = {"rho": r, "ci95": [lo, hi], "n_boot": nb}
    print(f"all {len(layers)} layers:   rho = {r:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    # drop the extreme layer (max CV); in the paper this is layer 0
    k = int(np.argmax(cv))
    m = np.ones(len(cv), bool); m[k] = False
    r2, lo2, hi2, nb2 = boot_ci(cv[m], df[m], args.n_boot)
    out["excluding_max_cv_layer"] = {"dropped_layer": int(layers[k]), "rho": r2,
                                     "ci95": [lo2, hi2], "n_boot": nb2}
    print(f"excluding layer {layers[k]} (max CV): rho = {r2:.3f}   95% CI [{lo2:.3f}, {hi2:.3f}]")
    # extremes coincide?
    out["argmax_coincide"] = bool(np.argmax(cv) == np.argmax(df))
    out["argmin_coincide"] = bool(np.argmin(cv) == np.argmin(df))
    print(f"extremes coincide: max {out['argmax_coincide']}, min {out['argmin_coincide']}")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
