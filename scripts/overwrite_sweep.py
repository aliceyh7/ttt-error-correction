"""E2: explicit overwrite task, swept over the effective step eta*||k||^2.

Reimplementation of the Sec. 4.6 overwrite task from its description in the paper:
  * dimension d=128, W0 = 0
  * one underlying unit key k; the two writes use noisy copies
        k_i = normalize(k + sigma * g_i),  g_i ~ N(0, I_d)
  * values v_old, v_new ~ random unit vectors
  * write 1 binds k_1 -> v_old, write 2 binds k_2 -> v_new (r times, for the repeated-write panel)
  * query at the underlying key k
  * margin = cos(W k, v_new) - cos(W k, v_old)
  * "additive"/Hebbian:  W += eta * v k^T
    "mse"/delta rule:    W += eta * (v - W k) k^T
Keys are unit norm, so eta == eta*||k||^2.

Validation: at eta=1 this should reproduce the paper's margins
  noise 0/0.02/0.05/0.10/0.20 -> 1.004, 0.952, 0.742, 0.380, 0.130 (MSE), |m|<0.003 (additive).
If it does, the task is the same one; if not, the convention differs and the sweep must be
rerun with the original code instead.

Usage:  python overwrite_sweep.py --out figdir
        python overwrite_sweep.py --out figdir --equal-counts   # Fig. 5(c) data only

`--equal-counts` writes overwrite_equal_counts.json: r writes of v_old followed by r writes
of v_new (equal counts, so vote-counting cannot favor either value), every write with its
own noisy key copy at sigma = 0.05, 400 trials per cell, eta in {0.04, 0.01, 0.004, 0.001},
r in {1, 2, 4, ..., 128}.
"""
import argparse
import json
import numpy as np

D = 128


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def run_trials(rule, eta, sigma, n_trials, r_new=1, seed=0):
    """Return per-trial margins for one (rule, eta, sigma, r_new) cell."""
    rng = np.random.default_rng(seed)
    margins = np.empty(n_trials)
    for t in range(n_trials):
        k = unit(rng.standard_normal(D))
        k1 = unit(k + sigma * rng.standard_normal(D))
        k2 = unit(k + sigma * rng.standard_normal(D))
        v_old = unit(rng.standard_normal(D))
        v_new = unit(rng.standard_normal(D))
        W = np.zeros((D, D))
        writes = [(k1, v_old)] + [(k2, v_new)] * r_new
        for kk, vv in writes:
            if rule == "additive":
                W += eta * np.outer(vv, kk)
            elif rule == "mse":
                W += eta * np.outer(vv - W @ kk, kk)
            else:
                raise ValueError(rule)
        out = W @ k
        n = np.linalg.norm(out)
        if n == 0:
            margins[t] = 0.0
        else:
            margins[t] = (out @ v_new - out @ v_old) / n
    return margins


def run_trials_equal(rule, eta, sigma, n_trials, r, seed=0):
    """Equal-count repeated writes: r x (k_i -> v_old) then r x (k_j -> v_new), each write
    with a fresh noisy key copy. Returns per-trial margins at the underlying key."""
    rng = np.random.default_rng(seed)
    margins = np.empty(n_trials)
    for t in range(n_trials):
        k = unit(rng.standard_normal(D))
        v_old = unit(rng.standard_normal(D))
        v_new = unit(rng.standard_normal(D))
        W = np.zeros((D, D))
        for vv in [v_old] * r + [v_new] * r:
            kk = unit(k + sigma * rng.standard_normal(D))
            if rule == "additive":
                W += eta * np.outer(vv, kk)
            elif rule == "mse":
                W += eta * np.outer(vv - W @ kk, kk)
            else:
                raise ValueError(rule)
        out = W @ k
        n = np.linalg.norm(out)
        margins[t] = 0.0 if n == 0 else (out @ v_new - out @ v_old) / n
    return margins


def summarize(m):
    return dict(mean=float(m.mean()), se=float(m.std(ddof=1) / np.sqrt(len(m))),
                frac_new=float((m > 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--equal-counts", action="store_true",
                    help="only generate the equal-count repeated-write data (Fig. 5c)")
    args = ap.parse_args()
    if args.equal_counts:
        rs = [1, 2, 4, 8, 16, 32, 64, 128]
        out = {}
        print("== equal-count repeated writes, noise=0.05, 400 trials ==")
        for e in (0.04, 0.01, 0.004, 0.001):
            out[str(e)] = {}
            for rule in ("additive", "mse"):
                out[str(e)][rule] = []
                for r in rs:
                    summ = summarize(run_trials_equal(rule, e, 0.05, 400, r, seed=7))
                    summ["r"] = r
                    out[str(e)][rule].append(summ)
                print(f"  eta={e:<6g} {rule:8s} " + "  ".join(f"r={x['r']}:{x['mean']:+.3f}" for x in out[str(e)][rule]))
        with open(f"{args.out}/overwrite_equal_counts.json", "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.out}/overwrite_equal_counts.json")
        return
    results = {}

    # --- Panel A: validation at eta = 1 across key noise (reproduces the paper's Fig. 5) ---
    noises = [0.0, 0.02, 0.05, 0.10, 0.20]
    results["eta1_vs_noise"] = {}
    print("== eta=1, vs key noise (paper: MSE 1.004 0.952 0.742 0.380 0.130; additive ~0) ==")
    for rule in ("additive", "mse"):
        results["eta1_vs_noise"][rule] = []
        for s in noises:
            summ = summarize(run_trials(rule, 1.0, s, args.n, seed=1))
            summ["noise"] = s
            results["eta1_vs_noise"][rule].append(summ)
            print(f"  {rule:8s} noise={s:.2f}  margin={summ['mean']:+.3f} ± {summ['se']:.3f}  "
                  f"prefers-new={summ['frac_new']:.3f}")

    # --- Panel B: eta sweep at fixed noise 0.05, single write of the new value ---
    etas = [1.0, 0.5, 0.2, 0.1, 0.04, 0.02, 0.01, 0.004, 0.001]
    results["eta_sweep_noise005"] = {}
    print("== noise=0.05, vs eta*||k||^2, r_new=1 ==")
    for rule in ("additive", "mse"):
        results["eta_sweep_noise005"][rule] = []
        for e in etas:
            summ = summarize(run_trials(rule, e, 0.05, args.n, seed=2))
            summ["eta"] = e
            results["eta_sweep_noise005"][rule].append(summ)
            print(f"  {rule:8s} eta={e:<6g} margin={summ['mean']:+.3f} ± {summ['se']:.3f}  "
                  f"prefers-new={summ['frac_new']:.3f}")

    # --- Panel C: repeated writes of the new value at trained-regime steps ---
    rs = [1, 2, 4, 8, 16, 32, 64, 128]
    results["repeated_writes"] = {}
    print("== noise=0.05, MSE, repeated writes of v_new at trained-regime eta ==")
    for e in (0.04, 0.01, 0.004, 0.001):
        results["repeated_writes"][str(e)] = {}
        for rule in ("additive", "mse"):
            results["repeated_writes"][str(e)][rule] = []
            for r in rs:
                summ = summarize(run_trials(rule, e, 0.05, max(200, args.n // 5), r_new=r, seed=3))
                summ["r"] = r
                results["repeated_writes"][str(e)][rule].append(summ)
            row = "  ".join(f"r={x['r']}:{x['mean']:+.3f}" for x in results["repeated_writes"][str(e)][rule])
            print(f"  eta={e:<6g} {rule:8s} {row}")

    with open(f"{args.out}/overwrite_sweep_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {args.out}/overwrite_sweep_results.json")


if __name__ == "__main__":
    main()
