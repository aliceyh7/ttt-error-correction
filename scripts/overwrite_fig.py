"""Make overwrite_fig.pdf (three panels) from the E2 results JSON files."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = json.load(open("results/overwrite_sweep_results.json"))
B = json.load(open("results/overwrite_equal_counts.json"))

BLUE, ORANGE, GRAY = "#1f77b4", "#ff7f0e", "#9a9a9a"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False})

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25))

# --- (a) eta = 1, vs key noise ---
ax = axes[0]
for rule, col, lab in (("additive", BLUE, "additive"), ("mse", ORANGE, "MSE (corrective)")):
    rows = A["eta1_vs_noise"][rule]
    x = [r["noise"] for r in rows]; y = [r["mean"] for r in rows]; e = [1.96 * r["se"] for r in rows]
    ax.errorbar(x, y, yerr=e, color=col, marker="o", ms=3.5, lw=1.4, capsize=2, label=lab)
ax.set_xlabel("key noise $\\sigma$")
ax.set_ylabel("overwrite margin\n$\\cos(\\cdot,v_{\\mathrm{new}})-\\cos(\\cdot,v_{\\mathrm{old}})$")
ax.set_title("(a) one write each, $\\eta\\|k\\|^2=1$")
ax.axhline(0, color=GRAY, lw=0.6)
ax.set_ylim(-0.08, 1.08)
ax.legend(frameon=False, loc="upper right")

# --- (b) eta sweep, single write, noise 0.05 ---
ax = axes[1]
ax.axvspan(0.001, 0.04, color="#000000", alpha=0.07, lw=0)
ax.text(0.0063, 0.97, "trained per-layer\nmedians (760M)", ha="center", va="top", fontsize=6.5, color="#444")
for rule, col in (("additive", BLUE), ("mse", ORANGE)):
    rows = A["eta_sweep_noise005"][rule]
    x = [r["eta"] for r in rows]; y = [r["mean"] for r in rows]; e = [1.96 * r["se"] for r in rows]
    ax.errorbar(x, y, yerr=e, color=col, marker="o", ms=3.5, lw=1.4, capsize=2)
ax.set_xscale("log")
ax.set_xlabel("effective step $\\eta\\|k\\|^2$")
ax.set_title("(b) one write each, noise $0.05$")
ax.axhline(0, color=GRAY, lw=0.6)
ax.set_ylim(-0.08, 1.08)
ax.set_xlim(7e-4, 1.4)

# --- (c) equal repeated writes, collapse on cumulative step ---
ax = axes[2]
ax.axvspan(0.001 * 1, 0.04 * 128, color="#000000", alpha=0.0, lw=0)
markers = {"0.04": "o", "0.01": "s", "0.004": "^", "0.001": "D"}
for eta, mk in markers.items():
    rows = B[eta]["mse"]
    x = [float(eta) * r["r"] for r in rows]; y = [r["mean"] for r in rows]
    ax.plot(x, y, color=ORANGE, marker=mk, ms=3.2, lw=0.9, alpha=0.9,
            label=f"MSE, $\\eta\\|k\\|^2={eta}$")
    rows = B[eta]["additive"]
    x = [float(eta) * r["r"] for r in rows]; y = [r["mean"] for r in rows]
    ax.plot(x, y, color=BLUE, marker=mk, ms=3.2, lw=0.9, alpha=0.9)
# theoretical new-value coefficient fraction 1-(1-eta)^r ~ 1-exp(-eta r)
xx = np.logspace(-3, np.log10(6), 100)
ax.plot(xx, 1 - np.exp(-xx), color=GRAY, lw=0.8, ls="--", label="$1-e^{-\\eta\\|k\\|^2 r}$")
ax.axhline(0, color=GRAY, lw=0.6)
ax.axvline(128 * 0.04, color="#444", lw=0.6, ls=":")
ax.text(128 * 0.04 * 0.9, 0.04, "128 chunk updates\nat 0.04", fontsize=6, ha="right", va="bottom", color="#444")
ax.set_xscale("log")
ax.set_xlabel("cumulative step $\\eta\\|k\\|^2\\, r$")
ax.set_title("(c) $r$ writes each, noise $0.05$")
ax.set_ylim(-0.08, 1.08)
ax.set_xlim(7e-4, 8)
ax.text(0.0012, 0.11, "additive (all $\\eta$)", color=BLUE, fontsize=6.5)
ax.legend(frameon=False, loc="upper left", fontsize=6, handlelength=1.6)

for ax in axes[1:]:
    ax.set_yticklabels([])
fig.tight_layout(w_pad=0.6)
fig.savefig("overwrite_fig.pdf")
fig.savefig("overwrite_fig.png", dpi=160)
print("wrote overwrite_fig.pdf/.png")
