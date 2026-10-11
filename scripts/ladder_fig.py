"""Regenerate ladder_fig.pdf from the Modal ladder JSONs: five arms on 760M and 1.3B,
plus the Hebbian / norm-matched ΔNLL vs context position on 760M."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results/modal"
d760 = json.load(open(f"{R}/ladder_linear-760m/ablation_results_linear760m.json"))
d13 = json.load(open(f"{R}/ladder_linear-1.3b/ablation_results_linear13b.json"))

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, DARKRED, ORANGE, GRAY = "#1f77b4", "#c0392b", "#7b241c", "#e67e22", "#7f7f7f"
arms = ["descent", "hebbian", "hebbian_nm", "ascent", "frozen"]
labels = ["descent\n(baseline)", "Hebbian\n(no corr.)", "Hebbian\nnorm-matched", "ascent", "frozen\n(no inner loop)"]
cols = [BLUE, RED, DARKRED, ORANGE, GRAY]

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1.15, 1]})

# (a) perplexity per arm, two checkpoints side by side
a = ax[0]
x = np.arange(len(arms)); w = 0.38
for i, (d, off, hatch, name) in enumerate([(d760, -w / 2, None, "TTT-Linear-760M"), (d13, w / 2, "////", "TTT-Linear-1.3B")]):
    ppl = [d["per_arm"][k]["ppl"] for k in arms]
    bars = a.bar(x + off, ppl, w, color=cols, hatch=hatch, edgecolor="white" if hatch is None else "black",
                 linewidth=0.4, alpha=0.95 if hatch is None else 0.55, label=name)
    for xi, p in zip(x + off, ppl):
        a.text(xi, p * 1.18, f"{p:,.1f}" if p < 100 else f"{p:,.0f}", ha="center", va="bottom", fontsize=5.8, rotation=0)
a.set_yscale("log"); a.set_ylim(8, 9000)
a.set_xticks(x); a.set_xticklabels(labels, fontsize=6.5)
a.set_ylabel("perplexity (log scale)")
a.set_title("(a) inference-time inner-update ablation")
from matplotlib.patches import Patch
a.legend(handles=[Patch(facecolor="#bbbbbb", label="TTT-Linear-760M"),
                  Patch(facecolor="#bbbbbb", hatch="////", alpha=0.55, edgecolor="black", label="TTT-Linear-1.3B")],
         frameon=False, loc="upper left")

# (b) Hebbian and norm-matched ΔNLL vs position (760M)
b = ax[1]
for arm, col, lab in (("hebbian", RED, "Hebbian"), ("hebbian_nm", DARKRED, "Hebbian, norm-matched")):
    bp = d760["deltas_vs_descent"][arm]["by_position"]
    xs = [p["pos_mid"] for p in bp]; ys = [p["delta_nll"] for p in bp]; es = [p["se"] for p in bp]
    b.errorbar(xs, ys, yerr=es, color=col, marker="o", ms=3, lw=1.3, capsize=2, label=lab,
               ls="-" if arm == "hebbian" else "--")
b.axhline(0, color=GRAY, lw=0.6)
b.set_xscale("log"); b.set_xlabel("context position (tokens seen)")
b.set_ylabel("$\\Delta$NLL vs. descent (nats)")
b.set_title("(b) cost of deleting error correction, 760M")
b.legend(frameon=False, loc="upper left")

fig.tight_layout(w_pad=1.0)
fig.savefig("ladder_fig.pdf"); fig.savefig("ladder_fig.png", dpi=160)
print("wrote ladder_fig.pdf/.png")
