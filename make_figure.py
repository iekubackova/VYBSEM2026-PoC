"""Figure 2 and Table 1 of the paper from results/*_per_gene.csv (written by evaluate.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

MODELS = ["Training mean", "Linear (PCA)", "Random forest (GO)"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
METRICS = ["MSE (all genes)", "MSE (top-20 DE)", "Pearson delta", "Top-20 overlap"]


def per_seed(dataset, metric):
    r = pd.read_csv(f"results/{dataset}_per_gene.csv")
    return r.groupby(["model", "seed"])[metric].mean()


def figure(out="results.pdf"):
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.linewidth": 0.6, "pdf.fonttype": 42})
    panels = [("Pearson delta", "Pearson $\\Delta$ (higher is better)"),
              ("MSE (top-20 DE)", "MSE, top-20 DE genes (lower is better)")]
    fig, axes = plt.subplots(1, 2, figsize=(3.33, 1.75))
    for ax, (metric, title) in zip(axes, panels):
        for di, d in enumerate(["adamson", "norman"]):
            ps = per_seed(d, metric)
            for mi, mod in enumerate(MODELS):
                x, v = di + (mi - 1) * 0.26, ps.xs(mod, level=0).values
                ax.bar(x, v.mean(), width=0.22, color=COLORS[mi], alpha=0.35, edgecolor="none", zorder=1)
                ax.scatter(np.full(len(v), x) + np.linspace(-0.06, 0.06, len(v)), v, s=7, color=COLORS[mi],
                           edgecolor="white", linewidth=0.4, zorder=3)
                ax.plot([x - 0.11, x + 0.11], [v.mean()] * 2, color=COLORS[mi], lw=1.2, zorder=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Adamson\n(CRISPRi)", "Norman\n(CRISPRa)"])
        ax.set_title(title, fontsize=6.5, pad=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e4e3df", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.5, 1.5)
    axes[0].set_ylim(0, 1)
    fig.legend([Patch(color=c) for c in COLORS], MODELS, loc="upper center", ncol=3, frameon=False,
               fontsize=6.5, bbox_to_anchor=(0.5, 1.02), handlelength=1, handleheight=0.8, columnspacing=1)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.2)
    fig.savefig(out)


def table():
    scale = {"MSE (all genes)": 1000}
    for d in ["adamson", "norman"]:
        r = pd.read_csv(f"results/{d}_per_gene.csv")
        ps = r.groupby(["model", "seed"])[METRICS].mean()
        for mod in MODELS:
            x = ps.xs(mod, level=0)
            cells = [f"{x[m].mean() * scale.get(m, 1):.3f} ({x[m].std() * scale.get(m, 1):.3f})" for m in METRICS]
            print(f"{d:8s} {mod:20s} " + "  ".join(cells))


if __name__ == "__main__":
    figure()
    table()
