"""Generate the data figures (Fig 3-6) from the real results ledger. Publication style."""
import json
import os
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "experiments", "results.jsonl")
RUNS = os.path.join(ROOT, "experiments", "runs")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 12,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
})
# pastel palette
C_CE, C_F1, C_F2, C_F3 = "#5B8FF9", "#9FB8E6", "#F6A6B2", "#E86A78"

def load():
    rows = {}
    with open(RESULTS) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["exp_id"]] = r
    return rows

R = load()

# ---- Fig 3: balanced accuracy vs ECE ----------------------------------------
def fig3():
    labels = ["CE", "Focal\n$\\gamma$=1", "Focal\n$\\gamma$=2", "Focal\n$\\gamma$=3"]
    exps = ["exp01", "exp02", "exp03", "exp04"]
    acc = [R[e]["test_acc"] * 100 for e in exps]
    ece = [R[e]["ece"] for e in exps]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    bars = ax1.bar(x - 0.2, acc, 0.4, label="Test accuracy (%)", color=C_CE)
    ax1.set_ylabel("Test accuracy (%)", color=C_CE)
    ax1.set_ylim(84, 89)
    ax1.tick_params(axis="y", labelcolor=C_CE)
    for b, v in zip(bars, acc):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.1f}", ha="center", fontsize=9, color=C_CE)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + 0.2, ece, 0.4, label="ECE", color=C_F3)
    ax2.set_ylabel("Expected Calibration Error", color=C_F3)
    ax2.set_ylim(0, 0.13)
    ax2.tick_params(axis="y", labelcolor=C_F3)
    ax2.grid(False)
    for b, v in zip(bars2, ece):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=9, color=C_F3)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_title("Balanced CIFAR-10: Accuracy vs Calibration")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_balanced.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ---- Fig 4: imbalanced CE vs focal, three metrics ---------------------------
def fig4():
    groups = ["Accuracy", "Macro-F1", "Worst-class acc"]
    ce = [R["exp05"]["test_acc"], R["exp05"]["macro_f1"], R["exp05"]["worst_class_acc"]]
    f2 = [R["exp06"]["test_acc"], R["exp06"]["macro_f1"], R["exp06"]["worst_class_acc"]]
    f3 = [R["exp07"]["test_acc"], R["exp07"]["macro_f1"], R["exp07"]["worst_class_acc"]]
    x = np.arange(len(groups)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    b1 = ax.bar(x - w, ce, w, label="Cross-entropy", color=C_CE)
    b2 = ax.bar(x, f2, w, label="Focal $\\gamma$=2", color=C_F1)
    b3 = ax.bar(x + w, f3, w, label="Focal $\\gamma$=3", color=C_F3)
    for bars in (b1, b2, b3):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{b.get_height():.3f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Score"); ax.set_ylim(0, 0.72)
    ax.set_title("Long-Tailed CIFAR-10-LT (imbalance 100): CE vs Focal Loss")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_imbalanced.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ---- Fig 5: per-class accuracy vs frequency (CE vs focal gamma=3) ------------
def fig5():
    hist = R["exp05"]["class_hist"]
    ce = R["exp05"]["per_class_acc"]
    fl = R["exp07"]["per_class_acc"]
    order = np.argsort(hist)[::-1]  # most frequent first
    ce_o = [ce[i] for i in order]; fl_o = [fl[i] for i in order]; hist_o = [hist[i] for i in order]
    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(x, ce_o, "o-", color=C_CE, label="Cross-entropy", linewidth=2)
    ax.plot(x, fl_o, "s-", color=C_F3, label="Focal $\\gamma$=3", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}" for h in hist_o], rotation=45, fontsize=9)
    ax.set_xlabel("Class (ordered by training frequency, n samples)")
    ax.set_ylabel("Per-class test accuracy")
    ax.set_title("Per-Class Accuracy vs Frequency (CIFAR-10-LT): Focal Rescues the Tail")
    ax.legend()
    ax.annotate("tail classes", xy=(8.5, 0.13), xytext=(6.2, 0.30),
                arrowprops=dict(arrowstyle="->", color="gray"), color="gray")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_perclass.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ---- Fig 6: test accuracy vs epoch ------------------------------------------
def fig6():
    def curve(exp):
        p = os.path.join(RUNS, exp, "curve.csv")
        ep, acc = [], []
        with open(p) as f:
            for row in csv.DictReader(f):
                ep.append(int(row["epoch"])); acc.append(float(row["test_acc"]) * 100)
        return ep, acc
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    styles = {"exp01": ("CE (balanced)", C_CE, "-"), "exp03": ("Focal $\\gamma$=2 (balanced)", C_F1, "-"),
              "exp05": ("CE (LT-100)", "#7A7A7A", "--"), "exp07": ("Focal $\\gamma$=3 (LT-100)", C_F3, "--")}
    for exp, (lab, col, ls) in styles.items():
        ep, acc = curve(exp)
        ax.plot(ep, acc, ls, color=col, label=lab, linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Training Dynamics: Test Accuracy per Epoch")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_curves.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    # remove the bogus rcParam key if present (guard)
    if "axes.spxxx" in plt.rcParams:
        pass
    fig3(); fig4(); fig5(); fig6()
    print("Saved:", sorted(os.listdir(OUT)))
