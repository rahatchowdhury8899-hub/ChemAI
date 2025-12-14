import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
PLOT_DIR = os.path.join(RESULTS_DIR, "final_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

baseline_path = os.path.join(RESULTS_DIR, "baseline_metrics.csv")
gnn_path = os.path.join(RESULTS_DIR, "gnn_plus_improved_metrics.csv")

base = pd.read_csv(baseline_path)
gnn = pd.read_csv(gnn_path)


def normalize_task(t):
    t = str(t).strip().lower()
    if "nr" in t:
        return "NR-AR"
    if "logp" in t:
        return "logP"
    if "logs" in t or "sol" in t:
        return "logS"
    return t


base["task"] = base["task"].apply(normalize_task)
gnn["task"] = gnn["task"].apply(normalize_task)

base["model"] = "RF_baseline"
if "model" not in gnn.columns:
    gnn["model"] = "GNN_PLUS_IMPROVED"

metrics = pd.concat([base, gnn], ignore_index=True)
metrics.to_csv(os.path.join(RESULTS_DIR, "final_comparison.csv"), index=False)


def bar_compare(task, cols, title, outname):
    df = metrics[metrics["task"] == task].copy()
    if df.empty:
        return

    keep = []
    for _, r in df.iterrows():
        keep.append(any(pd.notna(r.get(c, np.nan)) for c in cols))
    df = df[np.array(keep)]
    if df.empty:
        return

    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    plt.figure()
    offsets = np.linspace(-width, width, num=len(cols))

    for i, c in enumerate(cols):
        if c not in df.columns:
            continue
        vals = df[c].astype(float).values
        plt.bar(x + offsets[i], vals, width=width / len(cols) * 1.8, label=c)

    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, outname), dpi=200)
    plt.close()


bar_compare(
    "logP",
    ["R2", "MAE"],
    "Comparison on logP (higher R2, lower MAE)",
    "compare_logP_metrics.png",
)

bar_compare(
    "logS",
    ["R2", "MAE"],
    "Comparison on logS (higher R2, lower MAE)",
    "compare_logS_metrics.png",
)

bar_compare(
    "NR-AR",
    ["ROC_AUC", "PR_AUC", "ACC@0.5"],
    "Comparison on NR-AR",
    "compare_NRAR_metrics.png",
)


base_pred_path = os.path.join(RESULTS_DIR, "baseline_pred_test.csv")
gnn_pred_path = os.path.join(RESULTS_DIR, "gnn_plus_improved_pred_logP.csv")

bt = bpv = None
if os.path.exists(base_pred_path):
    bp = pd.read_csv(base_pred_path)
    for a, b in [
        ("y_logP", "pred_logP_RF"),
        ("y_true", "y_pred"),
    ]:
        if a in bp.columns and b in bp.columns:
            bt = bp[a].dropna().values
            bpv = bp[b].dropna().values
            break

gt = gpv = None
if os.path.exists(gnn_pred_path):
    gp = pd.read_csv(gnn_pred_path)
    if "y_true" in gp.columns and "y_pred" in gp.columns:
        gt = gp["y_true"].values
        gpv = gp["y_pred"].values

if bt is not None and bpv is not None:
    plt.figure()
    plt.scatter(bt, bpv, s=8)
    lo = min(bt.min(), bpv.min())
    hi = max(bt.max(), bpv.max())
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("True logP")
    plt.ylabel("Predicted logP (RF)")
    plt.title("RF Baseline: logP")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "scatter_logP_RF.png"), dpi=200)
    plt.close()

if gt is not None and gpv is not None:
    plt.figure()
    plt.scatter(gt, gpv, s=8)
    lo = min(gt.min(), gpv.min())
    hi = max(gt.max(), gpv.max())
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("True logP")
    plt.ylabel("Predicted logP (GNN+)")
    plt.title("GNN+ Improved: logP")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "scatter_logP_GNN_PLUS_IMPROVED.png"), dpi=200)
    plt.close()

print("All comparison plots saved to:", PLOT_DIR)
