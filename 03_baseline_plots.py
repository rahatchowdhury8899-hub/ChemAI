import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

RESULTS_DIR = "results"
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

pred_path = os.path.join(RESULTS_DIR, "baseline_pred_test.csv")
df = pd.read_csv(pred_path)

# LogP: true vs predicted scatter
if "y_logP" in df.columns and "pred_logP_RF" in df.columns:
    mask = ~df["y_logP"].isna()
    y_true = df.loc[mask, "y_logP"].astype(float)
    y_pred = df.loc[mask, "pred_logP_RF"].astype(float)

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max())
    ]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Observed logP")
    plt.ylabel("Predicted logP")
    plt.title("RF Baseline: logP (True vs Predicted)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "baseline_logP_true_vs_pred.png"), dpi=300)
    plt.close()

# NR-AR: ROC curve
if "y_nrar" in df.columns and "pred_nrar_RF" in df.columns:
    mask = ~df["y_nrar"].isna()
    y_true = df.loc[mask, "y_nrar"].astype(int)
    y_score = df.loc[mask, "pred_nrar_RF"].astype(float)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"RF (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("RF Baseline: NR-AR ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "baseline_NRAR_ROC.png"), dpi=300)
    plt.close()

print("Plots saved to:", PLOT_DIR)
