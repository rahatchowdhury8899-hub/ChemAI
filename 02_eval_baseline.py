import os
import numpy as np
import pandas as pd
from joblib import load

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.warning")


from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)

DATA_DIR = "data"
MODELS_DIR = "models"
OUT_DIR = "results"

os.makedirs(OUT_DIR, exist_ok=True)

BENCH_PATH = os.path.join(DATA_DIR, "benchmark.csv")
TRAIN_IDX_PATH = os.path.join(DATA_DIR, "train_idx.npy")
TEST_IDX_PATH = os.path.join(DATA_DIR, "test_idx.npy")

MODEL_LOGS = os.path.join(MODELS_DIR, "rf_esol_solubility.joblib")
MODEL_LOGP = os.path.join(MODELS_DIR, "rf_logp.joblib")
MODEL_NRAR = os.path.join(MODELS_DIR, "rf_tox21_NRAR.joblib")

RADIUS = 2
NBITS = 2048


def is_bad_smiles(s: str) -> bool:
    if s is None:
        return True
    s = str(s).strip()
    if not s or s.lower() in {"nan", "none"}:
        return True
    if "rdchem.mol" in s.lower() or s.startswith("<rdkit."):
        return True
    return False


def smiles_to_morgan(smiles_list, radius=2, nBits=2048):
    fps = []
    valid_mask = np.zeros(len(smiles_list), dtype=bool)

    for i, smi in enumerate(smiles_list):
        smi = str(smi).strip()
        if is_bad_smiles(smi):
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
        fps.append(np.array(fp, dtype=np.float32))
        valid_mask[i] = True

    if not fps:
        return np.empty((0, nBits), dtype=np.float32), valid_mask

    return np.vstack(fps), valid_mask


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def main():
    RDLogger.DisableLog("rdApp.error")

    bench = pd.read_csv(BENCH_PATH)
    test_idx = np.load(TEST_IDX_PATH)

    test = bench.iloc[test_idx].reset_index(drop=True)

    rf_logS = load(MODEL_LOGS) if os.path.exists(MODEL_LOGS) else None
    rf_logP = load(MODEL_LOGP) if os.path.exists(MODEL_LOGP) else None
    rf_NR = load(MODEL_NRAR) if os.path.exists(MODEL_NRAR) else None

    smiles = test["SMILES"].astype(str).values
    X_fp, valid_mask = smiles_to_morgan(smiles, radius=RADIUS, nBits=NBITS)

    if X_fp.shape[0] == 0:
        raise RuntimeError("No valid SMILES found in the test split.")

    test_valid = test.loc[valid_mask].copy().reset_index(drop=True)

    pred_df = pd.DataFrame({"SMILES": test_valid["SMILES"].values})
    metrics_rows = []

    if rf_logS is not None and "y_logS" in test_valid.columns:
        y = safe_numeric(test_valid["y_logS"]).values.astype(float)
        pred = np.full(len(test_valid), np.nan, dtype=float)

        mask = ~np.isnan(y)
        if mask.sum() > 5:
            pred[mask] = rf_logS.predict(X_fp[mask])
            r2 = r2_score(y[mask], pred[mask])
            mae = mean_absolute_error(y[mask], pred[mask])
        else:
            r2, mae = np.nan, np.nan

        pred_df["y_logS"] = y
        pred_df["pred_logS_RF"] = pred
        metrics_rows.append({"model": "RF_baseline", "task": "logS", "R2": r2, "MAE": mae})

    if rf_logP is not None and "y_logP" in test_valid.columns:
        y = safe_numeric(test_valid["y_logP"]).values.astype(float)
        pred = np.full(len(test_valid), np.nan, dtype=float)

        mask = ~np.isnan(y)
        if mask.sum() > 5:
            pred[mask] = rf_logP.predict(X_fp[mask])
            r2 = r2_score(y[mask], pred[mask])
            mae = mean_absolute_error(y[mask], pred[mask])
        else:
            r2, mae = np.nan, np.nan

        pred_df["y_logP"] = y
        pred_df["pred_logP_RF"] = pred
        metrics_rows.append({"model": "RF_baseline", "task": "logP", "R2": r2, "MAE": mae})

    if rf_NR is not None and "y_nrar" in test_valid.columns:
        y = safe_numeric(test_valid["y_nrar"]).values.astype(float)
        pred_prob = np.full(len(test_valid), np.nan, dtype=float)

        mask = ~np.isnan(y)
        if mask.sum() > 10 and len(np.unique(y[mask].astype(int))) == 2:
            if hasattr(rf_NR, "predict_proba"):
                pred_prob[mask] = rf_NR.predict_proba(X_fp[mask])[:, 1]
            else:
                pred_prob[mask] = rf_NR.predict(X_fp[mask])

            roc = roc_auc_score(y[mask].astype(int), pred_prob[mask])
            pr = average_precision_score(y[mask].astype(int), pred_prob[mask])
            acc = accuracy_score(y[mask].astype(int), (pred_prob[mask] >= 0.5).astype(int))
        else:
            roc, pr, acc = np.nan, np.nan, np.nan

        pred_df["y_nrar"] = y
        pred_df["pred_nrar_RF"] = pred_prob
        metrics_rows.append(
            {"model": "RF_baseline", "task": "NR-AR", "ROC_AUC": roc, "PR_AUC": pr, "ACC@0.5": acc}
        )

    metrics_df = pd.DataFrame(metrics_rows)

    metrics_path = os.path.join(OUT_DIR, "baseline_metrics.csv")
    pred_path = os.path.join(OUT_DIR, "baseline_pred_test.csv")

    metrics_df.to_csv(metrics_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    print("Evaluation completed")
    print("Metrics saved to:", metrics_path)
    print("Predictions saved to:", pred_path)
    print(metrics_df)


if __name__ == "__main__":
    main()
