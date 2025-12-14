import os
import numpy as np
import pandas as pd

DATA_DIR = "data"
RESULTS_DIR = "results"

BENCH_PATH = os.path.join(DATA_DIR, "benchmark.csv")
TRAIN_IDX_PATH = os.path.join(DATA_DIR, "train_idx.npy")
TEST_IDX_PATH  = os.path.join(DATA_DIR, "test_idx.npy")

OUT_PATH = os.path.join(RESULTS_DIR, "benchmark_audit.txt")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    df = pd.read_csv(BENCH_PATH)
    train_idx = np.load(TRAIN_IDX_PATH)
    test_idx = np.load(TEST_IDX_PATH)

    required = ["SMILES", "y_logS", "y_logP", "y_nrar"]
    missing = [c for c in required if c not in df.columns]

    lines = []
    lines.append("=== BENCHMARK AUDIT ===")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Columns: {list(df.columns)}")

    if missing:
        lines.append(f"Missing required columns: {missing}")
        txt = "\n".join(lines)
        print(txt)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            f.write(txt)
        return

    bad_smiles = df["SMILES"].isna().sum() + (df["SMILES"].astype(str).str.strip() == "").sum()
    lines.append(f"Bad/empty SMILES rows: {bad_smiles}")

    for c in ["y_logS", "y_logP", "y_nrar"]:
        lines.append(f"Non-null {c}: {df[c].notna().sum()}")

    tr = df.iloc[train_idx].copy()
    te = df.iloc[test_idx].copy()
    lines.append(f"\nTrain size: {len(tr)} | Test size: {len(te)}")

    for c in ["y_logS", "y_logP", "y_nrar"]:
        lines.append(f"Train non-null {c}: {tr[c].notna().sum()}")
        lines.append(f"Test  non-null {c}: {te[c].notna().sum()}")

    ytr = pd.to_numeric(tr["y_nrar"], errors="coerce")
    yte = pd.to_numeric(te["y_nrar"], errors="coerce")

    for name, y in [("Train", ytr), ("Test", yte)]:
        y = y.dropna()
        if len(y) == 0:
            lines.append(f"{name} NR-AR labels: none")
        else:
            pos = int((y == 1).sum())
            neg = int((y == 0).sum())
            lines.append(f"{name} NR-AR labels: pos={pos}, neg={neg}, total={pos + neg}")

    if tr["y_logS"].notna().sum() < 100 or te["y_logS"].notna().sum() < 30:
        lines.append("\nNote: logS labels are relatively sparse in this benchmark.")
        lines.append("Multitask training may be dominated by logP and NR-AR.")

    txt = "\n".join(lines)
    print(txt)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(txt)

    print("Audit saved to:", OUT_PATH)


if __name__ == "__main__":
    main()
