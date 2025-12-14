import os
import glob
import numpy as np
import pandas as pd

SEED = 42
DATA_DIR = "data"

OUT_BENCH = os.path.join(DATA_DIR, "benchmark.csv")
OUT_TRAIN = os.path.join(DATA_DIR, "train_idx.npy")
OUT_TEST  = os.path.join(DATA_DIR, "test_idx.npy")


def norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def find_col(df, candidates):
    col_map = {norm(c): c for c in df.columns}
    for cand in candidates:
        key = norm(cand)
        if key in col_map:
            return col_map[key]
    return None


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the data directory.")

    parts = []

    for path in csv_files:
        df = pd.read_csv(path)

        smiles_col = find_col(
            df,
            ["SMILES", "smiles", "canonical_smiles", "canon_smiles", "isomeric_smiles"]
        )
        if smiles_col is None:
            continue

        logS_col = find_col(
            df,
            [
                "logS", "esol", "solubility", "logs",
                "measured log solubility in mols per litre",
                "log solubility"
            ]
        )

        logP_col = find_col(
            df,
            [
                "logP", "logp", "exp", "expt",
                "exp_logp", "exp_log_p", "lipophilicity"
            ]
        )

        nrar_col = find_col(
            df,
            ["NR-AR", "nr-ar", "nrar", "nr_ar", "y", "label", "target"]
        )

        out = pd.DataFrame({"SMILES": df[smiles_col]})

        out["y_logS"] = pd.to_numeric(df[logS_col], errors="coerce") if logS_col else np.nan
        out["y_logP"] = pd.to_numeric(df[logP_col], errors="coerce") if logP_col else np.nan
        out["y_nrar"] = pd.to_numeric(df[nrar_col], errors="coerce") if nrar_col else np.nan

        parts.append(out)

    if not parts:
        raise RuntimeError("No valid SMILES columns were detected in the CSV files.")

    bench = pd.concat(parts, ignore_index=True)
    bench = (
        bench.dropna(subset=["SMILES"])
             .drop_duplicates("SMILES")
             .reset_index(drop=True)
    )

    bench.to_csv(OUT_BENCH, index=False)

    rng = np.random.default_rng(SEED)
    indices = np.arange(len(bench))
    rng.shuffle(indices)

    split = int(0.8 * len(indices))
    train_idx = np.sort(indices[:split])
    test_idx  = np.sort(indices[split:])

    np.save(OUT_TRAIN, train_idx)
    np.save(OUT_TEST, test_idx)

    print("Benchmark dataset saved to:", OUT_BENCH)
    print("Train indices:", len(train_idx))
    print("Test indices:", len(test_idx))


if __name__ == "__main__":
    main()
