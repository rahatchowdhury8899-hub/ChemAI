import os
import glob
import numpy as np
import pandas as pd
from typing import List, Union

SEED = 42
DATA_DIR = "data"

OUT_BENCH = os.path.join(DATA_DIR, "benchmark.csv")
OUT_TRAIN = os.path.join(DATA_DIR, "train_idx.npy")
OUT_TEST = os.path.join(DATA_DIR, "test_idx.npy")


def norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def find_col(df: pd.DataFrame, candidates: List[str]) -> Union[str, None]:
    col_map = {norm(c): c for c in df.columns}
    for cand in candidates:
        key = norm(cand)
        if key in col_map:
            return col_map[key]
    return None


def clean_smiles_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    bad = s.str.contains(r"rdchem\.Mol", case=False, na=False) | s.str.startswith("<rdkit.", na=False)
    s = s.mask(bad, np.nan)
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return s


def first_non_null(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce")
    x = x.dropna()
    return x.iloc[0] if len(x) else np.nan


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    if os.path.exists("multi_property_data.csv"):
        csv_files.append("multi_property_data.csv")

    if not csv_files:
        raise FileNotFoundError("No CSV files found in data/ and no multi_property_data.csv in project root.")

    parts = []
    used = 0
    skipped = 0

    for path in csv_files:
        try:
            df = pd.read_csv(path)
        except Exception:
            skipped += 1
            continue

        smiles_col = find_col(
            df,
            ["SMILES", "smiles", "canonical_smiles", "canon_smiles", "isomeric_smiles"],
        )
        if smiles_col is None:
            skipped += 1
            continue

        logS_col = find_col(
            df,
            [
                "y_logS", "logS", "logs", "esol", "solubility",
                "measured log solubility in mols per litre",
                "log solubility",
            ],
        )

        logP_col = find_col(
            df,
            [
                "y_logP", "LogP", "logP", "logp",
                "exp", "expt", "exp_logp", "exp_log_p", "lipophilicity",
            ],
        )

        nrar_col = find_col(
            df,
            [
                "y_nrar", "NR-AR", "nr-ar", "NRAR", "nrar", "nr_ar",
                "label", "target", "y",
            ],
        )

        out = pd.DataFrame({"SMILES": clean_smiles_series(df[smiles_col])})

        out["y_logS"] = pd.to_numeric(df[logS_col], errors="coerce") if logS_col else np.nan
        out["y_logP"] = pd.to_numeric(df[logP_col], errors="coerce") if logP_col else np.nan
        out["y_nrar"] = pd.to_numeric(df[nrar_col], errors="coerce") if nrar_col else np.nan

        parts.append(out)
        used += 1

    if not parts:
        raise RuntimeError("No valid SMILES columns were detected in the CSV files.")

    bench = pd.concat(parts, ignore_index=True)

    bench = bench.dropna(subset=["SMILES"]).copy()

    bench = (
        bench.groupby("SMILES", as_index=False)
        .agg(
            y_logS=("y_logS", first_non_null),
            y_logP=("y_logP", first_non_null),
            y_nrar=("y_nrar", first_non_null),
        )
        .reset_index(drop=True)
    )

    bench.to_csv(OUT_BENCH, index=False)

    rng = np.random.default_rng(SEED)
    indices = np.arange(len(bench))
    rng.shuffle(indices)

    split = int(0.8 * len(indices))
    train_idx = np.sort(indices[:split])
    test_idx = np.sort(indices[split:])

    np.save(OUT_TRAIN, train_idx)
    np.save(OUT_TEST, test_idx)

    print("=== BENCHMARK BUILD COMPLETE ===")
    print("Sources used:", used, "| skipped:", skipped)
    print("Rows:", len(bench))
    print("Columns:", bench.columns.tolist())
    print("Non-null y_logS:", int(bench["y_logS"].notna().sum()))
    print("Non-null y_logP:", int(bench["y_logP"].notna().sum()))
    print("Non-null y_nrar:", int(bench["y_nrar"].notna().sum()))
    print("Saved:", OUT_BENCH)
    print("Train indices:", len(train_idx))
    print("Test indices:", len(test_idx))


if __name__ == "__main__":
    main()