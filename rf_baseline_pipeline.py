"""ChemAI: multi-property chemical prediction (baseline Random Forest).

This script builds a cleaned dataset from a raw CSV containing SMILES and a target label,
computes a small set of RDKit descriptors, trains a multi-output Random Forest baseline,
and provides a simple single-molecule prediction helper.

Notes:
- PubChem name lookup is optional (requires `pubchempy`).
- Large artifacts (models, intermediate arrays) should be excluded from version control.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

try:
    import pubchempy as pcp  # type: ignore
except Exception:  # pragma: no cover
    pcp = None

RANDOM_STATE = 42
FP_BITS = 2048
FP_RADIUS = 2


TARGET_COLS: List[str] = [
    "Toxicity",
    "MolWt",
    "LogP",
    "TPSA",
    "HBA",
    "HBD",
    "NumRotatableBonds",
    "RingCount",
]


@dataclass
class QueryInfo:
    input: str
    kind: str
    smiles: str
    name: Optional[str] = None


def _smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def calc_rdkit_properties(smiles: str) -> Optional[Dict[str, float]]:
    mol = _smiles_to_mol(smiles)
    if mol is None:
        return None

    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "LogP": float(Descriptors.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBA": float(Lipinski.NumHAcceptors(mol)),
        "HBD": float(Lipinski.NumHDonors(mol)),
        "NumRotatableBonds": float(Lipinski.NumRotatableBonds(mol)),
        "RingCount": float(rdMolDescriptors.CalcNumRings(mol)),
    }


def build_dataset(
    raw_file: str = "toxicity_raw.csv",
    out_file: str = "multi_property_data.csv",
) -> pd.DataFrame:
    df_raw = pd.read_csv(raw_file)
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]

    if "SMILES" not in df.columns:
        raise ValueError("Input must contain a 'SMILES' column.")
    if "Toxicity" not in df.columns:
        raise ValueError("Input must contain a 'Toxicity' column.")

    df = df[["SMILES", "Toxicity"]].dropna(subset=["SMILES", "Toxicity"])
    df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

    props_rows: List[Dict[str, float]] = []
    keep_smiles: List[str] = []

    for smi in df["SMILES"].astype(str):
        props = calc_rdkit_properties(smi)
        if props is None:
            continue
        props_rows.append(props)
        keep_smiles.append(smi)

    df = df[df["SMILES"].isin(keep_smiles)].reset_index(drop=True)
    df_props = pd.DataFrame(props_rows)
    df_out = pd.concat([df.reset_index(drop=True), df_props.reset_index(drop=True)], axis=1)

    df_out.to_csv(out_file, index=False)
    print(f"Saved dataset: {out_file} | rows={len(df_out)} | cols={len(df_out.columns)}")
    return df_out


def smiles_to_morgan(smiles: str, radius: int = FP_RADIUS, n_bits: int = FP_BITS) -> Optional[np.ndarray]:
    mol = _smiles_to_mol(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def load_features_targets(
    data_file: str = "multi_property_data.csv",
    target_cols: List[str] = TARGET_COLS,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(data_file)
    df = df[["SMILES"] + target_cols].dropna()
    df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

    fps: List[np.ndarray] = []
    valid_idx: List[int] = []

    for i, smi in enumerate(df["SMILES"].astype(str)):
        fp = smiles_to_morgan(smi)
        if fp is None:
            continue
        fps.append(fp)
        valid_idx.append(i)

    if not fps:
        raise ValueError("No valid SMILES were found after cleaning.")

    X = np.stack(fps).astype(np.float32)
    df_clean = df.iloc[valid_idx].reset_index(drop=True)
    Y = df_clean[target_cols].values.astype(np.float32)
    return df_clean, X, Y, target_cols


def train_baseline_rf(
    X: np.ndarray,
    Y: np.ndarray,
    test_size: float = 0.2,
) -> Tuple[MultiOutputRegressor, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=RANDOM_STATE
    )

    base = RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)
    return model, (X_train, X_test, y_train, y_test)


def evaluate(
    model: MultiOutputRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_cols: List[str],
) -> Dict[str, Dict[str, float]]:
    y_pred = model.predict(X_test)

    metrics: Dict[str, Dict[str, float]] = {}

    overall_r2 = float(r2_score(y_test, y_pred, multioutput="variance_weighted"))
    overall_mae = float(mean_absolute_error(y_test, y_pred, multioutput="uniform_average"))

    print("-" * 50)
    print(f"Overall weighted R2: {overall_r2:.3f}")
    print(f"Overall mean MAE   : {overall_mae:.3f}")
    print("-" * 50)

    for i, col in enumerate(target_cols):
        r2 = float(r2_score(y_test[:, i], y_pred[:, i]))
        mae = float(mean_absolute_error(y_test[:, i], y_pred[:, i]))
        metrics[col] = {"R2": r2, "MAE": mae}
        print(f"{col:18s} | R2 = {r2:6.3f} | MAE = {mae:6.3f}")

    return metrics


def resolve_query(query: str) -> QueryInfo:
    q = str(query).strip()

    if pcp is None:
        return QueryInfo(input=q, kind="raw", smiles=q, name=None)

    try:
        hits = pcp.get_compounds(q, "smiles")
        if hits:
            c = hits[0]
            return QueryInfo(
                input=q,
                kind="smiles",
                smiles=c.connectivity_smiles or q,
                name=getattr(c, "iupac_name", None),
            )
    except Exception:
        pass

    try:
        hits = pcp.get_compounds(q, "name")
        if hits:
            c = hits[0]
            return QueryInfo(
                input=q,
                kind="name",
                smiles=c.connectivity_smiles,
                name=getattr(c, "iupac_name", None) or q,
            )
    except Exception:
        pass

    return QueryInfo(input=q, kind="raw", smiles=q, name=None)


def predict_molecule(
    model: MultiOutputRegressor,
    target_cols: List[str],
    query: str,
) -> Tuple[QueryInfo, Dict[str, float]]:
    info = resolve_query(query)
    fp = smiles_to_morgan(info.smiles)
    if fp is None:
        raise ValueError("Invalid SMILES after resolution.")

    preds = model.predict(fp.reshape(1, -1))[0]
    result = {k: float(v) for k, v in zip(target_cols, preds)}
    return info, result


def main() -> None:
    warnings.filterwarnings("ignore")
    np.random.seed(RANDOM_STATE)

    build_dataset(raw_file="toxicity_raw.csv", out_file="multi_property_data.csv")
    _, X, Y, target_cols = load_features_targets("multi_property_data.csv")
    model, (_, X_test, _, y_test) = train_baseline_rf(X, Y)
    evaluate(model, X_test, y_test, target_cols)

    import joblib

    joblib.dump(model, "rf_multi_property_model.joblib")
    print("Saved model: rf_multi_property_model.joblib")

    example = "CC(=O)OC1=CC=CC=C1C(=O)O"
    info, res = predict_molecule(model, target_cols, example)
    print(f"Example input: {info.input}")
    for k, v in res.items():
        print(f"{k:18s} {v:8.3f}")


if __name__ == "__main__":
    main()
