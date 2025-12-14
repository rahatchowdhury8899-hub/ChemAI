import os
import pandas as pd

os.makedirs("data", exist_ok=True)

try:
    import deepchem as dc
except ImportError:
    raise SystemExit("DeepChem is not installed. Install it using: pip install deepchem")

from deepchem.molnet.load_function.delaney_datasets import load_delaney

tasks, (train, valid, test), transformers = load_delaney(featurizer="Raw")

def to_df(dataset):
    smiles = [str(s) for s in dataset.X]
    y = dataset.y.reshape(-1)
    return pd.DataFrame({
        "SMILES": smiles,
        "logS": y
    })

df = pd.concat(
    [to_df(train), to_df(valid), to_df(test)],
    ignore_index=True
)

df = df.dropna(subset=["SMILES", "logS"])
df["SMILES"] = df["SMILES"].astype(str)

df = df[~df["SMILES"].str.startswith("<rdkit.Chem")]
df = df.drop_duplicates("SMILES").reset_index(drop=True)

out_csv = "data/ESOL_real.csv"
df.to_csv(out_csv, index=False)

print("ESOL dataset saved to:", out_csv)
print("Total molecules:", len(df))
print(df.head())
