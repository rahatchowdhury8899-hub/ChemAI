import os
import pandas as pd
import urllib.request

os.makedirs("data", exist_ok=True)

url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
out_csv = "data/ESOL_real.csv"

print("Downloading:", url)
urllib.request.urlretrieve(url, out_csv)
print("Downloaded to:", out_csv)

df = pd.read_csv(out_csv)

sm_col = None
for c in df.columns:
    if c.strip().lower() == "smiles":
        sm_col = c
        break

if sm_col is None:
    raise ValueError(f"SMILES column not found. Columns: {list(df.columns)}")

logS_col = None
for c in df.columns:
    if "measured log solubility" in c.strip().lower():
        logS_col = c
        break

if logS_col is None:
    raise ValueError(f"logS column not found. Columns: {list(df.columns)}")

df = df[[sm_col, logS_col]].rename(
    columns={sm_col: "SMILES", logS_col: "logS"}
)

df["SMILES"] = df["SMILES"].astype(str)
df["logS"] = pd.to_numeric(df["logS"], errors="coerce")

df = df.dropna(subset=["SMILES", "logS"])
df = df[~df["SMILES"].str.startswith("<rdkit.Chem")]
df = df.drop_duplicates("SMILES").reset_index(drop=True)

df.to_csv(out_csv, index=False)

print("ESOL dataset prepared successfully")
print("Total molecules:", len(df))
print(df.head())
