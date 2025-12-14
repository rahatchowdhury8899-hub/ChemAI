import os
import numpy as np
import pandas as pd

SEED = 42
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

ESOL_PATH = os.path.join(DATA_DIR, "ESOL_real.csv")
LOGP_PATH = os.path.join(DATA_DIR, "Lipophilicity.csv")
TOX_PATH  = os.path.join(DATA_DIR, "Tox21.csv")

# Backup existing benchmark and splits if present
for p in [
    os.path.join(DATA_DIR, "benchmark.csv"),
    os.path.join(DATA_DIR, "train_idx.npy"),
    os.path.join(DATA_DIR, "test_idx.npy")
]:
    if os.path.exists(p):
        os.rename(p, p + ".bak")


# ---------- Load ESOL (logS) ----------
esol = pd.read_csv(ESOL_PATH)

if "SMILES" not in esol.columns:
    raise ValueError("ESOL dataset does not contain a SMILES column")
if "logS" not in esol.columns:
    raise ValueError("ESOL dataset does not contain a logS column")

esol = (
    esol[["SMILES", "logS"]]
    .rename(columns={"logS": "y_logS"})
    .drop_duplicates("SMILES")
)


# ---------- Load LogP ----------
logp = pd.read_csv(LOGP_PATH)

sm_col = None
for c in logp.columns:
    if c.strip().lower() in [
        "smiles", "canonical_smiles", "canon_smiles",
        "isomeric_smiles", "smile"
    ]:
        sm_col = c
        break
if sm_col is None and "SMILES" in logp.columns:
    sm_col = "SMILES"
if sm_col is None:
    raise ValueError("LogP dataset does not contain a SMILES column")

logp_col = None
for c in logp.columns:
    k = c.strip().lower().replace(" ", "").replace("_", "")
    if k in ["logp", "explogp", "exptlogp", "exp", "expt"]:
        logp_col = c
        break
if logp_col is None and "LogP" in logp.columns:
    logp_col = "LogP"
if logp_col is None:
    raise ValueError("LogP dataset does not contain a LogP column")

logp = (
    logp[[sm_col, logp_col]]
    .rename(columns={sm_col: "SMILES", logp_col: "y_logP"})
)

logp["y_logP"] = pd.to_numeric(logp["y_logP"], errors="coerce")
logp = logp.drop_duplicates("SMILES")


# ---------- Load Tox21 NR-AR ----------
tox = pd.read_csv(TOX_PATH)

sm_col = None
for c in tox.columns:
    if c.strip().lower() in [
        "smiles", "canonical_smiles", "canon_smiles",
        "isomeric_smiles", "smile"
    ]:
        sm_col = c
        break
if sm_col is None and "SMILES" in tox.columns:
    sm_col = "SMILES"
if sm_col is None:
    raise ValueError("Tox21 dataset does not contain a SMILES column")

nrar_col = None
for c in tox.columns:
    k = c.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    if k in [
        "nrar", "nrarlabel", "nrarlbl",
        "nraractivity", "label", "y", "target"
    ]:
        nrar_col = c
        break
if nrar_col is None and "NR-AR" in tox.columns:
    nrar_col = "NR-AR"
if nrar_col is None:
    raise ValueError("Tox21 dataset does not contain an NR-AR column")

tox = (
    tox[[sm_col, nrar_col]]
    .rename(columns={sm_col: "SMILES", nrar_col: "y_nrar"})
)

tox["y_nrar"] = pd.to_numeric(tox["y_nrar"], errors="coerce")
tox = tox.drop_duplicates("SMILES")


# ---------- Merge benchmark ----------
bench = pd.DataFrame({
    "SMILES": pd.concat(
        [esol["SMILES"], logp["SMILES"], tox["SMILES"]]
    ).drop_duplicates()
})

bench = bench.merge(esol, on="SMILES", how="left")
bench = bench.merge(logp, on="SMILES", how="left")
bench = bench.merge(tox,  on="SMILES", how="left")

bench.to_csv(os.path.join(DATA_DIR, "benchmark.csv"), index=False)


# ---------- Train / test split ----------
rng = np.random.default_rng(SEED)
idx = np.arange(len(bench))
rng.shuffle(idx)

split = int(0.8 * len(idx))
train_idx = np.sort(idx[:split])
test_idx  = np.sort(idx[split:])

np.save(os.path.join(DATA_DIR, "train_idx.npy"), train_idx)
np.save(os.path.join(DATA_DIR, "test_idx.npy"), test_idx)

print("Benchmark dataset created:", os.path.join(DATA_DIR, "benchmark.csv"))
print("Total molecules:", len(bench))
print("logS available:", bench["y_logS"].notna().sum())
print("logP available:", bench["y_logP"].notna().sum())
print("NR-AR available:", bench["y_nrar"].notna().sum())
print("Train / test split saved")
