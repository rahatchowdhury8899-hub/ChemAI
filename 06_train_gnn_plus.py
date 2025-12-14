import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import rdchem

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    roc_auc_score,
    average_precision_score,
    accuracy_score
)
from sklearn.preprocessing import StandardScaler

SEED = 42
EPOCHS = 1
BATCH_SIZE = 64
LR = 1e-3

DATA_DIR = "data"
OUT_DIR = "results"
MODEL_DIR = "models"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ATOM_LIST = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]


def atom_features(atom: rdchem.Atom):
    one_hot = [int(atom.GetAtomicNum() == a) for a in ATOM_LIST]
    hyb = int(atom.GetHybridization())
    return one_hot + [
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        float(int(atom.GetIsAromatic())),
        float(hyb),
    ]


def bond_features(bond: rdchem.Bond):
    bt = bond.GetBondType()
    return [
        float(bt == rdchem.BondType.SINGLE),
        float(bt == rdchem.BondType.DOUBLE),
        float(bt == rdchem.BondType.TRIPLE),
        float(bt == rdchem.BondType.AROMATIC),
        float(int(bond.GetIsConjugated())),
        float(int(bond.IsInRing())),
    ]


def smiles_to_graph(smiles: str, y: torch.Tensor):
    mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_pairs = []
    edge_attr = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        bf = bond_features(b)
        edge_pairs += [[i, j], [j, i]]
        edge_attr += [bf, bf]

    if len(edge_pairs) == 0:
        return None

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.y = y
    return data


class GINMultiTask(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
        )
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
        )
        self.fc_shared = nn.Linear(128, 64)

        self.head_logS = nn.Linear(64, 1)
        self.head_logP = nn.Linear(64, 1)
        self.head_NR = nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc_shared(x))

        out_logS = self.head_logS(x).squeeze(-1)
        out_logP = self.head_logP(x).squeeze(-1)
        out_NR = self.head_NR(x).squeeze(-1)
        return out_logS, out_logP, out_NR


def y_as_matrix(y: torch.Tensor):
    if y.dim() == 1:
        if y.numel() % 3 != 0:
            raise RuntimeError(f"batch.y is 1D but not divisible by 3. numel={y.numel()}")
        y = y.view(-1, 3)
    return y


bench = pd.read_csv(os.path.join(DATA_DIR, "benchmark.csv"))
train_idx = np.load(os.path.join(DATA_DIR, "train_idx.npy"))
test_idx = np.load(os.path.join(DATA_DIR, "test_idx.npy"))

train_df = bench.iloc[train_idx].reset_index(drop=True)
test_df = bench.iloc[test_idx].reset_index(drop=True)

for c in ["y_logS", "y_logP", "y_nrar"]:
    if c in train_df.columns:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
    if c in test_df.columns:
        test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

scaler_logS = StandardScaler()
scaler_logP = StandardScaler()

train_df["y_logS_scaled"] = np.nan
train_df["y_logP_scaled"] = np.nan
test_df["y_logS_scaled"] = np.nan
test_df["y_logP_scaled"] = np.nan

has_logS = train_df["y_logS"].notna().sum() > 5
has_logP = train_df["y_logP"].notna().sum() > 5

if has_logS:
    m = train_df["y_logS"].notna()
    train_df.loc[m, "y_logS_scaled"] = scaler_logS.fit_transform(train_df.loc[m, ["y_logS"]]).ravel()
    m = test_df["y_logS"].notna()
    if m.sum() > 0:
        test_df.loc[m, "y_logS_scaled"] = scaler_logS.transform(test_df.loc[m, ["y_logS"]]).ravel()

if has_logP:
    m = train_df["y_logP"].notna()
    train_df.loc[m, "y_logP_scaled"] = scaler_logP.fit_transform(train_df.loc[m, ["y_logP"]]).ravel()
    m = test_df["y_logP"].notna()
    if m.sum() > 0:
        test_df.loc[m, "y_logP_scaled"] = scaler_logP.transform(test_df.loc[m, ["y_logP"]]).ravel()


def build_dataset(df, name="split"):
    data_list = []
    dropped = 0

    for _, r in df.iterrows():
        y = torch.tensor(
            [
                np.nan if pd.isna(r["y_logS_scaled"]) else float(r["y_logS_scaled"]),
                np.nan if pd.isna(r["y_logP_scaled"]) else float(r["y_logP_scaled"]),
                np.nan if pd.isna(r["y_nrar"]) else float(r["y_nrar"]),
            ],
            dtype=torch.float,
        )

        g = smiles_to_graph(r["SMILES"], y)
        if g is None:
            dropped += 1
            continue
        data_list.append(g)

    print(f"{name}: kept {len(data_list)} graphs, dropped {dropped}")
    return data_list


train_ds = build_dataset(train_df, "train")
test_ds = build_dataset(test_df, "test")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

y_nr = train_df["y_nrar"].dropna().values
pos = float((y_nr == 1).sum())
neg = float((y_nr == 0).sum())
pos_weight = torch.tensor([neg / pos], dtype=torch.float, device=DEVICE) if pos > 0 and neg > 0 else None

model = GINMultiTask(train_ds[0].x.shape[1]).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        batch = batch.to(DEVICE)
        opt.zero_grad()

        out_logS, out_logP, out_NR = model(batch)
        y = y_as_matrix(batch.y)

        loss = torch.tensor(0.0, device=DEVICE)

        mS = ~torch.isnan(y[:, 0])
        if mS.sum() > 0:
            loss = loss + F.mse_loss(out_logS[mS], y[mS, 0])

        mP = ~torch.isnan(y[:, 1])
        if mP.sum() > 0:
            loss = loss + F.mse_loss(out_logP[mP], y[mP, 1])

        mN = ~torch.isnan(y[:, 2])
        if mN.sum() > 0:
            if pos_weight is None:
                loss = loss + F.binary_cross_entropy_with_logits(out_NR[mN], y[mN, 2])
            else:
                loss = loss + F.binary_cross_entropy_with_logits(out_NR[mN], y[mN, 2], pos_weight=pos_weight)

        loss.backward()
        opt.step()
        total_loss += float(loss.item())

    print(f"Epoch {epoch}/{EPOCHS} | loss={total_loss:.4f}")

model.eval()

y_logS_true, y_logS_pred = [], []
y_logP_true, y_logP_pred = [], []
y_nr_true, y_nr_prob = [], []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(DEVICE)
        out_logS, out_logP, out_NR = model(batch)
        y = y_as_matrix(batch.y)

        m = ~torch.isnan(y[:, 0])
        if m.sum() > 0 and has_logS:
            y_true = scaler_logS.inverse_transform(y[m, 0].unsqueeze(1).cpu().numpy()).ravel()
            y_pred = scaler_logS.inverse_transform(out_logS[m].unsqueeze(1).cpu().numpy()).ravel()
            y_logS_true.extend(y_true.tolist())
            y_logS_pred.extend(y_pred.tolist())

        m = ~torch.isnan(y[:, 1])
        if m.sum() > 0 and has_logP:
            y_true = scaler_logP.inverse_transform(y[m, 1].unsqueeze(1).cpu().numpy()).ravel()
            y_pred = scaler_logP.inverse_transform(out_logP[m].unsqueeze(1).cpu().numpy()).ravel()
            y_logP_true.extend(y_true.tolist())
            y_logP_pred.extend(y_pred.tolist())

        m = ~torch.isnan(y[:, 2])
        if m.sum() > 0:
            y_nr_true.extend(y[m, 2].cpu().numpy().tolist())
            y_nr_prob.extend(torch.sigmoid(out_NR[m]).cpu().numpy().tolist())

rows = []

if len(y_logS_true) > 10:
    rows.append({
        "model": "GNN_PLUS",
        "task": "logS",
        "R2": r2_score(y_logS_true, y_logS_pred),
        "MAE": mean_absolute_error(y_logS_true, y_logS_pred),
    })

if len(y_logP_true) > 10:
    rows.append({
        "model": "GNN_PLUS",
        "task": "logP",
        "R2": r2_score(y_logP_true, y_logP_pred),
        "MAE": mean_absolute_error(y_logP_true, y_logP_pred),
    })

if len(y_nr_true) > 10 and len(np.unique(np.array(y_nr_true).astype(int))) == 2:
    rows.append({
        "model": "GNN_PLUS",
        "task": "NR-AR",
        "ROC_AUC": roc_auc_score(y_nr_true, y_nr_prob),
        "PR_AUC": average_precision_score(y_nr_true, y_nr_prob),
        "ACC@0.5": accuracy_score(
            np.array(y_nr_true).astype(int),
            (np.array(y_nr_prob) >= 0.5).astype(int),
        ),
    })

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(os.path.join(OUT_DIR, "gnn_metrics.csv"), index=False)

if len(y_logS_true) > 0:
    pd.DataFrame({"y_true": y_logS_true, "y_pred": y_logS_pred}).to_csv(
        os.path.join(OUT_DIR, "gnn_pred_logs.csv"), index=False
    )

if len(y_logP_true) > 0:
    pd.DataFrame({"y_true": y_logP_true, "y_pred": y_logP_pred}).to_csv(
        os.path.join(OUT_DIR, "gnn_pred_logp.csv"), index=False
    )

if len(y_nr_true) > 0:
    pd.DataFrame({"y_true": y_nr_true, "y_prob": y_nr_prob}).to_csv(
        os.path.join(OUT_DIR, "gnn_pred_nrar.csv"), index=False
    )

torch.save(model.state_dict(), os.path.join(MODEL_DIR, "gnn_multitask.pt"))

print("Training and evaluation completed")
print("Saved:")
print(" -", os.path.join(OUT_DIR, "gnn_metrics.csv"))
print(" -", os.path.join(OUT_DIR, "gnn_pred_logs.csv"))
print(" -", os.path.join(OUT_DIR, "gnn_pred_logp.csv"))
print(" -", os.path.join(OUT_DIR, "gnn_pred_nrar.csv"))
print(" -", os.path.join(MODEL_DIR, "gnn_multitask.pt"))
print(metrics_df)
