import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem
RDLogger.DisableLog("rdApp.*")

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)

SEED = 42
EPOCHS = 200
BATCH_SIZE = 64
LR = 2e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.25
VAL_FRAC = 0.10
PATIENCE = 25

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


def smiles_to_graph(smiles, y):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_pairs, edge_attr = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        edge_pairs += [[i, j], [j, i]]
        edge_attr += [bf, bf]

    if not edge_pairs:
        return None

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    d.y = y
    return d


def y_as_matrix(y):
    if y.dim() == 1:
        y = y.view(-1, 3)
    return y


class GNNPlus(nn.Module):
    def __init__(self, node_in, edge_in):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        self.conv1 = GINEConv(
            nn.Sequential(nn.Linear(node_in, 256), nn.ReLU(), nn.Linear(256, 256)),
            edge_dim=256,
        )
        self.conv2 = GINEConv(
            nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256)),
            edge_dim=256,
        )

        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 64)

        self.head_logS = nn.Linear(64, 1)
        self.head_logP = nn.Linear(64, 1)
        self.head_NR = nn.Linear(64, 1)

        self.drop = nn.Dropout(DROPOUT)

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        ea = self.edge_mlp(ea)

        x = F.relu(self.conv1(x, ei, ea))
        x = self.drop(x)
        x = F.relu(self.conv2(x, ei, ea))

        x = global_mean_pool(x, batch)
        x = self.drop(F.relu(self.lin1(x)))
        x = self.drop(F.relu(self.lin2(x)))

        return (
            self.head_logS(x).squeeze(-1),
            self.head_logP(x).squeeze(-1),
            self.head_NR(x).squeeze(-1),
        )


bench = pd.read_csv(os.path.join(DATA_DIR, "benchmark.csv"))
train_idx = np.load(os.path.join(DATA_DIR, "train_idx.npy"))
test_idx = np.load(os.path.join(DATA_DIR, "test_idx.npy"))

train_full = bench.iloc[train_idx].reset_index(drop=True)
test_df = bench.iloc[test_idx].reset_index(drop=True)

for c in ["y_logS", "y_logP", "y_nrar"]:
    train_full[c] = pd.to_numeric(train_full[c], errors="coerce")
    test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

n = len(train_full)
perm = np.random.permutation(n)
val_n = int(VAL_FRAC * n)
val_df = train_full.iloc[perm[:val_n]].reset_index(drop=True)
train_df = train_full.iloc[perm[val_n:]].reset_index(drop=True)

scS, scP = StandardScaler(), StandardScaler()
for df in [train_df, val_df, test_df]:
    df["y_logS_scaled"] = np.nan
    df["y_logP_scaled"] = np.nan

hasS = train_df["y_logS"].notna().sum() > 5
hasP = train_df["y_logP"].notna().sum() > 5

if hasS:
    m = train_df["y_logS"].notna()
    train_df.loc[m, "y_logS_scaled"] = scS.fit_transform(train_df.loc[m, ["y_logS"]]).ravel()
    for df in [val_df, test_df]:
        m = df["y_logS"].notna()
        if m.sum() > 0:
            df.loc[m, "y_logS_scaled"] = scS.transform(df.loc[m, ["y_logS"]]).ravel()

if hasP:
    m = train_df["y_logP"].notna()
    train_df.loc[m, "y_logP_scaled"] = scP.fit_transform(train_df.loc[m, ["y_logP"]]).ravel()
    for df in [val_df, test_df]:
        m = df["y_logP"].notna()
        if m.sum() > 0:
            df.loc[m, "y_logP_scaled"] = scP.transform(df.loc[m, ["y_logP"]]).ravel()


def build_ds(df):
    out, drop = [], 0
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
            drop += 1
            continue
        out.append(g)
    return out


train_ds = build_ds(train_df)
val_ds = build_ds(val_df)
test_ds = build_ds(test_df)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

y_nr = train_df["y_nrar"].dropna().values
pos = float((y_nr == 1).sum())
neg = float((y_nr == 0).sum())
pos_weight = torch.tensor([neg / pos], device=DEVICE) if pos > 0 and neg > 0 else None

model = GNNPlus(train_ds[0].x.shape[1], train_ds[0].edge_attr.shape[1]).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)


def batch_loss(batch):
    outS, outP, outN = model(batch)
    y = y_as_matrix(batch.y)

    loss = torch.tensor(0.0, device=DEVICE)

    m = ~torch.isnan(y[:, 0])
    if m.sum() > 0:
        loss += F.mse_loss(outS[m], y[m, 0])

    m = ~torch.isnan(y[:, 1])
    if m.sum() > 0:
        loss += F.mse_loss(outP[m], y[m, 1])

    m = ~torch.isnan(y[:, 2])
    if m.sum() > 0:
        if pos_weight is None:
            loss += F.binary_cross_entropy_with_logits(outN[m], y[m, 2])
        else:
            loss += F.binary_cross_entropy_with_logits(outN[m], y[m, 2], pos_weight=pos_weight)

    return loss


best_val = np.inf
best_state = None
bad = 0

for ep in range(1, EPOCHS + 1):
    model.train()
    for b in train_loader:
        b = b.to(DEVICE)
        opt.zero_grad()
        loss = batch_loss(b)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        vals = [batch_loss(b.to(DEVICE)).item() for b in val_loader]
    val_loss = float(np.mean(vals)) if vals else np.inf
    sch.step(val_loss)

    if val_loss < best_val - 1e-4:
        best_val = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        bad = 0
    else:
        bad += 1
        if bad >= PATIENCE:
            break

if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
logS_t, logS_p = [], []
logP_t, logP_p = [], []
nr_t, nr_p = [], []

with torch.no_grad():
    for b in test_loader:
        b = b.to(DEVICE)
        outS, outP, outN = model(b)
        y = y_as_matrix(b.y)

        m = ~torch.isnan(y[:, 0])
        if m.sum() > 0 and hasS:
            logS_t += scS.inverse_transform(y[m, 0].unsqueeze(1).cpu()).ravel().tolist()
            logS_p += scS.inverse_transform(outS[m].unsqueeze(1).cpu()).ravel().tolist()

        m = ~torch.isnan(y[:, 1])
        if m.sum() > 0 and hasP:
            logP_t += scP.inverse_transform(y[m, 1].unsqueeze(1).cpu()).ravel().tolist()
            logP_p += scP.inverse_transform(outP[m].unsqueeze(1).cpu()).ravel().tolist()

        m = ~torch.isnan(y[:, 2])
        if m.sum() > 0:
            nr_t += y[m, 2].cpu().tolist()
            nr_p += torch.sigmoid(outN[m]).cpu().tolist()

rows = []
if len(logS_t) > 20:
    rows.append({"model": "GNN_PLUS_IMPROVED", "task": "logS", "R2": r2_score(logS_t, logS_p), "MAE": mean_absolute_error(logS_t, logS_p)})
if len(logP_t) > 20:
    rows.append({"model": "GNN_PLUS_IMPROVED", "task": "logP", "R2": r2_score(logP_t, logP_p), "MAE": mean_absolute_error(logP_t, logP_p)})
if len(nr_t) > 20 and len(np.unique(np.array(nr_t).astype(int))) == 2:
    rows.append({"model": "GNN_PLUS_IMPROVED", "task": "NR-AR", "ROC_AUC": roc_auc_score(nr_t, nr_p), "PR_AUC": average_precision_score(nr_t, nr_p), "ACC@0.5": accuracy_score(np.array(nr_t).astype(int), (np.array(nr_p) >= 0.5).astype(int))})

pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "gnn_plus_improved_metrics.csv"), index=False)

if logS_t:
    pd.DataFrame({"y_true": logS_t, "y_pred": logS_p}).to_csv(os.path.join(OUT_DIR, "gnn_plus_improved_pred_logS.csv"), index=False)
if logP_t:
    pd.DataFrame({"y_true": logP_t, "y_pred": logP_p}).to_csv(os.path.join(OUT_DIR, "gnn_plus_improved_pred_logP.csv"), index=False)
if nr_t:
    pd.DataFrame({"y_true": nr_t, "y_prob": nr_p}).to_csv(os.path.join(OUT_DIR, "gnn_plus_improved_pred_NRAR.csv"), index=False)

torch.save(model.state_dict(), os.path.join(MODEL_DIR, "gnn_plus_improved.pt"))

print("Training completed")
