import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)

DATA_DIR = "data"
RESULTS_DIR = "results"
MODEL_DIR = "models"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

BENCH = os.path.join(DATA_DIR, "benchmark.csv")
TRAIN_IDX = os.path.join(DATA_DIR, "train_idx.npy")
TEST_IDX = os.path.join(DATA_DIR, "test_idx.npy")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def atom_features(atom):
    return torch.tensor(
        [
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            1.0 if atom.GetIsAromatic() else 0.0,
        ],
        dtype=torch.float,
    )


def mol_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    nodes = [atom_features(a) for a in mol.GetAtoms()]
    if not nodes:
        return None
    x = torch.stack(nodes)

    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

    if not edges:
        return None

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


class ChemDataset(Dataset):
    def __init__(self, df, indices):
        super().__init__()
        self.items = []

        subset = df.iloc[indices].reset_index(drop=True)

        for _, row in subset.iterrows():
            g = mol_to_graph(row["SMILES"])
            if g is None:
                continue

            g.y_logS = torch.tensor(row.get("y_logS", np.nan), dtype=torch.float)
            g.y_logP = torch.tensor(row.get("y_logP", np.nan), dtype=torch.float)
            g.y_nrar = torch.tensor(row.get("y_nrar", np.nan), dtype=torch.float)
            g.smiles = str(row["SMILES"])

            self.items.append(g)

    def len(self):
        return len(self.items)

    def get(self, idx):
        return self.items[idx]


class MultiTaskGNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, 64)
        self.conv2 = GCNConv(64, 128)

        self.head_logS = nn.Linear(128, 1)
        self.head_logP = nn.Linear(128, 1)
        self.head_nrar = nn.Linear(128, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)

        return {
            "logS": self.head_logS(x).squeeze(-1),
            "logP": self.head_logP(x).squeeze(-1),
            "nrar_logits": self.head_nrar(x).squeeze(-1),
        }


def masked_mse(pred, y):
    m = ~torch.isnan(y)
    return F.mse_loss(pred[m], y[m]) if m.sum() > 0 else torch.tensor(0.0, device=pred.device)


def masked_bce_logits(logits, y):
    m = ~torch.isnan(y)
    if m.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits[m], y[m])


df = pd.read_csv(BENCH)
train_idx = np.load(TRAIN_IDX)
test_idx = np.load(TEST_IDX)

train_ds = ChemDataset(df, train_idx)
test_ds = ChemDataset(df, test_idx)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

model = MultiTaskGNN(in_dim=4).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 30
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    for batch in train_loader:
        batch = batch.to(DEVICE)
        out = model(batch)

        loss = (
            masked_mse(out["logS"], batch.y_logS)
            + masked_mse(out["logP"], batch.y_logP)
            + masked_bce_logits(out["nrar_logits"], batch.y_nrar)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss.item())

    if epoch % 5 == 0:
        print(f"Epoch {epoch}/{EPOCHS} | loss={epoch_loss:.4f}")

model.eval()

smiles_all = []
y_logP_all, p_logP_all = [], []
y_nrar_all, p_nrar_all = [], []

with torch.no_grad():
    for batch in test_loader:
        smiles_all.extend(batch.smiles)
        batch = batch.to(DEVICE)
        out = model(batch)

        p_logP = out["logP"].cpu().numpy()
        p_nrar = torch.sigmoid(out["nrar_logits"]).cpu().numpy()

        y_logP_all.extend(batch.y_logP.cpu().numpy())
        p_logP_all.extend(p_logP)

        y_nrar_all.extend(batch.y_nrar.cpu().numpy())
        p_nrar_all.extend(p_nrar)

y_logP = np.array(y_logP_all, dtype=float)
p_logP = np.array(p_logP_all, dtype=float)
m = ~np.isnan(y_logP)
logP_r2 = r2_score(y_logP[m], p_logP[m]) if m.sum() > 2 else np.nan
logP_mae = mean_absolute_error(y_logP[m], p_logP[m]) if m.sum() > 2 else np.nan

y_nrar = np.array(y_nrar_all, dtype=float)
p_nrar = np.array(p_nrar_all, dtype=float)
m = ~np.isnan(y_nrar)

if m.sum() > 10 and len(np.unique(y_nrar[m].astype(int))) == 2:
    roc = roc_auc_score(y_nrar[m].astype(int), p_nrar[m])
    pr = average_precision_score(y_nrar[m].astype(int), p_nrar[m])
    acc = accuracy_score(y_nrar[m].astype(int), (p_nrar[m] >= 0.5).astype(int))
else:
    roc = pr = acc = np.nan

metrics = pd.DataFrame(
    [
        {"model": "GNN", "task": "logP", "R2": logP_r2, "MAE": logP_mae},
        {"model": "GNN", "task": "NR-AR", "ROC_AUC": roc, "PR_AUC": pr, "ACC@0.5": acc},
    ]
)
metrics.to_csv(os.path.join(RESULTS_DIR, "gnn_metrics.csv"), index=False)

pred_df = pd.DataFrame(
    {
        "SMILES": smiles_all,
        "y_logP": y_logP_all,
        "y_nrar": y_nrar_all,
        "pred_logP_GNN": p_logP_all,
        "pred_nrar_GNN": p_nrar_all,
    }
)
pred_df.to_csv(os.path.join(RESULTS_DIR, "gnn_pred_test.csv"), index=False)

torch.save(model.state_dict(), os.path.join(MODEL_DIR, "gnn_multitask.pt"))

print("GNN training and evaluation completed")
print(metrics)
