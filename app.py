import os
import numpy as np
import streamlit as st
from joblib import load

from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool

MODEL_DIR = "models"

RF_LOGP_PATH = os.path.join(MODEL_DIR, "rf_logp.joblib")
RF_LOGS_PATH = os.path.join(MODEL_DIR, "rf_esol_solubility.joblib")
RF_NRAR_PATH = os.path.join(MODEL_DIR, "rf_tox21_NRAR.joblib")

GNN_CANDIDATES = [
    os.path.join(MODEL_DIR, "gnn_plus_improved.pt"),
    os.path.join(MODEL_DIR, "gnn_multitask.pt"),
]
GNN_PATH = next((p for p in GNN_CANDIDATES if os.path.exists(p)), None)

fp_gen = GetMorganGenerator(radius=2, fpSize=2048)

def fp_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = fp_gen.GetFingerprint(mol)
    arr = np.zeros((2048,), dtype=np.float32)
    for i in range(2048):
        arr[i] = fp.GetBit(i)
    return arr

ATOM_LIST = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]

def atom_features(atom):
    one_hot = [int(atom.GetAtomicNum() == a) for a in ATOM_LIST]
    return one_hot + [
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        float(int(atom.GetIsAromatic())),
        float(int(atom.GetHybridization())),
    ]

def bond_features(bond):
    bt = bond.GetBondType()
    return [
        float(bt == Chem.rdchem.BondType.SINGLE),
        float(bt == Chem.rdchem.BondType.DOUBLE),
        float(bt == Chem.rdchem.BondType.TRIPLE),
        float(bt == Chem.rdchem.BondType.AROMATIC),
        float(int(bond.GetIsConjugated())),
        float(int(bond.IsInRing())),
    ]

def smiles_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
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
    d.batch = torch.zeros(d.x.size(0), dtype=torch.long)
    return d

class GNNPlus256(nn.Module):
    def __init__(self, node_in, edge_in):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, 256), nn.ReLU(), nn.Linear(256, 256)
        )

        nn1 = nn.Sequential(nn.Linear(node_in, 256), nn.ReLU(), nn.Linear(256, 256))
        nn2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))

        self.conv1 = GINEConv(nn=nn1, edge_dim=256)
        self.conv2 = GINEConv(nn=nn2, edge_dim=256)

        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 64)

        self.head_logS = nn.Linear(64, 1)
        self.head_logP = nn.Linear(64, 1)
        self.head_NR = nn.Linear(64, 1)

    def forward(self, data):
        ea = self.edge_mlp(data.edge_attr)
        x = F.relu(self.conv1(data.x, data.edge_index, ea))
        x = F.relu(self.conv2(x, data.edge_index, ea))
        x = global_mean_pool(x, data.batch)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return (
            self.head_logS(x).squeeze(-1),
            self.head_logP(x).squeeze(-1),
            self.head_NR(x).squeeze(-1),
        )

class GNNPlusImproved(nn.Module):
    def __init__(self, node_in, edge_in, hidden=128, shared_out=64):
        super().__init__()

        nn1 = nn.Sequential(nn.Linear(node_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        nn2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

        self.conv1 = GINEConv(nn=nn1, edge_dim=edge_in)
        self.conv2 = GINEConv(nn=nn2, edge_dim=edge_in)

        self.fc_shared = nn.Linear(hidden, shared_out)
        self.head_logS = nn.Linear(shared_out, 1)
        self.head_logP = nn.Linear(shared_out, 1)
        self.head_NR = nn.Linear(shared_out, 1)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = F.relu(self.conv2(x, data.edge_index, data.edge_attr))
        x = global_mean_pool(x, data.batch)
        x = F.relu(self.fc_shared(x))
        return (
            self.head_logS(x).squeeze(-1),
            self.head_logP(x).squeeze(-1),
            self.head_NR(x).squeeze(-1),
        )

def build_gnn_from_checkpoint(ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cpu")

    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    has_fc_shared = any(k.startswith("fc_shared.") for k in sd.keys())
    has_edge_mlp = any(k.startswith("edge_mlp.") for k in sd.keys())

    dummy = smiles_to_graph("CCO")
    if dummy is None:
        return None

    node_in = int(dummy.x.shape[1])
    edge_in = int(dummy.edge_attr.shape[1])

    if has_fc_shared and not has_edge_mlp:
        hidden = int(sd["conv1.nn.0.weight"].shape[0]) if "conv1.nn.0.weight" in sd else 128
        shared_out = int(sd["fc_shared.weight"].shape[0]) if "fc_shared.weight" in sd else 64
        model = GNNPlusImproved(node_in=node_in, edge_in=edge_in, hidden=hidden, shared_out=shared_out)
    else:
        model = GNNPlus256(node_in=node_in, edge_in=edge_in)

    model.load_state_dict(sd, strict=True)
    model.eval()
    return model

@st.cache_resource
def load_models():
    models = {
        "rf_logp": load(RF_LOGP_PATH) if os.path.exists(RF_LOGP_PATH) else None,
        "rf_logs": load(RF_LOGS_PATH) if os.path.exists(RF_LOGS_PATH) else None,
        "rf_nrar": load(RF_NRAR_PATH) if os.path.exists(RF_NRAR_PATH) else None,
        "gnn": None,
        "gnn_path": GNN_PATH,
    }

    if GNN_PATH is not None:
        try:
            models["gnn"] = build_gnn_from_checkpoint(GNN_PATH)
        except Exception as e:
            models["gnn"] = None
            models["gnn_error"] = str(e)

    return models

models = load_models()

st.set_page_config(page_title="ChemAI – Multi-property Predictor", layout="centered")
st.title("ChemAI – Multi-property Chemical Predictor")

st.write("Enter a **SMILES** string to obtain predictions for **logS**, **logP**, and **Tox21 NR-AR probability**.")

smiles = st.text_input("SMILES", value="CCO")
run = st.button("Predict")

if run:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES.")
        st.stop()

    fp = fp_from_smiles(smiles)
    if fp is None:
        st.error("Fingerprint generation failed.")
        st.stop()

    X = fp.reshape(1, -1)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Baseline (Random Forest)")

        if models["rf_logs"] is not None:
            st.write(f"logS (RF): {float(models['rf_logs'].predict(X)[0]):.3f}")
        else:
            st.write("logS model unavailable.")

        if models["rf_logp"] is not None:
            st.write(f"logP (RF): {float(models['rf_logp'].predict(X)[0]):.3f}")
        else:
            st.write("logP model unavailable.")

        if models["rf_nrar"] is not None:
            prob = models["rf_nrar"].predict_proba(X)[0, 1]
            st.write(f"NR-AR probability (RF): {float(prob):.3f}")
        else:
            st.write("NR-AR model unavailable.")

    with col2:
        st.subheader("GNN+ (Improved)")
        

        if models.get("gnn_path") is None:
            st.write("GNN model file not found in models/.")
        elif models["gnn"] is None:
            st.write("GNN model could not be loaded.")
            if "gnn_error" in models:
                st.code(models["gnn_error"])
        else:
            g = smiles_to_graph(smiles)
            if g is None:
                st.write("Graph construction failed.")
            else:
                with torch.no_grad():
                    outS, outP, outN = models["gnn"](g)
                st.write(f"logS (GNN): {float(outS.item()):.3f}")
                st.write(f"logP (GNN): {float(outP.item()):.3f}")
                st.write(f"NR-AR probability (GNN): {float(torch.sigmoid(outN).item()):.3f}")

st.caption("ChemAI © 2025 | Developed by Md Rahat Chowdhury")

st.caption(
    "Predictions are intended for research and educational use only."
)
