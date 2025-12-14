import os
import numpy as np
import pandas as pd
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

GNN_PATH = os.path.join(MODEL_DIR, "gnn_plus_improved.pt")
HAS_GNN = os.path.exists(GNN_PATH)


gen = GetMorganGenerator(radius=2, fpSize=2048)

def fp_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((2048,), dtype=np.int8)
    for i in range(2048):
        arr[i] = fp.GetBit(i)
    return arr.astype(np.float32)


ATOM_LIST = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]

def atom_features(atom):
    one_hot = [int(atom.GetAtomicNum() == a) for a in ATOM_LIST]
    hyb = int(atom.GetHybridization())
    return one_hot + [
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        float(int(atom.GetIsAromatic())),
        float(hyb),
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


class GNNPlus(nn.Module):
    def __init__(self, node_in, edge_in):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
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


@st.cache_resource
def load_models():
    models = {}
    models["rf_logp"] = load(RF_LOGP_PATH) if os.path.exists(RF_LOGP_PATH) else None
    models["rf_logs"] = load(RF_LOGS_PATH) if os.path.exists(RF_LOGS_PATH) else None
    models["rf_nrar"] = load(RF_NRAR_PATH) if os.path.exists(RF_NRAR_PATH) else None

    if HAS_GNN:
        dummy = smiles_to_graph("CCO")
        gnn = GNNPlus(dummy.x.shape[1], dummy.edge_attr.shape[1])
        gnn.load_state_dict(torch.load(GNN_PATH, map_location="cpu"))
        gnn.eval()
        models["gnn"] = gnn
    else:
        models["gnn"] = None

    return models


models = load_models()

st.set_page_config(page_title="ChemAI – Multi-property Predictor", layout="centered")
st.title("ChemAI – Multi-property Chemical Predictor")

st.write(
    "Enter a **SMILES** string to obtain predictions for **logS**, **logP**, "
    "and **Tox21 NR-AR probability**."
)

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
        st.subheader("Random Forest models")

        if models["rf_logs"] is not None:
            st.write(f"logS: {float(models['rf_logs'].predict(X)[0]):.3f}")
        else:
            st.write("logS model unavailable.")

        if models["rf_logp"] is not None:
            st.write(f"logP: {float(models['rf_logp'].predict(X)[0]):.3f}")
        else:
            st.write("logP model unavailable.")

        if models["rf_nrar"] is not None:
            prob = models["rf_nrar"].predict_proba(X)[0, 1]
            st.write(f"NR-AR probability: {float(prob):.3f}")
        else:
            st.write("NR-AR model unavailable.")

    with col2:
        st.subheader("GNN model")

        if models["gnn"] is None:
            st.write("GNN model unavailable.")
        else:
            g = smiles_to_graph(smiles)
            if g is None:
                st.write("Graph construction failed.")
            else:
                with torch.no_grad():
                    outS, outP, outN = models["gnn"](g)
                    st.write(f"logS: {float(outS.item()):.3f}")
                    st.write(f"logP: {float(outP.item()):.3f}")
                    st.write(f"NR-AR probability: {float(torch.sigmoid(outN).item()):.3f}")

st.caption(
    "ChemAI © 2025 | Developed by Md Rahat Chowdhury | Undergraduate Research Project, Chemistry"
)

with st.expander("About"):
    st.write(
        "ChemAI is a research prototype for multi-property chemical prediction. "
        "Baseline models use molecular fingerprints with Random Forest, while "
        "graph neural networks are evaluated as an alternative representation."
    )
