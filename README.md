ChemAI – Multi-property Chemical Prediction Framework
Overview

ChemAI is a unified chemoinformatics framework for multi-property chemical prediction using both classical machine-learning models and graph neural networks (GNNs). The project benchmarks a strong Random Forest (RF) baseline against an experimental GNN+ architecture across physicochemical and toxicological endpoints.

This repository accompanies an academic research project and is designed with a focus on reproducibility, transparent benchmarking, and methodological comparison between fingerprint-based and graph-based modeling approaches.

Predicted Properties

ChemAI currently supports prediction of the following molecular properties:

Aqueous solubility (logS) – regression

Lipophilicity (logP) – regression

Tox21 NR-AR activity – binary classification (probability output)

Baseline models rely on molecular fingerprints, while GNN+ models learn directly from molecular graph representations.

Modeling Approaches
Baseline: Random Forest (RF)

RDKit-generated molecular fingerprints

Scikit-learn Random Forest regressors and classifiers

Strong, interpretable baseline performance

Particularly effective under sparse or imbalanced data conditions

GNN+: Graph Neural Network (Experimental)

Multitask graph neural network architecture

Node- and edge-level molecular representations

End-to-end learning of structure–property relationships

Evaluated alongside RF models for direct comparison

Note: GNN regression outputs may be reported in scaled space unless inverse scaling is applied using training scalers.

```markdown
Repository Structure
ChemAI/
├── 00_fetch_esol_direct.py
├── 01_make_benchmark.py
├── 02_eval_baseline.py
├── 03_baseline_plots.py
├── 06_train_gnn_plus.py
├── 07_train_gnn_plus_improved.py
├── 08_make_comparison_plots.py
├── app.py                     # Streamlit demo application (local)
├── .gitignore
└── README.md


Large datasets, trained model weights, and intermediate result files are intentionally excluded from version control to keep the repository lightweight and publicly shareable.

Evaluation and Comparison

Model performance is assessed using standard metrics:

Regression: R², Mean Absolute Error (MAE)

Classification: ROC-AUC, PR-AUC, Accuracy

Comparative analyses include:

True vs. predicted scatter plots

ROC curves for NR-AR classification

Metric-based performance comparisons between RF and GNN+ models

All plots are generated using the provided analysis scripts.

Streamlit Web Application

A Streamlit-based web application is included for interactive demonstration.

Input:

SMILES string

Output:

Predictions from Random Forest (baseline)

Predictions from GNN+ (experimental)

The application is currently provided as a local demo for testing, visualization, and screenshots. Public deployment will follow additional validation and robustness testing.

Data and Code Availability

All datasets used in this project originate from publicly available sources.
Due to size constraints and licensing considerations, the following are not included in this repository:

Raw datasets

Trained model checkpoints

Intermediate result files

Scripts are provided to fully reproduce the modeling and evaluation pipeline, given access to the original data sources.

GitHub Repository:
https://github.com/rahatchowdhury8899-hub/ChemAI

Intended Use

This project is intended for:

Academic research and benchmarking

Educational use in chemoinformatics and machine learning

Methodological comparison between classical ML and GNN approaches

It is not intended for regulatory, clinical, or decision-making use.

Author

Md Rahat Chowdhury
Undergraduate Research Project, Chemistry
Begum Rokeya University, Rangpur, Bangladesh

License

This project is released under the MIT License.
See the LICENSE file for details.