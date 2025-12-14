ChemAI – Multi-property Chemical Prediction Framework

ChemAI is a unified chemoinformatics framework for predicting multiple chemical properties using both classical machine-learning models and graph neural networks (GNNs). The project integrates baseline Random Forest (RF) models with an experimental GNN+ architecture to benchmark performance across physicochemical and toxicological endpoints.

This repository accompanies an undergraduate research project in Chemistry and is designed for reproducibility, benchmarking, and academic dissemination.

Predicted Properties

ChemAI currently supports prediction of the following properties:

Aqueous solubility (logS) – regression

Lipophilicity (logP) – regression

Tox21 NR-AR probability – binary classification

Baseline models use molecular fingerprints, while the GNN+ models operate directly on molecular graphs.

Modeling Approaches
Baseline (Random Forest)

Molecular fingerprints (RDKit)

Scikit-learn Random Forest regressors/classifiers

Strong and interpretable baseline performance

GNN+ (Experimental)

Graph Neural Network with multitask learning

Node- and edge-level molecular representations

Evaluated alongside RF for comparative analysis

Note: GNN outputs for regression tasks may be in scaled space unless inverse transformation is applied.

Repository Structure
ChemAI/
├── 00_fetch_esol.py
├── 01_make_benchmark.py
├── 02_eval_baseline.py
├── 03_baseline_plots.py
├── 04_train_gnn_multitask.py
├── 06_train_gnn_plus.py
├── 07_train_gnn_plus_improved.py
├── 08_make_comparison_plots.py
├── app.py                     # Streamlit demo app (local)
├── chem-ai-2.ipynb             # Development notebook
├── .gitignore
└── README.md


Large datasets, trained models, and intermediate result files are excluded from version control.

Streamlit Web Application

A local Streamlit application is included for demonstration purposes:

Input: SMILES string

Output: Predictions from

Random Forest (baseline)

GNN+ (experimental)

The app is intended for local demo and screenshots.
Public deployment will be released after further validation and robustness testing.

Evaluation & Comparison

Model performance is assessed using:

Regression: R², MAE

Classification: ROC-AUC, PR-AUC, Accuracy

Comparative plots (RF vs GNN+) are generated using the provided analysis scripts.

Data & Model Availability

Due to size constraints and licensing considerations:

Raw datasets

Trained model weights

Intermediate result files

are not included in this repository.

Scripts are provided to reproduce the full pipeline given access to the original data sources.

Intended Use

This project is intended for:

Academic research and benchmarking

Educational use in chemoinformatics

Method comparison (classical ML vs GNNs)

It is not intended for regulatory or clinical decision-making.

Author

Md Rahat Chowdhury
Undergraduate Research Project, Chemistry
Begum Rokeya University, Rangpur, Bangladesh

License

This project is released for academic and research use only.
Commercial use requires explicit permission from the author.