# DTI-Guided-Transfer-Fusion-GNN

This repository provides the implementation of **DGTF**, a DTI-guided transfer fusion graph-learning framework for multimodal Alzheimer's disease (AD) diagnosis and mild cognitive impairment (MCI) staging.

DGTF combines **DTI-derived structural connectivity (SC)**, **rs-fMRI-derived functional connectivity (FC)**, and optional **clinical context**. DTI-derived node relevance is transferred to the functional branch through topology-preserving node reweighting, so structurally informative regions can guide functional graph learning without deleting brain regions or edges.

## 📌 Overview

The pipeline follows six main steps:

1. Train a DTI structural-connectivity graph model.
2. Estimate DTI node relevance using gradient saliency or the optional centrality selector.
3. Convert the selected regions into soft node weights while retaining the complete graph.
4. Transfer the aligned node weights to the rs-fMRI branch and train the functional graph model.
5. Extract DTI and rs-fMRI subject-level graph embeddings for subjects shared by both modalities.
6. Fuse imaging embeddings with optional demographic and PubMedBERT-derived clinical representations for prediction.

## 🧠 Framework Architecture

```text
DTI structural connectivity
        │
        ▼
   DTI GAT encoder
        │
        ├──► DTI embedding ───────────────────────┐
        │                                         │
        ▼                                         │
Node relevance / Top-K selection                  │
        │                                         │
        ▼                                         │
Soft node weights (γ)                             │
        │                                         │
        └──────────► rs-fMRI node reweighting     │
                         │                        │
                         ▼                        │
                    fMRI GAT encoder              │
                         │                        │
                         └──► fMRI embedding ─────┤
                                                  ▼
Clinical / demographic context ─────────────► Fusion MLP
                                                  │
                                                  ▼
                                      AD / MCI classification
```

## ✨ Key Features

- **Structure-guided functional learning:** DTI-derived relevance guides rs-fMRI representation learning.
- **Topology-preserving reweighting:** all graph nodes remain available; selected regions receive greater weight rather than being removed.
- **Multimodal fusion:** DTI, rs-fMRI, numeric clinical covariates, and optional PubMedBERT clinical embeddings can be combined.
- **Multiple diagnostic tasks:** the supplied implementation supports NC vs AD, NC vs MCI, EMCI vs LMCI, NC vs EMCI, NC vs EMCI vs LMCI, and MCI vs AD task definitions.
- **Reproducible evaluation utilities:** stratified splitting, deterministic seeding, early stopping, class-weighted cross-entropy, ROC/confusion-matrix reporting, and result export are included.

## 📁 Repository Structure

```text
DGTF/
├── README.md
├── demo.py                 # experiment configuration and executable workflow
├── data_processing.py      # data discovery, task construction, graph datasets
├── model.py                # GAT encoders, training loops, embedding extraction, fusion MLP
├── node_reweighting.py     # DTI saliency/centrality and DTI→fMRI node-weight transfer
├── clinical_encoder.py     # demographic handling and optional PubMedBERT encoding
├── utils.py                # configuration, logging, metrics, plotting, reproducibility
└── requirements.txt
```

## 🔧 Requirements

- Python 3.9+
- PyTorch
- PyTorch Geometric
- NumPy / pandas
- scikit-learn
- Transformers (for PubMedBERT clinical embeddings)
- Matplotlib
- NetworkX (only when the centrality-based node selector is used)
- openpyxl (for Excel demographic files)

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

> **PyTorch / PyTorch Geometric:** for CUDA environments, install the builds appropriate for your CUDA version before installing the remaining requirements when necessary.

## 📊 Expected Data Inputs

The supplied implementation expects the following data sources:

- DTI subject-specific structural-connectivity matrices.
- Optional DTI node-feature files.
- rs-fMRI dynamic functional-connectivity (`*_dfc.npy`) files.
- Diagnostic labels in JSON or the supported demographic source.
- Optional demographic/clinical information in Excel format.

The default paths in `demo.py` follow the Kaggle paths used in the original research script. Edit those paths to match your environment before running the code.

## ▶️ Usage

### 1. Configure the experiment

Open `demo.py` and edit the data paths and experiment settings. The default executable configuration preserves the supplied **NC vs AD / `dgtf_full`** single-ablation runner.

### 2. Run DGTF

```bash
python demo.py
```

Results are written to the configured result directory, including JSON summaries, figures, node-selection outputs, and downloadable archives when supported by the environment.

### 3. Change the diagnostic task

Set `TASK_NAME` in `demo.py` to one of the supported task keys, for example:

```python
TASK_NAME = "EMCI_LMCI"
```

Supported task keys are:

```text
NC_AD
NC_MCI
EMCI_LMCI
NC_EMCI
NC_EMCI_LMCI
MCI_AD
```

## ⚙️ Main DGTF Parameters

The principal experiment parameters are exposed through `MultiModalGNNConfig` and the configuration blocks in `demo.py`, including:

- `hidden_dim`
- `gnn_num_layers`
- `attention_heads`
- `dti_connectivity_threshold`
- `fmri_connectivity_threshold`
- `node_selection_topk`
- `node_reweight_factor`
- `dti_n_splits`
- `fmri_n_splits`
- `fusion_n_repeats`
- learning rates, weight decay, dropout, and early-stopping settings

## 🧪 Reproducibility

The implementation includes deterministic seed handling for Python/NumPy/PyTorch, stratified validation routines, saved configurations, fold-level metrics, aggregate summaries, and generated evaluation figures.

Dataset redistribution is **not** included in this repository. Users should obtain ADNI data through the appropriate ADNI access procedures and comply with its data-use requirements.

## 📝 Citation

If you use this implementation, please cite the associated manuscript:

```bibtex
@article{abdelaziz2026dgtf,
  title   = {Structural Connectivity-Guided Functional Graph Learning for Multimodal Alzheimer's Disease Diagnosis},
  author  = {Abdelaziz, Mohammed and Wang, Changmiao and Gorriz, Juan M. and Elazab, Ahmed},
  year    = {2026},
  note    = {Manuscript}
}
```

Update the citation with the final journal, volume, pages, and DOI after publication.

## 📜 License

Add the license that applies to your code before public release. The ADNI dataset is governed separately by ADNI's data-use terms and is not distributed here.
