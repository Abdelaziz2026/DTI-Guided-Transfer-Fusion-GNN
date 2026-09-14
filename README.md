# Diffusion Tensor Imaging-Guided Transfer Fusion Graph Neural Network (DGTF)

DGTF is a multimodal graph-learning framework for Alzheimer's disease (AD) diagnosis. It uses **diffusion tensor imaging (DTI)-derived structural connectivity (SC)** to guide **resting-state functional MRI (rs-fMRI)-derived functional connectivity (FC)** learning and combines imaging and clinical information for final prediction.

This repository contains the **NC vs AD** implementation of the DGTF framework.

## Method overview

DGTF consists of four main components:

### 1. DTI structural graph learning

Subject-specific DTI structural connectivity is represented as a brain graph and encoded using a graph attention network (GAT). The DTI branch produces a structural graph representation and estimates the relevance of individual brain regions.

### 2. DTI-guided node reweighting

Region relevance is estimated from the DTI branch using gradient-based saliency. The most relevant structural regions receive larger node weights, while all brain regions remain in the graph.

The resulting weights are transferred to anatomically corresponding rs-fMRI nodes. This provides structural guidance to the functional branch without removing nodes or changing the whole-brain graph topology.

### 3. rs-fMRI functional graph learning

Dynamic functional connectivity is aggregated into a subject-level functional connectivity representation. DTI-derived node weights are applied to the rs-fMRI node features before graph encoding.

The rs-fMRI branch then learns a functional graph representation under structural guidance.

### 4. Clinical context and multimodal fusion

Clinical information is represented using structured demographic/cognitive variables together with diagnosis-excluded clinical text encoded using frozen PubMedBERT.

The final representation combines:

```text
DTI embedding
+ rs-fMRI embedding
+ structured clinical information
+ PubMedBERT clinical embedding
```

A multilayer perceptron (MLP) produces the final **NC vs AD** prediction.

## Framework

```text
DTI structural connectivity
        |
        v
   DTI GAT encoder
        |
        +-----------------------> DTI embedding ----------------------+
        |                                                           |
        v                                                           |
Gradient-based node relevance                                       |
        |                                                           |
        v                                                           |
DTI-guided soft node weights                                        |
        |                                                           |
        +-------> rs-fMRI node reweighting                           |
                           |                                        |
                           v                                        |
                      rs-fMRI GAT                                   |
                           |                                        |
                           +------------> rs-fMRI embedding ---------+
                                                                    |
Structured clinical information ------------------------------------+
                                                                    |
Diagnosis-excluded PubMedBERT clinical embedding -------------------+
                                                                    |
                                                                    v
                                                               Fusion MLP
                                                                    |
                                                                    v
                                                               NC vs AD
```

## Data

The repository expects processed connectivity-level inputs rather than raw MRI volumes.

Required inputs include:

- subject-specific DTI structural-connectivity matrices;
- DTI node features when available;
- rs-fMRI dynamic functional-connectivity arrays;
- diagnostic labels;
- demographic and clinical information.

ADNI data are not redistributed in this repository. Users must obtain the data through the appropriate ADNI access procedures.

## Repository structure

```text
DTI-Guided-Transfer-Fusion-GNN/
├── README.md
├── demo.py
├── data_processing.py
├── model.py
├── node_reweighting.py
├── clinical_encoder.py
├── utils.py
├── requirements.txt
└── .gitignore
```

- `demo.py` — NC-vs-AD DGTF experiment runner.
- `data_processing.py` — data loading, task construction, graph preparation, and rs-fMRI connectivity aggregation.
- `model.py` — graph encoders, model training, embedding extraction, and fusion classifier.
- `node_reweighting.py` — DTI-based region relevance, soft node weighting, and transfer to rs-fMRI.
- `clinical_encoder.py` — clinical-text generation and PubMedBERT-based clinical representation.
- `utils.py` — configuration, metrics, logging, early stopping, and experiment utilities.

## Installation

```bash
pip install -r requirements.txt
```

Install PyTorch and PyTorch Geometric versions compatible with your CUDA environment when GPU execution is required.

## Run

Before execution, update the configured data paths in the code for your environment.

```bash
python demo.py
```

Results are saved under the configured output directory.
