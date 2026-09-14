# Diffusion Tensor Imaging-Guided Transfer Fusion Graph Neural Network (DGTF)

**Official implementation of the DGTF framework described in:**  
*Structural Connectivity-Guided Functional Graph Learning for Multimodal Alzheimer's Disease Diagnosis*

DGTF is a multimodal graph-learning framework for Alzheimer's disease (AD) diagnosis that uses **DTI-derived structural connectivity (SC)** to guide **rs-fMRI-derived functional connectivity (FC)** learning. The framework estimates region-level relevance from the DTI branch, transfers the resulting structural weights to anatomically aligned rs-fMRI nodes, preserves the complete graph topology, and combines structural, functional, and clinical representations for prediction.

> **Current release scope.** The executable entry point in `demo.py` is intentionally kept exactly as supplied and is configured for the **NC vs AD** `dgtf_full` experiment. The Python implementation has not been altered in this documentation revision.

## Method overview

The manuscript formulation and the corresponding implementation are organized around the following stages.

### 1. DTI structural-connectivity graph

Each subject is represented by a weighted structural graph. The supplied implementation loads a subject-specific SC matrix, replaces non-finite entries, symmetrizes the matrix, and constructs edges by retaining off-diagonal connections satisfying

```text
|W_ij| > tau_SC ,  i != j
```

When explicit DTI node features are available they are used directly; otherwise, the node connectivity profile is used as the node representation.

### 2. Structural graph encoder

The DTI branch uses a multilayer graph attention network (GAT). The manuscript configuration uses:

- 3 GAT layers
- hidden dimension = 128
- 8 attention heads
- edge weights supplied as edge attributes
- BatchNorm + ELU after GAT layers
- global mean pooling concatenated with global max pooling

The resulting graph-level representation is the DTI embedding.

### 3. DTI-guided node relevance and topology-preserving reweighting

Node relevance is estimated from the DTI branch using **gradient saliency**. Absolute input gradients are averaged over feature dimensions, subjects, and training batches to produce one relevance score per anatomical region.

The top-`K` DTI regions are selected and converted to soft weights:

```text
selected node     -> gamma
unselected node   -> 1
```

The main NC-vs-AD configuration uses:

```text
K = 25
gamma = 2.5
```

No selected or unselected region is deleted. The weighting changes the contribution of node features while retaining the complete graph topology. The same anatomically aligned weight vector is adapted to the rs-fMRI node dimension when necessary and transferred to the functional branch.

The code also contains auxiliary selector/prior utilities. The **main DGTF configuration uses gradient-based saliency**; alternative selector code paths should not be interpreted as the primary methodology unless explicitly enabled.

### 4. rs-fMRI functional-connectivity graph under structural guidance

The rs-fMRI input is a dynamic FC sequence. In the supplied implementation, the sequence is converted to a subject-level FC matrix by temporal averaging, followed by hyperbolic-tangent stabilization, symmetrization, and removal of diagonal self-connections.

Functional graph edges are retained when

```text
|C_final,ij| > tau_FC ,  i != j
```

The DTI-derived node weights are then applied multiplicatively to anatomically aligned rs-fMRI node features before GAT message passing. The functional branch uses the same GAT/pooling formulation to produce the rs-fMRI embedding.

### 5. Clinical context

The full DGTF configuration uses two forms of subject-level clinical information:

- a 5-dimensional structured vector based on age, sex, APOE4, MMSE, and CDR;
- a diagnosis-excluded clinical text representation encoded with frozen PubMedBERT.

The clinical text deliberately excludes the diagnostic label in the default full configuration (`clinical_no_diagnosis`) to reduce direct target leakage. The PubMedBERT `[CLS]` representation is projected before fusion.

### 6. Multimodal fusion

The final representation combines the available components in the following order:

```text
DTI embedding | rs-fMRI embedding | structured clinical covariates | clinical text embedding
```

A multilayer perceptron (MLP) produces the final diagnostic prediction.

## Architecture summary

```text
DTI structural connectivity
        |
        v
   DTI GAT encoder
        |
        +-------------------------> DTI embedding ---------------------+
        |                                                             |
        v                                                             |
Gradient-based node relevance                                         |
        |                                                             |
        v                                                             |
     Top-K nodes                                                       |
        |                                                             |
        v                                                             |
Soft node weights (gamma)                                             |
        |                                                             |
        +--------> aligned rs-fMRI node reweighting                    |
                             |                                        |
                             v                                        |
                        rs-fMRI GAT                                   |
                             |                                        |
                             +------------> rs-fMRI embedding --------+
                                                                      |
Structured clinical covariates ---------------------------------------+
                                                                      |
Diagnosis-excluded PubMedBERT clinical embedding ---------------------+
                                                                      |
                                                                      v
                                                                 Fusion MLP
                                                                      |
                                                                      v
                                                               NC vs AD output

## Important implementation and reproducibility notes

The manuscript describes DTI and rs-fMRI as modality-specific cohorts, with final fusion restricted to subjects shared by the two modalities. The current public `main_single_ablation()` entry point first constructs the paired subject intersection for the selected task and then passes that paired subset into the multimodal runner. Therefore, **this exact executable entry point should be described as the current NC-vs-AD paired-subject implementation rather than as a byte-for-byte reproduction of every cohort-level experiment reported in the manuscript**.

The multimodal runner also creates DTI, rs-fMRI, and fusion split plans independently. Consequently, the current code should **not** be described as guaranteeing end-to-end subject-level separation across every upstream branch and the downstream fusion evaluation. The fusion validation split is also used for validation/early stopping and subsequent metric computation in the current implementation.

These statements document the supplied code exactly; no algorithmic change has been made in this repository revision.

## Data and preprocessing

The repository expects **processed connectivity-level inputs**, not raw MRI volumes.

The manuscript preprocessing pipeline includes FSL/MRtrix3 operations such as motion/eddy correction, registration, tractography, SIFT2, and AAL2 parcellation. Those raw-image preprocessing stages are **not reimplemented by this Python repository**. The code consumes their downstream products, including:

- subject-specific DTI structural-connectivity matrices;
- optional DTI node-feature tables;
- rs-fMRI dynamic FC arrays (`*_dfc.npy`);
- diagnostic labels;
- demographic/clinical data.

ADNI data are not redistributed in this repository. Users must obtain data through the appropriate ADNI access procedures and comply with the applicable data-use requirements.

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

- `demo.py` — current NC-vs-AD full DGTF runner and experiment orchestration.
- `data_processing.py` — connectivity loading, validation, task construction, graph datasets, and rs-fMRI DFC aggregation.
- `model.py` — GAT graph encoders, training/evaluation routines, embedding extraction, and fusion MLP.
- `node_reweighting.py` — DTI gradient saliency, Top-K selection, soft node weighting, transfer/adaptation, and auxiliary prior utilities.
- `clinical_encoder.py` — clinical-text generation, frozen PubMedBERT encoding, and clinical embedding utilities.
- `utils.py` — experiment configuration, deterministic seeds, logging, metrics, plots, early stopping, and result utilities.

## Installation

Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
```

For GPU execution, install PyTorch and PyTorch Geometric builds compatible with your CUDA environment.

## Required inputs

The default code paths were written for the Kaggle experiment environment. Before execution, edit the configured paths to point to your data:

```text
DTI connectivity matrices
DTI node features
rs-fMRI DFC arrays
Diagnostic JSON
Demographic/clinical Excel file
```

## Run the supplied configuration

The current release is intentionally left as the **NC vs AD** full DGTF configuration.

```bash
python demo.py
```

Results are written under the configured results root and include configuration files, fold/iteration summaries, metrics, selected-node outputs, figures, and archived result artifacts when supported by the runtime.

## Methodological traceability

For the main DGTF configuration, the implementation contains the manuscript's central mechanisms:

- DTI-derived structural graph learning;
- gradient-based region relevance;
- Top-K soft node reweighting;
- topology preservation;
- transfer of structural relevance to rs-fMRI;
- temporal averaging and stabilization of FC;
- GAT-based structural and functional encoders;
- structured clinical covariates;
- diagnosis-excluded frozen PubMedBERT embeddings;
- late multimodal fusion through an MLP;
- AdamW, weighted cross-entropy, LR scheduling, gradient clipping, and early stopping.

Implementation-only safeguards and auxiliary utilities (for example, empty-graph safety edges, alternative selectors, and optional priors) should be regarded as engineering support unless they are explicitly activated in an experiment.

## License

Add the software license that applies to the source code before public release. ADNI data remain governed separately by ADNI data-use terms and are not distributed here.
