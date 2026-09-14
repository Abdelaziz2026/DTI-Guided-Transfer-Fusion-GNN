"""Configuration, reproducibility, logging, metrics, plotting, and utility helpers for DGTF."""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import zipfile
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score, roc_auc_score,
)

try:
    from IPython.display import FileLink, display  # type: ignore
    IPYTHON_DISPLAY_AVAILABLE = True
except Exception:
    FileLink = None  # type: ignore
    display = None  # type: ignore
    IPYTHON_DISPLAY_AVAILABLE = False
def set_deterministic_mode(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_safe_n_splits(y: np.ndarray, requested_splits: int) -> int:
    y = np.asarray(y, dtype=np.int64)
    if y.size < 2:
        raise ValueError("Need at least 2 samples for cross-validation.")
    counts = np.bincount(y)
    counts = counts[counts > 0]
    if counts.size < 2:
        raise ValueError("Need at least 2 classes for stratified cross-validation.")
    min_count = int(counts.min())
    safe = max(2, min(int(requested_splits), min_count))
    return safe


def get_modality_n_splits(cfg, modality_name: str) -> int:
    modality = str(modality_name).strip().lower()
    if modality == "dti":
        return int(getattr(cfg, "dti_n_splits", 10))
    if modality == "fmri":
        return int(getattr(cfg, "fmri_n_splits", 10))
    raise ValueError(f"Unknown modality for n_splits: {modality_name}")


def get_fusion_n_repeats(cfg) -> int:
    return int(getattr(cfg, "fusion_n_repeats", 5))


def get_fusion_test_size(cfg) -> float:
    return float(getattr(cfg, "fusion_test_size", 0.2))


def get_effective_gnn_dropout(cfg) -> float:
    return float(getattr(cfg, "gnn_dropout", 0.0)) if bool(getattr(cfg, "use_gnn_dropout", True)) else 0.0


def get_effective_fusion_dropout(cfg) -> float:
    return float(getattr(cfg, "fusion_dropout", 0.0)) if bool(getattr(cfg, "use_fusion_dropout", True)) else 0.0


def make_weighted_cross_entropy_from_labels(labels, num_classes: int, device: torch.device):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=int(num_classes)).astype(np.float32)
    weights = np.zeros(int(num_classes), dtype=np.float32)
    present = counts > 0
    if present.any():
        weights[present] = float(labels.size) / (float(int(num_classes)) * counts[present])
    else:
        weights[:] = 1.0
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.ones(int(num_classes), dtype=np.float32)
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


class UnifiedLogger:
    _instances: Dict[str, logging.Logger] = {}
    _initialized: bool = False

    @staticmethod
    def initialize(log_dir: str = "./logs", level: int = logging.INFO):
        if UnifiedLogger._initialized:
            return
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{log_dir}/multimodal_gnn_{timestamp}.log"
        logging.basicConfig(
            level=level,
            format="%(asctime)s - [%(levelname)-8s] - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        UnifiedLogger._initialized = True

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        if not UnifiedLogger._initialized:
            UnifiedLogger.initialize()
        if name not in UnifiedLogger._instances:
            UnifiedLogger._instances[name] = logging.getLogger(name)
        return UnifiedLogger._instances[name]


@dataclass
class MultiModalGNNConfig:
    experiment_name: str = "DTI_fMRI_MultiModalGNN_Fusion"
    results_base_dir: str = "./multimodal_gnn_results"
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # DATA PATHS
    # DTI
    dti_connectivity_dir: str = "/kaggle/input/dti-dataset/DTI_dataset/connectivity_matrices"
    dti_node_features_dir: str = "/kaggle/input/dti-dataset/DTI_dataset/node_features"

    # fMRI DFC dataset (single directory with *_dfc.npy files)
    # Example file: /kaggle/input/fmri-data-size-30-step-5/fmri_data_size_30_step_5/002_S_4229_dfc.npy
    fmri_base_dir: str = "/kaggle/input/fmri-data-size-30-step-5/fmri_data_size_30_step_5"

    # fMRI DFC dimensions
    fmri_time_steps: int = 30
    fmri_num_regions: int = 90

    # Labels
    diagnostic_json: str = "/kaggle/input/dti-diagnostic-groups/dti_diagnostic_groups.json"
    demographic_excel_path: str = "/kaggle/input/demographic/demographic.xlsx"


    # Demographics (optional covariates)
    
    # DemographicDataLoader (below) is used for LABELS (diagnosis) as a fallback.
    # DemographicFeatureLoader (below) is used for MODEL INPUT (age/sex/education/etc).
    use_demographics: bool = True
    demographics_in_unimodal: bool = False
    demographics_in_fusion: bool = False
    demographic_features_to_use: List[str] = field(default_factory=lambda: ["age", "sex", "education"])
    demographic_feature_dim: int = 5  # fixed (single demographic vector as in full_model_loss.py)

    
    # Clinical embedding via PubMedBERT (optional; subject-level from demographics)
    # Mirrors the "PersonalizedClinicalEncoder" logic in the reference implementation (PubMedBERT CLS embedding).
    # By default we DO NOT include diagnosis text to avoid label leakage.
    use_clinical_embedding: bool = True
    clinical_in_fusion: bool = True
    use_pubmedbert: bool = True
    pubmedbert_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    pubmedbert_local_files_only: bool = False
    clinical_device: str = "cuda" if torch.cuda.is_available() else "cpu"  # default; set to "cpu" if needed
    llm_embedding_dim: int = 768
    use_apoe4: bool = True
    clinical_embedding_mode: str = "clinical_no_diagnosis"  # demographics_only | clinical_no_diagnosis | clinical_with_diagnosis
    mask_diagnosis_in_embedding: bool = True

# PubMed / literature prior over nodes (optional; applied ONLY to SELECTED nodes)
    pubmed_node_prior_path: Optional[str] = None
    pubmed_alpha: float = 0.0  # scales PubMed modulation of selected-node weights

    # Dims (auto-detected after scan)
    dti_num_regions: int = 90
    fmri_num_regions: int = 90
    dti_node_feature_dim: Optional[int] = None
    fmri_node_feature_dim: Optional[int] = None

    # Model Architecture (shared)
    hidden_dim: int = 128
    gnn_num_layers: int = 3
    attention_heads: int = 8
    gnn_dropout: float = 0.15
    pooling: str = "meanmax"  # "meanmax", "mean", "max"

    # Graph construction (per modality thresholds)
    # Used to sparsify each modality graph (abs(conn) > threshold -> edge)
    # We tune DTI and fMRI separately via small grid sweeps (see sweep params below).
    dti_connectivity_threshold: float = 0.05
    fmri_connectivity_threshold: float = 0.05
    use_edge_weights: bool = True
    use_node_features: bool = True

    # HYPERPARAMETER SWEEPS
    # Tune DTI and fMRI thresholds separately (they often differ).
    # Keep these grids small; CV is expensive.
    run_hparam_sweep: bool = False
    sweep_mode: str = "sequential"  # "sequential" or "full"
    primary_sweep_metric: str = "accuracy"  # "accuracy" or "auc" (binary only)

    dti_connectivity_threshold_grid: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.15, 0.20])
    fmri_connectivity_threshold_grid: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.15, 0.20])

    node_selection_topk_grid: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 30, 40])
    node_reweight_factor_grid: List[float] = field(default_factory=lambda: [1.25, 1.5, 2.0, 3.0])

    # Safety: cap total sweep configurations (0 => no cap).
    max_sweep_configs: int = 0

    # Training (training)
    batch_size: int = 4
    epochs: int = 150
    learning_rate: float = 1e-3
    weight_decay: float = 5e-5

    # Optimizer / training stability
    # We use AdamW (cleaner decoupled weight decay).
    # ReduceLROnPlateau adapts LR based on validation loss.
    use_lr_scheduler: bool = True
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 10
    lr_scheduler_min_lr: float = 1e-6

    # Gradient clipping (global norm). Set 0.0 to disable.
    gradient_clip_norm: float = 1.0

    # Fusion training (also training)
    fusion_hidden_dim: int = 128
    fusion_dropout: float = 0.3
    fusion_epochs: int = 150
    fusion_learning_rate: float = 1e-3
    fusion_weight_decay: float = 5e-5

    # Regularization toggles
    use_gnn_dropout: bool = True
    use_fusion_dropout: bool = True

    # Cross-validation (separate per training stage)
    dti_n_splits: int = 10
    fmri_n_splits: int = 10
    fusion_n_repeats: int = 5
    fusion_test_size: float = 0.2

    # Early stopping controls
    use_early_stopping: bool = True
    early_stopping_tolerance: int = 7
    early_stopping_min_delta: float = 0.0

    # Outer loop iterations (optional)
    n_iterations: int = 10

    # Node selection (DTI-only)
    node_selection_method: str = "gradient"  # "gradient" (DTI saliency) or "centrality" (DTI graph centrality)
    node_selection_topk: int = 25
    node_reweight_factor: float = 2.5
    node_selection_max_batches: int = 0  # 0 => use all batches

    # Multi-class defaults (used before task filtering)
    num_classes: int = 4
    class_names: List[str] = field(default_factory=lambda: ["NC", "AD", "EMCI", "LMCI"])
    class_mapping: Dict[str, int] = field(default_factory=lambda: {"NC": 0, "AD": 1, "EMCI": 2, "LMCI": 3})

    def __post_init__(self):
        Path(self.results_base_dir).mkdir(parents=True, exist_ok=True)
        for subdir in [
            "figures",
            "subject_lists",
            "models",
            "dataset_analysis",
            "logs",
            "results",
            "checkpoints",
            "node_selection",
        ]:
            Path(f"{self.results_base_dir}/{subdir}").mkdir(parents=True, exist_ok=True)

        set_deterministic_mode(self.random_seed)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class EarlyStopping:
    """
    Early stopping on **best validation loss** .

    Stops training when `val_loss` has not improved by at least `min_delta`
    for `patience` consecutive epochs.

    Optionally caches & restores the best model weights (by val_loss).
    """
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.restore_best_weights = bool(restore_best_weights)

        self.counter = 0
        self.early_stop = False

        self.best_val_loss = float("inf")
        self.best_state_dict = None  # type: ignore
        self.best_epoch = -1

    def __call__(self, train_loss: float, validation_loss: float, model: Optional[nn.Module] = None, epoch: Optional[int] = None):
        # We keep the (train_loss, validation_loss) signature for minimal changes,
        # but only val_loss is used.
        val = float(validation_loss)

        improved = val < (self.best_val_loss - self.min_delta)

        if improved:
            self.best_val_loss = val
            self.counter = 0
            if epoch is not None:
                self.best_epoch = int(epoch)

            if self.restore_best_weights and (model is not None):
                # deepcopy is important; state_dict tensors are references.
                self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore(self, model: nn.Module):
        """Restore cached best weights into `model` (if available)."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
        return model


def build_early_stopping_from_config(cfg) -> "EarlyStopping":
    if bool(getattr(cfg, "use_early_stopping", True)):
        patience = int(getattr(cfg, "early_stopping_tolerance", 40))
    else:
        patience = int(10**9)
    return EarlyStopping(
        patience=patience,
        min_delta=float(getattr(cfg, "early_stopping_min_delta", 0.0)),
        restore_best_weights=True,
    )


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _sweep_score(result: Dict[str, Any], cfg: MultiModalGNNConfig) -> float:
    """Higher is better."""
    metric = str(getattr(cfg, "primary_sweep_metric", "accuracy")).strip().lower()

    if metric == "auc":
        auc_v = result.get("mean_fusion_auc", None)
        if auc_v is not None:
            v = _safe_float(auc_v)
            if math.isfinite(v):
                return v

    v = _safe_float(result.get("mean_fusion_accuracy", float("nan")))
    return v


def _fmt_tag_float(x: Any, ndp: int = 2) -> str:
    try:
        return f"{float(x):.{ndp}f}".replace(".", "p")
    except Exception:
        return "nan"


def make_run_tag(cfg: MultiModalGNNConfig, prefix: str = "") -> str:
    """Stable run tag used to avoid overwriting artifacts during sweeps."""
    parts = []
    if prefix:
        parts.append(prefix)

    parts.append(f"dtiT{_fmt_tag_float(cfg.dti_connectivity_threshold)}")
    parts.append(f"fmriT{_fmt_tag_float(cfg.fmri_connectivity_threshold)}")
    parts.append(f"k{int(cfg.node_selection_topk)}")
    parts.append(f"rf{_fmt_tag_float(cfg.node_reweight_factor)}")
    return "_".join(parts)


def auto_detect_flat_fmri_dir() -> str:
    explicit = os.environ.get("FMRI_BASE_DIR", "").strip()
    if explicit and Path(explicit).exists():
        return explicit

    preferred_names = [
        "fmri_data_size_30_step_5",
        "fmri_data_size_20_step_5",
        "fmri_data_size_40_step_5",
        "fmri_data_size_30_step_3",
        "fmri_data_size_30_step_10",
    ]
    search_roots = [Path("/kaggle/input"), Path("D:/Data/fmri_data"), Path("/mnt/d/Data/fmri_data"), Path(".")]

    for root in search_roots:
        if not root.exists():
            continue
        for name in preferred_names:
            p = root / name
            if p.exists() and p.is_dir():
                return str(p)
        for name in preferred_names:
            for p in root.rglob(name):
                if p.is_dir():
                    return str(p)

    pat = re.compile(r"^fmri_data_size_\d+_step_\d+$")
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_dir() and pat.match(p.name):
                return str(p)

    return "/kaggle/input/fmri-data/fmri_data_size_30_step_5"


def ensure_results_dirs(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["figures", "results", "node_selection", "logs", "models", "checkpoints", "subject_lists", "dataset_analysis", "artifacts", "artifacts/history", "artifacts/embeddings", "artifacts/connectivity"]:
        (base_dir / sub).mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob_pos: np.ndarray, cm: np.ndarray) -> Dict[str, Optional[float]]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    cm = np.asarray(cm)
    metrics: Dict[str, Optional[float]] = {}

    unique_classes = np.unique(np.concatenate([y_true, y_pred])) if (y_true.size or y_pred.size) else np.array([], dtype=np.int64)
    average_mode = "binary" if unique_classes.size <= 2 else "weighted"

    try:
        metrics["precision"] = float(precision_score(y_true, y_pred, average=average_mode, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, average=average_mode, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, average=average_mode, zero_division=0))
    except Exception:
        metrics["precision"] = None
        metrics["recall"] = None
        metrics["f1"] = None

    metrics["sensitivity"] = metrics["recall"]

    try:
        if cm.ndim == 2 and cm.shape[0] == cm.shape[1] and cm.shape[0] > 0:
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel().tolist()
                metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else None
            else:
                specificities = []
                total = float(cm.sum())
                for i in range(cm.shape[0]):
                    tp = float(cm[i, i])
                    fn = float(cm[i, :].sum() - tp)
                    fp = float(cm[:, i].sum() - tp)
                    tn = float(total - tp - fn - fp)
                    denom = tn + fp
                    if denom > 0:
                        specificities.append(tn / denom)
                metrics["specificity"] = float(np.mean(specificities)) if specificities else None
        else:
            metrics["specificity"] = None
    except Exception:
        metrics["specificity"] = None

    try:
        if (
            y_prob_pos is not None
            and len(y_prob_pos) == len(y_true)
            and len(np.unique(y_true)) == 2
        ):
            metrics["auc"] = float(roc_auc_score(y_true, y_prob_pos))
        else:
            metrics["auc"] = None
    except Exception:
        metrics["auc"] = None
    return metrics


def aggregate_scalar_metrics(values: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, vals in values.items():
        clean = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
        if clean:
            arr = np.asarray(clean, dtype=float)
            out[k] = {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "n": int(arr.size)}
        else:
            out[k] = {"mean": None, "std": None, "n": 0}
    return out


def plot_mean_roc_from_fold_results(fold_results: List[Dict[str, Any]], title: str, out_path: Path):
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    aucs = []
    for fr in fold_results:
        fpr = np.asarray(fr.get("fpr", []), dtype=float)
        tpr = np.asarray(fr.get("tpr", []), dtype=float)
        auc_val = fr.get("auc", None)
        if fpr.size > 1 and tpr.size > 1:
            order = np.argsort(fpr)
            fpr = fpr[order]
            tpr = tpr[order]
            tpr_i = np.interp(mean_fpr, fpr, tpr)
            tpr_i[0] = 0.0
            tprs.append(tpr_i)
            if auc_val is not None:
                aucs.append(float(auc_val))
    if not tprs:
        return {}
    tprs = np.stack(tprs, axis=0)
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = float(np.mean(aucs)) if aucs else None
    std_auc = float(np.std(aucs)) if aucs else None
    plt.figure(figsize=(6, 5))
    label = f"AUC = {mean_auc:.4f} ± {std_auc:.4f}" if mean_auc is not None and std_auc is not None else "Mean ROC"
    plt.plot(mean_fpr, mean_tpr, lw=2, label=label)
    plt.fill_between(mean_fpr, np.clip(mean_tpr - std_tpr, 0, 1), np.clip(mean_tpr + std_tpr, 0, 1), alpha=0.2)
    plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return {"mean_roc_path": str(out_path), "mean_auc": mean_auc, "std_auc": std_auc}


def plot_confusion_matrix(cm: np.ndarray, title: str, out_path: Path, class_names: List[str]):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha='right')
    plt.yticks(ticks, class_names)
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return str(out_path)


def create_downloadable_results_artifact(ablation_dir: Path) -> Optional[str]:
    """Zip this ablation folder into /kaggle/working and show a Kaggle/Jupyter FileLink."""
    try:
        import os
        import zipfile

        ablation_dir = Path(ablation_dir)
        working_dir = Path("/kaggle/working")
        if not working_dir.exists():
            working_dir = Path.cwd()

        zip_name = f"{ablation_dir.parent.name}_{ablation_dir.name}.zip"
        zip_path = working_dir / zip_name

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(ablation_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = os.path.relpath(file_path, ablation_dir.parent)
                    zipf.write(file_path, arcname)

        print(f"\n✅ Created: {zip_path}")
        print(f"📦 Total size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")

        if IPYTHON_DISPLAY_AVAILABLE and display is not None and FileLink is not None:
            try:
                display(FileLink(zip_name))
            except Exception as display_error:
                print(f"Could not render clickable download link: {display_error}")

        return str(zip_path)
    except Exception as e:
        print(f"Could not create downloadable archive for {ablation_dir}: {e}")
        return None
