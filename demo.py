"""Executable DGTF experiments.

By default this script runs the self-contained NC-vs-AD full DGTF ablation
configuration preserved from the supplied implementation. Edit the configuration
block near the bottom of this file to point to your local/Kaggle data paths or to
select another task/ablation.
"""

from __future__ import annotations

import copy
import gc
import itertools
import json
import math
import os
import re
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt

from utils import (
    EarlyStopping, MultiModalGNNConfig, UnifiedLogger,
    _sweep_score, aggregate_scalar_metrics, auto_detect_flat_fmri_dir,
    build_early_stopping_from_config, compute_binary_metrics,
    create_downloadable_results_artifact, ensure_results_dirs,
    get_effective_fusion_dropout, get_fusion_n_repeats, get_fusion_test_size,
    get_modality_n_splits, get_safe_n_splits, make_run_tag,
    make_weighted_cross_entropy_from_labels, plot_confusion_matrix,
    plot_mean_roc_from_fold_results, save_json, set_deterministic_mode,
)
from data_processing import (
    ConnectivityDataScanner, ConnectivityGraphDataset, DemographicDataLoader,
    DiagnosticGroupLoader, FMRIDFCScanner, build_demographics_matrix,
    build_task_subjects, collate_connectivity_batch, get_task_display_name,
    impute_and_scale_demographics,
)
from clinical_encoder import (
    PersonalizedClinicalEncoder, ClinicalDemographicDataLoader, TRANSFORMERS_AVAILABLE,
    build_clinical_embedding_matrix, scale_dense_embeddings,
)
from model import (
    ConnectivityGNN, FusionEmbeddingDataset, FusionMLP, collate_fusion,
    eval_gnn, extract_embeddings, train_and_evaluate_fusion,
    train_and_evaluate_gnn,
)
from node_reweighting import (
    CentralityNodeSelector, DTINodeSelector, PubMedNodePriorLoader,
    adapt_node_weights, apply_pubmed_prior_to_selected_nodes,
)
def run_multimodal_cv_for_task(
    dti_subjects_task: Dict[str, Dict],
    fmri_subjects_task: Dict[str, Dict],
    cfg: MultiModalGNNConfig,
    task_name: str,
    pubmed_prior_dti: Optional[np.ndarray] = None,
    clinical_encoder: Optional[Any] = None,
    run_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Cross-validate the *fusion* classifier using only common subjects.
    For each fold:
      1) Train base DTI model (no node weighting) on DTI train subjects.
      2) Compute DTI-only node selection (top-k) on DTI train set.
      3) Train weighted DTI model (reweight selected nodes; keep all nodes).
      4) Train weighted fMRI model using the *same* node weights.
      5) Extract embeddings for common train/val subjects.
      6) Train fusion MLP on train embeddings, eval on val embeddings.
    """
    tag_str = f".{run_tag}" if run_tag else ""
    logger = UnifiedLogger.get_logger(f"MultimodalCV.{task_name}{tag_str}")

    # Intersection of IDs for fusion
    common_ids = sorted(list(set(dti_subjects_task.keys()) & set(fmri_subjects_task.keys())))
    if len(common_ids) < 4:
        raise RuntimeError(f"Too few common subjects between DTI and fMRI for task={task_name}: n={len(common_ids)}")

    y_common = np.array([int(dti_subjects_task[sid]["label"]) for sid in common_ids], dtype=np.int64)

    dti_ids_all = sorted(list(dti_subjects_task.keys()))
    y_dti_all = np.array([int(dti_subjects_task[sid]["label"]) for sid in dti_ids_all], dtype=np.int64)
    fmri_ids_all = sorted(list(fmri_subjects_task.keys()))
    y_fmri_all = np.array([int(fmri_subjects_task[sid]["label"]) for sid in fmri_ids_all], dtype=np.int64)

    effective_fusion_repeats = int(get_fusion_n_repeats(cfg))
    effective_dti_splits = get_safe_n_splits(y_dti_all, get_modality_n_splits(cfg, "dti"))
    effective_fmri_splits = get_safe_n_splits(y_fmri_all, get_modality_n_splits(cfg, "fmri"))

    test_size = float(get_fusion_test_size(cfg))
    min_class_count_common = int(np.bincount(y_common).min())
    if min_class_count_common < 2:
        raise RuntimeError(f"Need at least 2 samples per class in common subjects for StratifiedShuffleSplit: min_class_count={min_class_count_common}")
    max_test_fraction = (min_class_count_common - 1) / float(len(common_ids))
    test_size = min(test_size, max_test_fraction)
    test_size = max(test_size, 1.0 / float(len(common_ids)))

    fusion_sss = StratifiedShuffleSplit(
        n_splits=effective_fusion_repeats,
        test_size=test_size,
        random_state=0,
    )
    dti_skf = StratifiedKFold(n_splits=effective_dti_splits, shuffle=True, random_state=0)
    fmri_skf = StratifiedKFold(n_splits=effective_fmri_splits, shuffle=True, random_state=0)

    fusion_split_plan = list(fusion_sss.split(np.arange(len(common_ids)), y_common))
    dti_split_plan = list(dti_skf.split(np.arange(len(dti_ids_all)), y_dti_all))
    fmri_split_plan = list(fmri_skf.split(np.arange(len(fmri_ids_all)), y_fmri_all))

    fold_results = []
    tsne_chunks = []
    tsne_labels = []
    tsne_ids: List[str] = []
    conn_store: Dict[str, Any] = {}
    cms_fusion = []
    aucs_fusion = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)

    device = torch.device(cfg.device)

    for fold_idx, (fusion_tr_idx, fusion_va_idx) in enumerate(fusion_split_plan, start=1):
        dti_plan_idx = (fold_idx - 1) % len(dti_split_plan)
        fmri_plan_idx = (fold_idx - 1) % len(fmri_split_plan)
        dti_tr_idx, dti_va_idx = dti_split_plan[dti_plan_idx]
        fmri_tr_idx, fmri_va_idx = fmri_split_plan[fmri_plan_idx]

        print("\n" + "=" * 90)
        print(
            f"[TASK {task_name}] REPEAT {fold_idx}/{effective_fusion_repeats} "
            f"| DTI fold {dti_plan_idx + 1}/{effective_dti_splits} "
            f"| fMRI fold {fmri_plan_idx + 1}/{effective_fmri_splits}"
        )
        print("=" * 90)

        train_common_ids = [common_ids[i] for i in fusion_tr_idx]
        val_common_ids = [common_ids[i] for i in fusion_va_idx]

        dti_train_ids = [dti_ids_all[i] for i in dti_tr_idx]
        dti_val_ids = [dti_ids_all[i] for i in dti_va_idx]
        fmri_train_ids = [fmri_ids_all[i] for i in fmri_tr_idx]
        fmri_val_ids = [fmri_ids_all[i] for i in fmri_va_idx]

        # Build unimodal train/val sets from EACH modality's own split plan.
        # Fusion keeps its own independent split on common subjects.
        dti_train_subjects = {sid: dti_subjects_task[sid] for sid in dti_train_ids}
        dti_val_subjects = {sid: dti_subjects_task[sid] for sid in dti_val_ids}

        fmri_train_subjects = {sid: fmri_subjects_task[sid] for sid in fmri_train_ids}
        fmri_val_subjects = {sid: fmri_subjects_task[sid] for sid in fmri_val_ids}

        
        # Stage 1: Train base DTI model (no weights) for node selection
        
        dti_train_ds = ConnectivityGraphDataset(
            dti_train_subjects,
            use_node_features=(cfg.use_node_features and cfg.dti_node_feature_dim is not None),
            use_demographics=(cfg.use_demographics and cfg.demographics_in_unimodal and cfg.demographic_feature_dim > 0),
            demographic_dim=int(cfg.demographic_feature_dim),
        )
        dti_val_ds = ConnectivityGraphDataset(
            dti_val_subjects,
            use_node_features=(cfg.use_node_features and cfg.dti_node_feature_dim is not None),
            use_demographics=(cfg.use_demographics and cfg.demographics_in_unimodal and cfg.demographic_feature_dim > 0),
            demographic_dim=int(cfg.demographic_feature_dim),
        )

        dti_train_loader = DataLoader(dti_train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_connectivity_batch)
        dti_val_loader = DataLoader(dti_val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_connectivity_batch)

        base_dti_model = ConnectivityGNN(
            num_regions=cfg.dti_num_regions,
            node_feature_dim=cfg.dti_node_feature_dim,
            num_classes=cfg.num_classes,
            connectivity_threshold=cfg.dti_connectivity_threshold,
            config=cfg,
            demographic_dim=(int(cfg.demographic_feature_dim) if (cfg.use_demographics and cfg.demographics_in_unimodal and cfg.demographic_feature_dim > 0) else 0),
        ).to(device)

        criterion = make_weighted_cross_entropy_from_labels([dti_train_subjects[sid]["label"] for sid in dti_train_subjects], cfg.num_classes, device)
        optimizer = optim.AdamW(base_dti_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = None
        if getattr(cfg, "use_lr_scheduler", True):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(cfg.lr_scheduler_factor),
                patience=int(cfg.lr_scheduler_patience),
                min_lr=float(cfg.lr_scheduler_min_lr),
            )
        early_stopping = build_early_stopping_from_config(cfg)

        base_dti_model, _, _, _, _, _, _, _ = train_and_evaluate_gnn(
            base_dti_model, optimizer, criterion,
            dti_train_loader, dti_val_loader,
            dti_train_ds, dti_val_ds,
            early_stopping, device,
            num_epochs=cfg.epochs,
            node_weights=None,
            scheduler=scheduler,
        )

        
        # Stage 2: DTI-only node selection (TOP-K) using base DTI model
        
        selector_method = str(getattr(cfg, 'node_selection_method', 'gradient')).strip().lower()
        if selector_method == 'centrality':
            selector = CentralityNodeSelector(
                topk=cfg.node_selection_topk,
                threshold=float(cfg.dti_connectivity_threshold),
                max_subjects=int(getattr(cfg, 'centrality_max_subjects', 0) or 0),
            )
            scores = selector.compute_node_scores(dti_train_subjects, num_regions=int(cfg.dti_num_regions))
        else:
            selector = DTINodeSelector(topk=cfg.node_selection_topk, max_batches=cfg.node_selection_max_batches)
            scores = selector.compute_node_scores(base_dti_model, dti_train_loader, device=device)
        top_idx, mask = selector.select_topk(scores)
        node_weights_np = selector.make_node_weights(mask, reweight_factor=cfg.node_reweight_factor)

        # Optional: modulate SELECTED node weights using an external PubMed prior
        if pubmed_prior_dti is not None and float(getattr(cfg, "pubmed_alpha", 0.0)) > 0.0:
            node_weights_np = apply_pubmed_prior_to_selected_nodes(
                node_weights_np,
                selected_mask=mask,
                pubmed_prior=pubmed_prior_dti,
                alpha=float(cfg.pubmed_alpha),
            )

        # Save node selection per fold
        fold_ns_path = Path(cfg.results_base_dir) / "node_selection" / f"node_selection_{task_name}{('_' + run_tag) if run_tag else ''}_fold{fold_idx}.json"
        with open(fold_ns_path, "w") as f:
            json.dump({
                "task": task_name,
                "repeat": fold_idx,
                "run_tag": run_tag,
                "dti_connectivity_threshold": float(cfg.dti_connectivity_threshold),
                "fmri_connectivity_threshold": float(cfg.fmri_connectivity_threshold),
                "topk": int(cfg.node_selection_topk),
                "reweight_factor": float(cfg.node_reweight_factor),
                "selected_indices": top_idx.tolist(),
                "scores": scores.tolist(),
            }, f, indent=2)
        print(f"[NodeSelection] Saved: {fold_ns_path}")

        # Prepare node weights tensors for DTI and fMRI
        node_weights_dti = torch.FloatTensor(node_weights_np).to(device)

        node_weights_fmri_np = adapt_node_weights(node_weights_np, target_num_regions=cfg.fmri_num_regions)
        node_weights_fmri = torch.FloatTensor(node_weights_fmri_np).to(device)

        
        # Stage 3: Train weighted DTI model (from scratch) using node_weights_dti
        
        weighted_dti_model = ConnectivityGNN(
            num_regions=cfg.dti_num_regions,
            node_feature_dim=cfg.dti_node_feature_dim,
            num_classes=cfg.num_classes,
            connectivity_threshold=cfg.dti_connectivity_threshold,
            config=cfg,
            demographic_dim=(int(cfg.demographic_feature_dim) if (cfg.use_demographics and cfg.demographics_in_unimodal and cfg.demographic_feature_dim > 0) else 0),
        ).to(device)
        optimizer = optim.AdamW(weighted_dti_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = None
        if getattr(cfg, "use_lr_scheduler", True):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(cfg.lr_scheduler_factor),
                patience=int(cfg.lr_scheduler_patience),
                min_lr=float(cfg.lr_scheduler_min_lr),
            )
        early_stopping = build_early_stopping_from_config(cfg)

        weighted_dti_model, dti_epoch_acc_train, dti_epoch_acc_val, _, _, _, dti_epoch_loss_train, dti_epoch_loss_val = train_and_evaluate_gnn(
            weighted_dti_model, optimizer, criterion,
            dti_train_loader, dti_val_loader,
            dti_train_ds, dti_val_ds,
            early_stopping, device,
            num_epochs=cfg.epochs,
            node_weights=node_weights_dti,
            scheduler=scheduler,
        )

        
        # Stage 4: Train weighted fMRI model using imported DTI node weights
        
        fmri_train_ds = ConnectivityGraphDataset(
            fmri_train_subjects,
            use_node_features=(cfg.use_node_features and cfg.fmri_node_feature_dim is not None),
            use_demographics=(cfg.use_demographics and cfg.demographics_in_unimodal and cfg.demographic_feature_dim > 0),
            demographic_dim=int(cfg.demographic_feature_dim),
        )
        fmri_val_ds = ConnectivityGraphDataset(
            fmri_val_subjects,
            use_node_features=(cfg.use_node_features and cfg.fmri_node_feature_dim is not None),
            use_demographics=(cfg.use_demographics and cfg.demographics_in_unimodal and cfg.demographic_feature_dim > 0),
            demographic_dim=int(cfg.demographic_feature_dim),
        )

        fmri_train_loader = DataLoader(fmri_train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_connectivity_batch)
        fmri_val_loader = DataLoader(fmri_val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_connectivity_batch)

        weighted_fmri_model = ConnectivityGNN(
            num_regions=cfg.fmri_num_regions,
            node_feature_dim=cfg.fmri_node_feature_dim,
            num_classes=cfg.num_classes,
            connectivity_threshold=cfg.fmri_connectivity_threshold,
            config=cfg,
            demographic_dim=(int(cfg.demographic_feature_dim) if (cfg.use_demographics and cfg.demographics_in_unimodal and cfg.demographic_feature_dim > 0) else 0),
        ).to(device)
        optimizer = optim.AdamW(weighted_fmri_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = None
        if getattr(cfg, "use_lr_scheduler", True):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(cfg.lr_scheduler_factor),
                patience=int(cfg.lr_scheduler_patience),
                min_lr=float(cfg.lr_scheduler_min_lr),
            )
        early_stopping = build_early_stopping_from_config(cfg)

        weighted_fmri_model, fmri_epoch_acc_train, fmri_epoch_acc_val, _, _, _, fmri_epoch_loss_train, fmri_epoch_loss_val = train_and_evaluate_gnn(
            weighted_fmri_model, optimizer, criterion,
            fmri_train_loader, fmri_val_loader,
            fmri_train_ds, fmri_val_ds,
            early_stopping, device,
            num_epochs=cfg.epochs,
            node_weights=node_weights_fmri,
            scheduler=scheduler,
        )

        
        # Stage 5: Extract embeddings for common train/val subjects (using weighted models)
        
        dti_common_train = {sid: dti_subjects_task[sid] for sid in train_common_ids}
        dti_common_val = {sid: dti_subjects_task[sid] for sid in val_common_ids}

        fmri_common_train = {sid: fmri_subjects_task[sid] for sid in train_common_ids}
        fmri_common_val = {sid: fmri_subjects_task[sid] for sid in val_common_ids}

        dti_emb_tr, y_tr, ids_tr = extract_embeddings(
            weighted_dti_model, dti_common_train, cfg, device,
            num_regions=cfg.dti_num_regions,
            node_feature_dim=cfg.dti_node_feature_dim,
            connectivity_threshold=cfg.dti_connectivity_threshold,
            node_weights=node_weights_dti,
        )
        fmri_emb_tr, y_tr2, ids_tr2 = extract_embeddings(
            weighted_fmri_model, fmri_common_train, cfg, device,
            num_regions=cfg.fmri_num_regions,
            node_feature_dim=cfg.fmri_node_feature_dim,
            connectivity_threshold=cfg.fmri_connectivity_threshold,
            node_weights=node_weights_fmri,
        )

        dti_emb_va, y_va, ids_va = extract_embeddings(
            weighted_dti_model, dti_common_val, cfg, device,
            num_regions=cfg.dti_num_regions,
            node_feature_dim=cfg.dti_node_feature_dim,
            connectivity_threshold=cfg.dti_connectivity_threshold,
            node_weights=node_weights_dti,
        )
        fmri_emb_va, y_va2, ids_va2 = extract_embeddings(
            weighted_fmri_model, fmri_common_val, cfg, device,
            num_regions=cfg.fmri_num_regions,
            node_feature_dim=cfg.fmri_node_feature_dim,
            connectivity_threshold=cfg.fmri_connectivity_threshold,
            node_weights=node_weights_fmri,
        )

        # Safety checks: IDs should match ordering
        assert ids_tr == ids_tr2, "DTI and fMRI train IDs are not aligned."
        assert ids_va == ids_va2, "DTI and fMRI val IDs are not aligned."
        assert np.array_equal(y_tr, y_tr2), "DTI and fMRI train labels differ."
        assert np.array_equal(y_va, y_va2), "DTI and fMRI val labels differ."

        X_tr = np.concatenate([dti_emb_tr, fmri_emb_tr], axis=1)
        X_va = np.concatenate([dti_emb_va, fmri_emb_va], axis=1)

        # Track fusion input segment dims (ordering must match FusionMLP slicing)
        dti_dim = int(dti_emb_tr.shape[1])
        fmri_dim = int(fmri_emb_tr.shape[1])
        demo_dim = 0
        clinical_dim = 0

        # Optional: add demographics covariates at fusion stage
        if cfg.use_demographics and cfg.demographics_in_fusion and int(cfg.demographic_feature_dim) > 0:
            Xdem_tr_raw = build_demographics_matrix(dti_subjects_task, ids_tr, int(cfg.demographic_feature_dim))
            Xdem_va_raw = build_demographics_matrix(dti_subjects_task, ids_va, int(cfg.demographic_feature_dim))
            Xdem_tr, Xdem_va = impute_and_scale_demographics(Xdem_tr_raw, Xdem_va_raw)
            demo_dim = int(Xdem_tr.shape[1])
            X_tr = np.concatenate([X_tr, Xdem_tr], axis=1)
            X_va = np.concatenate([X_va, Xdem_va], axis=1)

        # Optional: add PubMedBERT clinical embeddings at fusion stage (attached-code style)
        if (
            getattr(cfg, "use_clinical_embedding", False)
            and getattr(cfg, "clinical_in_fusion", True)
            and clinical_encoder is not None
            and int(getattr(cfg, "llm_embedding_dim", 0)) > 0
        ):
            Xclin_tr_raw = build_clinical_embedding_matrix(clinical_encoder, ids_tr, int(cfg.llm_embedding_dim))
            Xclin_va_raw = build_clinical_embedding_matrix(clinical_encoder, ids_va, int(cfg.llm_embedding_dim))
            Xclin_tr, Xclin_va = scale_dense_embeddings(Xclin_tr_raw, Xclin_va_raw)
            clinical_dim = int(Xclin_tr.shape[1])
            X_tr = np.concatenate([X_tr, Xclin_tr], axis=1)
            X_va = np.concatenate([X_va, Xclin_va], axis=1)


        # Stage 6: Train fusion classifier on common train subjects, eval on common val
        
        fusion_train_ds = FusionEmbeddingDataset(X_tr, y_tr, ids_tr)
        fusion_val_ds = FusionEmbeddingDataset(X_va, y_va, ids_va)

        fusion_train_loader = DataLoader(fusion_train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fusion)
        fusion_val_loader = DataLoader(fusion_val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fusion)

        fusion_model = FusionMLP(
            dti_dim=dti_dim,
            fmri_dim=fmri_dim,
            demo_dim=demo_dim,
            clinical_dim=clinical_dim,
            hidden_dim=cfg.fusion_hidden_dim,
            dropout=get_effective_fusion_dropout(cfg),
            num_classes=cfg.num_classes,
        ).to(device)

        fusion_criterion = make_weighted_cross_entropy_from_labels(y_tr, cfg.num_classes, device)
        fusion_optimizer = optim.AdamW(fusion_model.parameters(), lr=cfg.fusion_learning_rate, weight_decay=cfg.fusion_weight_decay)
        fusion_scheduler = None
        if getattr(cfg, "use_lr_scheduler", True):
            fusion_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                fusion_optimizer,
                mode="min",
                factor=float(cfg.lr_scheduler_factor),
                patience=int(cfg.lr_scheduler_patience),
                min_lr=float(cfg.lr_scheduler_min_lr),
            )
        fusion_early_stopping = build_early_stopping_from_config(cfg)

        fusion_model, acc_final, loss_final, cm, fpr, tpr, y_true, y_pred, y_prob_pos, fusion_history = train_and_evaluate_fusion(
            fusion_model,
            fusion_optimizer,
            fusion_criterion,
            fusion_train_loader,
            fusion_val_loader,
            train_len=len(fusion_train_ds),
            val_len=len(fusion_val_ds),
            early_stopping=fusion_early_stopping,
            device=device,
            num_epochs=cfg.fusion_epochs,
            scheduler=fusion_scheduler,
            grad_clip_norm=float(cfg.gradient_clip_norm),
        )

        if X_va is not None and len(X_va) > 0:
            tsne_chunks.append(np.asarray(X_va, dtype=np.float32))
            tsne_labels.append(np.asarray(y_va, dtype=np.int64))
            tsne_ids.extend(list(ids_va))
        _update_connectivity_aggregate(conn_store, dti_subjects_task, val_common_ids, "dti", list(getattr(cfg, "class_names", [])))
        _update_connectivity_aggregate(conn_store, fmri_subjects_task, val_common_ids, "fmri", list(getattr(cfg, "class_names", [])))

        # Compute fold AUC (binary only)
        fold_auc = float("nan")
        if cfg.num_classes == 2 and len(fpr) > 0 and len(tpr) > 0:
            try:
                fold_auc = auc(fpr, tpr)
                aucs_fusion.append(fold_auc)
                mean_tpr += np.interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
            except Exception:
                pass


        # compute fold metrics for this fold (fusion)
        metrics_fold = compute_binary_metrics(y_true, y_pred, y_prob_pos, cm)
        prec = metrics_fold["precision"] if metrics_fold["precision"] is not None else float("nan")
        rec = metrics_fold["recall"] if metrics_fold["recall"] is not None else float("nan")
        f1v = metrics_fold["f1"] if metrics_fold["f1"] is not None else float("nan")
        sensitivity = metrics_fold["sensitivity"] if metrics_fold["sensitivity"] is not None else float("nan")
        specificity = metrics_fold["specificity"] if metrics_fold["specificity"] is not None else float("nan")
        auc_value = metrics_fold["auc"]
        print(f"[Fusion Fold {fold_idx}] Acc={float(acc_final):.4f} | AUC={auc_value if auc_value is not None else fold_auc} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1v:.4f} | Spec={specificity:.4f}")

        fold_results.append({
            "repeat": fold_idx,
            "n_common_train": len(train_common_ids),
            "n_common_val": len(val_common_ids),
            "fusion_acc": float(acc_final),
            "fusion_loss": float(loss_final),
            "fusion_auc": auc_value,
            "fusion_precision": prec,
            "fusion_recall": rec,
            "fusion_sensitivity": sensitivity,
            "fusion_specificity": specificity,
            "fusion_f1": f1v,
            "fusion_confusion_matrix": cm.tolist(),
            "fusion_fpr": fpr.tolist() if hasattr(fpr, "tolist") else list(fpr),
            "fusion_tpr": tpr.tolist() if hasattr(tpr, "tolist") else list(tpr),
            "selected_nodes": top_idx.tolist(),
            "dti_train_history": {"acc": [float(v) for v in dti_epoch_acc_train], "loss": [float(v) for v in dti_epoch_loss_train]},
            "dti_val_history": {"acc": [float(v) for v in dti_epoch_acc_val], "loss": [float(v) for v in dti_epoch_loss_val]},
            "fmri_train_history": {"acc": [float(v) for v in fmri_epoch_acc_train], "loss": [float(v) for v in fmri_epoch_loss_train]},
            "fmri_val_history": {"acc": [float(v) for v in fmri_epoch_acc_val], "loss": [float(v) for v in fmri_epoch_loss_val]},
            "fusion_train_history": {"acc": [float(v) for v in fusion_history.get("train_acc", [])], "loss": [float(v) for v in fusion_history.get("train_loss", [])]},
            "fusion_val_history": {"acc": [float(v) for v in fusion_history.get("val_acc", [])], "loss": [float(v) for v in fusion_history.get("val_loss", [])]}
        })

        cms_fusion.append(cm)

        # cleanup
        del base_dti_model, weighted_dti_model, weighted_fmri_model, fusion_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # mean AUC curve
    if len(aucs_fusion) > 0:
        mean_tpr /= len(aucs_fusion)
        mean_tpr[-1] = 1.0
        mean_auc = float(auc(mean_fpr, mean_tpr))
    else:
        mean_auc = float("nan")

    # aggregate acc
    accs = [fr["fusion_acc"] for fr in fold_results]
    mean_acc = float(np.mean(accs)) if accs else float("nan")
    std_acc = float(np.std(accs)) if accs else float("nan")

    result = {
        "task": task_name,
        "run_tag": run_tag,
        "hyperparams": {
            "dti_connectivity_threshold": float(cfg.dti_connectivity_threshold),
            "fmri_connectivity_threshold": float(cfg.fmri_connectivity_threshold),
            "node_selection_topk": int(cfg.node_selection_topk),
            "node_reweight_factor": float(cfg.node_reweight_factor),
            "learning_rate": 1e-3,
            "weight_decay": float(cfg.weight_decay),
            "fusion_learning_rate": 1e-3,
            "fusion_weight_decay": float(cfg.fusion_weight_decay),
            "lr_scheduler_patience": int(getattr(cfg, "lr_scheduler_patience", 0)),
            "lr_scheduler_factor": float(getattr(cfg, "lr_scheduler_factor", 0.0)),
            "gradient_clip_norm": float(getattr(cfg, "gradient_clip_norm", 0.0)),
        },
        "n_common_subjects": len(common_ids),
        "dti_n_splits": int(getattr(cfg, "dti_n_splits", 10)),
        "fmri_n_splits": int(getattr(cfg, "fmri_n_splits", 10)),
        "fusion_n_repeats": int(getattr(cfg, "fusion_n_repeats", 5)),
        "fusion_test_size": float(getattr(cfg, "fusion_test_size", 0.2)),
        "fold_results": fold_results,
        "mean_fusion_accuracy": mean_acc,
        "std_fusion_accuracy": std_acc,
        "mean_fusion_auc": mean_auc if not math.isnan(mean_auc) else None,
    }

    payload: Dict[str, Any] = {"connectivity": _finalize_connectivity_aggregate(conn_store)}
    if tsne_chunks:
        payload["tsne_X"] = np.concatenate(tsne_chunks, axis=0).tolist()
        payload["tsne_y"] = np.concatenate(tsne_labels, axis=0).tolist()
        payload["tsne_ids"] = list(tsne_ids)
    result["_plot_payload"] = payload
    return result


def run_hparam_sweep_for_task(
    dti_subjects_task: Dict[str, Dict],
    fmri_subjects_task: Dict[str, Dict],
    base_cfg: MultiModalGNNConfig,
    task_name: str,
    pubmed_prior_dti: Optional[np.ndarray] = None,
    clinical_encoder: Optional[Any] = None,
) -> Tuple[MultiModalGNNConfig, Dict[str, Any], Dict[str, Any]]:
    """
    Runs a small sweep over:
      1) DTI threshold x fMRI threshold
      2) node_selection_topk x node_reweight_factor

    Two modes:
      - sequential : first tune thresholds using baseline (topk, rf),
        then tune (topk, rf) using best thresholds.
      - full: single sweep over all four hyperparams (expensive).

    Returns:
      best_cfg, best_result, sweep_summary
    """
    logger = UnifiedLogger.get_logger(f"Sweep.{task_name}")

    sweep_mode = str(getattr(base_cfg, "sweep_mode", "sequential")).strip().lower()
    max_cfg = int(getattr(base_cfg, "max_sweep_configs", 0) or 0)

    # Always deterministic per sweep run
    set_deterministic_mode(int(getattr(base_cfg, "random_seed", 42)))

    dti_thr_grid = list(getattr(base_cfg, "dti_connectivity_threshold_grid", [base_cfg.dti_connectivity_threshold]))
    fmri_thr_grid = list(getattr(base_cfg, "fmri_connectivity_threshold_grid", [base_cfg.fmri_connectivity_threshold]))
    topk_grid = list(getattr(base_cfg, "node_selection_topk_grid", [base_cfg.node_selection_topk]))
    rf_grid = list(getattr(base_cfg, "node_reweight_factor_grid", [base_cfg.node_reweight_factor]))

    metric_name = str(getattr(base_cfg, "primary_sweep_metric", "accuracy")).strip().lower()
    logger.info(
        f"[Sweep {task_name}] mode={sweep_mode} | metric={metric_name} | "
        f"thr_grid: DTI={dti_thr_grid} fMRI={fmri_thr_grid} | topk={topk_grid} | rf={rf_grid}"
    )

    sweep_summary: Dict[str, Any] = {
        "task": task_name,
        "sweep_mode": sweep_mode,
        "primary_metric": metric_name,
        "threshold_grid": {"dti": dti_thr_grid, "fmri": fmri_thr_grid},
        "topk_grid": topk_grid,
        "reweight_factor_grid": rf_grid,
        "max_sweep_configs": max_cfg,
        "runs": [],
        "best": None,
    }

    best_cfg: Optional[MultiModalGNNConfig] = None
    best_result: Optional[Dict[str, Any]] = None
    best_score: float = float("-inf")

    def _eval_one(cfg_run: MultiModalGNNConfig, prefix: str) -> Dict[str, Any]:
        tag = make_run_tag(cfg_run, prefix=prefix)
        logger.info(f"[Sweep {task_name}] RUN {tag}")
        # Ensure deterministic behavior across runs
        set_deterministic_mode(int(getattr(cfg_run, "random_seed", 42)))

        res = run_multimodal_cv_for_task(
            dti_subjects_task,
            fmri_subjects_task,
            cfg_run,
            task_name=task_name,
            pubmed_prior_dti=pubmed_prior_dti,
            clinical_encoder=clinical_encoder,
            run_tag=tag,
        )
        score = _sweep_score(res, cfg_run)

        sweep_summary["runs"].append({
            "run_tag": tag,
            "score": float(score) if math.isfinite(float(score)) else None,
            "mean_fusion_accuracy": float(res.get("mean_fusion_accuracy", float("nan"))),
            "mean_fusion_auc": res.get("mean_fusion_auc", None),
            "hyperparams": res.get("hyperparams", {}),
        })

        return res

    if sweep_mode == "full":
        combos = list(itertools.product(dti_thr_grid, fmri_thr_grid, topk_grid, rf_grid))
        if max_cfg > 0:
            combos = combos[:max_cfg]

        for dti_thr, fmri_thr, topk, rf in combos:
            cfg_run = copy.deepcopy(base_cfg)
            cfg_run.dti_connectivity_threshold = float(dti_thr)
            cfg_run.fmri_connectivity_threshold = float(fmri_thr)
            cfg_run.node_selection_topk = int(topk)
            cfg_run.node_reweight_factor = float(rf)

            res = _eval_one(cfg_run, prefix="FULL")
            score = _sweep_score(res, cfg_run)

            if math.isfinite(score) and score > best_score:
                best_score = float(score)
                best_cfg = cfg_run
                best_result = res

    else:
        
        # Stage 1: thresholds sweep
        
        thr_combos = list(itertools.product(dti_thr_grid, fmri_thr_grid))
        if max_cfg > 0:
            thr_combos = thr_combos[:max_cfg]

        best_thr_cfg: Optional[MultiModalGNNConfig] = None
        best_thr_result: Optional[Dict[str, Any]] = None
        best_thr_score: float = float("-inf")

        for dti_thr, fmri_thr in thr_combos:
            cfg_run = copy.deepcopy(base_cfg)
            cfg_run.dti_connectivity_threshold = float(dti_thr)
            cfg_run.fmri_connectivity_threshold = float(fmri_thr)

            # baseline node selection params for threshold tuning
            cfg_run.node_selection_topk = int(base_cfg.node_selection_topk)
            cfg_run.node_reweight_factor = float(base_cfg.node_reweight_factor)

            res = _eval_one(cfg_run, prefix="THR")
            score = _sweep_score(res, cfg_run)

            if math.isfinite(score) and score > best_thr_score:
                best_thr_score = float(score)
                best_thr_cfg = cfg_run
                best_thr_result = res

        if best_thr_cfg is None or best_thr_result is None:
            raise RuntimeError("Threshold sweep failed to produce a valid run/result.")

        logger.info(
            f"[Sweep {task_name}] Best thresholds: "
            f"DTI={best_thr_cfg.dti_connectivity_threshold} | fMRI={best_thr_cfg.fmri_connectivity_threshold} | "
            f"score={best_thr_score:.6f}"
        )

        
        # Stage 2: (topk, reweight_factor) sweep
        
        ns_combos = list(itertools.product(topk_grid, rf_grid))
        if max_cfg > 0:
            ns_combos = ns_combos[:max_cfg]

        for topk, rf in ns_combos:
            cfg_run = copy.deepcopy(base_cfg)
            cfg_run.dti_connectivity_threshold = float(best_thr_cfg.dti_connectivity_threshold)
            cfg_run.fmri_connectivity_threshold = float(best_thr_cfg.fmri_connectivity_threshold)
            cfg_run.node_selection_topk = int(topk)
            cfg_run.node_reweight_factor = float(rf)

            res = _eval_one(cfg_run, prefix="NS")
            score = _sweep_score(res, cfg_run)

            if math.isfinite(score) and score > best_score:
                best_score = float(score)
                best_cfg = cfg_run
                best_result = res

    if best_cfg is None or best_result is None:
        raise RuntimeError("Hyperparameter sweep did not find any valid configuration.")

    sweep_summary["best"] = {
        "score": float(best_score),
        "metric": metric_name,
        "run_tag": best_result.get("run_tag", None),
        "hyperparams": best_result.get("hyperparams", {}),
        "mean_fusion_accuracy": float(best_result.get("mean_fusion_accuracy", float("nan"))),
        "mean_fusion_auc": best_result.get("mean_fusion_auc", None),
    }

    # Save sweep summary
    out_path = Path(base_cfg.results_base_dir) / "results" / f"hparam_sweep_{task_name}.json"
    try:
        with open(out_path, "w") as f:
            json.dump(sweep_summary, f, indent=2, default=str)
        logger.info(f"[Sweep {task_name}] Saved sweep summary -> {out_path}")
    except Exception as e:
        logger.warning(f"[Sweep {task_name}] Could not save sweep summary: {e}")

    return best_cfg, best_result, sweep_summary


def main():
    print("\n" + "=" * 80)
    print("MULTI-MODAL DTI + fMRI CONNECTIVITY GNN (DTI-ONLY NODE SELECTION)")
    print(" - Independent unimodal training (DTI, fMRI)")
    print(" - Node selection on DTI only, imported to fMRI (reweight only)")
    print(" - Fusion classifier on common subjects only")
    print(" - Training/optimization: AdamW + CE + ReduceLROnPlateau + grad clipping")
    print("=" * 80 + "\n")

    
    # EDIT THESE:
    
    TASKS_TO_RUN = ["NC_AD", "NC_MCI", "EMCI_LMCI", "MCI_AD"]  # choose any subset of the four binary tasks
    RESULTS_DIR = "./multimodal_dti_fmri_results"

    cfg = MultiModalGNNConfig(
        results_base_dir=RESULTS_DIR,

        # DTI paths
        dti_connectivity_dir="/kaggle/input/dti-dataset/DTI_dataset/connectivity_matrices",
        dti_node_features_dir="/kaggle/input/dti-dataset/DTI_dataset/node_features",
        # fMRI DFC path
        fmri_base_dir="/kaggle/input/fmri-data-size-30-step-5/fmri_data_size_30_step_5",

        # Label sources
        diagnostic_json="/kaggle/input/dti-diagnostic-groups/dti_diagnostic_groups.json",
        demographic_excel_path="/kaggle/input/demographic/demographic.xlsx",

        # Clinical embedding (PubMedBERT; derived from demographics -> text -> CLS embedding)
        use_clinical_embedding=True,
        clinical_in_fusion=False,
        use_pubmedbert=True,
        pubmedbert_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        pubmedbert_local_files_only=False,
        clinical_device=("cuda" if torch.cuda.is_available() else "cpu"),  # default; set "cpu" if needed
        llm_embedding_dim=768,
        use_apoe4=True,
        clinical_embedding_mode="clinical_no_diagnosis",  # do NOT include diagnosis text (avoid leakage)
        mask_diagnosis_in_embedding=True,


        # Model
        hidden_dim=128,
        gnn_num_layers=3,
        attention_heads=8,
        gnn_dropout=0.3,
        pooling="meanmax",

        # Graph
        dti_connectivity_threshold=0.1,
        fmri_connectivity_threshold=0.1,
        use_edge_weights=True,
        use_node_features=True,

        # Train
        batch_size=8,
        epochs=80,
        learning_rate=1e-3,
        weight_decay=5e-4,

        # Fusion
        fusion_hidden_dim=128,
        fusion_dropout=0.3,
        fusion_epochs=80,
        fusion_learning_rate=1e-3,
        fusion_weight_decay=2e-3,

        # CV + early stop
        dti_n_splits=10,
        fmri_n_splits=10,
        fusion_n_repeats=5,
        fusion_test_size=0.2,
        use_early_stopping=True,
        early_stopping_tolerance=7,
        early_stopping_min_delta=0.0,

        # Node selection
        node_selection_topk=25,
        node_reweight_factor=2.5,
        node_selection_max_batches=50,

        n_iterations=1,
        random_seed=42,
    )

    UnifiedLogger.initialize(f"{cfg.results_base_dir}/logs")
    logger = UnifiedLogger.get_logger("Main")
    cfg.save(f"{cfg.results_base_dir}/config.json")

    device = torch.device(cfg.device)
    print("CUDA available:", torch.cuda.is_available(), "| Using device:", device)

    # label sources
    diagnostic_loader = DiagnosticGroupLoader(cfg.diagnostic_json, "DX")
    demographic_loader = DemographicDataLoader(cfg.demographic_excel_path)

    # scan DTI
    dti_scanner = ConnectivityDataScanner(
        modality_name="DTI",
        connectivity_dir=cfg.dti_connectivity_dir,
        node_features_dir=cfg.dti_node_features_dir,
        config=cfg,
        connectivity_threshold=cfg.dti_connectivity_threshold,
    )
    dti_subjects = dti_scanner.scan_and_validate(diagnostic_loader, demographic_loader)
    if len(dti_subjects) == 0:
        logger.error("No valid DTI subjects found.")
        return None

    # infer DTI dims
    dti_sample = next(iter(dti_subjects.values()))
    cfg.dti_num_regions = int(dti_sample["connectivity"].shape[0])
    if dti_sample.get("node_features") is not None:
        cfg.dti_node_feature_dim = int(dti_sample["node_features"].shape[1])
    else:
        cfg.dti_node_feature_dim = None

    # scan fMRI (DFC: *_dfc.npy)
    fmri_scanner = FMRIDFCScanner(
        base_dir=cfg.fmri_base_dir,
        config=cfg,
    )
    fmri_subjects = fmri_scanner.scan_and_validate(diagnostic_loader, demographic_loader)
    if len(fmri_subjects) == 0:
        logger.error("No valid fMRI subjects found. (Check fmri_* paths)")
        return None

    # infer fMRI dims
    fmri_sample = next(iter(fmri_subjects.values()))
    cfg.fmri_num_regions = int(fmri_sample["connectivity"].shape[0])
    if fmri_sample.get("node_features") is not None:
        cfg.fmri_node_feature_dim = int(fmri_sample["node_features"].shape[1])
    else:
        cfg.fmri_node_feature_dim = None
        # demographic covariates (MODEL INPUT)
        # Single demographics source (same Excel) for both DTI and fMRI, following full_model_loss.py.
        # Vectors are 5-D: [age_norm, gender_enc, apoe4, mmse_norm, cdr_norm]
        if cfg.use_demographics:
            cfg.demographic_feature_dim = 5
            # Ensure every subject has a demographics vector (or None if missing)
            for sdict in (dti_subjects, fmri_subjects):
                for sid, s in sdict.items():
                    if s.get("demographics") is None and demographic_loader is not None:
                        demo_info = demographic_loader.get_subject_info(sid)
                        if demo_info is not None:
                            s["demographics"] = demo_info.to_feature_vector()
                        else:
                            s["demographics"] = np.array([0.5, 0.5, 0.0, 0.5, 0.0], dtype=np.float32)
            print(f"[Demographics] Using single demographic Excel. dim={cfg.demographic_feature_dim} | path={cfg.demographic_excel_path}")
        else:
            cfg.demographic_feature_dim = 0
            for s in dti_subjects.values():
                s["demographics"] = None
            for s in fmri_subjects.values():
                s["demographics"] = None
            print("[Demographics] Disabled by config. Will ignore.")

        # optional PubMed / literature prior over nodes

    pubmed_prior_dti: Optional[np.ndarray] = None
    # If pubmed_alpha > 0, we try to load a per-node prior vector either from:
    # (a) cfg.pubmed_node_prior_path (preferred), or
    # (b) the demographic Excel file (auto-detected sheets/columns).
    if float(cfg.pubmed_alpha) > 0.0:
        pubmed_loader = PubMedNodePriorLoader(
            prior_path=cfg.pubmed_node_prior_path,
            demographic_excel_path=cfg.demographic_excel_path,
        )
        pubmed_prior_dti = pubmed_loader.load_vector(expected_len=int(cfg.dti_num_regions))
        if pubmed_prior_dti is None:
            print("[PubMedPrior] Not loaded (no prior found in file or demographic Excel). Will ignore.")
        else:
            print(f"[PubMedPrior] Loaded ({pubmed_loader.last_source}) | len={pubmed_prior_dti.shape[0]} | alpha={cfg.pubmed_alpha}")

    
    # optional PubMedBERT clinical embedding (from demographics text; like reference implementation)
    # This is used in the fusion classifier when cfg.clinical_in_fusion=True.
    clinical_encoder = None
    if getattr(cfg, "use_clinical_embedding", False) and getattr(cfg, "clinical_in_fusion", True):
        clinical_encoder = PersonalizedClinicalEncoder(cfg, demographic_loader)

        # Cache embeddings once (reused across folds/tasks)
        all_ids_for_clinical = sorted(list(set(dti_subjects.keys()) | set(fmri_subjects.keys())))
        n_cached = clinical_encoder.precompute_all_embeddings(all_ids_for_clinical)

        model_status = "loaded" if getattr(clinical_encoder, "model", None) is not None else "FALLBACK"
        print(
            f"[ClinicalEmbedding] Cached={n_cached} | transformers={TRANSFORMERS_AVAILABLE} | PubMedBERT={model_status} | mode={cfg.clinical_embedding_mode}"
        )
    else:
        print("[ClinicalEmbedding] Disabled by config.")

# quick summary
    dti_dist = Counter(v["class"] for v in dti_subjects.values())
    fmri_dist = Counter(v["class"] for v in fmri_subjects.values())
    print("\nDTI distribution:", dict(dti_dist))
    print("fMRI distribution:", dict(fmri_dist))
    print(f"DTI num_regions={cfg.dti_num_regions} | fMRI num_regions={cfg.fmri_num_regions}")

    
    
    # MULTI-ITERATION EXPERIMENTS (10 iterations x 10 folds)
    
    # We repeat the entire CV procedure `cfg.n_iterations` times with different seeds
    # (methodology unchanged) and aggregate metrics across all folds & iterations.

    n_iters = int(getattr(cfg, "n_iterations", 1) or 1)
    all_task_results: Dict[str, Any] = {}

    for task in TASKS_TO_RUN:
        task_iteration_results: List[Dict[str, Any]] = []

        for it in range(n_iters):
            # Different seed each iteration (keeps methodology identical; only randomness changes)
            iter_cfg = copy.deepcopy(cfg)
            iter_cfg.random_seed = int(cfg.random_seed) + int(it)
            set_deterministic_mode(iter_cfg.random_seed)

            logger.info(f"[{get_task_display_name(task)} | {task}] Iteration {it+1}/{n_iters} | seed={iter_cfg.random_seed}")

            # Filter per task
            dti_task, class_names, mapping = build_task_subjects(dti_subjects, task)
            fmri_task, class_names2, mapping2 = build_task_subjects(fmri_subjects, task)

            if class_names != class_names2:
                raise RuntimeError("Task class_names mismatch between modalities.")
            if mapping != mapping2:
                raise RuntimeError("Task label mapping mismatch between modalities.")

            if len(dti_task) == 0 or len(fmri_task) == 0:
                logger.warning(f"[{task}] No data after filtering for at least one modality. Skipping iteration.")
                continue

            # Update config for task (binary)
            task_cfg = copy.deepcopy(iter_cfg)
            task_cfg.num_classes = len(class_names)
            task_cfg.class_names = class_names
            task_cfg.class_mapping = mapping

            # Run fusion CV (optionally with hyperparameter sweep)
            if getattr(task_cfg, "run_hparam_sweep", False):
                best_cfg, result, sweep_summary = run_hparam_sweep_for_task(
                    dti_task,
                    fmri_task,
                    task_cfg,
                    task_name=task,
                    pubmed_prior_dti=pubmed_prior_dti,
                    clinical_encoder=clinical_encoder,
                )
                task_cfg = best_cfg
                result["sweep_summary_path"] = str(Path(task_cfg.results_base_dir) / "results" / f"hparam_sweep_{task}.json")
            else:
                result = run_multimodal_cv_for_task(
                    dti_task,
                    fmri_task,
                    task_cfg,
                    task_name=task,
                    pubmed_prior_dti=pubmed_prior_dti,
                    clinical_encoder=clinical_encoder,
                    run_tag=f"iter{it+1:02d}",
                )

            result["iteration"] = int(it)
            result["seed"] = int(task_cfg.random_seed)
            task_iteration_results.append(result)

            # Save per-iteration
            out_path_it = Path(task_cfg.results_base_dir) / "results" / f"fusion_results_{task}_iter{it+1:02d}.json"
            with open(out_path_it, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Saved iteration results: {out_path_it}")

        
        # Aggregate over iterations
        
        if not task_iteration_results:
            logger.warning(f"[{task}] No iteration results. Skipping aggregation.")
            continue

        # Collect per-fold metrics across all iterations (total = n_iters * n_splits folds)
        all_fold_metrics = {
            "accuracy": [],
            "auc": [],
            "precision": [],
            "recall": [],
            "sensitivity": [],
            "specificity": [],
            "f1": [],
        }
        # ROC averaging grid (across all iters+folds)
        mean_fpr = np.linspace(0, 1, 200)
        tprs_all = []
        aucs_all = []

        for r in task_iteration_results:
            for fr in r.get("fold_results", []):
                if "fusion_acc" in fr:
                    all_fold_metrics["accuracy"].append(float(fr["fusion_acc"]))
                if fr.get("fusion_auc", None) is not None:
                    all_fold_metrics["auc"].append(float(fr["fusion_auc"]))
                if fr.get("fusion_precision", None) is not None:
                    all_fold_metrics["precision"].append(float(fr["fusion_precision"]))
                if fr.get("fusion_recall", None) is not None:
                    all_fold_metrics["recall"].append(float(fr["fusion_recall"]))
                if fr.get("fusion_sensitivity", None) is not None:
                    all_fold_metrics["sensitivity"].append(float(fr["fusion_sensitivity"]))
                if fr.get("fusion_specificity", None) is not None:
                    all_fold_metrics["specificity"].append(float(fr["fusion_specificity"]))
                if fr.get("fusion_f1", None) is not None:
                    all_fold_metrics["f1"].append(float(fr["fusion_f1"]))

                # ROC curve (if present)
                fpr = np.asarray(fr.get("fusion_fpr", []), dtype=float)
                tpr = np.asarray(fr.get("fusion_tpr", []), dtype=float)
                if fpr.size > 1 and tpr.size > 1:
                    # sort + interpolate
                    order = np.argsort(fpr)
                    fpr_s = fpr[order]
                    tpr_s = tpr[order]
                    tpr_i = np.interp(mean_fpr, fpr_s, tpr_s)
                    tpr_i[0] = 0.0
                    tprs_all.append(tpr_i)

        def _mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
            if not vals:
                return None, None
            arr = np.asarray(vals, dtype=float)
            return float(np.mean(arr)), float(np.std(arr))

        agg = {}
        for k, v in all_fold_metrics.items():
            mu, sd = _mean_std(v)
            agg[k] = {"mean": mu, "std": sd, "n": int(len(v))}

        # Mean ROC + std band
        roc_figs = {}
        if tprs_all:
            tprs_all = np.stack(tprs_all, axis=0)
            mean_tpr = np.mean(tprs_all, axis=0)
            std_tpr = np.std(tprs_all, axis=0)
            mean_tpr[-1] = 1.0

            # AUC of mean curve
            try:
                mean_auc = float(auc(mean_fpr, mean_tpr))
            except Exception:
                mean_auc = None

            roc_figs["mean_auc_from_mean_curve"] = mean_auc

            import matplotlib.pyplot as plt

            # Plot: overall mean ROC with std band
            plt.figure()
            plt.plot(mean_fpr, mean_tpr)
            plt.fill_between(mean_fpr, np.clip(mean_tpr - std_tpr, 0, 1), np.clip(mean_tpr + std_tpr, 0, 1), alpha=0.2)
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"Mean ROC over {n_iters} iterations x {get_fusion_n_repeats(cfg)} fusion repeats ({get_task_display_name(task)})")
            roc_path = Path(cfg.results_base_dir) / "figures" / f"roc_mean_{task}_{n_iters}iters_{get_fusion_n_repeats(cfg)}fusionrepeats.png"
            plt.savefig(roc_path, dpi=200, bbox_inches="tight")
            plt.close()
            roc_figs["mean_roc_path"] = str(roc_path)

        # Plot: per-iteration mean ROC (one curve per iteration)
        try:
            import matplotlib.pyplot as plt

            plt.figure()
            for r in task_iteration_results:
                # average folds inside this iteration
                fprs = []
                tprs = []
                for fr in r.get("fold_results", []):
                    fpr = np.asarray(fr.get("fusion_fpr", []), dtype=float)
                    tpr = np.asarray(fr.get("fusion_tpr", []), dtype=float)
                    if fpr.size > 1 and tpr.size > 1:
                        order = np.argsort(fpr)
                        fpr_s = fpr[order]
                        tpr_s = tpr[order]
                        tpr_i = np.interp(mean_fpr, fpr_s, tpr_s)
                        tpr_i[0] = 0.0
                        tprs.append(tpr_i)
                if tprs:
                    tprs = np.stack(tprs, axis=0)
                    mean_tpr_it = np.mean(tprs, axis=0)
                    mean_tpr_it[-1] = 1.0
                    plt.plot(mean_fpr, mean_tpr_it)

            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"Per-iteration Mean ROC ({get_task_display_name(task)})")
            roc_it_path = Path(cfg.results_base_dir) / "figures" / f"roc_per_iteration_{task}_{n_iters}iters.png"
            plt.savefig(roc_it_path, dpi=200, bbox_inches="tight")
            plt.close()
            roc_figs["per_iteration_roc_path"] = str(roc_it_path)
        except Exception:
            pass

        aggregated_result = {
            "task": task,
            "n_iterations": n_iters,
            "dti_n_splits": int(getattr(cfg, "dti_n_splits", 10)),
            "fmri_n_splits": int(getattr(cfg, "fmri_n_splits", 10)),
            "fusion_n_repeats": int(getattr(cfg, "fusion_n_repeats", 5)),
        "fusion_test_size": float(getattr(cfg, "fusion_test_size", 0.2)),
            "n_total_repeats": int(get_fusion_n_repeats(cfg)) * int(n_iters),
            "aggregate_metrics_over_all_folds": agg,
            "roc_figures": roc_figs,
            "iterations": task_iteration_results,
        }

        # Save aggregated JSON
        out_path_agg = Path(cfg.results_base_dir) / "results" / f"fusion_results_{task}_AGG_{n_iters}iters.json"
        with open(out_path_agg, "w") as f:
            json.dump(aggregated_result, f, indent=2, default=str)

        all_task_results[task] = aggregated_result

        print("\n" + "-" * 90)
        print(f"[{task}] Aggregated over {n_iters} iterations x {get_fusion_n_repeats(cfg)} fusion repeats (N={get_fusion_n_repeats(cfg)*n_iters})")
        for k in ["accuracy", "auc", "precision", "recall", "f1", "sensitivity", "specificity"]:
            mu = aggregated_result["aggregate_metrics_over_all_folds"][k]["mean"]
            sd = aggregated_result["aggregate_metrics_over_all_folds"][k]["std"]
            n = aggregated_result["aggregate_metrics_over_all_folds"][k]["n"]
            if mu is not None:
                print(f"  {k:12s}: {mu:.4f} ± {sd:.4f}   (n={n})")
        if roc_figs.get("mean_roc_path"):
            print(f"  ROC(mean): {roc_figs['mean_roc_path']}")
        if roc_figs.get("per_iteration_roc_path"):
            print(f"  ROC(iters): {roc_figs['per_iteration_roc_path']}")
        print("-" * 90 + "\n")
    combined_path = Path(cfg.results_base_dir) / "results" / "all_tasks_fusion_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_task_results, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("DONE. Saved combined results to:")
    print(combined_path)
    print("=" * 80 + "\n")
    
    return all_task_results


RESULTS_ROOT = "./task_specific_ablation_results"


TASK_NAME = "NC_AD"


ABLATION_NAME = "dgtf_full"


ABLATION_CFG = {'mode': 'multimodal', 'use_demographics': True, 'demographics_in_unimodal': False, 'demographics_in_fusion': True, 'use_clinical_embedding': True, 'clinical_in_fusion': True, 'clinical_embedding_mode': 'clinical_no_diagnosis', 'pubmed_alpha': 0.0, 'node_selection_method': 'gradient', 'node_selection_topk': 25, 'node_reweight_factor': 2.5}


COMMON_OVERRIDES = {'n_iterations': 10, 'dti_n_splits': 10, 'fmri_n_splits': 10, 'fusion_n_repeats': 10, 'fusion_test_size': 0.2, 'batch_size': 8, 'epochs': 120, 'fusion_epochs': 120, 'learning_rate': 0.001, 'weight_decay': 0.0001, 'fusion_learning_rate': 0.001, 'fusion_weight_decay': 0.0005, 'node_selection_topk': 25, 'node_reweight_factor': 2.5, 'node_selection_max_batches': 50, 'dti_connectivity_threshold': 0.1, 'fmri_connectivity_threshold': 0.1, 'random_seed': 42}


DATA_PATH_OVERRIDES = {'fmri_base_dir': auto_detect_flat_fmri_dir()}


def _safe_connectivity_array(subject_entry: Dict[str, Any]) -> Optional[np.ndarray]:
    try:
        conn = np.asarray(subject_entry.get("connectivity"), dtype=np.float32)
        if conn.ndim == 2 and conn.shape[0] == conn.shape[1] and conn.size > 0:
            return conn
    except Exception:
        return None
    return None


def _update_connectivity_aggregate(store: Dict[str, Any], subject_map: Dict[str, Dict], subject_ids: List[str], prefix: str, class_names: Optional[List[str]] = None) -> None:
    if prefix not in store:
        store[prefix] = {"sum": None, "count": 0, "class_sums": {}, "class_counts": {}}
    bucket = store[prefix]
    for sid in subject_ids:
        subj = subject_map.get(sid)
        if not isinstance(subj, dict):
            continue
        conn = _safe_connectivity_array(subj)
        if conn is None:
            continue
        if bucket["sum"] is None:
            bucket["sum"] = np.zeros_like(conn, dtype=np.float64)
        bucket["sum"] += conn.astype(np.float64)
        bucket["count"] += 1
        label = subj.get("label", None)
        if label is None:
            continue
        try:
            label_idx = int(label)
        except Exception:
            continue
        label_name = class_names[label_idx] if class_names and 0 <= label_idx < len(class_names) else str(label_idx)
        if label_name not in bucket["class_sums"]:
            bucket["class_sums"][label_name] = np.zeros_like(conn, dtype=np.float64)
            bucket["class_counts"][label_name] = 0
        bucket["class_sums"][label_name] += conn.astype(np.float64)
        bucket["class_counts"][label_name] += 1


def _finalize_connectivity_aggregate(store: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for prefix, bucket in store.items():
        entry: Dict[str, Any] = {}
        if bucket.get("sum") is not None and int(bucket.get("count", 0)) > 0:
            entry["overall_mean"] = (bucket["sum"] / float(bucket["count"])).astype(np.float32)
            entry["count"] = int(bucket["count"])
        class_means: Dict[str, np.ndarray] = {}
        for cname, csum in bucket.get("class_sums", {}).items():
            ccount = int(bucket.get("class_counts", {}).get(cname, 0))
            if ccount > 0:
                class_means[cname] = (csum / float(ccount)).astype(np.float32)
        if class_means:
            entry["class_means"] = class_means
            entry["class_counts"] = {k: int(v) for k, v in bucket.get("class_counts", {}).items()}
        if entry:
            out[prefix] = entry
    return out


def save_iteration_plot_artifacts(result: Dict[str, Any], results_base_dir: str, iteration: int, task_name: str, ablation_name: str, class_names: Optional[List[str]] = None) -> Dict[str, str]:
    base_dir = Path(results_base_dir)
    artifact_dir = base_dir / "artifacts"
    history_dir = artifact_dir / "history"
    embed_dir = artifact_dir / "embeddings"
    conn_dir = artifact_dir / "connectivity"
    for d in [artifact_dir, history_dir, embed_dir, conn_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, str] = {}

    history_payload = {
        "task": task_name,
        "ablation_name": ablation_name,
        "iteration": int(iteration),
        "class_names": list(class_names or []),
        "fold_histories": [],
    }
    for fr in result.get("fold_results", []) or []:
        item = {
            "fold": fr.get("fold", fr.get("repeat")),
            "n_train": fr.get("n_train", fr.get("n_common_train")),
            "n_val": fr.get("n_val", fr.get("n_common_val")),
        }
        found = False
        for key in ["train_history", "val_history", "dti_train_history", "dti_val_history", "fmri_train_history", "fmri_val_history", "fusion_train_history", "fusion_val_history"]:
            if key in fr and fr[key]:
                item[key] = fr[key]
                found = True
        if found:
            history_payload["fold_histories"].append(item)
    if history_payload["fold_histories"]:
        history_path = history_dir / f"iter{int(iteration):02d}_epoch_history.json"
        save_json(history_path, history_payload)
        manifest["epoch_history_json"] = str(history_path)

    plot_payload = result.pop("_plot_payload", None) or {}
    X = plot_payload.get("tsne_X")
    y = plot_payload.get("tsne_y")
    ids = plot_payload.get("tsne_ids")
    if X is not None and y is not None:
        try:
            X_arr = np.asarray(X, dtype=np.float32)
            y_arr = np.asarray(y)
            if X_arr.ndim == 2 and X_arr.shape[0] >= 5 and y_arr.shape[0] == X_arr.shape[0]:
                embed_path = embed_dir / f"iter{int(iteration):02d}_fusion_embeddings.npz"
                np.savez_compressed(embed_path, embeddings=X_arr, labels=y_arr, subject_ids=np.asarray(ids if ids is not None else []), task=task_name, ablation=ablation_name, iteration=int(iteration), class_names=np.asarray(class_names or []))
                manifest["embedding_npz"] = str(embed_path)
        except Exception:
            pass

    connectivity = plot_payload.get("connectivity") or {}
    if connectivity:
        conn_arrays = {}
        meta = {"task": task_name, "ablation_name": ablation_name, "iteration": int(iteration), "class_names": list(class_names or [])}
        for prefix, entry in connectivity.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("overall_mean") is not None:
                conn_arrays[f"{prefix}_overall_mean"] = np.asarray(entry["overall_mean"], dtype=np.float32)
            for cname, arr in (entry.get("class_means") or {}).items():
                safe_cname = re.sub(r'[^A-Za-z0-9_]+', '_', str(cname))
                conn_arrays[f"{prefix}_class_{safe_cname}"] = np.asarray(arr, dtype=np.float32)
            meta[f"{prefix}_count"] = int(entry.get("count", 0))
            meta[f"{prefix}_class_counts"] = entry.get("class_counts", {})
        if conn_arrays:
            conn_path = conn_dir / f"iter{int(iteration):02d}_mean_connectivity.npz"
            np.savez_compressed(conn_path, **conn_arrays)
            meta_path = conn_dir / f"iter{int(iteration):02d}_mean_connectivity_meta.json"
            save_json(meta_path, meta)
            manifest["connectivity_npz"] = str(conn_path)
            manifest["connectivity_meta_json"] = str(meta_path)

    result["artifact_manifest"] = manifest
    return manifest


def make_cfg(results_dir: Path, extra_overrides: Optional[Dict[str, Any]] = None):
    cfg = MultiModalGNNConfig(results_base_dir=str(results_dir))
    for k, v in COMMON_OVERRIDES.items():
        setattr(cfg, k, v)
    for k, v in DATA_PATH_OVERRIDES.items():
        if v is not None:
            setattr(cfg, k, v)
    if extra_overrides is not None:
        for k, v in extra_overrides.items():
            if k in {"mode", "modality", "selector_patch"}:
                continue
            setattr(cfg, k, v)
    ensure_results_dirs(results_dir)
    return cfg


@contextmanager
def maybe_patch_random_selector(use_random: bool, seed: int = 42):
    if not use_random:
        yield
        return
    original_selector = DTINodeSelector
    class RandomNodeSelector:
        def __init__(self, topk: int = 20, max_batches: int = 0):
            self.topk = int(topk)
            self.max_batches = int(max_batches)
            self.rng = np.random.RandomState(seed)
        def compute_node_scores(self, model, dataloader, device=None):
            return self.rng.rand(int(model.num_regions)).astype(np.float32)
        def select_topk(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)
            topk = min(self.topk, scores.shape[0])
            if topk <= 0:
                return np.array([], dtype=np.int64), np.zeros(scores.shape[0], dtype=bool)
            idx = np.argsort(scores)[::-1][:topk]
            mask = np.zeros(scores.shape[0], dtype=bool)
            mask[idx] = True
            return idx, mask
        def make_node_weights(self, mask: np.ndarray, reweight_factor: float) -> np.ndarray:
            w = np.ones(mask.shape[0], dtype=np.float32)
            w[mask] = float(reweight_factor)
            return w
    globals()["DTINodeSelector"] = RandomNodeSelector
    try:
        yield
    finally:
        globals()["DTINodeSelector"] = original_selector


def prepare_shared_objects(cfg):
    UnifiedLogger.initialize(f"{cfg.results_base_dir}/logs")
    diagnostic_loader = DiagnosticGroupLoader(cfg.diagnostic_json, "DX")
    demographic_loader = DemographicDataLoader(cfg.demographic_excel_path)
    dti_scanner = ConnectivityDataScanner(modality_name="DTI", connectivity_dir=cfg.dti_connectivity_dir, node_features_dir=cfg.dti_node_features_dir, config=cfg, connectivity_threshold=cfg.dti_connectivity_threshold)
    dti_subjects = dti_scanner.scan_and_validate(diagnostic_loader, demographic_loader)
    if len(dti_subjects) == 0:
        raise RuntimeError("No valid DTI subjects found.")
    dti_sample = next(iter(dti_subjects.values()))
    cfg.dti_num_regions = int(dti_sample["connectivity"].shape[0])
    cfg.dti_node_feature_dim = int(dti_sample["node_features"].shape[1]) if dti_sample.get("node_features") is not None else None
    fmri_scanner = FMRIDFCScanner(base_dir=cfg.fmri_base_dir, config=cfg)
    fmri_subjects = fmri_scanner.scan_and_validate(diagnostic_loader, demographic_loader)
    if len(fmri_subjects) == 0:
        raise RuntimeError("No valid fMRI subjects found.")
    fmri_sample = next(iter(fmri_subjects.values()))
    cfg.fmri_num_regions = int(fmri_sample["connectivity"].shape[0])
    cfg.fmri_node_feature_dim = int(fmri_sample["node_features"].shape[1]) if fmri_sample.get("node_features") is not None else None
    cfg.demographic_feature_dim = 5
    for sdict in (dti_subjects, fmri_subjects):
        for sid, s in sdict.items():
            if s.get("demographics") is None and demographic_loader is not None:
                demo_info = demographic_loader.get_subject_info(sid)
                if demo_info is not None:
                    s["demographics"] = demo_info.to_feature_vector()
                else:
                    s["demographics"] = np.array([0.5, 0.5, 0.0, 0.5, 0.0], dtype=np.float32)
    return dti_subjects, fmri_subjects, demographic_loader


def maybe_load_pubmed_prior(cfg) -> Optional[np.ndarray]:
    if float(getattr(cfg, "pubmed_alpha", 0.0)) <= 0.0:
        return None
    loader = PubMedNodePriorLoader(prior_path=cfg.pubmed_node_prior_path, demographic_excel_path=cfg.demographic_excel_path)
    return loader.load_vector(expected_len=int(cfg.dti_num_regions))


def maybe_build_clinical_encoder(cfg, demographic_loader, dti_subjects, fmri_subjects):
    if not (getattr(cfg, "use_clinical_embedding", False) and getattr(cfg, "clinical_in_fusion", True)):
        return None
    encoder = PersonalizedClinicalEncoder(cfg, demographic_loader)
    all_ids = sorted(list(set(dti_subjects.keys()) | set(fmri_subjects.keys())))
    encoder.precompute_all_embeddings(all_ids)
    return encoder


def get_common_task_subjects(dti_subjects, fmri_subjects, task_name: str):
    dti_task, class_names, mapping = build_task_subjects(dti_subjects, task_name)
    fmri_task, class_names2, mapping2 = build_task_subjects(fmri_subjects, task_name)
    if class_names != class_names2 or mapping != mapping2:
        raise RuntimeError("Task metadata mismatch between modalities.")
    common_ids = sorted(list(set(dti_task.keys()) & set(fmri_task.keys())))
    if len(common_ids) < 4:
        raise RuntimeError(f"Too few common subjects for task={task_name}: n={len(common_ids)}")
    dti_common = {sid: dti_task[sid] for sid in common_ids}
    fmri_common = {sid: fmri_task[sid] for sid in common_ids}
    return dti_common, fmri_common, class_names, mapping, common_ids


def run_unimodal_cv(subjects_task: Dict[str, Dict], cfg, task_name: str, modality_name: str, run_tag: str):
    ids = sorted(list(subjects_task.keys()))
    y = np.array([int(subjects_task[sid]["label"]) for sid in ids], dtype=np.int64)
    effective_splits = get_safe_n_splits(y, get_modality_n_splits(cfg, modality_name))
    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=0)
    device = torch.device(cfg.device)
    if modality_name.lower() == "dti":
        num_regions = int(cfg.dti_num_regions)
        node_feature_dim = cfg.dti_node_feature_dim
        conn_threshold = float(cfg.dti_connectivity_threshold)
    else:
        num_regions = int(cfg.fmri_num_regions)
        node_feature_dim = cfg.fmri_node_feature_dim
        conn_threshold = float(cfg.fmri_connectivity_threshold)
    fold_results = []
    tsne_chunks = []
    tsne_labels = []
    tsne_ids: List[str] = []
    conn_store: Dict[str, Any] = {}
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.arange(len(ids)), y), start=1):
        train_ids = [ids[i] for i in tr_idx]
        val_ids = [ids[i] for i in va_idx]
        train_subjects = {sid: subjects_task[sid] for sid in train_ids}
        val_subjects = {sid: subjects_task[sid] for sid in val_ids}
        train_ds = ConnectivityGraphDataset(train_subjects, use_node_features=(cfg.use_node_features and node_feature_dim is not None), use_demographics=(cfg.use_demographics and cfg.demographics_in_unimodal and cfg.demographic_feature_dim > 0), demographic_dim=int(cfg.demographic_feature_dim))
        val_ds = ConnectivityGraphDataset(val_subjects, use_node_features=(cfg.use_node_features and node_feature_dim is not None), use_demographics=(cfg.use_demographics and cfg.demographics_in_unimodal and cfg.demographic_feature_dim > 0), demographic_dim=int(cfg.demographic_feature_dim))
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_connectivity_batch)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_connectivity_batch)
        model = ConnectivityGNN(num_regions=num_regions, node_feature_dim=node_feature_dim, num_classes=cfg.num_classes, connectivity_threshold=conn_threshold, config=cfg, demographic_dim=(int(cfg.demographic_feature_dim) if (cfg.use_demographics and cfg.demographics_in_unimodal and cfg.demographic_feature_dim > 0) else 0)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = None
        if getattr(cfg, "use_lr_scheduler", True):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=float(cfg.lr_scheduler_factor), patience=int(cfg.lr_scheduler_patience), min_lr=float(cfg.lr_scheduler_min_lr))
        early_stopping = EarlyStopping(patience=cfg.early_stopping_tolerance, min_delta=cfg.early_stopping_min_delta, restore_best_weights=True)
        model, epoch_acc_train, epoch_acc_val, _, _, _, epoch_loss_train, epoch_loss_val = train_and_evaluate_gnn(model, optimizer, criterion, train_loader, val_loader, train_ds, val_ds, early_stopping, device, num_epochs=cfg.epochs, node_weights=None, scheduler=scheduler)
        acc_final, loss_final, cm, fpr, tpr, y_true, y_pred, y_prob_pos = eval_gnn(model, criterion, val_loader, val_ds, device, node_weights=None)
        emb_va, y_emb, ids_emb = extract_embeddings(model, val_subjects, cfg, device, num_regions=num_regions, node_feature_dim=node_feature_dim, connectivity_threshold=conn_threshold, node_weights=None)
        if emb_va is not None and len(emb_va) > 0:
            tsne_chunks.append(np.asarray(emb_va, dtype=np.float32))
            tsne_labels.append(np.asarray(y_emb, dtype=np.int64))
            tsne_ids.extend(list(ids_emb))
        _update_connectivity_aggregate(conn_store, subjects_task, val_ids, modality_name.lower(), list(getattr(cfg, "class_names", [])))
        metrics = compute_binary_metrics(y_true, y_pred, y_prob_pos, cm)
        fold_results.append({"fold": fold_idx, "n_train": len(train_ids), "n_val": len(val_ids), "accuracy": float(acc_final), "loss": float(loss_final), "auc": metrics["auc"], "precision": metrics["precision"], "recall": metrics["recall"], "sensitivity": metrics["sensitivity"], "specificity": metrics["specificity"], "f1": metrics["f1"], "confusion_matrix": cm.tolist(), "fpr": fpr.tolist() if hasattr(fpr, "tolist") else list(fpr), "tpr": tpr.tolist() if hasattr(tpr, "tolist") else list(tpr), "modality": modality_name.lower(), "train_history": {"acc": [float(v) for v in epoch_acc_train], "loss": [float(v) for v in epoch_loss_train]}, "val_history": {"acc": [float(v) for v in epoch_acc_val], "loss": [float(v) for v in epoch_loss_val]}})
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {"task": task_name, "run_tag": run_tag, "fold_results": fold_results}
    payload: Dict[str, Any] = {"connectivity": _finalize_connectivity_aggregate(conn_store)}
    if tsne_chunks:
        payload["tsne_X"] = np.concatenate(tsne_chunks, axis=0).tolist()
        payload["tsne_y"] = np.concatenate(tsne_labels, axis=0).tolist()
        payload["tsne_ids"] = list(tsne_ids)
    result["_plot_payload"] = payload
    return result


def run_one_iteration(dti_common, fmri_common, cfg, demographic_loader, iter_idx: int):
    cfg_iter = copy.deepcopy(cfg)
    cfg_iter.random_seed = int(cfg.random_seed) + iter_idx
    set_deterministic_mode(cfg_iter.random_seed)
    run_tag = f"{ABLATION_NAME}_iter{iter_idx+1:02d}"
    if ABLATION_CFG["mode"] == "unimodal":
        modality = ABLATION_CFG["modality"]
        subjects_task = dti_common if modality == "dti" else fmri_common
        result = run_unimodal_cv(subjects_task, cfg_iter, TASK_NAME, modality_name=modality, run_tag=run_tag)
        result["iteration"] = iter_idx + 1
        result["seed"] = cfg_iter.random_seed
        save_iteration_plot_artifacts(result, cfg_iter.results_base_dir, iter_idx + 1, TASK_NAME, ABLATION_NAME, list(getattr(cfg_iter, "class_names", [])))
        return result
    pubmed_prior = maybe_load_pubmed_prior(cfg_iter)
    clinical_encoder = maybe_build_clinical_encoder(cfg_iter, demographic_loader, dti_common, fmri_common)
    use_random = ABLATION_CFG.get("selector_patch", None) == "random"
    with maybe_patch_random_selector(use_random=use_random, seed=int(cfg_iter.random_seed)):
        result = run_multimodal_cv_for_task(dti_common, fmri_common, cfg_iter, task_name=TASK_NAME, pubmed_prior_dti=pubmed_prior, clinical_encoder=clinical_encoder, run_tag=run_tag)
    result["iteration"] = iter_idx + 1
    result["seed"] = cfg_iter.random_seed
    return result


def aggregate_iterations(iter_results: List[Dict[str, Any]], out_dir: Path, class_names: List[str]):
    metric_store = {
        "accuracy": [], "auc": [], "precision": [], "recall": [],
        "sensitivity": [], "specificity": [], "f1": []
    }
    roc_fold_results = []
    confusion_sum = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    all_fold_results = []
    iter_summaries = []

    for result in iter_results:
        fold_results = result.get("fold_results", [])
        per_iter_metrics = {k: [] for k in metric_store}
        for fr in fold_results:
            if ABLATION_CFG["mode"] == "unimodal":
                metric_pairs = {
                    "accuracy": fr.get("accuracy"),
                    "auc": fr.get("auc"),
                    "precision": fr.get("precision"),
                    "recall": fr.get("recall"),
                    "sensitivity": fr.get("sensitivity"),
                    "specificity": fr.get("specificity"),
                    "f1": fr.get("f1"),
                }
                cm = np.asarray(fr.get("confusion_matrix", np.zeros((len(class_names), len(class_names)), dtype=np.int64).tolist()), dtype=np.int64)
                roc_fold_results.append({"fpr": fr.get("fpr", []), "tpr": fr.get("tpr", []), "auc": fr.get("auc", None)})
            else:
                metric_pairs = {
                    "accuracy": fr.get("fusion_acc"),
                    "auc": fr.get("fusion_auc"),
                    "precision": fr.get("fusion_precision"),
                    "recall": fr.get("fusion_recall"),
                    "sensitivity": fr.get("fusion_sensitivity"),
                    "specificity": fr.get("fusion_specificity"),
                    "f1": fr.get("fusion_f1"),
                }
                cm = np.asarray(fr.get("fusion_confusion_matrix", np.zeros((len(class_names), len(class_names)), dtype=np.int64).tolist()), dtype=np.int64)
                roc_fold_results.append({"fpr": fr.get("fusion_fpr", []), "tpr": fr.get("fusion_tpr", []), "auc": fr.get("fusion_auc", None)})
            if cm.shape != confusion_sum.shape:
                cm_fixed = np.zeros_like(confusion_sum)
                r = min(confusion_sum.shape[0], cm.shape[0])
                c = min(confusion_sum.shape[1], cm.shape[1])
                cm_fixed[:r, :c] = cm[:r, :c]
                cm = cm_fixed
            confusion_sum += cm
            for k, v in metric_pairs.items():
                if v is not None and not math.isnan(float(v)):
                    metric_store[k].append(float(v))
                    per_iter_metrics[k].append(float(v))
            all_fold_results.append(fr)
        iter_summaries.append({
            "iteration": result.get("iteration"),
            "seed": result.get("seed"),
            "aggregate_metrics": aggregate_scalar_metrics(per_iter_metrics),
        })

    final_metrics = aggregate_scalar_metrics(metric_store)
    roc_info = plot_mean_roc_from_fold_results(roc_fold_results, title=f"{TASK_NAME} - {ABLATION_NAME}", out_path=out_dir / "figures" / f"roc_mean_{ABLATION_NAME}.png")
    cm_path = plot_confusion_matrix(confusion_sum, title=f"{TASK_NAME} - {ABLATION_NAME}", out_path=out_dir / "figures" / f"confusion_matrix_sum_{ABLATION_NAME}.png", class_names=class_names)
    summary = {
        "task": TASK_NAME,
        "ablation_name": ABLATION_NAME,
        "mode": ABLATION_CFG["mode"],
        "n_iterations": int(COMMON_OVERRIDES["n_iterations"]),
        "dti_n_splits": int(COMMON_OVERRIDES["dti_n_splits"]),
        "fmri_n_splits": int(COMMON_OVERRIDES["fmri_n_splits"]),
        "fusion_n_repeats": int(COMMON_OVERRIDES["fusion_n_repeats"]),
        "n_total_folds": int(len(all_fold_results)),
        "aggregate_metrics_over_all_folds": final_metrics,
        "iteration_summaries": iter_summaries,
        "roc_figures": roc_info,
        "confusion_matrix_sum": confusion_sum.tolist(),
        "class_names": list(class_names),
        "confusion_matrix_figure": cm_path,
        "iterations": iter_results,
    }
    return summary


def main_single_ablation():
    scan_cfg_dir = Path(RESULTS_ROOT) / "_shared_scan_setup"
    scan_cfg = make_cfg(scan_cfg_dir, extra_overrides={"use_demographics": True, "use_clinical_embedding": False, "clinical_in_fusion": False, "pubmed_alpha": 0.0})
    dti_subjects, fmri_subjects, demographic_loader = prepare_shared_objects(scan_cfg)
    inferred = {"dti_num_regions": int(scan_cfg.dti_num_regions), "fmri_num_regions": int(scan_cfg.fmri_num_regions), "dti_node_feature_dim": scan_cfg.dti_node_feature_dim, "fmri_node_feature_dim": scan_cfg.fmri_node_feature_dim}
    ablation_dir = Path(RESULTS_ROOT) / TASK_NAME / ABLATION_NAME
    cfg_block = copy.deepcopy(ABLATION_CFG)
    cfg_block.update(inferred)
    cfg = make_cfg(ablation_dir, extra_overrides=cfg_block)
    dti_common, fmri_common, class_names, mapping, common_ids = get_common_task_subjects(dti_subjects, fmri_subjects, TASK_NAME)
    cfg.class_names = class_names
    cfg.class_mapping = mapping
    cfg.num_classes = len(class_names)
    cfg.experiment_name = f"{TASK_NAME}_{ABLATION_NAME}"
    cfg.save(str(ablation_dir / "config.json"))

    print("\n" + "#" * 100)
    print(f"Running task: {TASK_NAME}")
    print(f"Ablation: {ABLATION_NAME}")
    print(f"Common subjects: {len(common_ids)}")
    print(f"Iterations x split plans: {COMMON_OVERRIDES['n_iterations']} x (DTI={COMMON_OVERRIDES['dti_n_splits']}, fMRI={COMMON_OVERRIDES['fmri_n_splits']}, Fusion={COMMON_OVERRIDES['fusion_n_repeats']})")
    print("#" * 100 + "\n")

    iter_results = []
    for iter_idx in range(int(COMMON_OVERRIDES["n_iterations"])):
        print(f"\n>>> ITERATION {iter_idx+1}/{COMMON_OVERRIDES['n_iterations']} <<<\n")
        result = run_one_iteration(dti_common, fmri_common, cfg, demographic_loader, iter_idx)
        iter_results.append(result)
        save_json(ablation_dir / "results" / f"iter{iter_idx+1:02d}.json", result)

    final_summary = aggregate_iterations(iter_results, ablation_dir, class_names)
    save_json(ablation_dir / "results" / "final_summary.json", final_summary)
    downloadable_archive = create_downloadable_results_artifact(ablation_dir)

    print("\n" + "=" * 100)
    print(f"FINAL SUMMARY | {TASK_NAME} | {ABLATION_NAME}")
    for k in ["accuracy", "auc", "precision", "recall", "sensitivity", "specificity", "f1"]:
        blk = final_summary["aggregate_metrics_over_all_folds"][k]
        if blk["mean"] is not None:
            print(f"{k:12s}: {blk['mean']:.4f} ± {blk['std']:.4f}   (n={blk['n']})")
    print(f"Confusion matrix figure: {final_summary['confusion_matrix_figure']}")
    print(f"ROC figure: {final_summary['roc_figures'].get('mean_roc_path', 'N/A')}")
    print(f"Saved results to: {ablation_dir}")
    if downloadable_archive is not None:
        print(f"Downloadable archive path: {downloadable_archive}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main_single_ablation()
