"""Graph encoders, training loops, embedding extraction, and fusion classifier for DGTF."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score

from utils import EarlyStopping, MultiModalGNNConfig, get_effective_gnn_dropout
from data_processing import ConnectivityGraphDataset, collate_connectivity_batch
class ConnectivityGNN(nn.Module):
    def __init__(
        self,
        num_regions: int,
        node_feature_dim: Optional[int],
        num_classes: int,
        connectivity_threshold: float,
        config: MultiModalGNNConfig,
        demographic_dim: int = 0,
    ):
        super().__init__()
        self.config = config

        self.num_regions = int(num_regions)
        self.node_feature_dim = node_feature_dim
        self.num_classes = int(num_classes)
        self.connectivity_threshold = float(connectivity_threshold)

        # Demographics (optional)
        self.demographic_dim = int(demographic_dim) if demographic_dim else 0
        self.use_demographics = self.demographic_dim > 0

        self.use_node_features = config.use_node_features and (node_feature_dim is not None)
        input_dim = node_feature_dim if self.use_node_features else self.num_regions

        hidden_dim = config.hidden_dim
        heads = config.attention_heads

        # If we use edge weights, we treat them as 1D edge attributes (edge_dim=1).
        edge_dim = 1 if config.use_edge_weights else None

        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.gat_layers.append(
            GATConv(
                input_dim,
                hidden_dim // heads,
                heads=heads,
                dropout=get_effective_gnn_dropout(config),
                edge_dim=edge_dim,
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(config.gnn_num_layers - 1):
            self.gat_layers.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=get_effective_gnn_dropout(config),
                    edge_dim=edge_dim,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        if config.pooling == "meanmax":
            pooled_dim = hidden_dim * 2
        elif config.pooling in {"mean", "max"}:
            pooled_dim = hidden_dim
        else:
            raise ValueError(f"Unknown pooling: {config.pooling}")

        classifier_in_dim = pooled_dim + (self.demographic_dim if self.use_demographics else 0)

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(get_effective_gnn_dropout(config)),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(get_effective_gnn_dropout(config)),

            nn.Linear(hidden_dim // 2, self.num_classes)
        )

    def create_graph_from_connectivity(
        self,
        connectivity_matrices: torch.Tensor,
        node_features_batch: Optional[torch.Tensor],
        batch_size: int,
        node_weights: Optional[torch.Tensor] = None,
    ):
        """
        IMPORTANT: we do NOT remove unselected nodes.
        We ONLY REWEIGHT nodes by scaling their node features.
        """
        num_regions = connectivity_matrices.shape[1]
        device = connectivity_matrices.device

        # node features
        if self.use_node_features and node_features_batch is not None:
            x = node_features_batch.reshape(-1, node_features_batch.shape[-1])
        else:
            # fallback: each node gets its connectivity profile (row)
            x = connectivity_matrices.reshape(batch_size, num_regions, -1)
            x = x.reshape(-1, num_regions)

        # apply node reweighting (selected nodes get higher weights)
        if node_weights is not None:
            # node_weights should be shape [num_regions]
            if node_weights.numel() != num_regions:
                raise ValueError(f"node_weights has {node_weights.numel()} elements, but num_regions={num_regions}")
            w = node_weights.to(device).reshape(1, num_regions).repeat(batch_size, 1).reshape(-1, 1)
            x = x * w

        # edges
        edge_index_list = []
        edge_attr_list = []
        batch_vector = torch.repeat_interleave(torch.arange(batch_size, device=device), num_regions)

        thr = float(self.connectivity_threshold)
        for b in range(batch_size):
            offset = b * num_regions
            conn = connectivity_matrices[b]

            rows, cols = torch.where(torch.abs(conn) > thr)
            if rows.numel() == 0:
                continue
            # remove self-loops
            keep = rows != cols
            rows = rows[keep]
            cols = cols[keep]
            if rows.numel() == 0:
                continue

            ei = torch.stack([rows + offset, cols + offset], dim=0)
            edge_index_list.append(ei)

            if self.config.use_edge_weights:
                ew = conn[rows, cols].reshape(-1, 1)
                edge_attr_list.append(ew)

        # safety edges if graph ended up empty (e.g., too high threshold)
        if len(edge_index_list) == 0:
            # add a small chain per graph
            chain_i = torch.arange(0, min(num_regions - 1, 5), device=device)
            chain_j = chain_i + 1
            for b in range(batch_size):
                offset = b * num_regions
                ei = torch.stack(
                    [torch.cat([chain_i + offset, chain_j + offset]),
                     torch.cat([chain_j + offset, chain_i + offset])],
                    dim=0,
                )
                edge_index_list.append(ei)
                if self.config.use_edge_weights:
                    edge_attr_list.append(torch.full((ei.size(1), 1), 0.1, device=device))

        edge_index = torch.cat(edge_index_list, dim=1).long().contiguous()
        batch = batch_vector.long().contiguous()

        edge_attr = None
        if self.config.use_edge_weights:
            edge_attr = torch.cat(edge_attr_list, dim=0).float().contiguous() if len(edge_attr_list) > 0 else None
        return x, edge_index, edge_attr, batch

    def encode(
        self,
        connectivity_matrices: torch.Tensor,
        node_features_batch: Optional[torch.Tensor] = None,
        node_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns graph-level embedding (pooled representation) before classifier."""
        batch_size = connectivity_matrices.shape[0]
        x, edge_index, edge_attr, batch = self.create_graph_from_connectivity(
            connectivity_matrices, node_features_batch, batch_size, node_weights=node_weights
        )

        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            x = gat(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.elu(x)
            if i < len(self.gat_layers) - 1:
                x = F.dropout(x, p=get_effective_gnn_dropout(self.config), training=self.training)

        if self.config.pooling == "meanmax":
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x_pooled = torch.cat([x_mean, x_max], dim=1)
        elif self.config.pooling == "mean":
            x_pooled = global_mean_pool(x, batch)
        elif self.config.pooling == "max":
            x_pooled = global_max_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling: {self.config.pooling}")

        return x_pooled

    def forward(
        self,
        connectivity_matrices: torch.Tensor,
        node_features_batch: Optional[torch.Tensor] = None,
        demographics_batch: Optional[torch.Tensor] = None,
        node_weights: Optional[torch.Tensor] = None,
        return_embedding: bool = False,
    ):
        emb_graph = self.encode(connectivity_matrices, node_features_batch, node_weights=node_weights)

        if self.use_demographics:
            if demographics_batch is None:
                demo = torch.zeros((emb_graph.size(0), self.demographic_dim), device=emb_graph.device, dtype=emb_graph.dtype)
            else:
                demo = demographics_batch.to(emb_graph.device)
                if demo.dim() == 1:
                    demo = demo.unsqueeze(0)
                if demo.size(0) != emb_graph.size(0):
                    raise ValueError("demographics_batch batch size does not match connectivity batch size")
                if demo.size(1) != self.demographic_dim:
                    raise ValueError(f"demographics_batch has dim={demo.size(1)} but expected {self.demographic_dim}")
            emb_for_cls = torch.cat([emb_graph, demo], dim=1)
        else:
            emb_for_cls = emb_graph

        logits = self.classifier(emb_for_cls)
        if return_embedding:
            return logits, emb_graph
        return logits


def train_gnn(model, optimizer, criterion, dataloader, train_dataset, device, node_weights=None):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for connectivity, node_features, demo, labels, _ in dataloader:
        connectivity = connectivity.to(device)
        if node_features is not None:
            node_features = node_features.to(device)
        if demo is not None:
            demo = demo.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(connectivity, node_features, demographics_batch=demo, node_weights=node_weights)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        loss.backward()
        # Gradient clipping for stability
        clip_norm = float(getattr(model.config, "gradient_clip_norm", 0.0))
        if clip_norm and clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        running_loss += loss.item() * connectivity.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    return epoch_acc, epoch_loss


def eval_gnn(model, criterion, dataloader, valid_dataset, device, node_weights=None):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    all_labels = []
    all_preds = []
    all_probs_pos = []  # for binary ROC/AUC

    with torch.no_grad():
        for connectivity, node_features, demo, labels, _ in dataloader:
            connectivity = connectivity.to(device)
            if node_features is not None:
                node_features = node_features.to(device)
            if demo is not None:
                demo = demo.to(device)
            labels = labels.to(device)

            outputs = model(connectivity, node_features, demographics_batch=demo, node_weights=node_weights)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * connectivity.size(0)
            running_corrects += torch.sum(preds == labels.data)

            all_labels.append(labels.detach().cpu())
            all_preds.append(preds.detach().cpu())

            if outputs.size(1) == 2:
                all_probs_pos.append(probs[:, 1].detach().cpu())

    epoch_loss = running_loss / len(valid_dataset)
    epoch_acc = running_corrects.double() / len(valid_dataset)
    print(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    labels_np = torch.cat(all_labels).numpy() if all_labels else np.array([])
    preds_np = torch.cat(all_preds).numpy() if all_preds else np.array([])

    n_classes = int(getattr(model, "num_classes", 2))
    cm = confusion_matrix(labels_np, preds_np, labels=list(range(n_classes))) if labels_np.size else np.zeros((n_classes, n_classes), dtype=int)

    fpr = np.array([])
    tpr = np.array([])
    if all_probs_pos and labels_np.size:
        probs_pos_np = torch.cat(all_probs_pos).numpy()
        try:
            fpr, tpr, _ = roc_curve(labels_np, probs_pos_np)
        except Exception:
            pass

    probs_pos_np = torch.cat(all_probs_pos).numpy() if (all_probs_pos and labels_np.size) else np.array([])
    return epoch_acc, epoch_loss, cm, fpr, tpr, labels_np, preds_np, probs_pos_np


def train_and_evaluate_gnn(
    model,
    optimizer,
    criterion,
    trainloader,
    valloader,
    train_dataset,
    valid_dataset,
    early_stopping: EarlyStopping,
    device,
    num_epochs: int = 150,
    node_weights=None,
    scheduler=None,
):
    """Train a unimodal GNN with:
      - AdamW optimizer (passed in)
      - ReduceLROnPlateau scheduler (optional; passed in)
      - Early stopping on best validation loss (see EarlyStopping)
      - Optional node reweighting via `node_weights`
      - Optional gradient clipping (configured via model.config.gradient_clip_norm)
    """
    epoch_acc_train = []
    epoch_acc_val = []
    epoch_loss_train = []
    epoch_loss_val = []

    for epoch in range(int(num_epochs)):
        print(f"Epoch {epoch}/{int(num_epochs) - 1}")
        print("-" * 10)

        acc_tr, loss_tr = train_gnn(
            model, optimizer, criterion, trainloader, train_dataset, device, node_weights=node_weights
        )
        epoch_acc_train.append(float(acc_tr.detach().cpu()))
        epoch_loss_train.append(float(loss_tr))

        acc_va, loss_va, _, _, _, _, _, _ = eval_gnn(
            model, criterion, valloader, valid_dataset, device, node_weights=node_weights
        )
        epoch_acc_val.append(float(acc_va.detach().cpu()))
        epoch_loss_val.append(float(loss_va))

        # LR scheduler (val-loss driven)
        if scheduler is not None:
            try:
                scheduler.step(float(loss_va))
            except TypeError:
                # Some schedulers use step() without metric
                scheduler.step()

        # Early stopping (best val loss)
        early_stopping(loss_tr, loss_va, model=model, epoch=epoch)
        if early_stopping.early_stop:
            print(f"Early stop at epoch: {epoch} | best_val_loss={early_stopping.best_val_loss:.6f}")
            break

    # Restore best weights (by val loss)
    model = early_stopping.restore(model)

    # Final eval at best weights
    acc_final, loss_final, cm, fpr, tpr, _, _, _ = eval_gnn(
        model, criterion, valloader, valid_dataset, device, node_weights=node_weights
    )

    return (
        model,
        epoch_acc_train,
        epoch_acc_val,
        cm,
        fpr,
        tpr,
        epoch_loss_train,
        epoch_loss_val,
    )


@torch.no_grad()
def extract_embeddings(
    model: ConnectivityGNN,
    subjects: Dict[str, Dict],
    config: MultiModalGNNConfig,
    device: torch.device,
    num_regions: int,
    node_feature_dim: Optional[int],
    connectivity_threshold: float,
    node_weights: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      embeddings: [N, D]
      labels: [N]
      subject_ids: list length N
    """
    model.eval()

    dataset = ConnectivityGraphDataset(
        subjects,
        use_node_features=(config.use_node_features and node_feature_dim is not None),
        use_demographics=(config.use_demographics and config.demographics_in_unimodal and config.demographic_feature_dim > 0),
        demographic_dim=int(config.demographic_feature_dim),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=collate_connectivity_batch)

    embs = []
    labels_out = []
    ids_out = []

    for connectivity, node_features, demo, labels, subject_ids in loader:
        connectivity = connectivity.to(device)
        if node_features is not None:
            node_features = node_features.to(device)
        if demo is not None:
            demo = demo.to(device)

        logits, emb = model(connectivity, node_features, demographics_batch=demo, node_weights=node_weights, return_embedding=True)
        embs.append(emb.detach().cpu().numpy())
        labels_out.append(labels.detach().cpu().numpy())
        ids_out.extend(subject_ids)

    if embs:
        embs = np.concatenate(embs, axis=0)
        labels_out = np.concatenate(labels_out, axis=0)
    else:
        embs = np.zeros((0, config.hidden_dim * 2), dtype=np.float32)
        labels_out = np.zeros((0,), dtype=np.int64)

    return embs, labels_out, ids_out


class FusionMLP(nn.Module):
    """Fusion MLP with optional PubMedBERT clinical embedding projection.

    Input ordering MUST be:
      [DTI_embedding | fMRI_embedding | demographics(optional) | clinical_embedding(optional)]

    Where:
      - DTI_embedding is extracted from the weighted DTI model (DTI-only node selection).
      - fMRI_embedding is extracted from the weighted fMRI model (imported DTI weights).
      - demographics is the numeric covariates vector (optional).
      - clinical_embedding is PubMedBERT CLS embedding (optional).
    """

    def __init__(
        self,
        dti_dim: int,
        fmri_dim: int,
        demo_dim: int,
        clinical_dim: int,
        hidden_dim: int,
        dropout: float,
        num_classes: int,
    ):
        super().__init__()

        self.dti_dim = int(dti_dim)
        self.fmri_dim = int(fmri_dim)
        self.demo_dim = int(demo_dim)
        self.clinical_dim = int(clinical_dim)
        self.num_classes = int(num_classes)

        self.use_demo = self.demo_dim > 0
        self.use_clinical = self.clinical_dim > 0

        fusion_in = self.dti_dim + self.fmri_dim + (self.demo_dim if self.use_demo else 0)

        # If clinical embedding is enabled, we project it down before fusion (attached-code style).
        if self.use_clinical:
            self.clinical_proj = nn.Sequential(
                nn.Linear(self.clinical_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            fusion_in += hidden_dim
        else:
            self.clinical_proj = None

        self.net = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        start = 0

        dti = x[:, start:start + self.dti_dim]
        start += self.dti_dim

        fmri = x[:, start:start + self.fmri_dim]
        start += self.fmri_dim

        parts = [dti, fmri]

        if self.use_demo:
            demo = x[:, start:start + self.demo_dim]
            start += self.demo_dim
            parts.append(demo)

        if self.use_clinical:
            clinical = x[:, start:start + self.clinical_dim]
            start += self.clinical_dim
            clinical_h = self.clinical_proj(clinical)  # type: ignore
            parts.append(clinical_h)

        x_fused = torch.cat(parts, dim=1)

        return self.net(x_fused)


class FusionEmbeddingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, subject_ids: List[str]):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.subject_ids = subject_ids

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.subject_ids[idx]


def collate_fusion(batch):
    if not batch:
        return torch.tensor([]), torch.tensor([]), []
    X, y, ids = zip(*batch)
    return torch.stack(X), torch.stack(y), list(ids)


def train_fusion_epoch(model, optimizer, criterion, loader, dataset_len, device, grad_clip_norm: float = 0.0):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for X, y, _ in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        _, preds = torch.max(out, 1)
        loss.backward()
        # Gradient clipping for stability
        if grad_clip_norm and float(grad_clip_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        running_corrects += torch.sum(preds == y.data)

    epoch_loss = running_loss / max(dataset_len, 1)
    epoch_acc = running_corrects.double() / max(dataset_len, 1)
    print(f"Fusion Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    return epoch_acc, epoch_loss


@torch.no_grad()
def eval_fusion_epoch(model, criterion, loader, dataset_len, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    all_labels = []
    all_preds = []
    all_probs_pos = []

    for X, y, _ in loader:
        X = X.to(device)
        y = y.to(device)

        out = model(X)
        loss = criterion(out, y)

        probs = torch.softmax(out, dim=1)
        _, preds = torch.max(out, 1)

        running_loss += loss.item() * X.size(0)
        running_corrects += torch.sum(preds == y.data)

        all_labels.append(y.detach().cpu())
        all_preds.append(preds.detach().cpu())
        if out.size(1) == 2:
            all_probs_pos.append(probs[:, 1].detach().cpu())

    epoch_loss = running_loss / max(dataset_len, 1)
    epoch_acc = running_corrects.double() / max(dataset_len, 1)
    print(f"Fusion Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    labels_np = torch.cat(all_labels).numpy() if all_labels else np.array([])
    preds_np = torch.cat(all_preds).numpy() if all_preds else np.array([])
    n_classes = int(getattr(model, "num_classes", 2))
    cm = confusion_matrix(labels_np, preds_np, labels=list(range(n_classes))) if labels_np.size else np.zeros((n_classes, n_classes), dtype=int)

    fpr = np.array([])
    tpr = np.array([])
    if all_probs_pos and labels_np.size:
        probs_pos_np = torch.cat(all_probs_pos).numpy()
        try:
            fpr, tpr, _ = roc_curve(labels_np, probs_pos_np)
        except Exception:
            pass

    probs_pos_np = torch.cat(all_probs_pos).numpy() if (all_probs_pos and labels_np.size) else np.array([])
    return epoch_acc, epoch_loss, cm, fpr, tpr, labels_np, preds_np, probs_pos_np


def train_and_evaluate_fusion(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    train_len,
    val_len,
    early_stopping: EarlyStopping,
    device,
    num_epochs: int,
    scheduler=None,
    grad_clip_norm: float = 0.0,
):
    """Train fusion MLP with:
      - AdamW optimizer (passed in)
      - ReduceLROnPlateau scheduler (optional; passed in)
      - Early stopping on best validation loss
      - Optional gradient clipping
    """
    epoch_acc_train = []
    epoch_acc_val = []
    epoch_loss_train = []
    epoch_loss_val = []
    for epoch in range(int(num_epochs)):
        print(f"Fusion Epoch {epoch}/{int(num_epochs) - 1}")
        print("-" * 10)

        acc_tr, loss_tr = train_fusion_epoch(
            model, optimizer, criterion, train_loader, train_len, device, grad_clip_norm=grad_clip_norm
        )
        epoch_acc_train.append(float(acc_tr.detach().cpu() if hasattr(acc_tr, "detach") else acc_tr))
        epoch_loss_train.append(float(loss_tr))
        acc_va, loss_va, _, _, _, _, _, _ = eval_fusion_epoch(model, criterion, val_loader, val_len, device)
        epoch_acc_val.append(float(acc_va.detach().cpu() if hasattr(acc_va, "detach") else acc_va))
        epoch_loss_val.append(float(loss_va))

        # LR scheduler (val-loss driven)
        if scheduler is not None:
            try:
                scheduler.step(float(loss_va))
            except TypeError:
                scheduler.step()

        # Early stopping (best val loss)
        early_stopping(loss_tr, loss_va, model=model, epoch=epoch)
        if early_stopping.early_stop:
            print(f"Fusion early stop at epoch: {epoch} | best_val_loss={early_stopping.best_val_loss:.6f}")
            break

    # Restore best weights (by val loss)
    model = early_stopping.restore(model)

    acc_final, loss_final, cm, fpr, tpr, y_true, y_pred, y_prob_pos = eval_fusion_epoch(model, criterion, val_loader, val_len, device)
    return model, acc_final, loss_final, cm, fpr, tpr, y_true, y_pred, y_prob_pos, {"train_acc": epoch_acc_train, "val_acc": epoch_acc_val, "train_loss": epoch_loss_train, "val_loss": epoch_loss_val}
