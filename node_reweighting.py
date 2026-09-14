"""DTI-driven node relevance estimation and topology-preserving transfer to fMRI."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool

from utils import UnifiedLogger, get_effective_gnn_dropout
from model import ConnectivityGNN
class PubMedNodePriorLoader:
    """
    Loads a per-node prior vector (length = num_regions) derived from PubMed / literature.

    Two ways to provide the prior:
      1) External file via `prior_path` :
         - .npy: 1D array
         - .csv: first numeric column used
         - .json: list of scores, or dict with key 'scores'
      2) From the demographic Excel file (same file used for labels/covariates),
         by adding a sheet or columns that contain PubMed node scores.

    Excel conventions supported (auto-detected):
      - A sheet name containing "pubmed" (case-insensitive) with:
          * (expected_len rows) x (>=1 numeric column): uses first numeric column, OR
          * 2 columns: node index + score, OR
          * 1 row with expected_len numeric columns: uses that row as vector, OR
          * columns named like "pubmed_0", "pubmed_1", ... (or ROI/Node/Region variants):
            uses column-wise mean across rows.
      - If no PubMed sheet exists, we also scan the main sheet(s) for columns named like
        "pubmed_0" .. "pubmed_{N-1}" (mean across subjects).

    This prior is applied ONLY to SELECTED nodes (DTI-selected top-k) by scaling their weights:
        w_selected := w_selected * (1 + pubmed_alpha * norm_score)

    Unselected nodes remain unchanged (weight=1).
    """

    def __init__(self, prior_path: Optional[str], demographic_excel_path: Optional[str] = None):
        self.logger = UnifiedLogger.get_logger("PubMedNodePriorLoader")
        self.prior_path = Path(prior_path) if prior_path else None
        self.demographic_excel_path = Path(demographic_excel_path) if demographic_excel_path else None
        self.last_source: Optional[str] = None

    _COLNAME_INDEX_RE = re.compile(
        r"(?:^|[^a-z0-9])(?:pubmed|pmid|literature|node|roi|region)[_\s-]*([0-9]{1,4})(?:$|[^a-z0-9])",
        re.IGNORECASE,
    )

    def load_vector(self, expected_len: int) -> Optional[np.ndarray]:
        """
        Returns:
            np.ndarray of shape (expected_len,) if a prior is found; otherwise None.

        Priority:
            1) prior_path (file)
            2) demographic_excel_path (Excel)
        """
        # 1) external file
        if self.prior_path is not None:
            vec = self._load_from_path(self.prior_path, expected_len)
            if vec is not None:
                self.last_source = f"file:{self.prior_path}"
                return vec

        # 2) fallback: demographic excel
        if self.demographic_excel_path is not None and self.demographic_excel_path.exists():
            vec = self._load_from_demographic_excel(self.demographic_excel_path, expected_len)
            if vec is not None:
                return vec

        return None

    def _load_from_path(self, p: Path, expected_len: int) -> Optional[np.ndarray]:
        if not p.exists():
            self.logger.warning(f"PubMed prior file not found: {p}")
            return None
        try:
            if p.suffix.lower() == ".npy":
                vec = np.load(p)
            elif p.suffix.lower() == ".csv":
                df = pd.read_csv(p)
                num_cols = df.select_dtypes(include=[np.number]).columns
                if len(num_cols) == 0:
                    self.logger.warning("PubMed prior CSV has no numeric columns.")
                    return None
                vec = df[num_cols[0]].values
            elif p.suffix.lower() == ".json":
                with open(p, "r") as f:
                    obj = json.load(f)
                if isinstance(obj, list):
                    vec = np.asarray(obj)
                elif isinstance(obj, dict):
                    if "scores" in obj and isinstance(obj["scores"], list):
                        vec = np.asarray(obj["scores"])
                    else:
                        # try dict of index->score
                        try:
                            items = sorted(((int(k), float(v)) for k, v in obj.items()), key=lambda t: t[0])
                            vec = np.asarray([v for _, v in items])
                        except Exception:
                            self.logger.warning("Unsupported PubMed prior JSON format.")
                            return None
                else:
                    self.logger.warning("Unsupported PubMed prior JSON format.")
                    return None
            else:
                self.logger.warning(f"Unsupported PubMed prior format: {p.suffix}")
                return None

            vec = np.asarray(vec, dtype=np.float32).reshape(-1)
            return self._adapt_len(vec, expected_len)

        except Exception as e:
            self.logger.error(f"Failed to load PubMed prior from file: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _adapt_len(self, vec: np.ndarray, expected_len: int) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.shape[0] != expected_len:
            self.logger.warning(
                f"PubMed prior length {vec.shape[0]} != expected {expected_len}. "
                "Will adapt by truncation/padding."
            )
            if vec.shape[0] > expected_len:
                vec = vec[:expected_len]
            else:
                pad = np.zeros((expected_len,), dtype=np.float32)
                pad[: vec.shape[0]] = vec
                vec = pad
        return vec.astype(np.float32)

    def _extract_index_from_colname(self, colname: str) -> Optional[int]:
        m = self._COLNAME_INDEX_RE.search(str(colname))
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _try_vector_from_pubmed_columns(self, df: pd.DataFrame, expected_len: int) -> Optional[np.ndarray]:
        """Columns like pubmed_0..pubmed_{N-1} (or node/roi/region variants). Uses column-wise mean."""
        idx_to_col = {}
        for col in df.columns:
            idx = self._extract_index_from_colname(col)
            if idx is None:
                continue
            if idx < 0 or idx > expected_len:
                continue
            idx_to_col[idx] = col

        if not idx_to_col:
            return None

        # Detect 1-based indexing if all indices are in 1..N and 0 absent
        keys = sorted(idx_to_col.keys())
        one_based = (0 not in idx_to_col) and (min(keys) >= 1) and (max(keys) == expected_len)
        shift = -1 if one_based else 0

        vec = np.zeros((expected_len,), dtype=np.float32)
        filled = 0
        for idx, col in idx_to_col.items():
            j = idx + shift
            if j < 0 or j >= expected_len:
                continue
            col_vals = pd.to_numeric(df[col], errors="coerce").values.astype(np.float32)
            mval = float(np.nanmean(col_vals)) if np.isfinite(np.nanmean(col_vals)) else 0.0
            vec[j] = mval
            filled += 1

        if filled == 0:
            return None
        return vec.astype(np.float32)

    def _try_vector_from_node_score_table(self, df: pd.DataFrame, expected_len: int) -> Optional[np.ndarray]:
        """Try to interpret df as a node-index + score table."""
        if df.shape[1] < 2:
            return None

        cols = list(df.columns)

        for idx_col in cols:
            idx_series = pd.to_numeric(df[idx_col], errors="coerce")
            if idx_series.notna().sum() < max(5, expected_len // 3):
                continue

            for score_col in cols:
                if score_col == idx_col:
                    continue

                score_series = pd.to_numeric(df[score_col], errors="coerce")
                if score_series.notna().sum() < max(5, expected_len // 3):
                    continue

                good = np.isfinite(idx_series.values) & np.isfinite(score_series.values)
                idx_good = idx_series.values[good].astype(int)
                score_good = score_series.values[good].astype(np.float32)

                if idx_good.size < max(5, expected_len // 3):
                    continue

                # detect 1-based vs 0-based
                if idx_good.min() == 1 and idx_good.max() == expected_len:
                    idx_good = idx_good - 1

                vec = np.zeros((expected_len,), dtype=np.float32)
                for i, s in zip(idx_good, score_good):
                    if 0 <= i < expected_len:
                        vec[i] = float(s)
                return vec.astype(np.float32)

        return None

    def _try_vector_from_shape(self, df: pd.DataFrame, expected_len: int) -> Optional[np.ndarray]:
        """Infer vector from df shape if it looks like (N x 1) or (1 x N)."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            if df.shape[0] == expected_len:
                vec = df[num_cols[0]].values.astype(np.float32)
                return self._adapt_len(vec, expected_len)
            if df.shape[0] == 1 and len(num_cols) == expected_len:
                vec = df[num_cols].iloc[0].values.astype(np.float32)
                return self._adapt_len(vec, expected_len)

        # Try coercion if dtypes are object
        df_num = df.apply(pd.to_numeric, errors="coerce")
        num_cols2 = df_num.columns[df_num.notna().any(axis=0)]
        if len(num_cols2) > 0:
            if df_num.shape[0] == expected_len:
                vec = df_num[num_cols2[0]].values.astype(np.float32)
                return self._adapt_len(vec, expected_len)
            if df_num.shape[0] == 1 and len(num_cols2) == expected_len:
                vec = df_num[num_cols2].iloc[0].values.astype(np.float32)
                return self._adapt_len(vec, expected_len)

        return None

    def _parse_pubmed_prior_from_df(self, df: pd.DataFrame, expected_len: int) -> Optional[np.ndarray]:
        if df is None or df.empty:
            return None

        vec = self._try_vector_from_pubmed_columns(df, expected_len)
        if vec is not None:
            return self._adapt_len(vec, expected_len)

        vec = self._try_vector_from_node_score_table(df, expected_len)
        if vec is not None:
            return self._adapt_len(vec, expected_len)

        vec = self._try_vector_from_shape(df, expected_len)
        if vec is not None:
            return self._adapt_len(vec, expected_len)

        return None

    def _load_from_demographic_excel(self, excel_path: Path, expected_len: int) -> Optional[np.ndarray]:
        try:
            xls = pd.ExcelFile(excel_path)
        except Exception as e:
            self.logger.warning(f"Could not open demographic Excel for PubMed prior: {excel_path} ({e})")
            return None

        sheet_names = list(xls.sheet_names)
        pubmed_sheets = [s for s in sheet_names if "pubmed" in s.lower() or "literature" in s.lower()]

        # 1) Try PubMed sheets
        for sname in pubmed_sheets:
            try:
                df = pd.read_excel(xls, sheet_name=sname)
                vec = self._parse_pubmed_prior_from_df(df, expected_len)
                if vec is not None:
                    self.last_source = f"demographic_excel:{excel_path.name}:sheet={sname}"
                    self.logger.info(f"Loaded PubMed node prior from Excel sheet '{sname}' ({excel_path})")
                    return vec.astype(np.float32)
            except Exception:
                continue

        # 2) Fallback: scan all sheets for pubmed_* columns
        for sname in sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sname)
                vec = self._try_vector_from_pubmed_columns(df, expected_len)
                if vec is not None:
                    vec = self._adapt_len(vec, expected_len)
                    self.last_source = f"demographic_excel:{excel_path.name}:columns_in_sheet={sname}"
                    self.logger.info(f"Loaded PubMed node prior from pubmed_* columns in sheet '{sname}' ({excel_path})")
                    return vec.astype(np.float32)
            except Exception:
                continue

        self.logger.warning(
            "No PubMed node prior found in demographic Excel. "
            "Provide config.pubmed_node_prior_path or add a PubMed sheet/columns."
        )
        return None


def _minmax_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or abs(xmax - xmin) < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    x2 = (x - xmin) / (xmax - xmin)
    x2 = np.nan_to_num(x2, nan=0.0, posinf=0.0, neginf=0.0)
    return x2.astype(np.float32)


def apply_pubmed_prior_to_selected_nodes(
    node_weights: np.ndarray,
    selected_mask: np.ndarray,
    pubmed_prior: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Apply PubMed prior ONLY to selected nodes (keeps unselected untouched)."""
    if node_weights is None:
        return None
    if pubmed_prior is None:
        return node_weights

    w = np.asarray(node_weights, dtype=np.float32).copy()
    mask = np.asarray(selected_mask, dtype=bool)
    prior = np.asarray(pubmed_prior, dtype=np.float32).reshape(-1)

    if prior.shape[0] != w.shape[0]:
        # safety adapt
        prior = adapt_node_weights(prior, target_num_regions=w.shape[0]).astype(np.float32)

    prior_n = _minmax_norm(prior)
    scale = 1.0 + float(alpha) * prior_n
    # Apply only on selected
    w[mask] = w[mask] * scale[mask]
    return w.astype(np.float32)


class DTINodeSelector:
    """
    Computes node importance scores using gradient saliency w.r.t. the
    input node features x (before first GAT layer), averaged over batches.

    Selection is computed ONLY on DTI .
    """
    def __init__(self, topk: int = 20, max_batches: int = 0):
        self.topk = int(topk)
        self.max_batches = int(max_batches)

    @torch.no_grad()
    def _init_scores(self, num_regions: int):
        scores = torch.zeros(num_regions, dtype=torch.float32)
        counts = torch.zeros(num_regions, dtype=torch.float32)
        return scores, counts

    def compute_node_scores(
        self,
        model: ConnectivityGNN,
        dataloader: DataLoader,
        device: torch.device,
    ) -> np.ndarray:
        model.eval()
        num_regions = model.num_regions

        scores, counts = self._init_scores(num_regions)
        scores = scores.to(device)
        counts = counts.to(device)

        # We need grads => no torch.no_grad()
        batch_seen = 0
        for connectivity, node_features, demo, labels, _ in dataloader:
            batch_seen += 1
            if self.max_batches > 0 and batch_seen > self.max_batches:
                break

            connectivity = connectivity.to(device)
            if node_features is not None:
                node_features = node_features.to(device)
            labels = labels.to(device)

            batch_size = connectivity.shape[0]

            # Build graph and get input node feature matrix x
            x, edge_index, edge_attr, batch_vec = model.create_graph_from_connectivity(
                connectivity, node_features, batch_size, node_weights=None
            )

            # Enable grads on x only
            x = x.detach().clone().requires_grad_(True)

            # Forward manually through GNN to keep grad on x
            h = x
            for i, (gat, bn) in enumerate(zip(model.gat_layers, model.batch_norms)):
                h = gat(h, edge_index, edge_attr=edge_attr)
                h = bn(h)
                h = F.elu(h)
                if i < len(model.gat_layers) - 1:
                    h = F.dropout(h, p=get_effective_gnn_dropout(model.config), training=False)

            if model.config.pooling == "meanmax":
                h_mean = global_mean_pool(h, batch_vec)
                h_max = global_max_pool(h, batch_vec)
                h_pooled = torch.cat([h_mean, h_max], dim=1)
            elif model.config.pooling == "mean":
                h_pooled = global_mean_pool(h, batch_vec)
            elif model.config.pooling == "max":
                h_pooled = global_max_pool(h, batch_vec)
            else:
                raise ValueError(f"Unknown pooling: {model.config.pooling}")

            if getattr(model, "use_demographics", False) and getattr(model, "demographic_dim", 0) > 0:
                demo0 = torch.zeros((h_pooled.size(0), int(model.demographic_dim)), device=h_pooled.device, dtype=h_pooled.dtype)
                h_for_cls = torch.cat([h_pooled, demo0], dim=1)
            else:
                h_for_cls = h_pooled

            logits = model.classifier(h_for_cls)
            loss = F.cross_entropy(logits, labels)
            # Backprop to x
            loss.backward()

            grad = x.grad  # [batch_size*num_regions, feat_dim]
            if grad is None:
                continue

            grad = grad.detach().abs()

            # Aggregate per node within each graph
            grad = grad.view(batch_size, num_regions, -1)  # [B, R, F]
            node_scores_batch = grad.mean(dim=2)  # [B, R]
            scores += node_scores_batch.sum(dim=0)
            counts += float(batch_size)

            # cleanup grads
            model.zero_grad(set_to_none=True)

        scores = scores / torch.clamp(counts, min=1.0)
        return scores.detach().cpu().numpy()

    def select_topk(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if scores.ndim != 1:
            raise ValueError("scores must be 1D (num_regions,)")

        topk = min(self.topk, scores.shape[0])
        idx = np.argsort(scores)[::-1][:topk]
        mask = np.zeros(scores.shape[0], dtype=bool)
        mask[idx] = True
        return idx, mask

    def make_node_weights(self, mask: np.ndarray, reweight_factor: float) -> np.ndarray:
        w = np.ones(mask.shape[0], dtype=np.float32)
        w[mask] = float(reweight_factor)
        return w


class CentralityNodeSelector:
    """Centrality-based node importance computed from DTI connectivity only.

    This mirrors the *idea* in the attached full_model_loss.py:
      - Build a graph from the DTI connectivity matrix (thresholded)
      - Compute per-node scores (degree + PageRank + betweenness)
      - Average scores across TRAIN subjects
      - Select TOP-K nodes

    It is deterministic (given threshold) and faster than gradient saliency.
    """

    def __init__(self, topk: int = 20, threshold: float = 0.1, max_subjects: int = 0):
        self.topk = int(topk)
        self.threshold = float(threshold)
        self.max_subjects = int(max_subjects)  # 0 => use all

    def compute_node_scores(self, subjects_train: Dict[str, Dict], num_regions: int) -> np.ndarray:
        try:
            import networkx as nx
        except Exception as e:
            raise RuntimeError("Centrality selection requires networkx. Install: pip install networkx") from e

        ids = list(subjects_train.keys())
        if self.max_subjects > 0:
            ids = ids[: self.max_subjects]
        if len(ids) == 0:
            return np.zeros((num_regions,), dtype=np.float32)

        scores_sum = np.zeros((num_regions,), dtype=np.float64)
        used = 0

        for sid in ids:
            conn = subjects_train[sid].get("connectivity", None)
            if conn is None:
                continue
            conn = np.asarray(conn, dtype=np.float32)
            if conn.ndim != 2 or conn.shape[0] != conn.shape[1]:
                continue
            n = conn.shape[0]
            if n != num_regions:
                # adapt by trunc/pad
                tmp = np.zeros((num_regions, num_regions), dtype=np.float32)
                m = min(num_regions, n)
                tmp[:m, :m] = conn[:m, :m]
                conn = tmp

            # Build graph with abs(conn) > threshold
            G = nx.Graph()
            G.add_nodes_from(range(num_regions))
            mat = np.abs(conn)
            # Avoid self loops
            np.fill_diagonal(mat, 0.0)
            rows, cols = np.where(mat > self.threshold)
            for i, j in zip(rows.tolist(), cols.tolist()):
                if i >= j:
                    continue
                w = float(mat[i, j])
                if w <= 0:
                    continue
                G.add_edge(i, j, weight=w)

            # Safety: if graph has no edges, skip
            if G.number_of_edges() == 0:
                continue

            # Centralities
            deg = np.array([d for _, d in G.degree(weight='weight')], dtype=np.float64)
            try:
                pr = nx.pagerank(G, weight='weight')
                pr = np.array([pr[i] for i in range(num_regions)], dtype=np.float64)
            except Exception:
                pr = np.zeros((num_regions,), dtype=np.float64)

            try:
                btw = nx.betweenness_centrality(G, weight='weight', normalized=True)
                btw = np.array([btw[i] for i in range(num_regions)], dtype=np.float64)
            except Exception:
                btw = np.zeros((num_regions,), dtype=np.float64)

            # Normalize each component to [0,1] then average
            def _norm01(x: np.ndarray) -> np.ndarray:
                x = np.asarray(x, dtype=np.float64)
                lo = np.min(x)
                hi = np.max(x)
                if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
                    return np.zeros_like(x)
                return (x - lo) / (hi - lo)

            s = (_norm01(deg) + _norm01(pr) + _norm01(btw)) / 3.0
            scores_sum += s
            used += 1

        if used == 0:
            return np.zeros((num_regions,), dtype=np.float32)

        scores = (scores_sum / float(used)).astype(np.float32)
        return scores

    def select_topk(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        topk = min(self.topk, scores.shape[0])
        idx = np.argsort(scores)[::-1][:topk]
        mask = np.zeros(scores.shape[0], dtype=bool)
        mask[idx] = True
        return idx, mask

    def make_node_weights(self, mask: np.ndarray, reweight_factor: float) -> np.ndarray:
        w = np.ones(mask.shape[0], dtype=np.float32)
        w[mask] = float(reweight_factor)
        return w


def adapt_node_weights(weights: np.ndarray, target_num_regions: int) -> np.ndarray:
    """
    If fMRI has different num regions than DTI, adapt weights by truncation/padding.
    Unseen nodes get weight=1.0.
    """
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    if w.shape[0] == target_num_regions:
        return w
    if w.shape[0] > target_num_regions:
        return w[:target_num_regions].copy()
    out = np.ones(target_num_regions, dtype=np.float32)
    out[:w.shape[0]] = w
    return out
