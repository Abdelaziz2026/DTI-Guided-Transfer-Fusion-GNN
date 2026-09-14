"""Clinical and demographic semantic encoding for DGTF."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from utils import MultiModalGNNConfig, UnifiedLogger
from data_processing import DemographicInfo

try:
    from transformers import AutoTokenizer, AutoModel  # type: ignore
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None  # type: ignore
    AutoModel = None      # type: ignore
class ClinicalDemographicDataLoader:
    """
    Loads richer demographic / phenotype information (age/sex/APOE/MMSE/CDR/diagnosis)
    from the same demographic Excel file.

    This loader is specifically used to build PubMedBERT clinical text embeddings.
    """
    def __init__(self, excel_path: str):
        self.logger = UnifiedLogger.get_logger("ClinicalDemographicLoader")
        self.excel_path = Path(excel_path)
        self.subjects: Dict[str, DemographicInfo] = {}
        if self.excel_path.exists():
            self._load_data()
        else:
            self.logger.warning(f"Demographic file not found: {excel_path}")

    def _normalize_subject_id(self, raw_id: Any) -> Optional[str]:
        if pd.isna(raw_id):
            return None
        raw_id = str(raw_id).strip()
        for pattern in [re.compile(r"(\d{3}_S_\d{5})"), re.compile(r"(\d{3}_S_\d{4})")]:
            match = pattern.search(raw_id)
            if match:
                return match.group(1)
        return None

    def _parse_apoe_genotype(self, apoe_a1, apoe_a2) -> Tuple[Optional[int], Optional[int], int]:
        try:
            a1 = int(float(apoe_a1)) if not pd.isna(apoe_a1) else None
            a2 = int(float(apoe_a2)) if not pd.isna(apoe_a2) else None
            if a1 is not None and a1 not in [2, 3, 4]:
                a1 = None
            if a2 is not None and a2 not in [2, 3, 4]:
                a2 = None
            apoe4_count = sum(1 for a in [a1, a2] if a == 4)
            return a1, a2, apoe4_count
        except (ValueError, TypeError):
            return None, None, 0

    def _identify_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        columns_lower = {col.lower().strip(): col for col in df.columns}

        for pattern in ['subject id', 'subjectid', 'subject_id', 'id', 'ptid', 'rid', 'subject']:
            if pattern in columns_lower:
                mapping['subject_id'] = columns_lower[pattern]
                break

        # age
        for col_lower, col in columns_lower.items():
            if 'age' in col_lower:
                mapping['age'] = col
                break

        # gender
        for pattern in ['sex', 'gender']:
            if pattern in columns_lower:
                mapping['gender'] = columns_lower[pattern]
                break

        # APOE alleles
        for col_lower, col in columns_lower.items():
            if 'apoe' in col_lower and ('a1' in col_lower or col_lower.endswith('1')):
                mapping['apoe_a1'] = col
            if 'apoe' in col_lower and ('a2' in col_lower or col_lower.endswith('2')):
                mapping['apoe_a2'] = col

        # MMSE
        for col_lower, col in columns_lower.items():
            if 'mmse' in col_lower:
                mapping['mmse'] = col
                break

        # CDR
        for col_lower, col in columns_lower.items():
            if 'cdr' in col_lower or 'global cd' in col_lower:
                mapping['cdr'] = col
                break

        # diagnosis
        for pattern in ['research', 'group', 'diagnosis', 'dx', 'dx group', 'dx_group', 'label', 'class']:
            for col_lower, col in columns_lower.items():
                if pattern in col_lower:
                    mapping['research_group'] = col
                    break
            if 'research_group' in mapping:
                break

        return mapping

    def _extract_diagnosis(self, row: pd.Series, col_map: Dict[str, str]) -> Optional[str]:
        col = col_map.get('research_group')
        if col is None:
            return None
        value = row.get(col, None)
        if pd.isna(value):
            return None
        value = str(value).upper().strip()
        diagnosis_map = {
            'CN': 'NC', 'NC': 'NC', 'NORMAL': 'NC', 'NL': 'NC',
            'EMCI': 'EMCI', 'EARLY MCI': 'EMCI',
            'LMCI': 'LMCI', 'LATE MCI': 'LMCI', 'MCI': 'LMCI',
            'AD': 'AD', 'DEMENTIA': 'AD', 'ALZHEIMER': 'AD'
        }
        for key, dx in diagnosis_map.items():
            if key in value:
                return dx
        return None

    def _safe_float(self, value: Any) -> Optional[float]:
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _safe_gender(self, value: Any) -> Optional[str]:
        if pd.isna(value):
            return None
        s = str(value).strip().upper()
        if s and s[0] in ('M', 'F'):
            return s[0]
        # numeric fallback
        try:
            v = float(s)
            if v == 1.0:
                return 'M'
            if v == 2.0 or v == 0.0:
                return 'F'
        except Exception:
            pass
        return None

    def _load_data(self):
        self.logger.info(f"Loading clinical demographics from: {self.excel_path}")
        try:
            df = pd.read_excel(self.excel_path)
            self.logger.info(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")
            col_map = self._identify_columns(df)

            sid_col = col_map.get('subject_id')
            if sid_col is None:
                self.logger.warning("No subject_id column found in demographic sheet (clinical loader).")
                return

            valid_count = 0
            for _, row in df.iterrows():
                sid = self._normalize_subject_id(row.get(sid_col, None))
                if sid is None:
                    continue

                age = self._safe_float(row.get(col_map.get('age', '__NONE__'), None))
                gender = self._safe_gender(row.get(col_map.get('gender', '__NONE__'), None))

                a1, a2, apoe4_count = self._parse_apoe_genotype(
                    row.get(col_map.get('apoe_a1', '__NONE__'), None),
                    row.get(col_map.get('apoe_a2', '__NONE__'), None),
                )

                diagnosis = self._extract_diagnosis(row, col_map)
                mmse = self._safe_float(row.get(col_map.get('mmse', '__NONE__'), None))
                cdr = self._safe_float(row.get(col_map.get('cdr', '__NONE__'), None))

                info = DemographicInfo(
                    subject_id=sid,
                    age=age,
                    gender=gender,
                    apoe_a1=a1,
                    apoe_a2=a2,
                    apoe4_count=int(apoe4_count),
                    apoe4_carrier=bool(apoe4_count > 0),
                    diagnosis=diagnosis,
                    mmse=mmse,
                    global_cdr=cdr,
                )
                self.subjects[sid] = info
                valid_count += 1

            self.logger.info(f"Loaded clinical demographics for {valid_count} subjects")

        except Exception as e:
            self.logger.error(f"Error loading clinical demographics: {e}")
            import traceback
            traceback.print_exc()

    def get_subject_info(self, subject_id: str) -> Optional[DemographicInfo]:
        return self.subjects.get(subject_id)

    def get_diagnosis(self, subject_id: str) -> Optional[str]:
        info = self.subjects.get(subject_id)
        return info.diagnosis if info is not None else None

    def get_all_subject_ids(self) -> Set[str]:
        return set(self.subjects.keys())


class PersonalizedClinicalEncoder:
    """
    PubMedBERT-based clinical embedding (CLS token) with caching.

    - Uses the demographic loader to create per-subject text prompts.
    - Encodes each prompt via PubMedBERT (frozen).
    - If transformers/model not available, falls back to a deterministic numeric embedding.
    """
    def __init__(self, config: "MultiModalGNNConfig", demographic_loader: ClinicalDemographicDataLoader):
        self.config = config
        self.demographic_loader = demographic_loader
        self.logger = UnifiedLogger.get_logger("ClinicalEncoder")

        # Device dedicated for BERT (default: CPU to avoid GPU memory spikes)
        device_str = getattr(config, "clinical_device", None) or getattr(config, "device", "cpu")
        self.device = torch.device(device_str)

        self.tokenizer = None
        self.model = None
        self.embeddings_cache: Dict[str, torch.Tensor] = {}

        if (config.use_clinical_embedding and config.use_pubmedbert and TRANSFORMERS_AVAILABLE):
            self._initialize_model()

        # Safety: prevent diagnosis leakage if requested
        if getattr(config, "mask_diagnosis_in_embedding", True) and getattr(config, "clinical_embedding_mode", "") == "clinical_with_diagnosis":
            self.logger.warning("mask_diagnosis_in_embedding=True -> forcing clinical_embedding_mode='clinical_no_diagnosis'")
            self.config.clinical_embedding_mode = "clinical_no_diagnosis"

    def _initialize_model(self):
        try:
            self.logger.info(f"Loading PubMedBERT: {self.config.pubmedbert_model}")
            local_only = bool(getattr(self.config, "pubmedbert_local_files_only", False))
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.pubmedbert_model, local_files_only=local_only)
            self.model = AutoModel.from_pretrained(self.config.pubmedbert_model, local_files_only=local_only)
            self.model.to(self.device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self.logger.info("PubMedBERT loaded and frozen")
        except Exception as e:
            self.logger.error(f"Failed to load PubMedBERT (will use fallback embedding): {e}")
            self.model = None
            self.tokenizer = None

    @torch.no_grad()
    def encode_subject(self, subject_id: str) -> torch.Tensor:
        if not self.config.use_clinical_embedding:
            return torch.zeros(self.config.llm_embedding_dim, dtype=torch.float32)

        if subject_id in self.embeddings_cache:
            return self.embeddings_cache[subject_id]

        demo = self.demographic_loader.get_subject_info(subject_id)
        if demo is not None:
            description = demo.generate_clinical_description(
                use_apoe4=self.config.use_apoe4,
                mode=self.config.clinical_embedding_mode,
            )
        else:
            description = (
                "An elderly patient presenting for multimodal neuroimaging assessment using "
                "DTI structural connectivity and fMRI functional connectivity for cognitive evaluation."
            )

        if self.model is not None and self.tokenizer is not None:
            inputs = self.tokenizer(
                description,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].squeeze(0).detach().cpu().float()  # CLS
        else:
            emb = self._fallback_embedding(demo)

        # Ensure correct dim
        if emb.numel() != int(self.config.llm_embedding_dim):
            emb = emb.reshape(-1)
            if emb.numel() > int(self.config.llm_embedding_dim):
                emb = emb[: int(self.config.llm_embedding_dim)]
            else:
                pad = torch.zeros(int(self.config.llm_embedding_dim) - emb.numel(), dtype=torch.float32)
                emb = torch.cat([emb, pad], dim=0)

        self.embeddings_cache[subject_id] = emb
        return emb

    def _fallback_embedding(self, demo: Optional[DemographicInfo]) -> torch.Tensor:
        """Deterministic fallback embedding when transformers/model is unavailable."""
        if demo is not None:
            features = demo.to_feature_vector()
        else:
            dim = 3 if self.config.use_apoe4 else 2
            if self.config.clinical_embedding_mode in ("clinical_no_diagnosis", "clinical_with_diagnosis"):
                dim += 2
            features = np.full(dim, 0.5, dtype=np.float32)

        emb = np.zeros(int(self.config.llm_embedding_dim), dtype=np.float32)
        chunk = int(self.config.llm_embedding_dim) // max(int(len(features)), 1)
        for i, feat in enumerate(features):
            start = i * chunk
            end = min((i + 1) * chunk, int(self.config.llm_embedding_dim))
            emb[start:end] = float(feat)
        return torch.tensor(emb, dtype=torch.float32)

    def precompute_all_embeddings(self, subject_ids: List[str]) -> int:
        if not self.config.use_clinical_embedding:
            return 0
        self.logger.info(f"Precomputing clinical embeddings for {len(subject_ids)} subjects")
        for sid in subject_ids:
            if sid not in self.embeddings_cache:
                _ = self.encode_subject(sid)
        return len(self.embeddings_cache)


def build_clinical_embedding_matrix(
    clinical_encoder: Optional[Any],
    subject_ids: List[str],
    embed_dim: int,
) -> np.ndarray:
    """Stack PubMedBERT embeddings in the same order as subject_ids."""
    if clinical_encoder is None or embed_dim <= 0:
        return np.zeros((len(subject_ids), 0), dtype=np.float32)

    rows: List[np.ndarray] = []
    for sid in subject_ids:
        emb_t = clinical_encoder.encode_subject(sid)  # torch.Tensor [D]
        emb = emb_t.detach().cpu().numpy().astype(np.float32).reshape(-1)
        if emb.shape[0] != embed_dim:
            # safety truncate/pad
            out = np.zeros((embed_dim,), dtype=np.float32)
            out[: min(embed_dim, emb.shape[0])] = emb[: min(embed_dim, emb.shape[0])]
            emb = out
        rows.append(emb)
    return np.stack(rows, axis=0).astype(np.float32)


def scale_dense_embeddings(
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """StandardScale using TRAIN statistics (no leakage)."""
    if X_train.size == 0:
        return X_train, X_val
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(np.asarray(X_train, dtype=np.float32)).astype(np.float32)
    Xva = scaler.transform(np.asarray(X_val, dtype=np.float32)).astype(np.float32)
    return Xtr, Xva
