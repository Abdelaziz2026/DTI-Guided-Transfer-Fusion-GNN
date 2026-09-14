"""Data discovery, ADNI metadata handling, task construction, and graph-ready datasets for DGTF."""

from __future__ import annotations

import os
import copy
import re
import json
import math
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from utils import MultiModalGNNConfig, UnifiedLogger
@dataclass
class DemographicInfo:
    subject_id: str
    age: Optional[float] = None
    gender: Optional[str] = None  # 'M' or 'F'
    apoe_a1: Optional[int] = None
    apoe_a2: Optional[int] = None
    apoe4_count: int = 0
    apoe4_carrier: bool = False
    diagnosis: Optional[str] = None
    mmse: Optional[float] = None
    global_cdr: Optional[float] = None

    def to_feature_vector(self) -> np.ndarray:
        """Convert demographic info to a 5-D diagnosis-free clinical covariate vector."""
        age_norm = (self.age - 50.0) / 50.0 if self.age is not None else 0.5
        age_norm = float(np.clip(age_norm, 0.0, 1.0))
        gender_enc = 1.0 if self.gender == 'F' else 0.0
        apoe4 = float(self.apoe4_count)

        mmse_norm = (self.mmse / 30.0) if self.mmse is not None else 0.5
        mmse_norm = float(np.clip(mmse_norm, 0.0, 1.0))

        cdr_norm = (self.global_cdr / 3.0) if self.global_cdr is not None else 0.0
        cdr_norm = float(np.clip(cdr_norm, 0.0, 1.0))

        diagnosis_map = {'NC': 0, 'EMCI': 1, 'LMCI': 2, 'AD': 3}
        diagnosis_num = diagnosis_map.get(self.diagnosis, 0) / 3.0

        return np.asarray([age_norm, gender_enc, apoe4, mmse_norm, cdr_norm], dtype=np.float32)

    def generate_clinical_description(self, use_apoe4: bool = True, mode: str = "clinical_no_diagnosis") -> str:
        """
        mode:
          - "demographics_only": only age/gender + imaging context
          - "clinical_no_diagnosis": + APOE/MMSE/CDR but no diagnosis sentence
          - "clinical_with_diagnosis": includes diagnosis sentence (not used by default; may leak label)
        """
        parts: List[str] = []

        if self.age is not None and self.gender is not None:
            age_group = "elderly" if self.age >= 75 else "older adult"
            gender_str = "female" if self.gender == 'F' else "male"
            parts.append(f"A {int(self.age)}-year-old {gender_str} {age_group}")
        elif self.age is not None:
            parts.append(f"A {int(self.age)}-year-old patient")
        else:
            parts.append("An elderly patient")

        if mode == "clinical_with_diagnosis" and self.diagnosis is not None:
            diagnosis_desc = {
                'NC': "with normal cognitive function and no evidence of dementia",
                'EMCI': "presenting with early mild cognitive impairment",
                'LMCI': "diagnosed with late mild cognitive impairment",
                'AD': "diagnosed with Alzheimer's disease dementia",
            }
            if self.diagnosis in diagnosis_desc:
                parts.append(diagnosis_desc[self.diagnosis])

        if use_apoe4 and mode in ("clinical_no_diagnosis", "clinical_with_diagnosis"):
            if self.apoe4_count == 2:
                parts.append(
                    "who is homozygous for the APOE4 allele, conferring elevated genetic risk for neurodegenerative disease"
                )
            elif self.apoe4_count == 1:
                parts.append(
                    "who is heterozygous for the APOE4 allele, indicating some genetic risk"
                )
            else:
                parts.append("with no APOE4 alleles")

        if mode in ("clinical_no_diagnosis", "clinical_with_diagnosis"):
            if self.mmse is not None:
                parts.append(f"with a Mini-Mental State Examination score of {self.mmse:.0f} out of 30")
            if self.global_cdr is not None:
                parts.append(f"and a Clinical Dementia Rating of {self.global_cdr:.1f}")

        parts.append(
            "presenting for multimodal neuroimaging assessment using DTI structural connectivity and fMRI functional connectivity for cognitive evaluation"
        )
        return ". ".join(parts) + "."


class SubjectIDExtractor:
    """Extracts subject IDs from file paths."""

    def __init__(self):
        self.patterns = [
            re.compile(r'(\d{3}_S_\d{5})'),
            re.compile(r'(\d{3}_S_\d{4})'),
        ]

    def extract(self, filepath: str) -> Optional[str]:
        """Extract subject ID from filepath."""
        basename = os.path.basename(filepath)
        dirname = os.path.dirname(filepath)

        for pattern in self.patterns:
            match = pattern.search(basename)
            if match:
                return match.group(1)
            match = pattern.search(dirname)
            if match:
                return match.group(1)
        return None


    def extract(self, filepath: str) -> Optional[str]:
        text = str(filepath)
        for pattern in self.patterns:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None


class DiagnosticGroupLoader:
    def __init__(self, json_path: str, source_name: str = "DX"):
        self.logger = UnifiedLogger.get_logger(f"DiagnosticLoader_{source_name}")
        self.json_path = Path(json_path) if json_path else None
        self.source_name = source_name
        self.subject_info: Dict[str, Dict[str, Any]] = {}
        if json_path and Path(json_path).exists():
            self._load_data()
        else:
            self.logger.warning(f"Diagnostic JSON not found: {json_path}")

    def _load_data(self):
        try:
            with open(self.json_path, "r") as f:
                data = json.load(f)

            if "details" in data:
                self._parse_details(data["details"])
            elif "groups" in data:
                self._parse_details(data["groups"])
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        self._parse_category(key, value)

            self.logger.info(f"Loaded {len(self.subject_info)} subjects from {self.source_name} diagnostic JSON")
            category_counts = Counter(v["class"] for v in self.subject_info.values())
            for cat, count in sorted(category_counts.items()):
                self.logger.info(f"  {cat}: {count} subjects")

        except Exception as e:
            self.logger.error(f"Failed to load {self.json_path}: {e}")
            import traceback
            traceback.print_exc()

    def _parse_details(self, details_dict):
        for category, subjects_data in details_dict.items():
            self._parse_category(category, subjects_data)

    def _parse_category(self, category, subjects_data):
        if isinstance(subjects_data, dict):
            subjects_list = subjects_data.get("subjects", [])
            if not subjects_list:
                for k, v in subjects_data.items():
                    if isinstance(v, dict) and "subject_id" in v:
                        subjects_list.append(v)
                    elif re.match(r"\d{3}_S_\d{4,5}", str(k)):
                        subjects_list.append({"subject_id": k})
        elif isinstance(subjects_data, list):
            subjects_list = subjects_data
        else:
            return

        for entry in subjects_list:
            if isinstance(entry, dict):
                sid = entry.get("subject_id")
                if sid:
                    self.subject_info[sid] = {"class": category, "quality_score": entry.get("quality_score", 0.85)}
            elif isinstance(entry, str):
                self.subject_info[entry] = {"class": category, "quality_score": 0.85}

    def get_class(self, subject_id: str) -> Optional[str]:
        info = self.subject_info.get(subject_id, {})
        return info.get("class")

    def get_all_subject_ids(self) -> Set[str]:
        return set(self.subject_info.keys())


class DemographicDataLoader:
    """Loads and processes demographic data from Excel file (single demographics source)."""

    def __init__(self, excel_path: str):
        self.logger = UnifiedLogger.get_logger("DemographicLoader")
        self.excel_path = Path(excel_path)
        self.subjects: Dict[str, DemographicInfo] = {}
        self.statistics: Dict[str, Any] = {}

        if self.excel_path.exists():
            self._load_data()
        else:
            self.logger.warning(f"Demographic file not found: {excel_path}")

    def _normalize_subject_id(self, raw_id: Any) -> Optional[str]:
        if pd.isna(raw_id):
            return None
        raw_id = str(raw_id).strip()
        pattern = re.compile(r'(\d{3}_S_\d{4,5})')
        match = pattern.search(raw_id)
        if match:
            return match.group(1)
        return None

    def _safe_float(self, value: Any) -> Optional[float]:
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _safe_string(self, value: Any) -> Optional[str]:
        if pd.isna(value):
            return None
        v = str(value).strip().upper()
        return v[0] if v else None

    def _parse_apoe_genotype(self, apoe_a1: Any, apoe_a2: Any) -> Tuple[Optional[int], Optional[int], int]:
        try:
            a1 = int(float(apoe_a1)) if not pd.isna(apoe_a1) else None
            a2 = int(float(apoe_a2)) if not pd.isna(apoe_a2) else None

            if a1 is not None and a1 not in [2, 3, 4]:
                a1 = None
            if a2 is not None and a2 not in [2, 3, 4]:
                a2 = None

            apoe4_count = 0
            if a1 == 4:
                apoe4_count += 1
            if a2 == 4:
                apoe4_count += 1

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

        for pattern in ['age']:
            for col_lower, col in columns_lower.items():
                if pattern in col_lower:
                    mapping['age'] = col
                    break

        for pattern in ['sex', 'gender']:
            if pattern in columns_lower:
                mapping['gender'] = columns_lower[pattern]
                break

        for col_lower, col in columns_lower.items():
            if 'apoe' in col_lower and ('a1' in col_lower or col_lower.endswith('1')):
                mapping['apoe_a1'] = col
            if 'apoe' in col_lower and ('a2' in col_lower or col_lower.endswith('2')):
                mapping['apoe_a2'] = col

        for pattern in ['mmse', 'mini-mental']:
            for col_lower, col in columns_lower.items():
                if pattern in col_lower:
                    mapping['mmse'] = col
                    break

        for pattern in ['cdr', 'global cd']:
            for col_lower, col in columns_lower.items():
                if pattern in col_lower:
                    mapping['cdr'] = col
                    break

        for pattern in ['research group', 'research', 'group', 'diagnosis', 'dx']:
            for col_lower, col in columns_lower.items():
                if pattern in col_lower:
                    mapping['research_group'] = col
                    break
            if 'research_group' in mapping:
                break

        self.logger.info(f"Column mapping: {mapping}")
        return mapping

    def _extract_diagnosis(self, row: pd.Series, column_mapping: Dict[str, str]) -> Optional[str]:
        research_col = column_mapping.get('research_group')
        if research_col is None:
            return None

        value = row.get(research_col, None)
        if pd.isna(value):
            return None

        value = str(value).upper().strip()
        diagnosis_map = {
            'CN': 'NC', 'NC': 'NC', 'NORMAL': 'NC', 'NL': 'NC',
            'EMCI': 'EMCI', 'EARLY MCI': 'EMCI',
            'LMCI': 'LMCI', 'LATE MCI': 'LMCI', 'MCI': 'LMCI',
            'AD': 'AD', 'DEMENTIA': 'AD', 'ALZHEIMER': 'AD'
        }
        for key, diagnosis in diagnosis_map.items():
            if key in value:
                return diagnosis
        return None

    def _load_data(self):
        self.logger.info(f"Loading demographics from: {self.excel_path}")
        try:
            df = pd.read_excel(self.excel_path)
            self.logger.info(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")
            col_map = self._identify_columns(df)

            valid_count = 0
            for _, row in df.iterrows():
                sid_col = col_map.get('subject_id')
                if sid_col is None:
                    continue

                subject_id = self._normalize_subject_id(row.get(sid_col, None))
                if subject_id is None:
                    continue

                age = self._safe_float(row.get(col_map.get('age', 'Age'), None))
                gender = self._safe_string(row.get(col_map.get('gender', 'Sex'), None))

                a1, a2, apoe4_count = self._parse_apoe_genotype(
                    row.get(col_map.get('apoe_a1', 'APOE A1'), None),
                    row.get(col_map.get('apoe_a2', 'APOE A2'), None),
                )

                diagnosis = self._extract_diagnosis(row, col_map)
                mmse = self._safe_float(row.get(col_map.get('mmse', 'MMSE Total Score'), None))
                cdr = self._safe_float(row.get(col_map.get('cdr', 'Global CDR'), None))

                self.subjects[subject_id] = DemographicInfo(
                    subject_id=subject_id,
                    age=age,
                    gender=gender,
                    apoe_a1=a1,
                    apoe_a2=a2,
                    apoe4_count=apoe4_count,
                    apoe4_carrier=(apoe4_count > 0),
                    diagnosis=diagnosis,
                    mmse=mmse,
                    global_cdr=cdr,
                )
                valid_count += 1

            self.logger.info(f"Loaded {valid_count} subjects with demographics")
        except Exception as e:
            self.logger.error(f"Error loading demographic data: {e}")
            import traceback
            traceback.print_exc()

    def get_subject_info(self, subject_id: str) -> Optional[DemographicInfo]:
        return self.subjects.get(subject_id)

    def get_diagnosis(self, subject_id: str) -> Optional[str]:
        info = self.subjects.get(subject_id)
        return info.diagnosis if info is not None else None


class DemographicFeatureLoader:
    """
    Loads demographic covariates from the Excel sheet and builds a numeric feature vector
    per subject (e.g., age / sex / education).

    IMPORTANT:
    - This is different from DemographicDataLoader which is used ONLY to recover diagnosis labels.
    - These features are OPTIONAL and are intended to be concatenated at the graph (or fusion) level.
    """

    # Common patterns used in ADNI-like demographic sheets
    _DEFAULT_FEATURE_PATTERNS = {
        "age": ["age", "ptage", "age_bl", "ageat", "age at"],
        "sex": ["sex", "gender", "ptgender"],
        "education": ["educ", "education", "pteducat", "years of education"],
        "pubmed": ["pubmed", "pmid", "literature", "paper", "papers", "article", "articles", "citation", "citations"],
    }

    def __init__(self, excel_path: str, features_to_use: Optional[List[str]] = None):
        self.logger = UnifiedLogger.get_logger("DemographicFeatureLoader")
        self.excel_path = Path(excel_path)
        self.features_to_use = [f.lower().strip() for f in (features_to_use or ["age", "sex", "education"])]

        self.feature_columns: Dict[str, str] = {}  # feature_name -> column name in df
        self.subject_features: Dict[str, np.ndarray] = {}
        self.feature_dim: int = 0

        if self.excel_path.exists():
            self._load()
        else:
            self.logger.warning(f"Demographic feature file not found: {excel_path}")

    def _normalize_subject_id(self, raw_id: Any) -> Optional[str]:
        if pd.isna(raw_id):
            return None
        raw_id = str(raw_id).strip()
        for pattern in [re.compile(r"(\d{3}_S_\d{5})"), re.compile(r"(\d{3}_S_\d{4})")]:
            m = pattern.search(raw_id)
            if m:
                return m.group(1)
        return None

    def _identify_subject_id_column(self, df: pd.DataFrame) -> Optional[str]:
        columns_lower = {col.lower().strip(): col for col in df.columns}
        for pattern in ["subject id", "subjectid", "subject_id", "id", "ptid", "rid", "subject"]:
            if pattern in columns_lower:
                return columns_lower[pattern]
        # fallback: try to find any column containing "ptid"
        for col in df.columns:
            if "ptid" in col.lower():
                return col
        return None

    def _infer_feature_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        cols_l = {col.lower(): col for col in df.columns}
        out: Dict[str, str] = {}

        for feat in self.features_to_use:
            patterns = self._DEFAULT_FEATURE_PATTERNS.get(feat, [feat])
            found = None

            # exact match first
            for p in patterns:
                if p.lower() in cols_l:
                    found = cols_l[p.lower()]
                    break

            # substring match
            if found is None:
                for col in df.columns:
                    col_l = col.lower()
                    if any(p.lower() in col_l for p in patterns):
                        found = col
                        break

            if found is not None:
                out[feat] = found

        return out

    def _encode_value(self, feat: str, value: Any) -> float:
        if pd.isna(value):
            return float("nan")

        if feat == "sex":
            # Common encodings in ADNI-like sheets:
            # M/F, Male/Female, 1/2 (often 1=Male, 2=Female), or 0/1
            if isinstance(value, (int, float, np.integer, np.floating)):
                v = float(value)
                if v in (1.0,):
                    return 1.0
                if v in (2.0, 0.0):
                    return 0.0
                return float("nan")

            s = str(value).strip().upper()
            if s in {"M", "MALE"}:
                return 1.0
            if s in {"F", "FEMALE"}:
                return 0.0
            # try numeric string
            try:
                v = float(s)
                if v == 1.0:
                    return 1.0
                if v in (2.0, 0.0):
                    return 0.0
            except Exception:
                pass
            return float("nan")

        # numeric features (age, education, etc.)
        try:
            return float(value)
        except Exception:
            s = str(value).strip()
            try:
                return float(s)
            except Exception:
                return float("nan")

    def _load(self):
        self.logger.info(f"Loading demographic FEATURES from: {self.excel_path}")
        try:
            df = pd.read_excel(self.excel_path)
            sid_col = self._identify_subject_id_column(df)
            if sid_col is None:
                self.logger.warning("Could not identify subject id column in demographic sheet.")
                return

            self.feature_columns = self._infer_feature_columns(df)
            if not self.feature_columns:
                self.logger.warning(
                    "No demographic feature columns matched. "
                    "Set config.demographic_features_to_use or adjust patterns."
                )
                return

            self.logger.info(f"Demographic feature columns: {self.feature_columns}")

            # Build per-subject feature vectors
            for _, row in df.iterrows():
                sid = self._normalize_subject_id(row.get(sid_col, None))
                if sid is None:
                    continue

                vec = []
                for feat in self.features_to_use:
                    col = self.feature_columns.get(feat)
                    val = row.get(col, np.nan) if col is not None else np.nan
                    vec.append(self._encode_value(feat, val))

                self.subject_features[sid] = np.asarray(vec, dtype=np.float32)

            self.feature_dim = len(self.features_to_use)
            self.logger.info(f"Loaded demographic FEATURES for {len(self.subject_features)} subjects; dim={self.feature_dim}")

        except Exception as e:
            self.logger.error(f"Error loading demographic FEATURES: {e}")
            import traceback
            traceback.print_exc()

    def get_features(self, subject_id: str) -> Optional[np.ndarray]:
        return self.subject_features.get(subject_id)

    def attach_to_subject_dict(self, subjects: Dict[str, Dict]):
        """Add a 'demographics' key to each subject entry (if available)."""
        for sid, s in subjects.items():
            s["demographics"] = self.get_features(sid)


class SmartPathDiscovery:
    def __init__(self):
        self.logger = UnifiedLogger.get_logger("PathDiscovery")

    def find_connectivity_dir(self, base_paths: List[str]) -> Optional[str]:
        candidates = []
        for base in base_paths:
            candidates.extend([
                base,
                os.path.join(base, "connectivity_matrices"),
                os.path.join(base, "connectivity"),
                os.path.join(base, "DTI_dataset", "connectivity_matrices"),
                os.path.join(base, "fMRI_dataset", "connectivity_matrices"),
                os.path.join(base, "dataset", "connectivity_matrices"),
            ])

        for candidate in candidates:
            p = Path(candidate)
            if p.exists() and p.is_dir():
                subdirs = [d for d in p.iterdir() if d.is_dir()]
                if subdirs:
                    csv_files = list(subdirs[0].glob("*.csv"))
                    npy_files = list(subdirs[0].glob("*.npy"))
                    if csv_files or npy_files:
                        self.logger.info(f"Found connectivity dir: {candidate} ({len(subdirs)} subject dirs)")
                        return str(p)

                csv_files = list(p.glob("*.csv"))
                npy_files = list(p.glob("*.npy"))
                if csv_files or npy_files:
                    self.logger.info(f"Found connectivity dir (flat): {candidate}")
                    return str(p)

        self.logger.warning("Could not find connectivity directory")
        return None

    def find_node_features_dir(self, base_paths: List[str]) -> Optional[str]:
        candidates = []
        for base in base_paths:
            candidates.extend([
                base,
                os.path.join(base, "node_features"),
                os.path.join(base, "DTI_dataset", "node_features"),
                os.path.join(base, "fMRI_dataset", "node_features"),
            ])

        for candidate in candidates:
            p = Path(candidate)
            if p.exists() and p.is_dir():
                subdirs = [d for d in p.iterdir() if d.is_dir()]
                csv_files = list(p.glob("**/*.csv"))
                if subdirs or csv_files:
                    self.logger.info(f"Found node features dir: {candidate}")
                    return str(p)

        self.logger.info("Node features directory not found (optional)")
        return None


class ConnectivityDataScanner:
    """
    Scan a modality dataset directory, load connectivity matrices and optional node features,
    validate numeric integrity, and assign labels.
    """

    def __init__(self, modality_name: str, connectivity_dir: str, node_features_dir: str,
                 config: MultiModalGNNConfig, connectivity_threshold: float):
        self.modality_name = modality_name
        self.connectivity_dir = Path(connectivity_dir)
        self.node_features_dir = Path(node_features_dir) if node_features_dir else Path("")
        self.config = config
        self.connectivity_threshold = connectivity_threshold

        self.logger = UnifiedLogger.get_logger(f"{modality_name}.DataScanner")
        self.id_extractor = SubjectIDExtractor()

    def scan_and_validate(self, diagnostic_loader: DiagnosticGroupLoader, demographic_loader: DemographicDataLoader) -> Dict[str, Dict]:
        self.logger.info("=" * 60)
        self.logger.info(f"SCANNING AND VALIDATING {self.modality_name} DATASET")
        self.logger.info("=" * 60)
        self.logger.info(f"Connectivity dir: {self.connectivity_dir} | exists={self.connectivity_dir.exists()}")
        self.logger.info(f"Node features dir: {self.node_features_dir} | exists={self.node_features_dir.exists()}")

        # Smart discovery if missing
        if not self.connectivity_dir.exists():
            self.logger.warning(f"Primary connectivity path not found: {self.connectivity_dir}")
            self.logger.info("Attempting smart path discovery...")

            discoverer = SmartPathDiscovery()
            found_conn = discoverer.find_connectivity_dir([
                str(self.connectivity_dir),
                "/kaggle/input",
                "/kaggle/input/dti-dataset",
                "/kaggle/input/fmri-dataset",
                "/kaggle/input/fMRI-dataset",
                "/kaggle/input/dataset",
            ])
            if found_conn:
                self.connectivity_dir = Path(found_conn)
                self.logger.info(f"Auto-discovered connectivity dir: {found_conn}")
            else:
                self.logger.error("Could not find connectivity matrices directory!")
                return {}

            found_nf = discoverer.find_node_features_dir([
                str(self.node_features_dir),
                "/kaggle/input",
                "/kaggle/input/dti-dataset",
                "/kaggle/input/fmri-dataset",
            ])
            if found_nf:
                self.node_features_dir = Path(found_nf)
                self.logger.info(f"Auto-discovered node features dir: {found_nf}")

        all_discovered = self._discover_subjects()
        if not all_discovered:
            self.logger.error("No subjects discovered!")
            return {}

        self.logger.info(f"Discovered {len(all_discovered)} subjects in {self.modality_name}")

        labeled_subjects = self._assign_labels(all_discovered, diagnostic_loader, demographic_loader)
        if not labeled_subjects:
            self.logger.error("No subjects could be labeled!")
            return {}

        valid_subjects = self._validate_data(labeled_subjects)
        if not valid_subjects:
            self.logger.error("No subjects passed validation!")
            return {}

        class_counts = Counter(v["class"] for v in valid_subjects.values())
        self.logger.info(f"FINAL CLASS DISTRIBUTION ({self.modality_name}):")
        for cls_name in ["NC", "AD", "EMCI", "LMCI"]:
            self.logger.info(f"  {cls_name}: {class_counts.get(cls_name, 0)}")

        return valid_subjects

    # discovery
    def _discover_subjects(self) -> Dict[str, Dict]:
        all_discovered: Dict[str, Dict] = {}

        if not self.connectivity_dir.exists():
            return all_discovered

        # Strategy 1: subject subdirectories
        subject_dirs = [d for d in self.connectivity_dir.iterdir() if d.is_dir()]
        self.logger.info(f"Found {len(subject_dirs)} subdirectories in connectivity dir")

        for subject_dir in subject_dirs:
            subject_id = self.id_extractor.extract(subject_dir.name)
            if subject_id is None:
                dirname = subject_dir.name.strip()
                if re.match(r"\d{3}_S_\d{4,5}", dirname):
                    subject_id = dirname
                else:
                    continue

            conn_file = self._find_connectivity_file(subject_dir, subject_id)
            if conn_file is None:
                continue

            node_features_path = self._find_node_features(subject_id)
            all_discovered[subject_id] = {
                "subject_id": subject_id,
                "connectivity_path": str(conn_file),
                "node_features_path": node_features_path,
            }

        # Strategy 2: flat files
        if not all_discovered:
            csv_files = list(self.connectivity_dir.glob("*.csv"))
            npy_files = list(self.connectivity_dir.glob("*.npy"))
            all_files = csv_files + npy_files
            self.logger.info(f"Found {len(all_files)} files in connectivity dir (flat)")
            for f in all_files:
                subject_id = self.id_extractor.extract(f.name)
                if subject_id is None:
                    continue
                node_features_path = self._find_node_features(subject_id)
                all_discovered[subject_id] = {
                    "subject_id": subject_id,
                    "connectivity_path": str(f),
                    "node_features_path": node_features_path,
                }

        # Strategy 3: recursive
        if not all_discovered:
            files = list(self.connectivity_dir.rglob("*.csv")) + list(self.connectivity_dir.rglob("*.npy"))
            self.logger.info(f"Found {len(files)} files recursively")
            for f in files:
                subject_id = self.id_extractor.extract(str(f))
                if subject_id is None:
                    continue
                if subject_id not in all_discovered:
                    node_features_path = self._find_node_features(subject_id)
                    all_discovered[subject_id] = {
                        "subject_id": subject_id,
                        "connectivity_path": str(f),
                        "node_features_path": node_features_path,
                    }

        return all_discovered

    def _find_connectivity_file(self, subject_dir: Path, subject_id: str) -> Optional[Path]:
        candidates = [
            subject_dir / f"{subject_id}_connectivity_matrix.csv",
            subject_dir / f"{subject_id}_connectivity.csv",
            subject_dir / f"{subject_id}.csv",
            subject_dir / "connectivity_matrix.csv",
            subject_dir / "connectivity.csv",
            subject_dir / f"{subject_id}_connectivity_matrix.npy",
            subject_dir / f"{subject_id}.npy",
            subject_dir / "connectivity_matrix.npy",
            subject_dir / "connectivity.npy",
        ]
        for c in candidates:
            if c.exists():
                return c

        # fallback: first csv or npy
        csvs = list(subject_dir.glob("*.csv"))
        if csvs:
            return csvs[0]
        npys = list(subject_dir.glob("*.npy"))
        if npys:
            return npys[0]
        return None

    def _find_node_features(self, subject_id: str) -> Optional[str]:
        if not self.node_features_dir.exists():
            return None

        subject_nf_dir = self.node_features_dir / subject_id
        if subject_nf_dir.exists():
            candidates = [
                subject_nf_dir / f"{subject_id}_node_features.csv",
                subject_nf_dir / f"{subject_id}.csv",
                subject_nf_dir / "node_features.csv",
            ]
            for c in candidates:
                if c.exists():
                    return str(c)
            csv_files = list(subject_nf_dir.glob("*.csv"))
            if csv_files:
                return str(csv_files[0])

        flat_candidates = [
            self.node_features_dir / f"{subject_id}_node_features.csv",
            self.node_features_dir / f"{subject_id}.csv",
        ]
        for c in flat_candidates:
            if c.exists():
                return str(c)

        return None

    # labels
    def _assign_labels(self, all_discovered: Dict[str, Dict],
                       diagnostic_loader: DiagnosticGroupLoader,
                       demographic_loader: DemographicDataLoader) -> Dict[str, Dict]:
        labeled_subjects: Dict[str, Dict] = {}
        label_source_counts = Counter()

        for subject_id, info in all_discovered.items():
            label = None
            source = None

            diag_class = diagnostic_loader.get_class(subject_id)
            if diag_class is not None and diag_class in self.config.class_mapping:
                label = diag_class
                source = "diagnostic_json"

            if label is None:
                demo_class = demographic_loader.get_diagnosis(subject_id)
                if demo_class is not None and demo_class in self.config.class_mapping:
                    label = demo_class
                    source = "demographic"

            if label is None:
                continue

            info["class"] = label
            info["label"] = self.config.class_mapping[label]
            info["label_source"] = source
            labeled_subjects[subject_id] = info
            label_source_counts[source] += 1

        self.logger.info(f"Labeled subjects: {len(labeled_subjects)} | label sources: {dict(label_source_counts)}")
        return labeled_subjects

    # validation
    def _validate_data(self, labeled_subjects: Dict[str, Dict]) -> Dict[str, Dict]:
        valid_subjects: Dict[str, Dict] = {}
        reference_shape = None

        for subject_id, info in labeled_subjects.items():
            connectivity = self._load_and_validate_connectivity(info["connectivity_path"])
            if connectivity is None:
                continue

            if reference_shape is None:
                reference_shape = connectivity.shape
            elif connectivity.shape != reference_shape:
                connectivity = self._resize_matrix(connectivity, reference_shape[0])

            node_features = None
            if self.config.use_node_features and info.get("node_features_path"):
                node_features = self._load_and_validate_node_features(info["node_features_path"], connectivity.shape[0])

            info["connectivity"] = connectivity
            info["node_features"] = node_features
            valid_subjects[subject_id] = info

        # Infer dims (stored back to config by caller)
        return valid_subjects

    def _load_and_validate_connectivity(self, filepath: str) -> Optional[np.ndarray]:
        try:
            fp = Path(filepath)
            if fp.suffix.lower() == ".npy":
                connectivity = np.load(fp).astype(np.float32)
            else:
                # CSV
                try:
                    df = pd.read_csv(fp, index_col=0)
                except Exception:
                    df = pd.read_csv(fp)

                df_numeric = df.apply(pd.to_numeric, errors="coerce")
                df_numeric = df_numeric.dropna(axis=0, how="all")
                df_numeric = df_numeric.dropna(axis=1, how="all")
                if df_numeric.empty:
                    return None

                if df_numeric.shape[0] != df_numeric.shape[1]:
                    common = df_numeric.index.intersection(df_numeric.columns)
                    if len(common) > 0:
                        df_numeric = df_numeric.loc[common, common]

                connectivity = df_numeric.values.astype(np.float32)

            # Ensure 2D square
            if connectivity.ndim != 2:
                return None
            if connectivity.shape[0] != connectivity.shape[1]:
                min_dim = min(connectivity.shape)
                connectivity = connectivity[:min_dim, :min_dim]
                if min_dim < 10:
                    return None

            if connectivity.shape[0] < 10:
                return None

            if np.any(np.isinf(connectivity)):
                connectivity = np.nan_to_num(connectivity, nan=0.0, posinf=0.0, neginf=0.0)
            connectivity = np.nan_to_num(connectivity, nan=0.0)

            if np.allclose(connectivity, 0.0):
                return None

            # Symmetrize
            connectivity = (connectivity + connectivity.T) / 2.0
            return connectivity

        except Exception:
            return None

    def _load_and_validate_node_features(self, filepath: str, num_nodes: int) -> Optional[np.ndarray]:
        try:
            fp = Path(filepath)
            # Only CSV for node features
            try:
                df = pd.read_csv(fp, index_col=0)
            except Exception:
                df = pd.read_csv(fp)

            if df.shape[1] > 1 and df.iloc[:, 0].dtype == object:
                df = df.iloc[:, 1:]

            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                return None

            node_features = df[numeric_columns].values.astype(np.float32)
            node_features = np.nan_to_num(node_features, nan=0.0, posinf=1.0, neginf=-1.0)

            # shape fix
            if node_features.shape[0] != num_nodes:
                if node_features.shape[0] > num_nodes:
                    node_features = node_features[:num_nodes, :]
                else:
                    padded = np.zeros((num_nodes, node_features.shape[1]), dtype=np.float32)
                    padded[:node_features.shape[0], :] = node_features
                    node_features = padded

            scaler = StandardScaler()
            node_features = scaler.fit_transform(node_features)
            return node_features.astype(np.float32)

        except Exception:
            return None

    def _resize_matrix(self, matrix: np.ndarray, target_size: int) -> np.ndarray:
        current_size = matrix.shape[0]
        if current_size == target_size:
            return matrix
        elif current_size > target_size:
            return matrix[:target_size, :target_size]
        else:
            padded = np.zeros((target_size, target_size), dtype=np.float32)
            padded[:current_size, :current_size] = matrix
            return padded


TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "NC_AD": {
        "display_name": "NC vs AD",
        "class_names": ["NC", "AD"],
        "source_to_target": {"NC": "NC", "AD": "AD"},
    },
    "NC_MCI": {
        "display_name": "NC vs MCI",
        "class_names": ["NC", "MCI"],
        "source_to_target": {"NC": "NC", "EMCI": "MCI", "LMCI": "MCI"},
    },
    "EMCI_LMCI": {
        "display_name": "EMCI vs LMCI",
        "class_names": ["EMCI", "LMCI"],
        "source_to_target": {"EMCI": "EMCI", "LMCI": "LMCI"},
    },
    "NC_EMCI": {
        "display_name": "NC vs EMCI",
        "class_names": ["NC", "EMCI"],
        "source_to_target": {"NC": "NC", "EMCI": "EMCI"},
    },
    "NC_EMCI_LMCI": {
        "display_name": "NC vs EMCI vs LMCI",
        "class_names": ["NC", "EMCI", "LMCI"],
        "source_to_target": {"NC": "NC", "EMCI": "EMCI", "LMCI": "LMCI"},
    },
    "MCI_AD": {
        "display_name": "MCI vs AD",
        "class_names": ["MCI", "AD"],
        "source_to_target": {"EMCI": "MCI", "LMCI": "MCI", "AD": "AD"},
    },
}


TASK_ALIASES: Dict[str, str] = {
    "NC_AD": "NC_AD",
    "NCVSAD": "NC_AD",
    "NC-AD": "NC_AD",
    "NC_VS_AD": "NC_AD",
    "NC_MCI": "NC_MCI",
    "NCVSMCI": "NC_MCI",
    "NC-MCI": "NC_MCI",
    "NC_VS_MCI": "NC_MCI",
    "EMCI_LMCI": "EMCI_LMCI",
    "EMCIVSLMCI": "EMCI_LMCI",
    "EMCI-LMCI": "EMCI_LMCI",
    "EMCI_VS_LMCI": "EMCI_LMCI",
    "NC_EMCI": "NC_EMCI",
    "NCVSEMCI": "NC_EMCI",
    "NC-EMCI": "NC_EMCI",
    "NC_VS_EMCI": "NC_EMCI",
    "NC_EMCI_LMCI": "NC_EMCI_LMCI",
    "NCVSEMCIVSLMCI": "NC_EMCI_LMCI",
    "NC-EMCI-LMCI": "NC_EMCI_LMCI",
    "NC_VS_EMCI_VS_LMCI": "NC_EMCI_LMCI",
    "MCI_AD": "MCI_AD",
    "MCIVSAD": "MCI_AD",
    "MCI-AD": "MCI_AD",
    "MCI_VS_AD": "MCI_AD",
}


def normalize_task_name(task: str) -> str:
    task_norm = str(task).upper().strip()
    task_norm = task_norm.replace("/", "_")
    task_norm = task_norm.replace(" ", "")
    task_norm = task_norm.replace("VS", "_VS_")
    task_norm = task_norm.replace("-", "_")
    while "__" in task_norm:
        task_norm = task_norm.replace("__", "_")
    if task_norm in TASK_SPECS:
        return task_norm
    if task_norm in TASK_ALIASES:
        return TASK_ALIASES[task_norm]
    compact = task_norm.replace("_", "")
    if compact in TASK_ALIASES:
        return TASK_ALIASES[compact]
    raise ValueError(
        f"Unknown task '{task}'. Use one of: {', '.join(TASK_SPECS.keys())}"
    )


def get_task_display_name(task: str) -> str:
    task_key = normalize_task_name(task)
    return TASK_SPECS[task_key]["display_name"]


def build_task_subjects(valid_subjects: Dict[str, Dict], task: str) -> Tuple[Dict[str, Dict], List[str], Dict[str, int]]:
    """
    Supported task options:
      - "NC_AD"      : NC vs AD
      - "NC_MCI"     : NC vs (EMCI + LMCI) mapped to MCI
      - "EMCI_LMCI"  : EMCI vs LMCI
      - "MCI_AD"     : (EMCI + LMCI) mapped to MCI vs AD

    The function keeps the original subject payload and only rewrites the
    task-specific class / label when required.
    """
    task_key = normalize_task_name(task)
    spec = TASK_SPECS[task_key]
    class_names = list(spec["class_names"])
    mapping = {name: idx for idx, name in enumerate(class_names)}
    source_to_target = dict(spec["source_to_target"])

    out: Dict[str, Dict] = {}
    for sid, s in valid_subjects.items():
        src_class = s.get("class")
        if src_class not in source_to_target:
            continue
        target_class = source_to_target[src_class]
        s2 = copy.deepcopy(s)
        s2["original_class"] = src_class
        s2["class"] = target_class
        s2["label"] = mapping[target_class]
        out[sid] = s2

    return out, class_names, mapping


class FMRIDFCScanner:
    """
    Scan fMRI DFC dataset where each subject is stored as a single NumPy file:
        <SUBJECT_ID>_dfc.npy
    with shape (T, N, N).

    We convert the DFC sequence into a *static* connectivity matrix for this pipeline by
    averaging over time: connectivity = mean(dfc[:T], axis=0).

    This matches the rest of the codebase which expects one connectivity matrix per subject.
    """

    def __init__(self, base_dir: str, config: MultiModalGNNConfig):
        self.base_dir = Path(base_dir)
        self.config = config
        self.logger = UnifiedLogger.get_logger("FMRIDFCScanner")
        self._id_patterns = [
            re.compile(r"(\d{3}_S_\d{5})"),
            re.compile(r"(\d{3}_S_\d{4})"),
        ]

    def _extract_subject_id(self, filename: str) -> Optional[str]:
        for pat in self._id_patterns:
            m = pat.search(filename)
            if m:
                return m.group(1)
        return None

    def scan_and_validate(self, diagnostic_loader: DiagnosticGroupLoader, demographic_loader: Optional[DemographicDataLoader] = None) -> Dict[str, Dict]:
        if not self.base_dir.exists():
            self.logger.error(f"fMRI base dir not found: {self.base_dir}")
            return {}

        dfc_files = sorted(self.base_dir.glob("*_dfc.npy"))
        if len(dfc_files) == 0:
            # Some datasets might use uppercase or other suffix conventions
            dfc_files = sorted(self.base_dir.glob("*dfc*.npy"))

        self.logger.info(f"[fMRI] Found {len(dfc_files)} DFC files in {self.base_dir}")

        subjects: Dict[str, Dict] = {}
        for fpath in dfc_files:
            subject_id = self._extract_subject_id(fpath.name)
            if subject_id is None:
                continue

            # label (match DTI logic: use diagnostic JSON first, fallback to demographics)
            cls = None
            diag_class = diagnostic_loader.get_class(subject_id)
            if diag_class is not None and diag_class in self.config.class_mapping:
                cls = diag_class
            elif demographic_loader is not None:
                demo_class = demographic_loader.get_diagnosis(subject_id)
                if demo_class is not None and demo_class in self.config.class_mapping:
                    cls = demo_class

            if cls is None:
                continue

            try:
                dfc = np.load(str(fpath))
            except Exception as e:
                self.logger.warning(f"[fMRI] Could not load {fpath}: {e}")
                continue

            if not (isinstance(dfc, np.ndarray) and dfc.ndim == 3 and dfc.shape[1] == dfc.shape[2]):
                self.logger.warning(f"[fMRI] Unexpected shape for {fpath.name}: {getattr(dfc, 'shape', None)}")
                continue

            # Time steps: truncate/pad to config.fmri_time_steps
            T_target = int(getattr(self.config, "fmri_time_steps", 30))
            if dfc.shape[0] > T_target:
                dfc = dfc[:T_target]
            elif dfc.shape[0] < T_target:
                # pad by repeating last frame
                pad = np.repeat(dfc[-1:,:,:], T_target - dfc.shape[0], axis=0)
                dfc = np.concatenate([dfc, pad], axis=0)

            # Regions: take first 90 to match DTI (and/or pad if needed)
            R_target = int(getattr(self.config, "fmri_num_regions", 90))
            if dfc.shape[1] > R_target:
                dfc = dfc[:, :R_target, :R_target]
            elif dfc.shape[1] < R_target:
                tmp = np.zeros((dfc.shape[0], R_target, R_target), dtype=np.float32)
                r = dfc.shape[1]
                tmp[:, :r, :r] = dfc
                dfc = tmp

            dfc = dfc.astype(np.float32, copy=False)
            dfc = np.nan_to_num(dfc, nan=0.0, posinf=0.0, neginf=0.0)

            # Convert dynamic FC -> static connectivity for this model
            connectivity = dfc.mean(axis=0)
            # squash outliers slightly
            connectivity = np.tanh(connectivity)

            # Ensure symmetric (some DFC estimators yield small asymmetries)
            connectivity = (connectivity + connectivity.T) / 2.0
            np.fill_diagonal(connectivity, 0.0)

            demo_vec = None
            if demographic_loader is not None:
                demo_info = demographic_loader.get_subject_info(subject_id)
                if demo_info is not None:
                    demo_vec = demo_info.to_feature_vector()
                else:
                    demo_vec = np.array([0.5, 0.5, 0.0, 0.5, 0.0], dtype=np.float32)

            subjects[subject_id] = {
                "subject_id": subject_id,
                "connectivity": connectivity,
                "node_features": None,  # dataset has no separate node features
                "class": cls,
                "label": int(self.config.class_mapping[cls]),
                "demographics": demo_vec,
                "source_path": str(fpath),
                "dfc_shape": tuple(dfc.shape),
            }

        self.logger.info(f"[fMRI] Valid subjects with labels: {len(subjects)}")
        return subjects


class ConnectivityGraphDataset(Dataset):
    def __init__(
        self,
        subjects: Dict[str, Dict],
        use_node_features: bool = True,
        use_demographics: bool = False,
        demographic_dim: int = 0,
    ):
        self.valid_data = []
        self.labels = []
        self.use_node_features = bool(use_node_features)
        self.use_demographics = bool(use_demographics and demographic_dim > 0)
        self.demographic_dim = int(demographic_dim)

        for subject_id, subject_data in subjects.items():
            demo_vec = None
            if self.use_demographics:
                raw = subject_data.get("demographics", None)
                if raw is None:
                    demo_vec = np.full((self.demographic_dim,), np.nan, dtype=np.float32)
                else:
                    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
                    if raw.shape[0] != self.demographic_dim:
                        tmp = np.full((self.demographic_dim,), np.nan, dtype=np.float32)
                        tmp[: min(self.demographic_dim, raw.shape[0])] = raw[: min(self.demographic_dim, raw.shape[0])]
                        demo_vec = tmp
                    else:
                        demo_vec = raw.astype(np.float32)

            self.valid_data.append({
                "connectivity": subject_data["connectivity"],
                "node_features": subject_data.get("node_features", None),
                "demographics": demo_vec,
                "subject_id": subject_id,
                "class": subject_data["class"],
                "label": int(subject_data["label"]),
            })
            self.labels.append(int(subject_data["label"]))

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        data = self.valid_data[idx]
        connectivity = torch.FloatTensor(data["connectivity"])
        label = torch.LongTensor([data["label"]])[0]

        node_features = None
        if self.use_node_features and data["node_features"] is not None:
            node_features = torch.FloatTensor(data["node_features"])

        demo = None
        if self.use_demographics and data["demographics"] is not None:
            demo = torch.FloatTensor(data["demographics"])

        return connectivity, node_features, demo, label, data["subject_id"]


def collate_connectivity_batch(batch):
    """Collate: (connectivity, node_features, demographics, label, subject_id)."""
    if not batch:
        return torch.tensor([]), None, None, torch.tensor([]), []

    connectivities, node_features_list, demo_list, labels, subject_ids = [], [], [], [], []

    has_node_features = (batch[0][1] is not None)
    has_demo = (batch[0][2] is not None)

    for connectivity, node_features, demo, label, subject_id in batch:
        connectivities.append(connectivity)

        if has_node_features and node_features is not None:
            node_features_list.append(node_features)

        if has_demo and demo is not None:
            demo_list.append(demo)

        labels.append(label)
        subject_ids.append(subject_id)

    connectivities = torch.stack(connectivities)
    labels = torch.stack(labels)

    node_features_batch = None
    if has_node_features and len(node_features_list) > 0:
        node_features_batch = torch.stack(node_features_list)

    demo_batch = None
    if has_demo and len(demo_list) > 0:
        demo_batch = torch.stack(demo_list)

    return connectivities, node_features_batch, demo_batch, labels, subject_ids


def build_demographics_matrix(
    subjects: Dict[str, Dict],
    subject_ids: List[str],
    feature_dim: int,
) -> np.ndarray:
    """Stack demographics vectors in the same order as subject_ids (NaN if missing)."""
    if feature_dim <= 0:
        return np.zeros((len(subject_ids), 0), dtype=np.float32)

    rows = []
    for sid in subject_ids:
        vec = None
        if sid in subjects:
            vec = subjects[sid].get("demographics", None)
        if vec is None:
            vec = np.full((feature_dim,), np.nan, dtype=np.float32)
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.shape[0] != feature_dim:
            # adapt by trunc/pad with nan
            out = np.full((feature_dim,), np.nan, dtype=np.float32)
            out[: min(feature_dim, vec.shape[0])] = vec[: min(feature_dim, vec.shape[0])]
            vec = out
        rows.append(vec)
    return np.stack(rows, axis=0).astype(np.float32)


def impute_and_scale_demographics(
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Impute NaNs using train medians then StandardScale using train statistics."""
    if X_train.size == 0:
        return X_train, X_val

    Xtr = np.asarray(X_train, dtype=np.float32).copy()
    Xva = np.asarray(X_val, dtype=np.float32).copy()

    # median imputation (train only)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)

    # fill NaNs
    inds_tr = np.where(~np.isfinite(Xtr))
    if inds_tr[0].size > 0:
        Xtr[inds_tr] = np.take(med, inds_tr[1])

    inds_va = np.where(~np.isfinite(Xva))
    if inds_va[0].size > 0:
        Xva[inds_va] = np.take(med, inds_va[1])

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr).astype(np.float32)
    Xva_s = scaler.transform(Xva).astype(np.float32)

    return Xtr_s, Xva_s
