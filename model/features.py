"""
Shared feature preparation for training, prediction, and explanations.
"""

from __future__ import annotations

import os
import warnings
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.preprocessing import LabelEncoder

from model.artifacts import (
    LEGACY_ENCODERS_PATH,
    LEGACY_FEATURE_COLUMNS_PATH,
    artifact_path,
    atomic_write_json,
    read_active_version,
    read_json,
)


EXCLUDE_COLUMNS = {
    "employee_id",
    "manager_id",
    "first_name",
    "last_name",
    "hire_date",
    "last_promotion_date",
    "attrition_risk",
    "exit_note",
    "survey_blurb",
    "name",
    "risk_score",
    "risk_badge",
}

CATEGORICAL_COLUMNS = ["department", "job_title", "level"]
VOCABULARIES_PATH = str(artifact_path("vocabularies.json"))
FEATURE_SCHEMA_PATH = str(artifact_path("schema.json"))
ENCODERS_PATH = LEGACY_ENCODERS_PATH
FEATURE_COLUMNS_PATH = LEGACY_FEATURE_COLUMNS_PATH

logger = logging.getLogger(__name__)


def _atomic_joblib_dump(obj, path: str) -> None:
    """Write a joblib artifact via a temp file before replacing the target."""
    target = Path(path)
    if target.parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with target.open("r+b"):
            pass
    else:
        with target.open("xb"):
            pass
        target.unlink()

    with tempfile.NamedTemporaryFile(
        suffix=target.suffix,
        dir=target.parent or Path("."),
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        joblib.dump(obj, tmp_path)
        tmp_path.replace(target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    """Return model feature columns from a raw employee dataframe."""
    return [col for col in data.columns if col not in EXCLUDE_COLUMNS]


def _normalize_vocabularies(vocabularies: Dict[str, Any] | None) -> Dict[str, list[str]]:
    """Normalize JSON vocabularies or legacy LabelEncoders into lists."""
    if not vocabularies:
        return {}

    normalized = {}
    for col, vocab in vocabularies.items():
        if hasattr(vocab, "classes_"):
            normalized[col] = [str(value) for value in vocab.classes_]
        else:
            normalized[col] = [str(value) for value in vocab]
    return normalized


def prepare_features(
    data: pd.DataFrame,
    label_encoders: Dict[str, Any] | None = None,
    training_columns: list[str] | None = None,
    fit: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, list[str]]]:
    """
    Convert raw employee data into the numeric feature matrix used by the model.
    """
    X = data[get_feature_columns(data)].copy()
    vocabularies = _normalize_vocabularies(label_encoders)

    for col in CATEGORICAL_COLUMNS:
        if col not in X.columns:
            continue

        values = X[col].astype(str)
        if fit or col not in vocabularies:
            vocabulary = sorted(values.dropna().unique().tolist())
            vocabularies[col] = vocabulary
        else:
            vocabulary = vocabularies[col]
            unknown_values = sorted(set(values) - set(vocabulary))
            if unknown_values:
                logger.warning(
                    "Unknown categorical values in %s mapped to -1: %s",
                    col,
                    unknown_values,
                )
        known_mapping = {cls: i for i, cls in enumerate(vocabularies[col])}
        X[col] = values.map(lambda value: known_mapping.get(value, -1))

    for col in X.columns:
        if col not in CATEGORICAL_COLUMNS:
            X[col] = pd.to_numeric(X[col], errors="raise")

    if training_columns is not None:
        missing = [col for col in training_columns if col not in X.columns]
        extra = [col for col in X.columns if col not in training_columns]
        if missing or extra:
            logger.warning(
                "Feature schema mismatch. Missing=%s Extra=%s",
                missing,
                extra,
            )
        X = X.reindex(columns=training_columns, fill_value=0)

    return X, vocabularies


def load_category_vocabularies(path: str = VOCABULARIES_PATH) -> Dict[str, list[str]]:
    """Load JSON category vocabularies, falling back to legacy encoders."""
    active_path = artifact_path("vocabularies.json", read_active_version())
    path_to_load = Path(path)
    if active_path.exists():
        path_to_load = active_path

    if path_to_load.exists():
        return _normalize_vocabularies(read_json(path_to_load))

    legacy_encoders = load_label_encoders()
    if legacy_encoders:
        logger.warning(
            "Using legacy pickle label encoders; retrain to write JSON vocabularies."
        )
    return _normalize_vocabularies(legacy_encoders)


def save_category_vocabularies(
    vocabularies: Dict[str, list[str]],
    path: str = VOCABULARIES_PATH,
) -> None:
    """Persist category vocabularies as JSON."""
    atomic_write_json(path, _normalize_vocabularies(vocabularies))


def load_label_encoders(path: str = ENCODERS_PATH) -> Dict[str, LabelEncoder]:
    """Load legacy sklearn label encoders, returning an empty mapping when absent."""
    if not os.path.exists(path):
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        return joblib.load(path)


def save_label_encoders(
    label_encoders: Dict[str, LabelEncoder],
    path: str = ENCODERS_PATH,
) -> None:
    """Persist label encoders used during training."""
    _atomic_joblib_dump(label_encoders, path)


def load_feature_columns(path: str = FEATURE_SCHEMA_PATH) -> list[str] | None:
    """Load the training feature column order, if it has been persisted."""
    active_path = artifact_path("schema.json", read_active_version())
    path_to_load = Path(path)
    if active_path.exists():
        path_to_load = active_path

    if path_to_load.exists():
        schema = read_json(path_to_load)
        return schema.get("feature_columns")

    if os.path.exists(FEATURE_COLUMNS_PATH):
        return joblib.load(FEATURE_COLUMNS_PATH)
    return None


def save_feature_columns(
    feature_columns: list[str],
    path: str = FEATURE_SCHEMA_PATH,
) -> None:
    """Persist the exact training feature schema as JSON."""
    schema = {
        "feature_columns": feature_columns,
        "categorical_columns": CATEGORICAL_COLUMNS,
    }
    atomic_write_json(path, schema)


def get_model_feature_names(model) -> list[str] | None:
    """Return feature names embedded in an XGBoost model when available."""
    try:
        return model.get_booster().feature_names
    except AttributeError:
        return None
