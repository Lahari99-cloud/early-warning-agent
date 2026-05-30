"""
Model training for attrition prediction.
"""

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import os
import sys
import json
import logging
import tempfile
from pathlib import Path

# Add the project root to the path so we can import config
sys.path.append(str(Path(__file__).parent.parent))
from config import FEATURE_SCHEMA_VERSION, MODEL_VERSION, SEED
from model.artifacts import (
    LEGACY_METADATA_PATH,
    LEGACY_MODEL_PATH,
    active_version_path,
    artifact_path,
    atomic_write_text,
    atomic_write_json,
    read_active_version,
)


METADATA_PATH = str(artifact_path("metadata.json"))
logger = logging.getLogger(__name__)


def train_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier for attrition prediction.

    Args:
        X: Feature matrix.
        y: Target vector (attrition risk binary or probability).

    Returns:
        Trained XGBoost classifier.
    """
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        seed=SEED,
        eval_metric='logloss',
        n_jobs=1,
    )

    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model


def evaluate_model(model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Evaluate the trained model and return metrics.

    Args:
        model: Trained XGBoost classifier.
        X: Feature matrix.
        y: Target vector.

    Returns:
        Dictionary of evaluation metrics.
    """
    # Make predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    # Calculate metrics
    auc_score = roc_auc_score(y, y_pred_proba)

    # For binary classification, we can also calculate accuracy, precision, recall, f1
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    return {
        'auc': auc_score,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def save_model(model: xgb.XGBClassifier, path: str = str(artifact_path("model.json"))) -> None:
    """
    Save the trained model to disk.

    Args:
        model: Trained XGBoost classifier.
        path: File path to save the model.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    target = Path(path)
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
        model.save_model(str(tmp_path))
        tmp_path.replace(target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def save_artifact_metadata(
    feature_columns: list[str],
    path: str = METADATA_PATH,
) -> None:
    """Persist compatibility metadata for trained artifacts."""
    metadata = {
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": feature_columns,
        "pandas_version": pd.__version__,
        "xgboost_version": xgb.__version__,
    }
    atomic_write_json(path, metadata)
    atomic_write_text(active_version_path(), MODEL_VERSION)


def load_artifact_metadata(path: str = METADATA_PATH) -> dict:
    """Load artifact metadata and validate known compatibility versions."""
    active_path = artifact_path("metadata.json", read_active_version())
    path_to_load = Path(path)
    if active_path.exists():
        path_to_load = active_path
    elif Path(LEGACY_METADATA_PATH).exists():
        path_to_load = Path(LEGACY_METADATA_PATH)

    if not path_to_load.exists():
        logger.warning("Artifact metadata missing; treating model as legacy.")
        return {}

    with path_to_load.open(encoding="utf-8") as f:
        metadata = json.load(f)

    expected = {
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }
    mismatches = {
        key: {"expected": expected[key], "actual": metadata.get(key)}
        for key in expected
        if metadata.get(key) != expected[key]
    }
    if mismatches:
        raise ValueError(f"Artifact metadata version mismatch: {mismatches}")

    return metadata


def load_model(path: str | None = None) -> xgb.XGBClassifier:
    """
    Load a trained model from disk.

    Args:
        path: File path to the saved model.

    Returns:
        Trained XGBoost classifier.
    """
    if path is None:
        active_model_path = artifact_path("model.json", read_active_version())
        path = str(active_model_path if active_model_path.exists() else Path(LEGACY_MODEL_PATH))

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    load_artifact_metadata()
    model = xgb.XGBClassifier()
    model.load_model(path)
    logger.info("Loaded model artifact from %s", path)
    return model
