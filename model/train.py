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
from pathlib import Path

# Add the project root to the path so we can import config
sys.path.append(str(Path(__file__).parent.parent))
from config import SEED


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
        eval_metric='logloss'
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


def save_model(model: xgb.XGBClassifier, path: str = "model/xgboost_model.json") -> None:
    """
    Save the trained model to disk.

    Args:
        model: Trained XGBoost classifier.
        path: File path to save the model.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)


def load_model(path: str = "model/xgboost_model.json") -> xgb.XGBClassifier:
    """
    Load a trained model from disk.

    Args:
        path: File path to the saved model.

    Returns:
        Trained XGBoost classifier.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model