"""
Model explanation using SHAP for attrition prediction.
"""

import shap
import pandas as pd
from typing import Tuple, Dict, Any
import logging

from data.generate import load_data_from_parquet
from model.features import (
    get_model_feature_names,
    load_category_vocabularies,
    load_feature_columns,
    prepare_features,
)


logger = logging.getLogger(__name__)
_EXPLAINER_CACHE = {}


def get_background_data(
    category_vocabularies: dict,
    training_columns: list[str] | None,
) -> pd.DataFrame:
    """Load a representative encoded background sample for SHAP baselines."""
    background_raw = load_data_from_parquet()
    background_encoded, _ = prepare_features(
        background_raw,
        label_encoders=category_vocabularies,
        training_columns=training_columns,
    )
    return background_encoded.head(100)


def get_explainer(model, background: pd.DataFrame):
    """Cache SHAP explainers by model object to avoid rebuilding in Streamlit."""
    cache_key = (id(model), tuple(background.columns))
    if cache_key not in _EXPLAINER_CACHE:
        _EXPLAINER_CACHE[cache_key] = shap.TreeExplainer(
            model,
            data=background,
            feature_perturbation="interventional",
        )
    return _EXPLAINER_CACHE[cache_key]


def explain_prediction(model, X: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Generate SHAP explanations for model predictions.

    Args:
        model: Trained model (XGBoost classifier).
        X: Feature matrix for which to generate explanations.
           Note: This should be the raw data with string categorical columns.
                 The function will encode categorical columns to match the model's training data.

    Returns:
        Tuple of (explanation_dict, shap_values_df, X_encoded):
        - explanation_dict: Dictionary containing expected value and feature contributions.
        - shap_values_df: DataFrame of SHAP values per feature per sample.
        - X_encoded: DataFrame with encoded categorical columns (used for model prediction).
    """
    category_vocabularies = load_category_vocabularies()
    training_columns = load_feature_columns() or get_model_feature_names(model)
    X_encoded, _ = prepare_features(
        X,
        label_encoders=category_vocabularies,
        training_columns=training_columns,
    )

    # Create SHAP explainer for tree-based models (XGBoost)
    background = get_background_data(category_vocabularies, training_columns)
    explainer = get_explainer(model, background)

    # Calculate SHAP values
    try:
        shap_values = explainer.shap_values(X_encoded)
    except Exception:
        logger.exception("SHAP explanation failed")
        raise

    # For binary classification, shap_values is a list of two arrays [class_0, class_1]
    # We want the SHAP values for the positive class (attrition risk)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class

    # Import numpy for array handling
    import numpy as np

    # Convert to numpy array for consistent handling
    shap_values = np.array(shap_values)

    # Binary classifier with class dimension
    if shap_values.ndim == 3:
        # shape: (samples, features, classes)
        shap_values = shap_values[:, :, 1]

    # Single sample edge case
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    # Create DataFrame with SHAP values (using original column names for clarity)
    shap_values_df = pd.DataFrame(
        shap_values,
        columns=X_encoded.columns,
        index=X.index
    )

    # Calculate expected value (base value)
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]  # Positive class

    # Create explanation dictionary
    explanation_dict = {
        'expected_value': expected_value,
        'feature_names': list(X_encoded.columns),
        'shap_values': shap_values  # This is the array for the positive class
    }

    return explanation_dict, shap_values_df, X_encoded
