"""
Model explanation using SHAP for attrition prediction.
"""

import shap
import pandas as pd
from typing import Tuple, Dict, Any
import os
import joblib


def explain_prediction(model, X: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Generate SHAP explanations for model predictions.

    Args:
        model: Trained model (XGBoost classifier).
        X: Feature matrix for which to generate explanations.
           Note: This should be the raw data with string categorical columns.
                 The function will encode categorical columns to match the model's training data.

    Returns:
        Tuple of (explanation_dict, shap_values_df):
        - explanation_dict: Dictionary containing expected value and feature contributions.
        - shap_values_df: DataFrame of SHAP values per feature per sample.
    """
    # Load label encoders if they exist
    encoders_path = 'model/label_encoders.pkl'
    if os.path.exists(encoders_path):
        label_encoders = joblib.load(encoders_path)
    else:
        label_encoders = {}

    # Make a copy to avoid modifying the original
    X_encoded = X.copy()

    # Encode categorical columns if we have encoders for them
    categorical_cols = ['department', 'job_title', 'level']
    for col in categorical_cols:
        if col in label_encoders and col in X_encoded.columns:
            # Transform using the saved label encoder
            # Handle unseen labels by mapping to -1 or a default? We'll use the encoder's transform
            # and if there's an unseen label, we'll catch the error and map to -1.
            try:
                X_encoded[col] = label_encoders[col].transform(X_encoded[col].astype(str))
            except ValueError:
                # If there are unseen labels, map them to -1
                # We'll create a mapping from known classes to integers, and then map unknowns to -1
                known_mapping = {cls: i for i, cls in enumerate(label_encoders[col].classes_)}
                X_encoded[col] = X_encoded[col].apply(lambda x: known_mapping.get(x, -1))

    # Create SHAP explainer for tree-based models (XGBoost)
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_encoded)

    # For binary classification, shap_values is a list of two arrays [class_0, class_1]
    # We want the SHAP values for the positive class (attrition risk)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class

    # Create DataFrame with SHAP values (using original column names for clarity)
    shap_values_df = pd.DataFrame(
        shap_values,
        columns=X.columns,  # Use original column names
        index=X.index
    )

    # Calculate expected value (base value)
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]  # Positive class

    # Create explanation dictionary
    explanation_dict = {
        'expected_value': expected_value,
        'feature_names': list(X.columns),  # Original feature names
        'shap_values': shap_values  # This is the array for the positive class
    }

    return explanation_dict, shap_values_df