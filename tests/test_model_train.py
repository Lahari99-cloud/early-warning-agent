"""
Tests for model training module.
"""

import pytest
import pandas as pd
import numpy as np
from model.train import train_model, evaluate_model


def test_train_model():
    """Test that model training works."""
    # Create simple test data
    X = pd.DataFrame(np.random.rand(100, 5))
    y = pd.Series(np.random.randint(0, 2, 100))

    # Test that model training returns an XGBClassifier
    model = train_model(X, y)
    assert model is not None
    # Check it's an XGBoost classifier
    from xgboost import XGBClassifier
    assert isinstance(model, XGBClassifier)


def test_evaluate_model():
    """Test that model evaluation works."""
    # Create simple test data
    X = pd.DataFrame(np.random.rand(100, 5))
    y = pd.Series(np.random.randint(0, 2, 100))

    # Train a model to evaluate
    model = train_model(X, y)

    # Evaluate the model
    metrics = evaluate_model(model, X, y)

    # Check that we get expected metrics
    assert 'auc' in metrics
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics

    # Check that metrics are reasonable values
    assert 0 <= metrics['auc'] <= 1
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1'] <= 1