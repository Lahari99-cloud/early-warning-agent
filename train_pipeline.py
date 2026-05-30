#!/usr/bin/env python3
"""
Training pipeline for attrition early-warning agent.
Generate data, train model, and save artifacts.
"""
import sys
import os
import logging
import random
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.generate import generate_synthetic_data, save_data_to_parquet, load_data_from_parquet
from model.train import evaluate_model, save_artifact_metadata, save_model, train_model
from model.features import prepare_features, save_category_vocabularies, save_feature_columns
from model.artifacts import artifact_path
from config import ACTIVE_MODEL_VERSION, SEED
import pandas as pd
import numpy as np
from xgboost.core import XGBoostError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("=== Attrition Early-Warning Agent Training Pipeline ===")

    # Step 1: Generate synthetic data
    print("\n1. Generating synthetic employee data (1500 records)...")
    df = generate_synthetic_data(1500)
    print(f"   Generated data shape: {df.shape}")

    # Step 2: Save to parquet
    print("\n2. Saving data to parquet...")
    os.makedirs('data/cache', exist_ok=True)
    cache_path = 'data/cache/synthetic_employee_data.parquet'
    cache_updated = False
    try:
        save_data_to_parquet(df, cache_path)
        cache_updated = True
        print(f"   Data saved to {cache_path}")
    except PermissionError as exc:
        logger.warning("Could not update %s: %s", cache_path, exc)
        print(f"   Warning: could not update {cache_path}: {exc}")
        print("   Continuing with freshly generated in-memory data.")

    # Step 3: Load data for training
    print("\n3. Loading data for training...")
    if cache_updated:
        data = load_data_from_parquet(cache_path)
    else:
        data = df
    print(f"   Loaded data shape: {data.shape}")

    # Step 4: Prepare features and target
    print("\n4. Preparing features and target...")
    X_encoded, category_vocabularies = prepare_features(data, fit=True)
    y = data['attrition_risk'].copy()

    print(f"   Features ({len(X_encoded.columns)}): {list(X_encoded.columns)}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")

    print(f'   Encoded features dtypes:')
    print(f'   {X_encoded.dtypes.to_string()}')

    # Step 5: Train model
    print("\n5. Training XGBoost model...")
    model = train_model(X_encoded, y)

    # Step 6: Evaluate model
    print("\n6. Evaluating model...")
    metrics = evaluate_model(model, X_encoded, y)
    print("   Metrics:")
    for metric_name, value in metrics.items():
        print(f"     {metric_name.upper()}: {value:.4f}")

    # Step 7: Save model
    print("\n7. Saving model...")
    os.makedirs('model', exist_ok=True)
    try:
        model_path = artifact_path("model.json", ACTIVE_MODEL_VERSION)
        save_model(model, str(model_path))
        print(f"   Model saved to {model_path}")
    except (PermissionError, XGBoostError) as exc:
        logger.warning("Could not update model artifact: %s", exc)
        print(f"   Warning: could not update model artifact: {exc}")
        print("   The trained in-memory model is still returned by this run.")

    try:
        vocab_path = artifact_path("vocabularies.json", ACTIVE_MODEL_VERSION)
        save_category_vocabularies(category_vocabularies, str(vocab_path))
        print(f"   Category vocabularies saved to {vocab_path}")
    except PermissionError as exc:
        logger.warning("Could not update category vocabularies: %s", exc)
        print(f"   Warning: could not update category vocabularies: {exc}")

    try:
        feature_columns = list(X_encoded.columns)
        schema_path = artifact_path("schema.json", ACTIVE_MODEL_VERSION)
        metadata_path = artifact_path("metadata.json", ACTIVE_MODEL_VERSION)
        save_feature_columns(feature_columns, str(schema_path))
        save_artifact_metadata(feature_columns, str(metadata_path))
        print(f"   Feature schema saved to {schema_path}")
        print(f"   Artifact metadata saved to {metadata_path}")
    except PermissionError as exc:
        logger.warning("Could not update feature schema metadata: %s", exc)
        print(f"   Warning: could not update feature schema metadata: {exc}")

    print("\n=== Training Pipeline Complete ===")
    return model, metrics

if __name__ == '__main__':
    main()
