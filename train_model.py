"""
Script to train and save the attrition prediction model.
"""
import sys
import os
sys.path.append('.')

from data.generate import load_data_from_parquet
from model.train import train_model, evaluate_model, save_model
import pandas as pd
import numpy as np

def main():
    print("Loading data...")
    # Load the cached data
    data = load_data_from_parquet('data/cache/synthetic_employee_data.parquet')
    print(f'Data shape: {data.shape}')

    # Prepare features and target
    # Exclude non-feature columns
    exclude_cols = ['employee_id', 'first_name', 'last_name', 'hire_date',
                   'last_promotion_date', 'exit_note', 'survey_blurb']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    X = data[feature_cols]
    y = data['attrition_risk']

    print(f'Features ({len(feature_cols)}): {list(feature_cols)}')
    print(f'Target distribution: {y.value_counts().to_dict()}')

    # Convert categorical variables to numeric for XGBoost
    # For simplicity, we'll label encode the categorical columns
    from sklearn.preprocessing import LabelEncoder

    X_encoded = X.copy()
    label_encoders = {}

    categorical_cols = ['department', 'job_title', 'level']
    for col in categorical_cols:
        if col in X_encoded.columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            label_encoders[col] = le

    print(f'Encoded features dtypes:')
    print(X_encoded.dtypes)

    # Train model
    print('Training model...')
    model = train_model(X_encoded, y)

    # Evaluate
    print('Evaluating model...')
    metrics = evaluate_model(model, X_encoded, y)
    print('Metrics:', metrics)

    # Save model
    print('Saving model...')
    os.makedirs('model', exist_ok=True)
    save_model(model, 'model/xgboost_model.json')
    print('Model saved successfully to model/xgboost_model.json')

    # Also save label encoders for later use in explanation
    import joblib
    joblib.dump(label_encoders, 'model/label_encoders.pkl')
    print('Label encoders saved to model/label_encoders.pkl')

if __name__ == '__main__':
    main()