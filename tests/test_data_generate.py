"""
Tests for data generation module.
"""

import pytest
import pandas as pd
import numpy as np
from data.generate import generate_synthetic_data, save_data_to_parquet, load_data_from_parquet
import os
import tempfile


def test_generate_synthetic_data():
    """Test that synthetic data generation works correctly."""
    df = generate_synthetic_data(100)

    # Check shape
    assert df.shape[0] == 100
    expected_columns = [
        'employee_id', 'first_name', 'last_name', 'department', 'job_title',
        'hire_date', 'salary', 'engagement_score', 'last_promotion_date',
        'years_at_company', 'manager_id', 'attrition_risk', 'level',
        'comp_vs_band_pct', 'months_since_promo', 'perf_trend', 'pto_usage',
        'one_on_one_freq', 'internal_moves', 'exit_note', 'survey_blurb'
    ]
    assert list(df.columns) == expected_columns

    # Check data types
    assert df['employee_id'].dtype in ['int64', 'int32']
    assert df['first_name'].dtype == 'object'
    assert df['last_name'].dtype == 'object'
    assert df['department'].dtype == 'object'
    assert df['job_title'].dtype == 'object'
    assert pd.api.types.is_datetime64_any_dtype(df['hire_date'])
    assert df['salary'].dtype in ['float64', 'float32']
    assert df['engagement_score'].dtype in ['float64', 'float32']
    assert pd.api.types.is_datetime64_any_dtype(df['last_promotion_date'])
    assert df['years_at_company'].dtype in ['float64', 'float32']
    assert df['manager_id'].dtype in ['int64', 'int32']
    assert df['attrition_risk'].dtype in ['int64', 'int32']
    assert set(df['attrition_risk'].unique()).issubset({0, 1})
    assert df['level'].dtype == 'object'
    assert df['comp_vs_band_pct'].dtype in ['float64', 'float32']
    assert df['months_since_promo'].dtype in ['int64', 'int32']
    assert df['perf_trend'].dtype in ['float64', 'float32']
    assert df['pto_usage'].dtype in ['float64', 'float32']
    assert df['one_on_one_freq'].dtype in ['float64', 'float32']
    assert df['internal_moves'].dtype in ['int64', 'int32']
    assert df['exit_note'].dtype == 'object'
    assert df['survey_blurb'].dtype == 'object'

    # Check value ranges
    assert (df['years_at_company'] >= 0).all() and (df['years_at_company'] <= 5).all()
    assert (df['salary'] > 0).all()
    assert (df['engagement_score'] >= 0).all() and (df['engagement_score'] <= 100).all()
    assert (df['comp_vs_band_pct'] >= 70).all() and (df['comp_vs_band_pct'] <= 130).all()
    assert (df['months_since_promo'] >= 0).all()
    assert (df['perf_trend'] >= -1).all() and (df['perf_trend'] <= 1).all()
    assert (df['pto_usage'] >= 0).all() and (df['pto_usage'] <= 100).all()
    assert (df['one_on_one_freq'] >= 0).all() and (df['one_on_one_freq'] <= 4).all()
    assert df['internal_moves'].isin([0, 1, 2, 3]).all()

    # Check attrition rate is approximately 12% (allowing for randomness)
    attrition_rate = df['attrition_risk'].mean()
    assert 0.08 <= attrition_rate <= 0.16  # Allow some variance from 12%

    # Check that no employee is their own manager
    assert (df['employee_id'] != df['manager_id']).all(), "Found employee who is their own manager"


def test_save_and_load_parquet():
    """Test that saving and loading parquet works correctly."""
    # Generate small dataset
    df_original = generate_synthetic_data(50)

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        temp_path = tmp.name

    try:
        # Save to parquet
        save_data_to_parquet(df_original, temp_path)

        # Load from parquet
        df_loaded = load_data_from_parquet(temp_path)

        # Check that dataframes are equal
        pd.testing.assert_frame_equal(df_original, df_loaded)
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_generate_synthetic_data_default_size():
    """Test that default size is 1500 employees as specified."""
    df = generate_synthetic_data()
    assert df.shape[0] == 1500


def test_signal_correlation():
    """Test that attrition label correlates with risk factors as intended."""
    # Generate a larger dataset to reduce randomness
    df = generate_synthetic_data(1000)

    # Calculate the risk components as used in the generation
    # Note: We replicate the logic from generate_synthetic_data to compute risk scores
    comp_risk = (130 - df['comp_vs_band_pct']) / (130 - 70)  # 0 when comp=130, 1 when comp=70
    perf_risk = (df['perf_trend'] + 1) / 2  # 0 when perf_trend=-1, 1 when perf_trend=+1
    max_months = df['years_at_company'] * 12
    time_risk = df['months_since_promo'] / np.maximum(max_months, 1)  # proportion of tenure since promo

    # Combined risk score (same weights as in generation)
    risk_score = (0.4 * comp_risk + 0.3 * perf_risk + 0.3 * time_risk)

    # Compare attrition risk group vs non-attrition risk group
    attrition_group = df[df['attrition_risk'] == 1]
    non_attrition_group = df[df['attrition_risk'] == 0]

    # With probabilistic labels, individual drivers may vary by sample, but the
    # combined risk signal should remain higher for the attrition group.
    attrition_risk_score = risk_score[df['attrition_risk'] == 1]
    non_attrition_risk_score = risk_score[df['attrition_risk'] == 0]
    assert attrition_risk_score.mean() > non_attrition_risk_score.mean(), \
        "Attrition group should have higher combined risk score on average"


def test_reproducibility():
    """Test that seeding produces reproducible results."""
    # Generate two datasets with same seed
    df1 = generate_synthetic_data(100)
    df2 = generate_synthetic_data(100)

    # They should be identical
    pd.testing.assert_frame_equal(df1, df2)


def test_save_creates_directory():
    """Test that save_data_to_parquet creates directory if it doesn't exist."""
    df = generate_synthetic_data(10)

    # Use a path in a non-existent directory
    test_dir = "test_nonexistent_dir"
    test_path = os.path.join(test_dir, "test.parquet")

    try:
        # This should create the directory
        save_data_to_parquet(df, test_path)

        # Check that directory and file were created
        assert os.path.exists(test_dir)
        assert os.path.exists(test_path)

        # Verify we can load the data back
        df_loaded = load_data_from_parquet(test_path)
        pd.testing.assert_frame_equal(df, df_loaded)
    finally:
        # Clean up
        if os.path.exists(test_path):
            os.unlink(test_path)
        if os.path.exists(test_dir):
            os.rmdir(test_dir)
