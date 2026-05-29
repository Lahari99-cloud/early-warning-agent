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