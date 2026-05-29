"""
Tests for data generation module.
"""

import pytest
from data.generate import generate_synthetic_data, save_data_to_parquet, load_data_from_parquet


def test_generate_synthetic_data():
    """Test that synthetic data generation works."""
    # This test should fail initially since the function is not implemented
    with pytest.raises(NotImplementedError):
        generate_synthetic_data(10)


def test_save_data_to_parquet():
    """Test that saving data to parquet works."""
    # This test should fail initially since the function is not implemented
    with pytest.raises(NotImplementedError):
        save_data_to_parquet(None, "test.parquet")


def test_load_data_from_parquet():
    """Test that loading data from parquet works."""
    # This test should fail initially since the function is not implemented
    with pytest.raises(NotImplementedError):
        load_data_from_parquet("test.parquet")