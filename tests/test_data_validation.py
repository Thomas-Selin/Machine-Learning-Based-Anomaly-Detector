"""Tests for data validation module."""

import numpy as np
import pandas as pd
import pytest

from ml_monitoring_service.data_validation import (
    DataValidationError,
    validate_combined_dataset,
    validate_dataframe_schema,
    validate_metrics_data,
    validate_model_input,
)


def test_validate_dataframe_schema_valid():
    """Test schema validation with valid DataFrame."""
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4], "col3": [5, 6]})
    required_columns = ["col1", "col2"]

    # Should not raise exception
    validate_dataframe_schema(df, required_columns)


def test_validate_dataframe_schema_missing_columns():
    """Test schema validation with missing columns."""
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    required_columns = ["col1", "col2", "col3"]

    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_dataframe_schema(df, required_columns)


def test_validate_metrics_data_empty():
    """Test metrics validation with empty DataFrame."""
    df = pd.DataFrame()

    with pytest.raises(DataValidationError, match="Empty DataFrame"):
        validate_metrics_data(df, "test_service")


def test_validate_metrics_data_missing_required_columns():
    """Test metrics validation with missing required columns."""
    df = pd.DataFrame({"value": [1, 2, 3]})

    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_metrics_data(df, "test_service")


def test_validate_metrics_data_invalid_timestamp():
    """Test metrics validation with invalid timestamp format."""
    df = pd.DataFrame(
        {
            "timestamp": ["invalid", "timestamp"],
            "service": ["svc1", "svc2"],
            "value": [1, 2],
        }
    )

    with pytest.raises(DataValidationError, match="Invalid timestamp format"):
        validate_metrics_data(df, "test_service")


def test_validate_metrics_data_valid():
    """Test metrics validation with valid data."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=3, freq="h"),
            "service": ["svc1", "svc1", "svc1"],
            "cpu_usage": [0.5, 0.6, 0.7],
            "memory_usage": [1000, 1100, 1200],
        }
    )

    result = validate_metrics_data(df, "svc1")
    assert not result.empty
    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])


def test_validate_metrics_data_infinite_values():
    """Test metrics validation handles infinite values."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=3, freq="h"),
            "service": ["svc1", "svc1", "svc1"],
            "value": [1.0, float("inf"), 3.0],
        }
    )

    result = validate_metrics_data(df, "svc1")
    # Infinite values should be replaced with NaN
    assert result["value"].isna().sum() == 1


def test_validate_combined_dataset_valid():
    """Test combined dataset validation with valid data."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=6, freq="h"),
            "service": ["svc1", "svc1", "svc2", "svc2", "svc3", "svc3"],
            "timestamp_nanoseconds": [1, 2, 1, 2, 1, 2],
            "value": [1, 2, 3, 4, 5, 6],
        }
    )

    expected_services = ["svc1", "svc2", "svc3"]
    # Should not raise exception
    validate_combined_dataset(df, expected_services)


def test_validate_combined_dataset_missing_services():
    """Test combined dataset validation with missing services."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=2, freq="h"),
            "service": ["svc1", "svc1"],
            "timestamp_nanoseconds": [1, 2],
            "value": [1, 2],
        }
    )

    expected_services = ["svc1", "svc2", "svc3"]

    with pytest.raises(DataValidationError, match="Missing data for services"):
        validate_combined_dataset(df, expected_services)


def test_validate_combined_dataset_missing_columns():
    """Test combined dataset validation with missing required columns."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=2, freq="h"),
            "service": ["svc1", "svc1"],
            "value": [1, 2],
        }
    )

    expected_services = ["svc1"]

    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_combined_dataset(df, expected_services)


def test_validate_model_input_valid():
    """Test model input validation with valid data."""
    data = np.random.rand(100, 5, 10)  # 100 timesteps, 5 services, 10 features
    expected_shape = (100, 5, 10)
    service_names = ["svc1", "svc2", "svc3", "svc4", "svc5"]

    # Should not raise exception
    validate_model_input(data, expected_shape, service_names)


def test_validate_model_input_wrong_type():
    """Test model input validation with wrong data type."""
    data = [[1, 2], [3, 4]]  # List instead of numpy array
    expected_shape = (2, 2, 1)
    service_names = ["svc1", "svc2"]

    with pytest.raises(DataValidationError, match="must be numpy array"):
        validate_model_input(data, expected_shape, service_names)


def test_validate_model_input_wrong_shape():
    """Test model input validation with wrong shape."""
    data = np.random.rand(100, 5, 10)
    expected_shape = (100, 5, 8)  # Wrong feature dimension
    service_names = ["svc1", "svc2", "svc3", "svc4", "svc5"]

    with pytest.raises(DataValidationError, match="doesn't match expected"):
        validate_model_input(data, expected_shape, service_names)


def test_validate_model_input_nan_values():
    """Test model input validation with NaN values."""
    data = np.random.rand(10, 5, 10)
    data[5, 2, 3] = np.nan  # Insert NaN
    expected_shape = (10, 5, 10)
    service_names = ["svc1", "svc2", "svc3", "svc4", "svc5"]

    with pytest.raises(DataValidationError, match="contains .* NaN values"):
        validate_model_input(data, expected_shape, service_names)


def test_validate_model_input_inf_values():
    """Test model input validation with infinite values."""
    data = np.random.rand(10, 5, 10)
    data[5, 2, 3] = np.inf  # Insert Inf
    expected_shape = (10, 5, 10)
    service_names = ["svc1", "svc2", "svc3", "svc4", "svc5"]

    with pytest.raises(DataValidationError, match="contains infinite values"):
        validate_model_input(data, expected_shape, service_names)
