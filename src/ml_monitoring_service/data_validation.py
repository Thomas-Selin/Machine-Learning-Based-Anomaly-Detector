"""Data validation module for ensuring data quality and schema compliance.

This module provides validation functions to check incoming data from Prometheus and Splunk
before it's used for training or inference.
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails."""

    pass


def validate_dataframe_schema(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Validate that DataFrame has all required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present

    Raises:
        DataValidationError: If any required columns are missing
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise DataValidationError(
            f"DataFrame missing required columns: {missing_columns}"
        )


def validate_metrics_data(df: pd.DataFrame, service_name: str) -> pd.DataFrame:
    """Validate and clean metrics data from Prometheus/Splunk.

    Args:
        df: Raw metrics DataFrame
        service_name: Name of the service for logging context

    Returns:
        Validated and cleaned DataFrame

    Raises:
        DataValidationError: If data is invalid or empty
    """
    if df.empty:
        raise DataValidationError(
            f"Empty DataFrame received for service {service_name}"
        )

    # Validate required columns
    required_columns = ["timestamp", "service"]
    validate_dataframe_schema(df, required_columns)

    # Check for valid timestamps
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except (ValueError, TypeError) as e:
            raise DataValidationError(
                f"Invalid timestamp format for service {service_name}: {e}"
            ) from e

    # Check for duplicate timestamps
    duplicate_timestamps = df[
        df.duplicated(subset=["timestamp", "service"], keep=False)
    ]
    if not duplicate_timestamps.empty:
        logger.warning(
            f"Found {len(duplicate_timestamps)} duplicate timestamp entries for service {service_name}"
        )

    # Validate numeric columns (metrics should be numeric)
    numeric_columns = df.select_dtypes(include=["number"]).columns
    if len(numeric_columns) == 0:
        logger.warning(f"No numeric columns found for service {service_name}")

    # Check for infinite values
    if (df[numeric_columns] == float("inf")).any().any():
        logger.warning(
            f"Infinite values detected for service {service_name}, replacing with NaN"
        )
        df = df.replace([float("inf"), float("-inf")], float("nan"))

    # Check for negative values (some metrics shouldn't be negative)
    negative_counts = (df[numeric_columns] < 0).sum()
    if negative_counts.any():
        logger.info(
            f"Negative values found for service {service_name}: {negative_counts.to_dict()}"
        )

    return df


def validate_combined_dataset(df: pd.DataFrame, expected_services: list[str]) -> None:
    """Validate combined dataset before model training/inference.

    Args:
        df: Combined dataset DataFrame
        expected_services: List of expected service names

    Raises:
        DataValidationError: If validation fails
    """
    # Validate required columns
    required_columns = ["timestamp", "service", "timestamp_nanoseconds"]
    validate_dataframe_schema(df, required_columns)

    # Check all expected services are present
    actual_services = set(df["service"].unique())
    missing_services = set(expected_services) - actual_services
    if missing_services:
        raise DataValidationError(f"Missing data for services: {missing_services}")

    extra_services = actual_services - set(expected_services)
    if extra_services:
        logger.warning(f"Unexpected services in dataset: {extra_services}")

    # Validate data coverage
    for service in expected_services:
        service_data = df[df["service"] == service]
        if len(service_data) == 0:
            raise DataValidationError(f"No data found for service: {service}")

        # Check for sufficient data points
        if len(service_data) < 10:
            logger.warning(
                f"Service {service} has only {len(service_data)} data points (minimum recommended: 10)"
            )


def validate_model_input(
    data: Any, expected_shape: tuple[int, int, int], service_names: list[str]
) -> None:
    """Validate model input data shape and content.

    Args:
        data: Input data array
        expected_shape: Expected shape tuple (timesteps, services, features)
        service_names: List of service names for context

    Raises:
        DataValidationError: If validation fails
    """
    import numpy as np

    if not isinstance(data, np.ndarray):
        raise DataValidationError(f"Model input must be numpy array, got {type(data)}")

    if data.shape != expected_shape:
        raise DataValidationError(
            f"Model input shape {data.shape} doesn't match expected {expected_shape}"
        )

    # Check for NaN/Inf
    if np.isnan(data).any():
        nan_count = np.isnan(data).sum()
        raise DataValidationError(
            f"Model input contains {nan_count} NaN values. Data must be cleaned before inference."
        )

    if np.isinf(data).any():
        raise DataValidationError(
            "Model input contains infinite values. Data must be cleaned before inference."
        )

    logger.debug(
        f"Model input validated: shape={data.shape}, services={len(service_names)}"
    )
