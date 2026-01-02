import logging
import os
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import ml_monitoring_service.configuration as conf

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails.

    This exception is raised when input data doesn't meet the requirements
    for model training or inference, such as missing columns, incorrect shapes,
    or invalid values (NaN, Inf).
    """


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
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise DataValidationError(
            f"DataFrame missing required columns: {missing_columns}"
        )

    # Check all expected services are present
    actual_services = set(df["service"].unique())
    missing_services = set(expected_services) - actual_services
    if missing_services:
        logger.warning(
            f"Missing data for services: {missing_services}. "
            "This may affect model training/inference quality."
        )

    extra_services = actual_services - set(expected_services)
    if extra_services:
        logger.warning(f"Unexpected services in dataset: {extra_services}")


def validate_model_input(
    data: np.ndarray, expected_shape: tuple[int, int, int], service_names: list[str]
) -> None:
    """Validate model input data shape and content.

    Args:
        data: Input data array
        expected_shape: Expected shape tuple (timesteps, services, features)
        service_names: List of service names for context

    Raises:
        DataValidationError: If validation fails
    """
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


def ensure_timestamp_nanoseconds_ns(df: pd.DataFrame) -> np.ndarray:
    """Ensure `timestamp_nanoseconds` exists and is normalized to int64 nanoseconds.

    Historical datasets in this repo sometimes store `timestamp_nanoseconds` as an ISO datetime
    string (despite the name). Production pipeline should treat it as an integer nanoseconds key.

    This function mutates the input DataFrame in-place.

    Returns:
        Sorted unique timestamp keys as int64 nanoseconds.
    """
    if "timestamp_nanoseconds" not in df.columns:
        raise KeyError("DataFrame must contain 'timestamp_nanoseconds'")

    series = df["timestamp_nanoseconds"]

    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        # Treat numeric values as ns since epoch.
        ns = pd.to_numeric(series, errors="coerce")
    else:
        # Treat as datetime-like (string/datetime).
        dt = pd.to_datetime(series, errors="coerce")
        # pandas Series.view is deprecated; use astype and preserve NaT as missing.
        ns = pd.Series(dt.astype("int64"), index=df.index).where(dt.notna())

    # Fill rows that couldn't be converted using the parsed timestamp column when available.
    if ns.isna().any() and "timestamp" in df.columns:
        fallback_dt = pd.to_datetime(df["timestamp"], errors="coerce")
        ns = ns.where(
            ~ns.isna(),
            other=pd.Series(fallback_dt.astype("int64"), index=df.index).where(
                fallback_dt.notna()
            ),
        )

    if ns.isna().any():
        raise ValueError("Failed to normalize 'timestamp_nanoseconds' for some rows")

    df["timestamp_nanoseconds"] = ns.astype("int64")
    return np.sort(df["timestamp_nanoseconds"].unique())


def get_ordered_timepoints(df: pd.DataFrame) -> np.ndarray:
    """Return ordered unique timepoints used as the model's time axis.

    The pipeline often contains multiple rows per timestamp (one per service).
    The model input is constructed over unique timepoints; this helper ensures
    deterministic ordering and consistent alignment across training/thresholding/inference.

    Args:
        df: DataFrame containing at least 'timestamp_nanoseconds'.

    Returns:
        numpy array of datetime64-compatible timepoints (sorted ascending).
    """
    timestamp_ns = ensure_timestamp_nanoseconds_ns(df)
    # Convert ns keys to datetime64 for time features.
    return pd.to_datetime(timestamp_ns, unit="ns").to_numpy()


def check_for_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Check for NaN values in the DataFrame and fill them with appropriate values

    Args:
        df: DataFrame to check for NaN values

    Returns:
        DataFrame with NaN values filled with 0
    """
    if df.isnull().values.any():
        nan_counts = df.isnull().sum()
        nan_columns = nan_counts[nan_counts > 0]
        logger.warning(f"Data contains NaN values in columns: {nan_columns.to_dict()}")

        # Fill NaN values with 0 for numeric columns
        df = df.fillna(0)
        logger.info("NaN values filled with 0")
    else:
        logger.debug("No NaN values in data")
    return df


def normalize_feature(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """Normalize a feature using Min-Max normalization

    Args:
        df: DataFrame containing the feature
        feature_name: Name of the feature column to normalize

    Returns:
        DataFrame with normalized feature (values between 0 and 1)
    """
    if feature_name not in df.columns:
        logger.warning(f"Feature '{feature_name}' not found in DataFrame")
        return df

    min_value = df[feature_name].min()
    max_value = df[feature_name].max()

    if pd.isna(min_value) or pd.isna(max_value):
        logger.warning(
            f"Feature '{feature_name}' contains only NaN values, setting to 0"
        )
        df[feature_name] = 0
    elif min_value == max_value:
        # Feature has constant value - normalize to 0.5 to indicate "middle" value
        logger.debug(
            f"Feature '{feature_name}' has constant value {min_value}, normalizing to 0.5"
        )
        df[feature_name] = 0.5
    else:
        df[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

    return df


def convert_to_model_input(
    active_set: str,
    df: pd.DataFrame,
    *,
    approach: Literal["grid", "event"] = "grid",
    time_grid_freq: str | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Convert DataFrame to model input format

    Args:
        active_set: Name of the service set
        df: DataFrame with service metrics

    Returns:
        tuple: (data array [time_steps, num_services, num_features], services list, features list)

    Raises:
        DataValidationError: If data validation fails
    """
    original_df = df

    # Validate combined dataset
    services = conf.config.get_services(active_set)

    # Ensure timestamp is in datetime format
    original_df["timestamp"] = pd.to_datetime(
        original_df["timestamp"], format="mixed", errors="coerce"
    )
    if original_df["timestamp"].isna().any():
        raise DataValidationError("Some rows have invalid timestamps")

    # Ensure timestamp_nanoseconds exists (some test/fixtures may omit it)
    if "timestamp_nanoseconds" not in original_df.columns:
        original_df["timestamp_nanoseconds"] = original_df["timestamp"].astype("int64")

    validate_combined_dataset(original_df, services)

    # Time features are handled separately in the Dataset class
    # Only include the actual metrics and severity level
    features = conf.config.get_config(active_set).metrics + ["severity_level"]

    # ---------------------------------------------------------------------
    # Align per-service timestamps so the model sees a dense tensor:
    #   [time, service, feature]
    #
    # - approach="grid": floor timestamps to a fixed grid and create a complete
    #   date_range from min..max at `TIME_GRID_FREQ`.
    # - approach="event": keep original (irregular) timestamps but ensure the
    #   full (timestamp, service) product exists at observed timestamps.
    #
    # Both approaches aggregate duplicates per (timestamp, service), forward-fill
    # within each service, normalize features, then fill remaining NaNs with 0.
    # ---------------------------------------------------------------------
    from ml_monitoring_service.constants import TIME_GRID_FREQ

    if approach not in ("grid", "event"):
        raise ValueError(
            f"Unsupported approach: {approach}. Expected 'grid' or 'event'."
        )

    effective_freq = time_grid_freq or TIME_GRID_FREQ

    working_df = original_df
    if approach == "grid":
        working_df["timestamp"] = working_df["timestamp"].dt.floor(effective_freq)

    # Keep only relevant columns for aggregation/alignment.
    keep_cols = ["timestamp", "service"] + features
    keep_cols = [c for c in keep_cols if c in working_df.columns]
    working_df = working_df[keep_cols].copy()

    agg: dict[str, str] = {f: "mean" for f in features if f != "severity_level"}
    if "severity_level" in features:
        # Severity is categorical/ordinal; keep the most severe event in the bucket.
        agg["severity_level"] = "max"

    # Collapse duplicates within the same timestamp bucket.
    aligned_df = working_df.groupby(["timestamp", "service"], as_index=False).agg(agg)
    if aligned_df.empty:
        raise DataValidationError("No valid rows after preprocessing")

    if approach == "grid":
        # Build a complete time grid from min..max and align all services.
        start = aligned_df["timestamp"].min()
        end = aligned_df["timestamp"].max()
        if pd.isna(start) or pd.isna(end):
            raise DataValidationError("No valid timestamps after preprocessing")

        all_timestamps = pd.date_range(start=start, end=end, freq=effective_freq)
    else:
        # Keep observed timestamps only (irregular axis).
        all_timestamps = pd.DatetimeIndex(
            pd.to_datetime(aligned_df["timestamp"]).unique()
        ).sort_values()

    full_index = pd.MultiIndex.from_product(
        [all_timestamps, services], names=["timestamp", "service"]
    )

    aligned_df = aligned_df.set_index(["timestamp", "service"]).reindex(full_index)

    # Forward-fill within each service to reduce artificial zeros from missing rows.
    # Any leading gaps will remain NaN and be set to 0 after normalization.
    aligned_df = aligned_df.groupby(level=1).ffill()
    aligned_df = aligned_df.reset_index()
    aligned_df["timestamp_nanoseconds"] = aligned_df["timestamp"].astype("int64")

    # Normalize features (after resampling; NaNs are preserved through normalization)
    for feature in features:
        aligned_df = normalize_feature(aligned_df, feature)

    # Fill remaining missing values with 0 after normalization.
    aligned_df = aligned_df.fillna(0)

    # Deterministic ordering for reshape.
    aligned_df["service"] = pd.Categorical(
        aligned_df["service"], categories=services, ordered=True
    )
    aligned_df = aligned_df.sort_values(
        ["timestamp_nanoseconds", "service"], kind="stable"
    )

    timestamp_ns = np.sort(aligned_df["timestamp_nanoseconds"].unique())
    timestamps = timestamp_ns
    expected_rows = len(timestamps) * len(services)
    if len(aligned_df) != expected_rows:
        raise DataValidationError(
            f"Aligned dataset shape mismatch: rows={len(aligned_df)} expected_rows={expected_rows}. "
            "This indicates inconsistent timestamp/service alignment."
        )

    data = (
        aligned_df[features]
        .to_numpy()
        .reshape(len(timestamps), len(services), len(features))
    )

    # Mutate the input DataFrame in-place to reflect the aligned time grid.
    # Downstream code relies on get_ordered_timepoints(df) matching the returned data.
    original_df.drop(index=original_df.index, inplace=True)
    original_df.drop(columns=list(original_df.columns), inplace=True, errors="ignore")
    for col in aligned_df.columns:
        original_df[col] = aligned_df[col].to_numpy()

    # Validate model input
    expected_shape = (len(timestamps), len(services), len(features))
    validate_model_input(data, expected_shape, services)

    logger.info(
        f"Converted data shape: {data.shape} (timesteps={len(timestamps)}, services={len(services)}, features={len(features)})"
    )
    return data, services, features


def get_microservice_data_from_file(file_path: str) -> pd.DataFrame:
    """Read microservice metrics data from a JSON file

    Args:
        file_path: Path to the JSON file containing metrics data

    Returns:
        DataFrame with microservice metrics and timestamps
    """
    df = pd.read_json(file_path)

    # Ensure the timestamp column is in datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

    return df


def get_timestamp_of_latest_data(active_set: str) -> str | None:
    """
    Finds the timestamp of the most recent data point for the specified service set.
    This helps in determining appropriate time ranges for new data downloads.

    Args:
        active_set (str): The name of the service set to check

    Returns:
        str: Timestamp of the most recent data in format MM/DD/YYYY:HH:MM:SS, or None if no data exists
    """
    logger.info(
        f"Checking for existing data timestamps for service set '{active_set}'..."
    )

    # Define path to check for existing training data
    training_path = f"output/{active_set}/training_dataset.json"

    latest_timestamp = None

    # Check if the training file exists
    if os.path.exists(training_path):
        try:
            # Read the file and find the most recent timestamp
            df = pd.read_json(training_path)

            if "timestamp" in df.columns:
                max_time = pd.to_datetime(df["timestamp"], format="mixed").max()
                min_time = pd.to_datetime(df["timestamp"], format="mixed").min()
                latest_timestamp = max_time
                logger.info(
                    f"Found data in {training_path} with latest timestamp: {max_time}"
                )
                logger.info(
                    f"Found data in {training_path} with earliest timestamp: {min_time}"
                )
        except Exception as e:
            logger.warning(f"Error reading timestamps from {training_path}: {e}")

    if latest_timestamp:
        # Format the timestamp as MM/DD/YYYY:HH:MM:SS
        formatted_timestamp = latest_timestamp.strftime("%m/%d/%Y:%H:%M:%S")
        logger.info(
            f"Most recent data timestamp for '{active_set}': {formatted_timestamp}"
        )
        return formatted_timestamp
    else:
        logger.info(
            f"No existing training data found for '{active_set}', will download from scratch"
        )
        return None


class ServiceMetricsDataset(Dataset):
    """PyTorch Dataset for service metrics time series data

    This dataset handles windowed sequences of service metrics along with their timestamps,
    extracting temporal features (hour, minute, day of week, second) for model training.

    Attributes:
        metrics: Tensor of metrics data [time_steps, num_services, num_features]
        timestamps: Pandas datetime series with timestamps for each time step
        window_size: Number of time steps to include in each sample
    """

    def __init__(self, metrics: np.ndarray, timestamps: pd.Series, window_size: int):
        """
        Args:
            metrics: numpy array of shape [time_steps, num_services, num_features]
            timestamps: pandas DatetimeIndex or Series containing timestamps
            window_size: number of time steps to consider for each sample
        """
        self.metrics = torch.FloatTensor(metrics)
        self.timestamps = pd.to_datetime(
            timestamps
        )  # Ensure timestamps are pandas datetime
        self.window_size = window_size

    def __len__(self) -> int:
        """Returns the number of valid windows in the dataset"""
        return len(self.metrics) - self.window_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a windowed sample with temporal features

        Args:
            idx: Index of the window to retrieve

        Returns:
            tuple: (window, window, time_features)
                - window: Metrics data for the window [window_size, num_services, num_features]
                - window: Same as first (for autoencoder target)
                - time_features: Temporal features [window_size, 4] (hour, minute, day, second)
        """
        window = self.metrics[idx : idx + self.window_size]
        timestamp_window = self.timestamps[idx : idx + self.window_size]

        # Pre-extract time features from timestamps
        hours = (
            torch.tensor([ts.hour / 24.0 for ts in timestamp_window])
            .float()
            .unsqueeze(1)
        )
        minutes = (
            torch.tensor([ts.minute / 60.0 for ts in timestamp_window])
            .float()
            .unsqueeze(1)
        )
        days = (
            torch.tensor([ts.dayofweek / 7.0 for ts in timestamp_window])
            .float()
            .unsqueeze(1)
        )
        seconds = (
            torch.tensor([ts.second / 60.0 for ts in timestamp_window])
            .float()
            .unsqueeze(1)
        )

        # Stack time features [seq_len, 4]
        time_features = torch.cat([hours, minutes, days, seconds], dim=1)

        return window, window, time_features
