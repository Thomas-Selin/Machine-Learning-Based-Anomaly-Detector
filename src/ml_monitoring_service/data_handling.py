import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import ml_monitoring_service.configuration as conf
from ml_monitoring_service.data_validation import (
    validate_combined_dataset,
    validate_model_input,
)

logger = logging.getLogger(__name__)


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
        ns = pd.Series(dt.view("int64"), index=df.index)

    # Fill rows that couldn't be converted using the parsed timestamp column when available.
    if ns.isna().any() and "timestamp" in df.columns:
        fallback_dt = pd.to_datetime(df["timestamp"], errors="coerce")
        ns = ns.where(
            ~ns.isna(), other=pd.Series(fallback_dt.view("int64"), index=df.index)
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
    active_set: str, df: pd.DataFrame
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
    # Validate combined dataset
    services = conf.get_services(active_set)
    validate_combined_dataset(df, services)

    # Ensure timestamp is in datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Time features are handled separately in the Dataset class
    # Only include the actual metrics and severity level
    features = conf.get_metrics(active_set) + ["severity_level"]

    # Normalize features
    for feature in features:
        df = normalize_feature(df, feature)

    # Normalize timestamp key column to int64 nanoseconds for stable equality checks.
    timestamp_ns = ensure_timestamp_nanoseconds_ns(df)
    timestamps = timestamp_ns
    service_to_idx = {service: idx for idx, service in enumerate(services)}
    data = np.zeros((len(timestamps), len(services), len(features)))

    for t, timestamp in enumerate(timestamps):
        for service in services:
            service_data = df[
                (df["timestamp_nanoseconds"] == timestamp) & (df["service"] == service)
            ]
            if not service_data.empty:
                feature_values = service_data[features].values
                if feature_values.shape[0] > 1:
                    # Average multiple entries for the same service at the same timestamp
                    feature_values = feature_values.mean(axis=0)
                else:
                    # Flatten to 1D array if only one row
                    feature_values = feature_values.flatten()
                data[t, service_to_idx[service]] = feature_values

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
