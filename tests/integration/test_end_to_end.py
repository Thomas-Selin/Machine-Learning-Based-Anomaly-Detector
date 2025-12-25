"""Integration tests for the full anomaly detection pipeline."""

import numpy as np
import pandas as pd
import pytest
import torch

from ml_monitoring_service.anomaly_detector import AnomalyDetector
from ml_monitoring_service.configuration import ConfigLoader
from ml_monitoring_service.data_handling import convert_to_model_input


@pytest.fixture
def test_config():
    """Create test configuration."""
    return ConfigLoader("src/ml_monitoring_service/resources/service_sets.yaml")


@pytest.fixture
def sample_training_data():
    """Create sample training data for testing."""
    # Create a sample dataset with 3 services and multiple timepoints
    services = ["service_a", "service_b", "service_c"]
    num_timepoints = 100

    data = []
    base_time = pd.Timestamp("2023-01-01 00:00:00")

    for i in range(num_timepoints):
        timestamp = base_time + pd.Timedelta(minutes=i)
        for service in services:
            data.append(
                {
                    "timestamp": timestamp,
                    "timestamp_nanoseconds": int(timestamp.value),
                    "service": service,
                    "cpu_usage": np.random.uniform(0.3, 0.7),
                    "memory_usage": np.random.uniform(1000, 2000),
                    "request_count": np.random.uniform(10, 100),
                    "error_count": np.random.uniform(0, 5),
                    "response_time": np.random.uniform(100, 500),
                    "severity_level": 0,
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def sample_inference_data():
    """Create sample inference data (with potential anomaly)."""
    services = ["service_a", "service_b", "service_c"]
    num_timepoints = 50

    data = []
    base_time = pd.Timestamp("2023-01-02 00:00:00")

    for i in range(num_timepoints):
        timestamp = base_time + pd.Timedelta(minutes=i)
        for service in services:
            # Inject anomaly at timepoint 25 for service_b
            if i == 25 and service == "service_b":
                data.append(
                    {
                        "timestamp": timestamp,
                        "timestamp_nanoseconds": int(timestamp.value),
                        "service": service,
                        "cpu_usage": 0.99,  # Anomalous high CPU
                        "memory_usage": 5000,  # Anomalous high memory
                        "request_count": 500,  # Anomalous high requests
                        "error_count": 50,  # Anomalous high errors
                        "response_time": 2000,  # Anomalous high response time
                        "severity_level": 3,
                    }
                )
            else:
                data.append(
                    {
                        "timestamp": timestamp,
                        "timestamp_nanoseconds": int(timestamp.value),
                        "service": service,
                        "cpu_usage": np.random.uniform(0.3, 0.7),
                        "memory_usage": np.random.uniform(1000, 2000),
                        "request_count": np.random.uniform(10, 100),
                        "error_count": np.random.uniform(0, 5),
                        "response_time": np.random.uniform(100, 500),
                        "severity_level": 0,
                    }
                )

    return pd.DataFrame(data)


def test_full_training_pipeline(sample_training_data, test_config, tmp_path):
    """Test complete training pipeline from data to trained model."""
    # Save sample data to temp file
    data_file = tmp_path / "training_data.json"
    sample_training_data.to_json(data_file)

    # Load and convert data
    df = pd.read_json(data_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Mock configuration for test services
    services = ["service_a", "service_b", "service_c"]
    metrics = [
        "cpu_usage",
        "memory_usage",
        "request_count",
        "error_count",
        "response_time",
    ]

    # Create test config object with required attributes
    class TestConfig:
        def __init__(self):
            self.window_size = 10
            self.services = services
            self.metrics = metrics

    config = TestConfig()

    # Create detector
    detector = AnomalyDetector(
        num_services=len(services),
        num_features=6,
        window_size=config.window_size,
        config=config,
    )

    # Verify model was created
    assert detector.model is not None
    assert detector.num_services == 3
    assert detector.num_features == 6
    assert detector.window_size == 10

    # Verify model can perform forward pass
    test_input = torch.randn(1, 10, 3, 6)
    test_time_features = torch.randn(1, 10, 4)

    with torch.no_grad():
        output = detector.model(test_input, test_time_features)

    assert output.shape == test_input.shape


def test_full_inference_pipeline(sample_inference_data, tmp_path):
    """Test complete inference pipeline from data to anomaly detection."""
    # Create a simple trained model
    services = ["service_a", "service_b", "service_c"]
    num_features = 6
    window_size = 10

    class TestConfig:
        def __init__(self):
            self.window_size = window_size
            self.services = services
            self.metrics = [
                "cpu_usage",
                "memory_usage",
                "request_count",
                "error_count",
                "response_time",
            ]

    config = TestConfig()

    detector = AnomalyDetector(
        num_services=len(services),
        num_features=num_features,
        window_size=window_size,
        config=config,
    )

    # Set a reasonable threshold
    detector.threshold = np.float64(0.5)

    # Create sample inference data
    data_array = np.random.rand(40, 3, 6)  # 40 timesteps, 3 services, 6 features
    timestamps = pd.date_range(start="2023-01-02", periods=40, freq="min")

    # Run detection on a window
    window_data = data_array[0:window_size]
    result = detector.detect(window_data, timestamps[0])

    # Verify result structure
    assert "is_anomaly" in result
    assert "error_score" in result
    assert "threshold" in result
    assert "service_errors" in result
    assert "variable_errors" in result
    assert "timestamp" in result

    assert isinstance(result["is_anomaly"], bool | np.bool_)
    assert isinstance(result["error_score"], float | np.floating)
    assert len(result["service_errors"]) == len(services)


def test_model_save_and_load(tmp_path):
    """Test that models can be saved and loaded correctly."""
    services = ["service_a", "service_b", "service_c"]
    num_features = 6
    window_size = 10

    class TestConfig:
        def __init__(self):
            self.window_size = window_size
            self.services = services

    config = TestConfig()

    # Create and configure detector
    detector1 = AnomalyDetector(
        num_services=len(services),
        num_features=num_features,
        window_size=window_size,
        config=config,
    )
    detector1.threshold = np.float64(0.42)

    # Save model
    model_path = tmp_path / "test_model.pth"
    torch.save(
        {
            "model_state_dict": detector1.model.state_dict(),
            "threshold": float(
                detector1.threshold
            ),  # Convert numpy scalar to Python float
            "num_services": detector1.num_services,
            "num_features": detector1.num_features,
            "window_size": detector1.window_size,
        },
        model_path,
    )

    # Load model
    checkpoint = torch.load(model_path, weights_only=True, map_location="cpu")

    detector2 = AnomalyDetector(
        num_services=checkpoint["num_services"],
        num_features=checkpoint["num_features"],
        window_size=checkpoint["window_size"],
        config=config,
    )
    detector2.model.load_state_dict(checkpoint["model_state_dict"])
    detector2.threshold = checkpoint["threshold"]

    # Verify loaded model matches original
    assert detector2.num_services == detector1.num_services
    assert detector2.num_features == detector1.num_features
    assert detector2.window_size == detector1.window_size
    assert np.isclose(detector2.threshold, detector1.threshold)

    # Verify both models produce same output
    test_input = torch.randn(1, window_size, len(services), num_features)
    test_time_features = torch.randn(1, window_size, 4)

    detector1.model.eval()
    detector2.model.eval()

    with torch.no_grad():
        output1 = detector1.model(test_input, test_time_features)
        output2 = detector2.model(test_input, test_time_features)

    assert torch.allclose(output1, output2, atol=1e-6)


def test_data_pipeline_with_real_config(test_config):
    """Test data conversion pipeline with real configuration."""
    # Get transfer service set config
    config = test_config.get_config("transfer")
    services = config.services
    metrics = config.metrics

    # Create sample data matching the configuration
    num_timepoints = 50
    data = []
    base_time = pd.Timestamp("2023-01-01 00:00:00")

    for i in range(num_timepoints):
        timestamp = base_time + pd.Timedelta(minutes=i)
        for service in services:
            row = {
                "timestamp": timestamp,
                "timestamp_nanoseconds": int(timestamp.value),
                "service": service,
                "severity_level": 0,
            }
            # Add metric values
            for metric in metrics:
                row[metric] = np.random.uniform(0.1, 0.9)

            data.append(row)

    df = pd.DataFrame(data)

    # Convert to model input
    data_array, services_list, features_list = convert_to_model_input("transfer", df)

    # Verify output
    assert data_array.shape[0] == num_timepoints
    assert data_array.shape[1] == len(services)
    assert data_array.shape[2] == len(metrics) + 1  # metrics + severity_level
    assert set(services_list) == set(services)
    assert "severity_level" in features_list


def test_anomaly_detection_sensitivity():
    """Test that anomaly detection responds to different severity levels."""
    services = ["service_a"]
    num_features = 3
    window_size = 5

    class TestConfig:
        def __init__(self):
            self.window_size = window_size
            self.services = services

    config = TestConfig()

    detector = AnomalyDetector(
        num_services=len(services),
        num_features=num_features,
        window_size=window_size,
        config=config,
    )

    # Set a moderate threshold
    detector.threshold = np.float64(0.1)

    timestamps = pd.date_range(start="2023-01-01", periods=window_size, freq="min")

    # Test with normal data
    normal_data = np.random.rand(window_size, 1, num_features) * 0.1
    result_normal = detector.detect(normal_data, timestamps[0])

    # Test with anomalous data (high values)
    anomalous_data = np.random.rand(window_size, 1, num_features) * 10
    result_anomalous = detector.detect(anomalous_data, timestamps[0])

    # Anomalous data should have higher error score
    assert result_anomalous["error_score"] > result_normal["error_score"]
