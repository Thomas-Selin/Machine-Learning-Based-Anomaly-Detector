"""Integration tests for the full anomaly detection pipeline."""

import numpy as np
import pandas as pd
import pytest
import torch

from ml_monitoring_service.anomaly_detector import AnomalyDetector
from ml_monitoring_service.configuration import ConfigLoader
from ml_monitoring_service.data_handling import convert_to_model_input

pytestmark = pytest.mark.integration


@pytest.fixture
def test_config():
    """Create test configuration."""
    return ConfigLoader("src/ml_monitoring_service/resources/service_sets.yaml")


def _create_test_data(num_services=3, num_timepoints=50, inject_anomaly=False):
    """Helper to create test data - reduces duplication."""
    services = [f"service_{chr(97 + i)}" for i in range(num_services)]
    data = []
    base_time = pd.Timestamp("2023-01-01 00:00:00")

    for i in range(num_timepoints):
        timestamp = base_time + pd.Timedelta(minutes=i)
        for service in services:
            # Inject anomaly if requested
            if inject_anomaly and i == num_timepoints // 2 and service == services[1]:
                data.append(
                    {
                        "timestamp": timestamp,
                        "timestamp_nanoseconds": int(timestamp.value),
                        "service": service,
                        "cpu_usage": 0.99,
                        "memory_usage": 5000,
                        "request_count": 500,
                        "error_count": 50,
                        "response_time": 2000,
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


def test_end_to_end_pipeline():
    """Test complete pipeline from model creation through anomaly detection."""
    services = ["service_a", "service_b", "service_c"]
    num_features = 6
    window_size = 10

    class TestConfig:
        def __init__(self):
            self.window_size = window_size
            self.services = services

    config = TestConfig()

    # Create detector
    detector = AnomalyDetector(
        num_services=len(services),
        num_features=num_features,
        window_size=window_size,
        config=config,
    )

    # Verify model creation
    assert detector.model is not None
    assert detector.num_services == 3
    assert detector.num_features == 6

    # Set threshold
    detector.threshold = 0.5

    # Test detection
    data_array = np.random.rand(window_size, 3, 6)
    timestamps = pd.date_range(start="2023-01-02", periods=window_size, freq="min")
    result = detector.detect(data_array, timestamps[0].isoformat())

    # Verify result structure
    assert "is_anomaly" in result
    assert "error_score" in result
    assert "threshold" in result
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

    # Create and save detector
    detector1 = AnomalyDetector(
        num_services=len(services),
        num_features=num_features,
        window_size=window_size,
        config=config,
    )
    detector1.threshold = 0.42

    model_path = tmp_path / "test_model.pth"
    torch.save(
        {
            "model_state_dict": detector1.model.state_dict(),
            "threshold": float(detector1.threshold),
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

    # Verify loaded model matches
    assert detector2.num_services == detector1.num_services
    assert detector1.threshold is not None
    assert detector2.threshold is not None
    assert float(detector2.threshold) == pytest.approx(float(detector1.threshold))

    # Verify same outputs
    test_input = torch.randn(1, window_size, len(services), num_features)
    test_time_features = torch.randn(1, window_size, 4)

    detector1.model.eval()
    detector2.model.eval()

    with torch.no_grad():
        output1 = detector1.model(test_input, test_time_features)
        output2 = detector2.model(test_input, test_time_features)

    assert torch.allclose(output1, output2, atol=1e-6)


def test_data_pipeline_with_config(test_config):
    """Test data conversion pipeline with real configuration."""
    config = test_config.get_config("transfer")
    services = config.services
    metrics = config.metrics

    # Create sample data
    data = []
    base_time = pd.Timestamp("2023-01-01 00:00:00")

    for i in range(50):
        timestamp = base_time + pd.Timedelta(minutes=i)
        for service in services:
            row = {
                "timestamp": timestamp,
                "timestamp_nanoseconds": int(timestamp.value),
                "service": service,
                "severity_level": 0,
            }
            for metric in metrics:
                row[metric] = np.random.uniform(0.1, 0.9)
            data.append(row)

    df = pd.DataFrame(data)

    # Convert to model input
    data_array, services_list, features_list = convert_to_model_input("transfer", df)

    # Verify output shapes
    assert data_array.shape[0] == 50
    assert data_array.shape[1] == len(services)
    assert data_array.shape[2] == len(metrics) + 1
    assert set(services_list) == set(services)
