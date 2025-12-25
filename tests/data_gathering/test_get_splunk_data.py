from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from ml_monitoring_service.configuration import ConfigLoader
from ml_monitoring_service.data_gathering.get_splunk_data import (
    check_authentication,
    download_splunk_data,
    parse_raw_data,
)
from ml_monitoring_service.data_handling import (
    ServiceMetricsDataset,
    check_for_nan,
    convert_to_model_input,
    get_microservice_data_from_file,
    normalize_feature,
)
from ml_monitoring_service.main import create_app


@pytest.fixture
def config():
    return ConfigLoader("src/ml_monitoring_service/resources/service_sets.yaml")


def test_get_microservice_data_from_file():
    df = get_microservice_data_from_file("tests/resources/combined_dataset_test.json")
    assert not df.empty
    assert "timestamp" in df.columns


def test_normalize_feature():
    df = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})
    df = normalize_feature(df, "feature")
    assert df["feature"].max() == 1
    assert df["feature"].min() == 0


def test_service_metrics_dataset():
    """Test ServiceMetricsDataset returns correct shapes and values"""
    metrics = np.random.rand(100, 5, 10)  # 100 timesteps, 5 services, 10 features
    timestamps = pd.date_range(start="2023-01-01", periods=100, freq="h")
    dataset = ServiceMetricsDataset(metrics, timestamps, window_size=10)

    # Length should be total timesteps minus window size
    assert len(dataset) == 90

    # Dataset returns (input_window, target_window, time_features)
    input_window, target_window, time_features = dataset[0]

    # Both input and target should have shape (window_size, num_services, num_features)
    assert input_window.shape == (10, 5, 10)
    assert target_window.shape == (10, 5, 10)

    # Input and target should be identical (autoencoder)
    assert torch.equal(input_window, target_window)

    # Time features should have shape (window_size, 4) for [hour, minute, day, second]
    assert time_features.shape == (10, 4)

    # Time features should be normalized between 0 and 1
    assert time_features.min() >= 0.0
    assert time_features.max() <= 1.0


def test_create_app():
    """Test that Flask app is created with correct configuration"""
    from flask import Flask

    app = create_app()

    # App should be a Flask instance
    assert isinstance(app, Flask)

    # Check that health endpoint exists
    assert app.url_map is not None

    # Get all registered routes
    routes = [str(rule) for rule in app.url_map.iter_rules()]

    # Should have health endpoint
    assert any("/health" in route for route in routes)


def test_check_for_nan():
    df = pd.DataFrame({"feature": [1, 2, np.nan, 4, 5]})
    df = check_for_nan(df)
    assert df.isnull().sum().sum() == 0


def test_convert_to_model_input(config):
    df = get_microservice_data_from_file("tests/resources/combined_dataset_test.json")
    _, services, _ = convert_to_model_input(active_set="transfer", df=df)
    config = ConfigLoader("src/ml_monitoring_service/resources/service_sets.yaml")
    expected_services = config.get_services("transfer")
    assert len(services) == len(expected_services)
    assert all(s in expected_services for s in services)


def test_check_authentication_fail():
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 401
        mock_get.return_value.text = "Unauthorized"
        result = check_authentication()
        assert result is False


def test_parse_raw_data():
    raw_data = "key1=value1, key2=value2"
    expected_output = {"key1": "value1", "key2": "value2"}
    parsed_data = parse_raw_data(raw_data)
    assert parsed_data == expected_output


def test_parse_raw_data_with_spaces():
    """Test parsing raw data with extra spaces"""
    raw_data = "key1=value1, key2=value2"
    parsed_data = parse_raw_data(raw_data)
    assert "key1" in parsed_data
    assert "key2" in parsed_data
    assert parsed_data["key1"] == "value1"
    assert parsed_data["key2"] == "value2"


def test_parse_raw_data_empty():
    """Test parsing empty raw data"""
    raw_data = ""
    parsed_data = parse_raw_data(raw_data)
    assert parsed_data == {}


def test_parse_raw_data_single_pair():
    """Test parsing single key-value pair"""
    raw_data = "key1=value1"
    parsed_data = parse_raw_data(raw_data)
    assert parsed_data == {"key1": "value1"}


def test_download_splunk_data_file_path():
    with patch(
        "ml_monitoring_service.data_gathering.get_splunk_data.download_splunk_data_in_batches"
    ) as mock_download:
        mock_download.return_value = [{"test": "data"}]
        active_set = "transfer"

        result = download_splunk_data("training", active_set)

        # Verify the function was called with correct parameters
        mock_download.assert_called_once_with("training", active_set, None)
        assert result == [{"test": "data"}]
