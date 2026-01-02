import datetime
import os
from pathlib import Path

import pytest
import requests

from ml_monitoring_service.configuration import ConfigLoader
from ml_monitoring_service.data_gathering.get_prometheus_data import (
    create_session_with_retries,
    download_prometheus_data,
    fill_missing_data,
    is_numeric,
    main,
    process_results,
    query_prometheus,
)


@pytest.fixture
def config():
    return ConfigLoader("src/ml_monitoring_service/resources/service_sets.yaml")


def test_create_session_with_retries():
    session = create_session_with_retries()
    assert isinstance(session, requests.Session)


def test_query_prometheus(monkeypatch):
    def mock_get(*args, **kwargs):
        class MockResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"data": {"result": [{"value": [1622470425.0, "1.0"]}]}}

        return MockResponse()

    session = create_session_with_retries()
    monkeypatch.setattr(session, "get", mock_get)
    results = query_prometheus("query", 1622470425.0, session)
    assert results == [{"value": [1622470425.0, "1.0"]}]


def test_is_numeric():
    assert is_numeric("1.0")
    assert not is_numeric("abc")


def test_process_results():
    results = [{"value": [1622470425.0, "1.0"]}]
    processed_data = process_results(results, "cpu")
    assert processed_data == [
        {"timestamp": datetime.datetime(2021, 5, 31, 16, 13, 45), "cpu": 1.0}
    ]


def test_fill_missing_data():
    data = [{"timestamp": datetime.datetime(2022, 1, 1), "cpu": 1.0}]
    start_time = datetime.datetime(2022, 1, 1)
    end_time = datetime.datetime(2022, 1, 1, 0, 2)
    df = fill_missing_data(data, start_time, end_time)
    assert len(df) == 5


def test_download_prometheus_data(monkeypatch, config):
    # Mock the prometheus query
    def mock_query_prometheus(*args, **kwargs):
        return [{"value": [1622470425.0, "1.0"]}]

    monkeypatch.setattr(
        "ml_monitoring_service.data_gathering.get_prometheus_data.query_prometheus",
        mock_query_prometheus,
    )

    # Create test service directory
    service_name = "test-service"
    service_name_underscore = service_name.replace(
        "-", "_"
    )  # Note: service name casing follows Prometheus convention (lowercase with underscores)
    active_set = "test-microservice-set"
    service_dir = Path(
        f"output/{active_set}/prometheus_data_training/{service_name_underscore}"
    )
    service_dir.mkdir(parents=True, exist_ok=True)

    # Test download
    download_prometheus_data(
        "training", service_name, "cpu", ["2022-01-01 00:00:00.000"], active_set
    )

    # Check if file exists in correct location
    assert (service_dir / f"cpu-{service_name_underscore}.json").exists()


def test_main_with_service_set(monkeypatch, config):
    # Mock splunk data
    mock_splunk_data = [
        {"service": "service1", "timestamp": "2022-01-01 00:00:00.000"},
        {"service": "service2", "timestamp": "2022-01-01 00:00:00.000"},
    ]

    def mock_read_splunk_data(*args):
        return mock_splunk_data

    def mock_download_prometheus_data(*args, **kwargs):
        return True

    # Mock functions
    monkeypatch.setattr(
        "ml_monitoring_service.data_gathering.get_prometheus_data.read_splunk_data",
        mock_read_splunk_data,
    )
    monkeypatch.setattr(
        "ml_monitoring_service.data_gathering.get_prometheus_data.download_prometheus_data",
        mock_download_prometheus_data,
    )

    # Create splunk data file
    active_set = "transfer"
    Path(f"output/{active_set}").mkdir(parents=True, exist_ok=True)
    with open(f"output/{active_set}/splunk_results.json", "w") as f:
        f.write("[]")

    # Run main
    main("training", active_set)


def test_service_specific_metrics(monkeypatch, config):
    # Test that different service sets use their specific metrics
    metrics = config.get_config(
        "transfer"
    ).metrics  # Get metrics for transfer service set
    assert "cpu" in metrics
    assert "memory" in metrics
    assert "latency" in metrics


def test_process_results_with_invalid_values():
    """Test that invalid values like Inf are filtered out"""
    results = [
        {"value": [1622470425.0, "1.0"]},
        {"value": [1622470426.0, "Inf"]},
        {"value": [1622470427.0, "+Inf"]},
        {"value": [1622470428.0, "-Inf"]},
        {"value": [1622470429.0, "2.5"]},
    ]
    processed_data = process_results(results, "cpu")
    assert len(processed_data) == 2
    assert processed_data[0]["cpu"] == 1.0
    assert processed_data[1]["cpu"] == 2.5


def test_process_results_with_non_numeric():
    """Test that non-numeric values are filtered out"""
    results = [
        {"value": [1622470425.0, "abc"]},
        {"value": [1622470426.0, "1.0"]},
    ]
    processed_data = process_results(results, "memory")
    assert len(processed_data) == 1
    assert processed_data[0]["memory"] == 1.0


def test_process_results_empty():
    """Test processing empty results"""
    results = []
    processed_data = process_results(results, "cpu")
    assert processed_data == []


def test_convert_to_timestamp():
    """Test timestamp conversion"""
    from ml_monitoring_service.data_gathering.get_prometheus_data import (
        convert_to_timestamp,
    )

    dt = datetime.datetime(2022, 1, 1, 12, 0, 0)
    timestamp = convert_to_timestamp(dt)
    assert isinstance(timestamp, int)
    assert timestamp > 0


def test_convert_to_timestamp_with_milliseconds():
    """Test timestamp conversion with milliseconds"""
    from ml_monitoring_service.data_gathering.get_prometheus_data import (
        convert_to_timestamp_with_milliseconds,
    )

    dt = datetime.datetime(2022, 1, 1, 12, 0, 0)
    timestamp = convert_to_timestamp_with_milliseconds(dt)
    assert isinstance(timestamp, float)
    assert timestamp > 0


def test_get_timestamps_for_service():
    """Test extracting timestamps for a specific service"""
    from ml_monitoring_service.data_gathering.get_prometheus_data import (
        get_timestamps_for_service,
    )

    splunk_data = [
        {"service": "service1", "timestamp": "2022-01-01 00:00:00.000"},
        {"service": "service2", "timestamp": "2022-01-01 00:01:00.000"},
        {"service": "service1", "timestamp": "2022-01-01 00:02:00.000"},
    ]
    timestamps = get_timestamps_for_service(splunk_data, "service1")
    assert len(timestamps) == 2
    assert "2022-01-01 00:00:00.000" in timestamps
    assert "2022-01-01 00:02:00.000" in timestamps


def test_fill_missing_data_with_gaps():
    """Test filling missing data with time gaps"""
    data = [
        {"timestamp": datetime.datetime(2022, 1, 1, 0, 0, 0), "cpu": 1.0},
        {"timestamp": datetime.datetime(2022, 1, 1, 0, 1, 0), "cpu": 2.0},
    ]
    start_time = datetime.datetime(2022, 1, 1, 0, 0, 0)
    end_time = datetime.datetime(2022, 1, 1, 0, 1, 0)
    df = fill_missing_data(data, start_time, end_time)
    assert len(df) == 3  # 0:00, 0:00:30, 0:01


@pytest.fixture(autouse=True, scope="session")
def cleanup_fixture():
    yield  # Run tests
    # Cleanup after all tests
    import shutil

    if os.path.exists("output/test_service"):
        shutil.rmtree("output/test_service")
    if os.path.exists("output/test-microservice-set"):
        shutil.rmtree("output/test-microservice-set")
    if os.path.exists("output/transfer/splunk_results.json"):
        os.remove("output/transfer/splunk_results.json")
