from datetime import timedelta

import pytest

from ml_monitoring_service.configuration import (
    ConfigLoader,
    ServiceSetConfig,
    get_metrics,
    get_relationships,
    get_services,
)


@pytest.fixture
def config_loader():
    return ConfigLoader("src/ml_monitoring_service/resources/service_sets.yaml")


def test_load_config(config_loader):
    # assert "default" in config_loader.service_sets
    assert "transfer" in config_loader.service_sets


def test_service_set_config():
    config = ServiceSetConfig(
        relationships={"service1": ["service2"]},
        metrics=["cpu"],
        training_lookback_hours=24,
        inference_lookback_minutes=5,
        window_size=30,
        anomaly_threshold_percentile=95,
    )
    assert config.retraining_interval == timedelta(hours=24)
    assert config.inference_interval == timedelta(minutes=5)


def test_get_services():
    services = get_services("transfer")
    assert "mobile-bff" in services
    assert "transfer-service" in services


def test_get_metrics():
    metrics = get_metrics("transfer")
    assert "cpu" in metrics
    assert "memory" in metrics
    # assert "latency" in metrics


# def test_active_set():
#     set_active_sets("transfer")
#     assert get_active_sets() == "transfer"
#     services = get_services()  # Should use active set
#     assert "transfer-service" in services

# def test_invalid_service_set():
#     with pytest.raises(ValueError):
#         get_services("nonexistent_set")


def test_config_validation():
    with pytest.raises(ValueError):
        ServiceSetConfig(
            relationships={},
            metrics=[],
            training_lookback_hours=0,  # Invalid
            inference_lookback_minutes=5,
            window_size=30,
            anomaly_threshold_percentile=95,
        )


def test_get_relationships():
    relationships = get_relationships("transfer")
    assert "mobile-bff" in relationships
    assert isinstance(relationships["mobile-bff"], list)


def test_max_depth_calculation():
    """Test max_depth property calculation"""
    # Test with simple chain: A -> B -> C
    config = ServiceSetConfig(
        relationships={"A": ["B"], "B": ["C"], "C": []},
        metrics=["cpu"],
        training_lookback_hours=24,
        inference_lookback_minutes=5,
        window_size=30,
        anomaly_threshold_percentile=95,
    )
    assert config.max_depth >= 0  # Should never be negative

    # Test with single service (no relationships)
    config_single = ServiceSetConfig(
        relationships={"A": []},
        metrics=["cpu"],
        training_lookback_hours=24,
        inference_lookback_minutes=5,
        window_size=30,
        anomaly_threshold_percentile=95,
    )
    assert config_single.max_depth == 0  # Single node has depth 0
