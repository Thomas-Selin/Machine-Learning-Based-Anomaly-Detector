import pytest
from ml_monitoring_service.anomaly_detector import AnomalyDetector
from ml_monitoring_service.data_handling import convert_to_model_input, get_microservice_data_from_file
from ml_monitoring_service.data_handling import get_ordered_timepoints
from ml_monitoring_service.anomaly_analyser import analyse_anomalies
from ml_monitoring_service.configuration import ConfigLoader, ServiceSetConfig

@pytest.fixture
def config():
    config_loader = ConfigLoader("src/ml_monitoring_service/resources/service_sets.yaml")
    return config_loader.get_config("transfer")

@pytest.fixture
def sample_data():
    df = get_microservice_data_from_file('tests/resources/combined_dataset_test.json')
    return df

@pytest.fixture
def model_input(sample_data, config):
    data, services, features = convert_to_model_input("transfer", sample_data)
    return data, services, features, sample_data 

def test_train(model_input, config):
    data, services, features, df = model_input
    timepoints = get_ordered_timepoints(df)
    detector = AnomalyDetector(
        num_services=len(services),
        num_features=len(features),
        window_size=config.window_size,
        config=config
    )
    
    # Split data into train and validation sets
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    detector.train(train_data, val_data, df, active_set="transfer", max_epochs=1, timepoints=timepoints)
    assert detector.model is not None

def test_set_threshold(model_input, config):
    data, services, features, df = model_input
    timepoints = get_ordered_timepoints(df)
    detector = AnomalyDetector(
        num_services=len(services),
        num_features=len(features),
        window_size=config.window_size,
        config=config
    )
    
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    detector.train(train_data, val_data, df, active_set="transfer", max_epochs=1, timepoints=timepoints)
    detector.set_threshold(val_data, timepoints=timepoints[train_size:], percentile=config.anomaly_threshold_percentile)
    assert detector.threshold is not None

def test_detect(model_input, config):
    data, services, features, df = model_input
    timepoints = get_ordered_timepoints(df)
    detector = AnomalyDetector(
        num_services=len(services),
        num_features=len(features),
        window_size=config.window_size,
        config=config
    )
    
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    detector.train(train_data, val_data, df, active_set="transfer", max_epochs=2, timepoints=timepoints)
    detector.set_threshold(val_data, timepoints=timepoints[train_size:], percentile=config.anomaly_threshold_percentile)
    
    result = detector.detect(data[:config.window_size], '2022-01-01 00:00:00')
    
    assert 'is_anomaly' in result
    assert 'error_score' in result
    assert 'threshold' in result
    assert 'service_errors' in result
    assert 'variable_errors' in result
    assert 'timestamp' in result

def test_analyze_anomalies(model_input, config):
    data, services, features, df = model_input
    timepoints = get_ordered_timepoints(df)
    detector = AnomalyDetector(
        num_services=len(services),
        num_features=len(features),
        window_size=config.window_size,
        config=config
    )
    
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    detector.train(train_data, val_data, df, active_set="transfer", max_epochs=2, timepoints=timepoints)
    detector.set_threshold(val_data, timepoints=timepoints[train_size:], percentile=config.anomaly_threshold_percentile)
    
    result = detector.detect(data[:config.window_size], '2022-01-01 00:00:00')
    explanations = analyse_anomalies(result, config.relationships, services, features)
    
    assert len(explanations) > 0

def test_config_loading():
    config_loader = ConfigLoader("src/ml_monitoring_service/resources/service_sets.yaml")
    config = config_loader.get_config("transfer")
    
    assert isinstance(config, ServiceSetConfig)
    assert config.window_size > 0
    assert 0 <= config.anomaly_threshold_percentile <= 100
    assert isinstance(config.relationships, dict)
    assert isinstance(config.metrics, list)

def test_analyse_anomalies_with_single_feature():
    """Test that analyse_anomalies handles single feature case without crashing"""
    result = {
        'is_anomaly': True,
        'error_score': 0.95,
        'threshold': 0.85,
        'service_errors': [0.8, 0.6],
        'variable_errors': [0.9],  # Only one feature
        'timestamp': '2022-01-01 00:00:00'
    }
    services = ['service1', 'service2']
    features = ['cpu']  # Only one feature
    relationships = {'service1': ['service2'], 'service2': []}
    
    # Should not crash with single feature
    explanations = analyse_anomalies(result, relationships, services, features)
    assert len(explanations) > 0
    # Check that it mentions the single variable
    explanations_text = '\n'.join(explanations)
    assert 'cpu' in explanations_text

def test_analyse_anomalies_with_no_features():
    """Test that analyse_anomalies handles empty features gracefully"""
    result = {
        'is_anomaly': True,
        'error_score': 0.95,
        'threshold': 0.85,
        'service_errors': [0.8, 0.6],
        'variable_errors': [],  # No features
        'timestamp': '2022-01-01 00:00:00'
    }
    services = ['service1', 'service2']
    features = []  # No features
    relationships = {'service1': ['service2'], 'service2': []}
    
    # Should not crash with no features
    explanations = analyse_anomalies(result, relationships, services, features)
    assert len(explanations) > 0
