import logging
import torch
from datetime import datetime
from typing import Optional

import ml_monitoring_service.configuration as conf
from ml_monitoring_service.data_handling import (
    check_for_nan, get_timestamp_of_latest_data, get_microservice_data_from_file, 
    convert_to_model_input
)
from ml_monitoring_service.data_gathering.get_prometheus_data import main as download_prometheus_data
from ml_monitoring_service.data_gathering.get_splunk_data import download_splunk_data
from ml_monitoring_service.data_gathering.combine import combine_services
from ml_monitoring_service.anomaly_detector import AnomalyDetector
from ml_monitoring_service.constants import DOWNLOAD_ENABLED, MAX_EPOCHS, Colors

logger = logging.getLogger(__name__)


def create_and_train_model(active_set: str) -> Optional[AnomalyDetector]:
    """Create and train an anomaly detection model for the specified service set
    
    Args:
        active_set: Name of the service set to train the model for
        
    Returns:
        Trained AnomalyDetector instance, or None if training fails
    """


    age_latest_data = get_timestamp_of_latest_data(active_set)

    logger.info(Colors.blue(f"\n⏱️  MODEL CREATION PROCESS STARTED FOR SERVICE SET: {active_set}. TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
    
    try:
        if DOWNLOAD_ENABLED:
            download_splunk_data("training", active_set, age_latest_data)
            download_prometheus_data("training", active_set)
            combine_services("training", active_set, age_latest_data)
            logger.info("Data download and combination completed successfully.")
        else:
            logger.warning("DOWNLOAD env var is set to false, skipping data download")
    except Exception as e:
        logger.error(f"An error occurred during training data download: {e}", exc_info=True)
        return None

    df = get_microservice_data_from_file(f"output/{active_set}/training_dataset.json")
    
    # Check for NaN values in data
    df = check_for_nan(df)

    data, services, features = convert_to_model_input(active_set, df)
    
    logger.info("\nDataset details:")
    logger.info(f"Number of services: {len(services)}")
    logger.info(f"Features per service: {features}")
    logger.info(f"Total timepoints: {len(data)}")
    
    # Split data into train/validation sets (70/30 split)
    train_size = int(0.7 * len(data))
    val_size = len(data) - train_size
    
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    logger.info(f"Data split: {train_size} training samples, {val_size} validation samples")
    
    config = conf.get_config(active_set)

    # Create and train detector
    logger.info("\nInitializing anomaly detector...")
    detector = AnomalyDetector(
        num_services=len(services),
        num_features=len(features),
        window_size=30,
        config=config
    )

    # # Add visualization before training
    # logger.info("\nGenerating model architecture visualization...")
    # visualize_model_architecture(detector.model)
    
    logger.info("Training model...")
    detector.train(train_data, val_data, df, active_set, max_epochs=MAX_EPOCHS)
    
    logger.info("Setting threshold using validation data...")
    detector.set_threshold(val_data, df, percentile=config.anomaly_threshold_percentile)
    
    # Save the trained model with all necessary information
    model_filename = f'output/{active_set}/best_model_{active_set}.pth'
    torch.save({
        'model_state_dict': detector.model.state_dict(),
        'threshold': detector.threshold,
        'num_services': len(services),
        'num_features': len(features),
        'window_size': detector.window_size
    }, model_filename)
    logger.info(f"Model saved as {model_filename}")
    logger.info(Colors.blue(f"✅ MODEL CREATION PROCESS FINISHED FOR SERVICE SET: {active_set}."))
    logger.info(Colors.blue(f"TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))