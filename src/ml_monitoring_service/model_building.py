import logging
import os
from datetime import datetime

import mlflow

import ml_monitoring_service.configuration as conf
from ml_monitoring_service.anomaly_detector import AnomalyDetector
from ml_monitoring_service.constants import DOWNLOAD_ENABLED, MAX_EPOCHS, Colors
from ml_monitoring_service.data_gathering.combine import combine_services
from ml_monitoring_service.data_gathering.get_prometheus_data import (
    main as download_prometheus_data,
)
from ml_monitoring_service.data_gathering.get_splunk_data import download_splunk_data
from ml_monitoring_service.data_handling import (
    check_for_nan,
    convert_to_model_input,
    get_microservice_data_from_file,
    get_ordered_timepoints,
    get_timestamp_of_latest_data,
)

logger = logging.getLogger(__name__)


def create_and_train_model(active_set: str) -> AnomalyDetector | None:
    """Create and train an anomaly detection model for the specified service set

    Args:
        active_set: Name of the service set to train the model for

    Returns:
        Trained AnomalyDetector instance, or None if training fails
    """

    age_latest_data = get_timestamp_of_latest_data(active_set)

    logger.info(
        Colors.blue(
            f"\n⏱️  MODEL CREATION PROCESS STARTED FOR SERVICE SET: {active_set}. TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    )

    run_name = f"Model training: {active_set}-microservice-set"
    with mlflow.start_run(run_name=run_name, log_system_metrics=True):
        try:
            if DOWNLOAD_ENABLED:
                download_splunk_data("training", active_set, age_latest_data)
                download_prometheus_data("training", active_set)
                combine_services("training", active_set, age_latest_data)
                logger.info("Data download and combination completed successfully.")
            else:
                logger.warning(
                    "DOWNLOAD env var is set to false, skipping data download"
                )
        except Exception as e:
            logger.error(
                f"An error occurred during training data download: {e}", exc_info=True
            )
            mlflow.log_param("training_error", str(e))
            return None

        training_dataset_path = f"output/{active_set}/training_dataset.json"
        if os.path.exists(training_dataset_path):
            mlflow.log_artifact(training_dataset_path, artifact_path="data")

        df = get_microservice_data_from_file(training_dataset_path)

        # Check for NaN values in data
        df = check_for_nan(df)

        data, services, features = convert_to_model_input(active_set, df)
        timepoints = get_ordered_timepoints(df)

        logger.info("\nDataset details:")
        logger.info(f"Number of services: {len(services)}")
        logger.info(f"Features per service: {features}")
        logger.info(f"Total timepoints: {len(data)}")

        # Split data into train/validation sets (70/30 split)
        train_size = int(0.7 * len(data))
        val_size = len(data) - train_size

        train_data = data[:train_size]
        val_data = data[train_size:]

        logger.info(
            f"Data split: {train_size} training samples, {val_size} validation samples"
        )

        config = conf.get_config(active_set)

        # Create and train detector
        logger.info("\nInitializing anomaly detector...")
        detector = AnomalyDetector(
            num_services=len(services),
            num_features=len(features),
            window_size=config.window_size,
            config=config,
        )

        logger.info("Training model...")
        detector.train(
            train_data,
            val_data,
            df,
            active_set,
            max_epochs=MAX_EPOCHS,
            timepoints=timepoints,
        )

        logger.info("Setting threshold using validation data...")
        detector.set_threshold(
            val_data,
            timepoints=timepoints[train_size:],
            percentile=config.anomaly_threshold_percentile,
        )

        # Ensure the final model checkpoint exists and is logged (detector.train logs best checkpoint).
        model_filename = f"output/{active_set}/best_model_{active_set}.pth"
        if os.path.exists(model_filename):
            mlflow.log_artifact(model_filename, artifact_path="model")

        logger.info(
            Colors.blue(
                f"✅ MODEL CREATION PROCESS FINISHED FOR SERVICE SET: {active_set}."
            )
        )
        logger.info(
            Colors.blue(f"TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        )
        return detector
