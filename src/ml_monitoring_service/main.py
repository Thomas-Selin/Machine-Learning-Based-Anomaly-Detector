import logging
import os
import platform
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import mlflow
import torch
from apscheduler.schedulers.background import BackgroundScheduler

# Load environment variables BEFORE importing any modules that use them
from dotenv import find_dotenv, load_dotenv
from flask import Flask, Response, jsonify
from waitress import serve

load_dotenv(find_dotenv())

import ml_monitoring_service.configuration as conf
from ml_monitoring_service.anomaly_analyser import analyse_anomalies
from ml_monitoring_service.anomaly_detector import AnomalyDetector
from ml_monitoring_service.constants import (
    DEFAULT_TRAINING_INTERVAL_MINUTES,
    DOWNLOAD_ENABLED,
    INFERENCE_DELAY_OFFSET_MINUTES,
    LOG_LEVEL,
    MLFLOW_EXPERIMENT_NAME,
    RECALCULATE_THRESHOLD_ON_INFERENCE,
    Colors,
)
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
from ml_monitoring_service.model_building import create_and_train_model
from ml_monitoring_service.utils import (
    ConditionalFormatter,
    HealthCheckFilter,
    get_memory_usage,
)
from ml_monitoring_service.visualisation import visualize_microservice_graph

handler = logging.StreamHandler()
handler.setFormatter(ConditionalFormatter(datefmt="%H:%M"))
logging.basicConfig(level=getattr(logging, LOG_LEVEL), handlers=[handler])
logger = logging.getLogger(__name__)


def _try_restore_model_from_mlflow(active_set: str, model_filename: str) -> bool:
    """Try to download the latest trained model checkpoint from MLflow to a local cache path.

    Returns:
        True if a checkpoint was downloaded to model_filename, else False.
    """
    try:
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            logger.info(
                f"MLflow experiment '{MLFLOW_EXPERIMENT_NAME}' not found; cannot restore model."
            )
            return False

        run_name = f"Model training: {active_set}-microservice-set"
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            logger.info(f"No MLflow training runs found for '{active_set}'.")
            return False

        run_id = runs[0].info.run_id
        artifact_rel = f"model/best_model_{active_set}.pth"
        downloaded = client.download_artifacts(run_id=run_id, path=artifact_rel)

        Path(model_filename).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded, model_filename)
        logger.info(
            f"Restored model checkpoint from MLflow run {run_id} to {model_filename}"
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to restore model from MLflow: {e}")
        return False


def load_model(
    model_filename: str,
    num_features: int,
    num_services: int,
    config: conf.ServiceSetConfig,
    override_threshold: float | None = None,
) -> AnomalyDetector:
    """Load a trained model from disk

    Args:
        model_filename: Path to the saved model file
        num_features: Number of features per service
        num_services: Number of services
        config: Configuration object
        override_threshold: Optional threshold to override the saved one (useful for testing different sensitivity levels)

    Returns:
        Loaded AnomalyDetector instance

    Raises:
        FileNotFoundError: If model file doesn't exist
        KeyError: If checkpoint is missing required keys
    """
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file not found: {model_filename}")

    logger.info(f"Memory usage before loading model: {get_memory_usage()}")

    # Security note: torch.load with weights_only=False can execute arbitrary code.
    # Only load models from trusted sources (your own training runs or verified artifacts).
    # For additional safety in production, consider:
    # - Validating checkpoint signatures/hashes before loading
    # - Using weights_only=True if model architecture is known
    # - Sandboxing model loading in isolated environments
    try:
        checkpoint = torch.load(model_filename, weights_only=False, map_location="cpu")

        # Validate checkpoint structure
        required_keys = {
            "model_state_dict",
            "threshold",
            "num_services",
            "num_features",
            "window_size",
        }
        missing_keys = required_keys - set(checkpoint.keys())
        if missing_keys:
            raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

        window_size = checkpoint.get("window_size", config.window_size)

        detector = AnomalyDetector(
            num_services=num_services,
            num_features=num_features,
            window_size=window_size,
            config=config,
        )

        logger.debug(f"Checkpoint keys: {list(checkpoint.keys())}")
        detector.model.load_state_dict(checkpoint["model_state_dict"])

        # Use override threshold if provided, otherwise use saved threshold
        if override_threshold is not None:
            import numpy as np

            detector.threshold = np.float64(override_threshold)
            logger.info(
                f"Using override threshold: {override_threshold:.6f} (saved threshold was: {checkpoint['threshold']:.6f})"
            )
        else:
            detector.threshold = checkpoint["threshold"]
            logger.info(f"Using saved threshold: {detector.threshold:.6f}")

        detector.model.to(detector.device)

        logger.info(
            f"Memory usage after loading model '{model_filename}': {get_memory_usage()}"
        )
        return detector
    except KeyError as e:
        logger.error(f"Checkpoint missing required key: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Failed to load model from {model_filename}: {e}", exc_info=True)
        raise


def inference(active_set: str, model_filename: str) -> None:
    """Run inference on latest data to detect anomalies

    Args:
        active_set: Name of the service set to run inference on
        model_filename: Path to the trained model file
    """
    # Ensure model is available locally (output/ is treated as a cache; MLflow is the source of truth).
    if not os.path.exists(model_filename):
        _try_restore_model_from_mlflow(active_set, model_filename)

    if not os.path.exists(model_filename):
        logger.warning(
            f"No trained model found at {model_filename}. Skipping inference."
        )
        return

    config = conf.get_config(active_set)
    age_latest_data = get_timestamp_of_latest_data(active_set)

    with mlflow.start_run(run_name=f"Model inference: {active_set}-microservice-set"):
        # Log start time
        start_time = datetime.now()
        mlflow.log_param("inference_start_time", start_time.isoformat())
        mlflow.log_param("active_set", active_set)
        logger.info(
            Colors.blue(
                f"⏱️  INFERENCE PROCESS STARTED FOR SERVICE SET: {active_set}. TIME: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        )

        try:
            if DOWNLOAD_ENABLED:
                download_splunk_data("inference", active_set, age_latest_data)
                download_prometheus_data("inference", active_set)
                combine_services("inference", active_set, age_latest_data)
                logger.info("Data download and combination completed successfully.")
            else:
                logger.warning(
                    "DOWNLOAD env var is set to false, skipping data download"
                )
        except Exception as e:
            logger.error(f"An error occurred during data download: {e}", exc_info=True)
            mlflow.log_param("inference_error", str(e))
            return

        # Read sample data from file
        logger.info("Reading sample data from file...")
        inference_dataset_path = f"output/{active_set}/inference_dataset.json"
        df = get_microservice_data_from_file(inference_dataset_path)
        if mlflow.active_run() and os.path.exists(inference_dataset_path):
            mlflow.log_artifact(inference_dataset_path, artifact_path="data")

        # Check for NaN values in data
        df = check_for_nan(df)

        data, services, features = convert_to_model_input(active_set, df)
        timepoints = get_ordered_timepoints(df)

        logger.info("\nDataset details:")
        logger.info(f"Number of services: {len(services)}")
        logger.info(f"Features per service: {features}")
        logger.info(f"Total timepoints: {len(data)}")

        # Load the model
        num_features = len(features)
        num_services = len(services)

        # For testing/debugging: You can manually set a very low threshold to force anomaly detection
        # Set to None to use the saved threshold from training
        # Example: override_threshold = 0.0001  # Very low threshold for testing
        override_threshold = None  # Use saved threshold

        detector = load_model(
            model_filename, num_features, num_services, config, override_threshold
        )

        # Option: Recalculate threshold if you want to use a different percentile without retraining
        # This will use the inference data to recalculate the threshold based on the current percentile setting
        if RECALCULATE_THRESHOLD_ON_INFERENCE:
            logger.warning(
                "Recalculating threshold on inference data is enabled. "
                "This can reduce sensitivity to real anomalies if the inference window contains them."
            )
            logger.info(
                f"Recalculating threshold with percentile={config.anomaly_threshold_percentile}"
            )
            detector.set_threshold(
                data,
                timepoints=timepoints,
                percentile=int(config.anomaly_threshold_percentile),
            )
            logger.info(f"New threshold: {detector.threshold}")
        else:
            logger.info(
                f"Using persisted threshold {detector.threshold:.6f}. "
                "(Set RECALCULATE_THRESHOLD_ON_INFERENCE=true to recompute on inference data.)"
            )

        # Perform inference
        logger.info("Running anomaly detection...")
        service_errors = {service: 0 for service in services}
        timestamps = timepoints
        num_anomalies = 0
        num_windows_processed = 0

        # Calculate how many windows we'll process
        total_possible_windows = len(data) - detector.window_size
        if total_possible_windows <= 0:
            logger.warning(
                f"Not enough data for inference. Data length: {len(data)}, Window size: {detector.window_size}"
            )
            mlflow.log_param("inference_error", "Insufficient data for windowing")
            return

        logger.info(
            f"Processing {total_possible_windows} possible windows with step size {detector.window_size}"
        )
        logger.info(f"Anomaly threshold: {detector.threshold}")

        for i in range(0, len(data) - detector.window_size, detector.window_size):
            inference_window = data[i : i + detector.window_size]
            timestamp = timestamps[i]
            result = detector.detect(inference_window, timestamp)
            num_windows_processed += 1

            # Log each window's result for debugging
            logger.debug(
                f"Window {num_windows_processed}: Error score: {result['error_score']:.6f}, Threshold: {result['threshold']:.6f}, Is anomaly: {result['is_anomaly']}"
            )

            if result["is_anomaly"]:
                logger.warning(
                    f"🚨 ANOMALY DETECTED in window {num_windows_processed} at timestamp {timestamp}"
                )
                logger.warning(
                    f"   Error score: {result['error_score']:.6f} (threshold: {result['threshold']:.6f})"
                )
                explanations = analyse_anomalies(
                    result, config.relationships, services, features
                )
                logger.info("\n".join(explanations))
                num_anomalies += 1

            # Aggregate service errors
            for service, error in zip(services, result["service_errors"], strict=False):
                service_errors[service] += error

        # Log summary of detection
        if num_anomalies == 0:
            logger.info(
                f"✅ No anomalies detected in {num_windows_processed} windows processed."
            )
            logger.info(
                f"   All error scores were below threshold of {detector.threshold:.6f}"
            )
        else:
            logger.warning(
                f"⚠️  Detected {num_anomalies} anomalies out of {num_windows_processed} windows processed ({100 * num_anomalies / num_windows_processed:.1f}%)"
            )

        # Log inference metrics
        end_time = datetime.now()
        inference_duration = (end_time - start_time).total_seconds()
        mlflow.log_metric("inference_duration_seconds", inference_duration)
        mlflow.log_metric("num_windows_processed", num_windows_processed)
        mlflow.log_metric("num_anomalies_detected", num_anomalies)
        mlflow.log_metric("anomaly_rate", num_anomalies / max(1, num_windows_processed))

        # You could also log a summary of service errors
        for service, error in service_errors.items():
            mlflow.log_metric(f"service_error_{service}", error)

        logger.info(
            Colors.blue(f"✅ INFERENCE PROCESS FINISHED FOR SERVICE SET: {active_set}.")
        )
        logger.info(
            Colors.blue(f"TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        )

        # Visualize the microservice graph with error information
        logger.info("Visualizing microservice graph...")
        result_graph_path = f"output/{active_set}/microservice_graph_with_results.png"
        visualize_microservice_graph(
            config.relationships,
            result_graph_path,
            {k: float(v) for k, v in service_errors.items()},
        )
        if os.path.exists(result_graph_path):
            mlflow.log_artifact(result_graph_path)


def create_app() -> Flask:
    """Create and configure Flask application

    Returns:
        Configured Flask app instance
    """
    app = Flask(__name__)
    health_check_filter = HealthCheckFilter()

    # Add filter to the root logger
    for handler in logging.getLogger().handlers:
        handler.addFilter(health_check_filter)

    app.logger.addFilter(health_check_filter)

    @app.route("/ml-based-anomaly-detector/admin/healthcheck", methods=["GET"])
    def health() -> Response:
        return jsonify(status="up")

    @app.route("/ml-based-anomaly-detector/version", methods=["GET"])
    def version() -> str:
        return "Python Version: " + sys.version

    return app


if __name__ == "__main__":
    conf.set_active_sets(conf.get_available_sets())
    active_sets = conf.get_active_sets()

    flask_app = create_app()

    if torch.cuda.is_available():
        processor = "GPU (Cuda) 🤓"
    else:
        if (
            platform.system() == "Darwin"
            and platform.machine().startswith("arm")
            and torch.backends.mps.is_available()
        ):
            processor = "MacBook Metal acceleration 🍏"
        else:
            processor = "CPU 🤬"
    logger.info(f"Processor: {processor}")

    # Schedule the model creation task
    scheduler = BackgroundScheduler(
        job_defaults={"max_instances": 1},
        executors={"default": {"type": "threadpool", "max_workers": 5}},
    )
    logger.info("Scheduler created")

    # Schedule the model creation and inference tasks for each active set
    for active_set in active_sets:
        # Retrieve the entire configuration object for a specific service set
        service_set_config = conf.get_config(active_set)

        training_lookback_hours = conf.get_training_lookback_hours(active_set)
        inference_lookback_minutes = conf.get_inference_lookback_minutes(active_set)

        # Access attributes of the configuration object
        logger.info(f"---------------Config for {active_set} service set:")
        logger.info(f"Relationships: {service_set_config.relationships}")
        logger.info(f"Metrics: {service_set_config.metrics}")
        logger.info(
            f"Training Lookback Hours: {service_set_config.training_lookback_hours}"
        )
        logger.info(
            f"Inference Lookback Minutes: {service_set_config.inference_lookback_minutes}"
        )
        logger.info(f"Window Size: {service_set_config.window_size}")
        logger.info(
            f"Anomaly Threshold Percentile: {service_set_config.anomaly_threshold_percentile}"
        )
        logger.info(f"Training Lookback Hours: {training_lookback_hours}")
        logger.info(f"Inference Lookback Minutes: {inference_lookback_minutes}")
        logger.info(f"---------------End of config for {active_set} service set.")

        # Schedule the model creation task (every 45 minutes)
        scheduler.add_job(
            name=f"training {active_set}",
            func=create_and_train_model,
            trigger="interval",
            next_run_time=datetime.now(),
            minutes=DEFAULT_TRAINING_INTERVAL_MINUTES,
            args=[active_set],
        )

        # Schedule the inference task
        model_filename = f"output/{active_set}/best_model_{active_set}.pth"
        delay = INFERENCE_DELAY_OFFSET_MINUTES
        scheduler.add_job(
            name=f"inference {active_set}",
            func=inference,
            trigger="interval",
            next_run_time=datetime.now() + timedelta(minutes=delay),
            minutes=inference_lookback_minutes,
            args=[active_set, model_filename],
        )
        delay += INFERENCE_DELAY_OFFSET_MINUTES
        logger.info(f"Training and inference scheduled for '{active_set}' service set.")

    scheduler.start()

    try:
        logger.info("Starting production WSGI server on 0.0.0.0:8080")
        serve(flask_app, host="0.0.0.0", port=8080)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
    logger.info("App stopped running")
