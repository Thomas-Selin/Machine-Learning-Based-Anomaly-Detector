import copy
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import ml_monitoring_service.configuration as conf
from ml_monitoring_service.data_handling import ServiceMetricsDataset
from ml_monitoring_service.model import HybridAutoencoderTransformerModel

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Anomaly detector for microservice metrics using transformer-based autoencoder

    Attributes:
        num_services: Number of microservices to monitor
        num_features: Number of features per service
        window_size: Size of the sliding window for detection
        config: Configuration object with service relationships and settings
        device: PyTorch device (cuda/cpu)
        model: The transformer-based autoencoder model
        threshold: Anomaly detection threshold
    """

    def __init__(
        self,
        num_services: int,
        num_features: int,
        window_size: int,
        config: Any = None,
    ) -> None:
        """Initialize anomaly detector.

        Args:
            num_services: Number of services to monitor
            num_features: Number of features per service
            window_size: Size of sliding window for detection
            config: Optional configuration object
        """
        self.num_services = num_services
        self.num_features = num_features
        self.window_size = window_size
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HybridAutoencoderTransformerModel(
            num_services=num_services,
            num_features=num_features,
        ).to(self.device)
        self.threshold: float | None = None

        logger.info(f"Initialized AnomalyDetector on device: {self.device}")

    def train(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray,
        df: Any,
        active_set: str,
        max_epochs: int,
        timepoints: list[Any] | np.ndarray,
        batch_size: int = 32,
        patience: int = 15,
        *,
        save_checkpoint: bool = True,
    ) -> None:
        """Train the anomaly detection model.

        Args:
            train_data: Training data array
            val_data: Validation data array
            df: DataFrame with training data
            active_set: Name of the service set
            max_epochs: Maximum number of training epochs
            timepoints: List or ndarray of timepoints
            batch_size: Batch size for training
            patience: Early stopping patience
        """
        # Check if datasets are large enough
        min_samples = max(self.window_size + 1, batch_size)
        if len(train_data) <= min_samples:
            logger.error(
                f"Training data too small ({len(train_data)} samples) for window_size={self.window_size} and batch_size={batch_size}"
            )
            return

        if len(val_data) <= self.window_size:
            logger.error(
                f"Validation data too small ({len(val_data)} samples) for window_size={self.window_size}"
            )
            return

        logger.info(
            f"Training with {len(train_data)} training samples and {len(val_data)} validation samples"
        )

        # MLflow logging (always enabled). Reuse an existing run if present.
        import tempfile

        import mlflow
        import mlflow.pytorch

        mlflow.pytorch.autolog(log_models=False, log_every_n_epoch=1)  # type: ignore[attr-defined]

        run_ctx = nullcontext()
        if mlflow.active_run() is None:
            run_ctx = mlflow.start_run(
                run_name=f"Model training: {active_set}-microservice-set",
                log_system_metrics=True,
            )

        with run_ctx:
            # Log custom parameters
            mlflow.log_param("num_services", self.num_services)
            mlflow.log_param("num_features", self.num_features)
            mlflow.log_param("window_size", self.window_size)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("max_epochs", max_epochs)

            # Log training/val data arrays and metadata directly to MLflow using temp files
            if mlflow.active_run():
                with tempfile.TemporaryDirectory() as tmpdir:
                    train_data_path = Path(tmpdir) / "train_data.npy"
                    val_data_path = Path(tmpdir) / "val_data.npy"
                    service_names_path = Path(tmpdir) / "service_names.txt"

                    np.save(train_data_path, train_data)
                    np.save(val_data_path, val_data)
                    service_names_path.write_text(
                        "\n".join(conf.config.get_services(active_set))
                    )

                    mlflow.log_artifact(str(train_data_path), artifact_path="data")
                    mlflow.log_artifact(str(val_data_path), artifact_path="data")
                    mlflow.log_artifact(str(service_names_path), artifact_path="data")

            # Also log a simple, always-viewable architecture snapshot as an image.
            # MLflow doesn't have a dedicated "model graph" viewer, but it can display image artifacts.
            try:
                import matplotlib.pyplot as plt

                model_text = (
                    "HybridAutoencoderTransformerModel\n"
                    f"input: (batch, window={self.window_size}, services={self.num_services}, features={self.num_features})\n\n"
                    + repr(self.model)
                )
                # Keep the artifact reasonably sized.
                model_lines = model_text.splitlines()
                if len(model_lines) > 400:
                    model_text = "\n".join(
                        model_lines[:220]
                        + ["", "... (truncated) ...", ""]
                        + model_lines[-120:]
                    )
                lines = model_text.count("\n") + 1
                fig_height = min(30.0, max(6.0, lines * 0.20))
                fig_width = 14.0

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir_p = Path(tmpdir)
                    txt_path = tmpdir_p / "model_architecture.txt"
                    png_path = tmpdir_p / "model_architecture.png"

                    txt_path.write_text(model_text)

                    fig = plt.figure(figsize=(fig_width, fig_height))
                    fig.text(
                        0.01,
                        0.99,
                        model_text,
                        va="top",
                        ha="left",
                        family="monospace",
                        fontsize=8,
                    )
                    plt.axis("off")
                    fig.savefig(png_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)

                    mlflow.log_artifact(
                        str(txt_path), artifact_path="model/architecture"
                    )
                    mlflow.log_artifact(
                        str(png_path), artifact_path="model/architecture"
                    )
                    mlflow.set_tag("model.architecture_image", "true")
            except Exception as e:
                logger.warning(f"Failed to export model architecture image: {e}")

            # Manual diagram bundle (overview + deep dives) as SVG artifacts.
            # These are intentionally hand-crafted; update if the architecture changes.
            try:
                from ml_monitoring_service.architecture_diagrams import (
                    write_model_architecture_diagrams,
                )

                with tempfile.TemporaryDirectory() as tmpdir:
                    out_dir = Path(tmpdir)
                    write_model_architecture_diagrams(
                        model=self.model,
                        num_services=self.num_services,
                        num_features=self.num_features,
                        window_size=self.window_size,
                        out_dir=out_dir,
                    )
                    mlflow.log_artifacts(
                        str(out_dir), artifact_path="model/architecture/diagrams"
                    )
                    mlflow.set_tag("model.architecture_diagrams", "true")
            except Exception as e:
                logger.warning(f"Failed to export manual architecture diagrams: {e}")

            # Align timestamps to the model time axis (unique ordered timepoints)
            train_size = len(train_data)
            val_size = len(val_data)
            if len(timepoints) < (train_size + val_size):
                raise ValueError(
                    f"Not enough timepoints ({len(timepoints)}) for train_size={train_size} and val_size={val_size}"
                )

            train_timestamps = pd.Series(timepoints[:train_size])
            val_timestamps = pd.Series(timepoints[train_size : train_size + val_size])

            dataset = ServiceMetricsDataset(
                train_data, train_timestamps, self.window_size
            )
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            val_dataset = ServiceMetricsDataset(
                val_data, val_timestamps, self.window_size
            )
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=patience
            )

            best_loss = float("inf")
            patience_counter = 0
            best_model_state = None

            self.model.train()
            for epoch in range(max_epochs):
                total_loss = 0
                self.model.train()
                for batch_x, batch_y, batch_timestamps in dataloader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(batch_x, batch_timestamps)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    optimizer.step()
                    total_loss += loss.item()

                avg_train_loss = total_loss / len(dataloader)

                # Validation loss
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y, batch_timestamps in val_dataloader:
                        batch_x, batch_y = (
                            batch_x.to(self.device),
                            batch_y.to(self.device),
                        )
                        output = self.model(batch_x, batch_timestamps)
                        loss = criterion(output, batch_y)
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_dataloader)

                # Log metrics manually in case autolog misses something
                mlflow.log_metrics(
                    {"train_loss": avg_train_loss, "val_loss": avg_val_loss},
                    step=epoch,
                )

                logger.info(
                    f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
                )

                # Check for early stopping
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0
                    # Save the best model state in memory
                    # Deep-copy state dict so later training steps can't mutate it.
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    mlflow.log_metric("best_val_loss", avg_val_loss)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info("Early stopping triggered")
                        break

                # Step the scheduler
                scheduler.step(avg_val_loss)

            # Load best model state after training
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
                logger.info(f"Loaded best model with validation loss: {best_loss:.6f}")

                # Set threshold before saving (use aligned timepoints for correct time features)
                self.set_threshold(val_data, timepoints=val_timestamps)

                if save_checkpoint:
                    # Generate model version (timestamp-based for uniqueness)
                    from datetime import datetime

                    model_version = datetime.now().strftime("%Y%m%d_%H%M%S")

                    mlflow.set_tag("model_version", model_version)
                    mlflow.set_tag("service_set", active_set)

                    # Save model checkpoint to temp file (optionally log)
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        mode="wb", suffix=".pth", delete=False
                    ) as tmp:
                        torch.save(
                            {
                                "model_state_dict": best_model_state,
                                "threshold": self.threshold,
                                "num_services": self.num_services,
                                "num_features": self.num_features,
                                "window_size": self.window_size,
                                "model_version": model_version,
                                "training_date": datetime.now().isoformat(),
                            },
                            tmp,
                        )
                        temp_model_path = tmp.name

                    # Also save to output/ as a local cache for convenience
                    model_path = f"output/{active_set}/best_model_{active_set}.pth"
                    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                    import shutil

                    shutil.copy(temp_model_path, model_path)
                    logger.info(
                        f"Best model saved to {model_path} (local cache) with threshold: {self.threshold:.6f}"
                    )

                    # Log model artifact to MLflow with versioning
                    mlflow.log_artifact(temp_model_path, artifact_path="model")

                    # Clean up temp file
                    Path(temp_model_path).unlink()

            # Log the already-computed threshold
            if self.threshold is not None:
                mlflow.log_metric("anomaly_threshold", float(self.threshold))
                logger.info(f"Anomaly detection threshold set to: {self.threshold:.6f}")
            logger.info(f"\n{self.model}\n")

    def set_threshold(
        self,
        validation_data: np.ndarray,
        df: Any = None,
        timepoints: list[Any] | pd.Series | np.ndarray | None = None,
        percentile: int = 99,
    ) -> None:
        """Set anomaly threshold based on validation data.

        Args:
            validation_data: Validation dataset
            df: Optional DataFrame with data
            timepoints: Optional list, Series, or ndarray of timepoints
            percentile: Percentile for threshold calculation
        """
        self.model.eval()

        if len(validation_data) <= self.window_size:
            logger.warning(
                "Validation dataset is too small for windowing "
                f"(len={len(validation_data)} <= window_size={self.window_size}). "
                "Computing threshold from a single padded window."
            )

        if timepoints is not None:
            val_timestamps = pd.to_datetime(timepoints).to_numpy()
        elif df is not None:
            # Best-effort fallback: derive from df's ordered unique timepoints
            from ml_monitoring_service.data_handling import get_ordered_timepoints

            ordered = get_ordered_timepoints(df)
            val_timestamps = pd.to_datetime(ordered[: len(validation_data)]).to_numpy()
        else:
            # Fallback to generating synthetic timestamps
            logger.warning(
                "No timestamps provided for thresholding; using synthetic timestamps"
            )
            val_timestamps = pd.date_range(
                start=pd.Timestamp("2025-03-27"),
                periods=len(validation_data),
                freq="1min",
            ).values

        # If we can't create at least one window, pad up to window_size.
        if len(validation_data) <= self.window_size:
            pad_len = self.window_size - len(validation_data)
            if pad_len > 0:
                validation_data = np.pad(
                    validation_data,
                    pad_width=((0, pad_len), (0, 0), (0, 0)),
                    mode="edge",
                )
                if len(val_timestamps) > 0:
                    val_timestamps = np.concatenate(
                        [val_timestamps, np.repeat(val_timestamps[-1], pad_len)]
                    )
            # Use a batch of one window and compute errors directly.
            with torch.no_grad():
                x = torch.FloatTensor(validation_data[: self.window_size]).unsqueeze(0)
                x = x.to(self.device)
                ts = torch.FloatTensor(
                    ServiceMetricsDataset(
                        validation_data,
                        pd.Series(val_timestamps),
                        self.window_size,
                    )[0][2]
                ).unsqueeze(0)
                out = self.model(x, ts)
                error = torch.mean((x - out) ** 2, dim=(2, 3))
                self.threshold = float(np.percentile(error.cpu().numpy(), percentile))
            return

        dataset = ServiceMetricsDataset(
            validation_data, pd.Series(val_timestamps), self.window_size
        )
        dataloader = DataLoader(dataset, batch_size=32)

        reconstruction_errors = []
        with torch.no_grad():
            for batch_x, _, batch_timestamps in dataloader:  # Unpack 3 values
                batch_x = batch_x.to(self.device)
                output = self.model(batch_x, batch_timestamps)  # Pass timestamps
                error = torch.mean((batch_x - output) ** 2, dim=(2, 3))
                reconstruction_errors.extend(error.cpu().numpy())

        self.threshold = float(
            np.percentile(np.array(reconstruction_errors), percentile)
        )

    def detect(
        self, metrics_window: np.ndarray, window_timepoints: list[Any] | np.ndarray
    ) -> dict[str, Any]:
        """Detect anomalies in current metrics window.

        Args:
            metrics_window: Window of metrics to analyze
            window_timepoints: Sequence of timestamps for the window (length=window_size)

        Returns:
            Dictionary with detection results
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(metrics_window).unsqueeze(0).to(self.device)

            timestamps = pd.to_datetime(window_timepoints).to_list()
            if len(timestamps) != self.window_size:
                raise ValueError(
                    f"window_timepoints must have length window_size={self.window_size}, got {len(timestamps)}"
                )

            # Extract time features (hours, minutes, day_of_the_week, seconds)
            hours = (
                torch.tensor([ts.hour / 24.0 for ts in timestamps]).float().unsqueeze(1)
            )
            minutes = (
                torch.tensor([ts.minute / 60.0 for ts in timestamps])
                .float()
                .unsqueeze(1)
            )
            days = (
                torch.tensor([ts.dayofweek / 7.0 for ts in timestamps])
                .float()
                .unsqueeze(1)
            )
            seconds = (
                torch.tensor([ts.second / 60.0 for ts in timestamps])
                .float()
                .unsqueeze(1)
            )

            # Stack time features [seq_len, 4] and add batch dimension
            time_features = torch.cat([hours, minutes, days, seconds], dim=1).unsqueeze(
                0
            )

            # Now pass both x and time_features to the model
            output = self.model(x, time_features)

            # Rest of the method remains the same
            error = torch.mean((x - output) ** 2, dim=(2, 3))
            error_scalar = torch.mean(error).item()
            service_errors = torch.mean((x - output) ** 2, dim=(0, 1, 3)).cpu().numpy()
            variable_errors = torch.mean((x - output) ** 2, dim=(0, 1, 2)).cpu().numpy()

            if self.threshold is None:
                raise ValueError(
                    "Threshold not set. Call set_threshold() before detect()."
                )

            is_anomaly = error_scalar > self.threshold

            return {
                "is_anomaly": bool(is_anomaly),
                "error_score": error_scalar,
                "threshold": float(self.threshold),
                "service_errors": service_errors,
                "variable_errors": variable_errors,
                "timestamp": str(pd.to_datetime(timestamps[0])),
            }
