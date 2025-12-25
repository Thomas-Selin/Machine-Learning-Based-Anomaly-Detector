"""
Constants and configuration values used throughout the ML Based Anomaly Detector.

This module centralizes environment variables and magic numbers to improve
maintainability and provide a single source of truth.
"""

import os

# ============================================================================
# Environment Variables
# ============================================================================

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Data download configuration
DOWNLOAD_ENABLED = os.getenv("DOWNLOAD", "true").lower() != "false"

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000/")
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "ML Based Anomaly Detector"
)

# Prometheus configuration
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL")

# Splunk configuration
SPLUNK_URL = os.getenv("SPLUNK_URL")
SPLUNK_AUTH_TOKEN = os.getenv("SPLUNK_AUTH_TOKEN")
SPLUNK_HEC_TOKEN = os.getenv("SPLUNK_HEC_TOKEN")

# Data subset/sampling configuration for testing/development
DATA_SUBSET = os.getenv("DATA_SUBSET")
DATA_SAMPLING = os.getenv("DATA_SAMPLING")

# Training configuration
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", "100"))

# Active service sets configuration
# Comma-separated list of service sets to activate (e.g., "default,transfer")
# If not set, all available service sets from service_sets.yaml will be activated
ACTIVE_SERVICE_SETS = os.getenv("ACTIVE_SERVICE_SETS")


# ============================================================================
# ANSI Color Codes
# ============================================================================


class Colors:
    """ANSI color codes for terminal output formatting"""

    BLUE = "\033[94m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    @classmethod
    def blue(cls, text: str) -> str:
        """Wrap text in blue color"""
        return f"{cls.BLUE}{text}{cls.RESET}"

    @classmethod
    def red(cls, text: str) -> str:
        """Wrap text in red color"""
        return f"{cls.RED}{text}{cls.RESET}"

    @classmethod
    def green(cls, text: str) -> str:
        """Wrap text in green color"""
        return f"{cls.GREEN}{text}{cls.RESET}"

    @classmethod
    def yellow(cls, text: str) -> str:
        """Wrap text in yellow color"""
        return f"{cls.YELLOW}{text}{cls.RESET}"


# ============================================================================
# Scheduler Configuration
# ============================================================================

# Delay offset in minutes between inference jobs for different service sets
# This prevents all inference jobs from starting simultaneously
INFERENCE_DELAY_OFFSET_MINUTES = 4

# Default training interval in minutes
DEFAULT_TRAINING_INTERVAL_MINUTES = 45


# ============================================================================
# HTTP Configuration
# ============================================================================

# Request timeout for external API calls (in seconds)
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# Prometheus-specific timeout (for long-running queries)
PROMETHEUS_TIMEOUT = int(os.getenv("PROMETHEUS_TIMEOUT", "900"))

# Splunk-specific timeout
SPLUNK_TIMEOUT = int(os.getenv("SPLUNK_TIMEOUT", "30"))

# Whether requests should verify TLS certificates (recommended true in production).
# Set REQUESTS_VERIFY=false for local testing with self-signed certs.
REQUESTS_VERIFY = os.getenv("REQUESTS_VERIFY", "true").lower() != "false"

# Recompute anomaly threshold from inference data (not recommended by default).
RECALCULATE_THRESHOLD_ON_INFERENCE = (
    os.getenv("RECALCULATE_THRESHOLD_ON_INFERENCE", "false").lower() == "true"
)

# ============================================================================
# Training config
# ============================================================================

# Max number of epochs the training goes on for. Might stop earlier because of early-stopping mechanism
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", 20))

# ============================================================================
# Scheduler config
# ============================================================================

# Maximum number of concurrent job instances
SCHEDULER_MAX_INSTANCES = int(os.getenv("SCHEDULER_MAX_INSTANCES", "1"))

# Number of worker threads for background scheduler
SCHEDULER_MAX_WORKERS = int(os.getenv("SCHEDULER_MAX_WORKERS", "5"))

# Job coalescing: skip jobs if previous run is still active
SCHEDULER_COALESCE = os.getenv("SCHEDULER_COALESCE", "true").lower() == "true"

# Maximum job execution time in seconds (1 hour default)
SCHEDULER_MAX_EXECUTION_TIME = int(os.getenv("SCHEDULER_MAX_EXECUTION_TIME", "3600"))
