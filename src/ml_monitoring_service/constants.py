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
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Data download configuration
DOWNLOAD_ENABLED = os.getenv('DOWNLOAD', 'true').lower() != 'false'

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000/')

# Prometheus configuration
PROMETHEUS_URL = os.getenv('PROMETHEUS_URL')

# Splunk configuration
SPLUNK_URL = os.getenv('SPLUNK_URL')
SPLUNK_AUTH_TOKEN = os.getenv('SPLUNK_AUTH_TOKEN')
SPLUNK_HEC_TOKEN = os.getenv('SPLUNK_HEC_TOKEN')

# Data subset/sampling configuration
DATA_SUBSET = os.getenv('SUBSET')
DATA_SAMPLING = os.getenv('SAMPLING')


# ============================================================================
# ANSI Color Codes
# ============================================================================

class Colors:
    """ANSI color codes for terminal output formatting"""
    BLUE = '\033[94m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    
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
REQUEST_TIMEOUT = 10

# ============================================================================
# Training config
# ============================================================================

# Max number of epochs the training goes on for. Might stop earlier because of early-stopping mechanism
MAX_EPOCHS = int(os.getenv('MAX_EPOCHS', 20))
