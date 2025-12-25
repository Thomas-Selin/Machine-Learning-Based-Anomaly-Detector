"""Utility functions and classes for ML monitoring service."""

import logging


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds color to log level names."""

    COLORS = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[91m",  # Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with color coding."""
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def get_gpu_memory_usage() -> str | None:
    """Get current GPU memory usage if GPU is available.

    Returns:
        String representation of GPU memory usage (e.g., "1.23 GB") or None if no GPU
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        # Get memory allocated on the current GPU device
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
    except ImportError:
        return None
