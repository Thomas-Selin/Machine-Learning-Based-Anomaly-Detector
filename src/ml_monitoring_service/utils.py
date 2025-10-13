import os
import logging
import psutil

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds color to log level names
    
    Uses ANSI escape codes to colorize log levels for better readability in terminals.
    """
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[91m',  # Red
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[91m', # Red
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)

class ConditionalFormatter(ColoredFormatter):
    """Formatter that ensures consistent timestamp formatting across all log levels"""
    
    def format(self, record):
        # Always show timestamp before log level for all log levels
        self._style._fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        return super().format(record)

def get_memory_usage() -> str:
    """Get current memory usage of Python process
    
    Returns:
        String representation of memory usage in MB (e.g., "123.45 MB")
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Convert to MB for readability
    memory_mb = memory_info.rss / 1024 / 1024
    return f"{memory_mb:.2f} MB"

class HealthCheckFilter(logging.Filter):
    """Logging filter that suppresses health check endpoint logs to reduce noise"""
    
    def filter(self, record):
        return '/ml-based-anomaly-detector/admin/healthcheck' not in record.getMessage()