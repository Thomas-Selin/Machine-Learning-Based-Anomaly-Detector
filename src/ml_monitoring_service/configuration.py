from dataclasses import dataclass
from datetime import timedelta
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ServiceSetConfig:
    relationships: dict[str, list[str]]
    metrics: list[str]
    training_lookback_hours: int
    inference_lookback_minutes: int
    window_size: int
    anomaly_threshold_percentile: float

    @property
    def services(self) -> list[str]:
        """Get all unique services from relationships"""
        services = set()
        services.update(self.relationships.keys())
        for targets in self.relationships.values():
            services.update(targets)
        return sorted(list(services))

    def __post_init__(self) -> None:
        """Initialize and validate the configuration after creation"""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values

        Raises:
            ValueError: If any configuration value is invalid
        """
        if self.training_lookback_hours <= 0:
            raise ValueError("training_lookback_hours must be positive")
        if self.inference_lookback_minutes <= 0:
            raise ValueError("inference_lookback_minutes must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0 <= self.anomaly_threshold_percentile <= 100:
            raise ValueError("anomaly_threshold_percentile must be between 0 and 100")

    @property
    def retraining_interval(self) -> timedelta:
        return timedelta(hours=self.training_lookback_hours)

    @property
    def inference_interval(self) -> timedelta:
        return timedelta(minutes=self.inference_lookback_minutes)

    @property
    def max_depth(self) -> int:
        """Calculate maximum depth of service relationships graph

        Returns:
            Maximum depth of the service dependency graph
        """

        def dfs(node: str, visited: set) -> int:
            if node in visited:
                return 0
            visited.add(node)
            max_child_depth = 0
            for child in self.relationships.get(node, []):
                max_child_depth = max(max_child_depth, dfs(child, visited.copy()))
            return 1 + max_child_depth

        if not self.services:
            return 0

        max_depth = 0
        for service in self.services:
            depth = dfs(service, set())
            max_depth = max(max_depth, depth)

        # Depth should be at least 0 (single node) and subtract 1 since we count edges not nodes
        return max(0, max_depth - 1)


class ConfigLoader:
    """Loads and manages service set configurations from YAML file

    Attributes:
        config_path: Path to the service sets configuration file
        config: Raw configuration dictionary
        service_sets: Dictionary of ServiceSetConfig instances by name
        active_sets: List of currently active service set names
    """

    def __init__(self, config_path: str | None = None):
        """Initialize the ConfigLoader with the specified configuration file

        Args:
            config_path: Optional filesystem path to YAML config (for testing/override).
                        If None, loads from package resources.
        """
        self.config_path = Path(config_path) if config_path else None
        self._load_config()
        self.active_sets = ["default", "transfer"]

    def _load_config(self) -> None:
        """Load and parse the configuration file

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the configuration file is invalid
        """
        if self.config_path:
            # Explicit path provided (testing/dev override)
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
            with open(self.config_path) as f:
                self.config: dict[str, Any] = yaml.safe_load(f)
        else:
            # Load from package resources
            resource = files("ml_monitoring_service").joinpath(
                "resources/service_sets.yaml"
            )
            with resource.open("r") as f:
                self.config: dict[str, Any] = yaml.safe_load(f)

        if "service_sets" not in self.config:
            raise ValueError("Config file must contain 'service_sets' key")

        self.service_sets: dict[str, ServiceSetConfig] = {}
        for set_name, set_config in self.config["service_sets"].items():
            config = ServiceSetConfig(**set_config)
            config.validate()
            self.service_sets[set_name] = config

    def set_active(self, service_sets: list[str]) -> None:
        """Change the active service sets

        Args:
            service_sets: List of service set names to activate

        Raises:
            ValueError: If any service set name is not found
        """
        for service_set in service_sets:
            if service_set not in self.service_sets:
                raise ValueError(f"Service set '{service_set}' not found")
        self.active_sets = service_sets

    def add_active_set(self, service_set: str) -> None:
        """Add a service set to the active sets

        Args:
            service_set: Name of the service set to add

        Raises:
            ValueError: If the service set name is not found
        """
        if service_set not in self.service_sets:
            raise ValueError(f"Service set '{service_set}' not found")
        if service_set not in self.active_sets:
            self.active_sets.append(service_set)

    def remove_active_set(self, service_set: str) -> None:
        """Remove a service set from the active sets

        Args:
            service_set: Name of the service set to remove
        """
        if service_set in self.active_sets:
            self.active_sets.remove(service_set)

    def get_active(self) -> list[str]:
        """Get the current active service sets

        Returns:
            List of active service set names

        Raises:
            ValueError: If any active service set is not found in configuration
        """
        for service_set in self.active_sets:
            if service_set not in self.service_sets:
                raise ValueError(f"Active service set '{service_set}' not found")
        return self.active_sets

    def get_config(self, service_set: str | None = None) -> ServiceSetConfig:
        """Get the configuration for a specific service set

        Args:
            service_set: Name of the service set

        Returns:
            ServiceSetConfig instance for the requested service set

        Raises:
            ValueError: If the service set is not found
        """
        if service_set not in self.service_sets:
            raise ValueError(f"Service set '{service_set}' not found")
        return self.service_sets[service_set]

    def get_services(self, service_set: str | None = None) -> list[str]:
        """Get all services for a service set or all active sets

        Args:
            service_set: Specific service set name, or None for all active sets

        Returns:
            Sorted list of unique service names
        """
        if service_set is not None:
            # Return services only for the specified service_set
            config = self.get_config(service_set)
            services = set()
            services.update(config.relationships.keys())
            for targets in config.relationships.values():
                services.update(targets)
            return sorted(list(services))  # Sort for consistent ordering
        else:
            # Return services for all active sets if no specific set is provided
            services = set()
            for active_set in self.active_sets:
                config = self.get_config(active_set)
                services.update(config.relationships.keys())
                for targets in config.relationships.values():
                    services.update(targets)
            return sorted(list(services))  # Sort for consistent ordering

    def get_available_sets(self) -> list[str]:
        """Get list of available service sets

        Returns:
            Sorted list of all configured service set names
        """
        return sorted(list(self.service_sets.keys()))


# Create global config instance
config = ConfigLoader()

# ============================================================================
# Helper Functions - Convenience wrappers around ConfigLoader
# ============================================================================


def get_services(service_set: str | None = None) -> list[str]:
    """Get all services for a service set or all active sets"""
    return config.get_services(service_set)


def get_relationships(service_set: str | None = None) -> dict[str, list[str]]:
    """Get service relationships for a specific service set"""
    return config.get_config(service_set).relationships


def get_metrics(service_set: str | None = None) -> list[str]:
    """Get metrics for a specific service set"""
    return config.get_config(service_set).metrics


def get_training_lookback_hours(service_set: str | None = None) -> int:
    """Get training lookback hours for a specific service set"""
    return config.get_config(service_set).training_lookback_hours


def get_inference_lookback_minutes(service_set: str | None = None) -> int:
    """Get inference lookback minutes for a specific service set"""
    return config.get_config(service_set).inference_lookback_minutes


def get_available_sets() -> list[str]:
    """Get list of all available service sets"""
    return config.get_available_sets()


def set_active_sets(service_sets: list[str]) -> None:
    """Set the active service sets"""
    config.set_active(service_sets)


def add_active_set(service_set: str) -> None:
    """Add a service set to the active sets"""
    config.add_active_set(service_set)


def remove_active_set(service_set: str) -> None:
    """Remove a service set from the active sets"""
    config.remove_active_set(service_set)


def get_active_sets() -> list[str]:
    """Get the current active service sets"""
    return config.get_active()


def get_config(service_set: str) -> ServiceSetConfig:
    """Get the configuration for a specific service set"""
    return config.get_config(service_set)
