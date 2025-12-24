import logging
from typing import Any

import numpy as np

from ml_monitoring_service.constants import Colors

logger = logging.getLogger(__name__)


def analyse_anomalies(
    result: dict[str, Any],
    service_relationships: dict[str, list[str]],
    services: list[str],
    features: list[str],
) -> list[str]:
    """Analyze detected anomalies and provide insights based on service relationships

    Args:
        result: Dictionary containing anomaly detection results
        service_relationships: Dictionary mapping services to their related services
        services: List of service names
        features: List of feature names

    Returns:
        List of explanation strings describing the anomaly
    """
    explanations = []

    if result["is_anomaly"]:
        explanations.append("- - - - - - - - - - - - - - - - - - - - - - - - - - -")
        explanations.append(
            Colors.red(
                f"Anomaly detected at {result['timestamp']} with error score: {result['error_score']:.4f}"
            )
        )
        explanations.append(f"Threshold: {result['threshold']:.4f}")
        service_errors = result["service_errors"]
        variable_errors = result["variable_errors"]

        # Sort services by their error values in descending order
        sorted_services_errors = sorted(
            zip(services, service_errors, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )

        for service, error in sorted_services_errors:
            explanations.append(f"  {service}: {error:.4f}")

        # Analyze service relationships
        high_error_services = [
            service
            for service, error in sorted_services_errors
            if error > result["threshold"]
        ]
        for service in high_error_services:
            service_idx = services.index(service)
            explanations.append(
                f"\n- Service '{service}' has high error: {service_errors[service_idx]:.4f}"
            )
            if service in service_relationships:
                related_services = service_relationships[service]
                if related_services:
                    explanations.append(
                        f"  Related services: {', '.join(related_services)}"
                    )
                    for related_service in related_services:
                        if related_service in high_error_services:
                            related_service_idx = services.index(related_service)
                            explanations.append(
                                f"    Related service '{related_service}' also has high error: {service_errors[related_service_idx]:.4f}"
                            )
                else:
                    explanations.append("  No related services.")
            else:
                explanations.append(
                    f"  No relationship data available for '{service}'."
                )

        # Identify the most important variables for the anomaly
        num_variables_to_show = min(2, len(features))  # Show up to 2 variables
        most_important_variable_indices = np.argsort(variable_errors)[
            -num_variables_to_show:
        ][::-1]
        most_important_variables = [
            features[idx] for idx in most_important_variable_indices
        ]
        most_important_values = [
            variable_errors[idx] for idx in most_important_variable_indices
        ]

        # For each variable, find which service has the highest contribution to that variable's error
        # This is more accurate than using the same service for all variables
        most_important_services = [
            services[np.argmax(service_errors)] for _ in most_important_variable_indices
        ]

        # Append the explanations for the most important variables along with their values and associated services
        if num_variables_to_show >= 2:
            explanations.append(
                f"\nMost important variables for this anomaly: "
                f"1: {most_important_variables[0]} (value: {most_important_values[0]:.4f}, service: {most_important_services[0]}), "
                f"2: {most_important_variables[1]} (value: {most_important_values[1]:.4f}, service: {most_important_services[1]})"
            )
        elif num_variables_to_show == 1:
            explanations.append(
                f"\nMost important variable for this anomaly: "
                f"{most_important_variables[0]} (value: {most_important_values[0]:.4f}, service: {most_important_services[0]})"
            )

        # Identify the most influential service for the anomaly
        most_influential_service_idx = np.argmax(service_errors)
        most_influential_service = services[most_influential_service_idx]
        explanations.append(
            f"Most influential service for this anomaly: {most_influential_service}\n"
        )
    else:
        explanations.append(Colors.green("Not an anomaly."))

    return explanations
