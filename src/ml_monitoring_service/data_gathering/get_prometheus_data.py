import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import ml_monitoring_service.configuration as conf
from ml_monitoring_service.constants import (
    PROMETHEUS_TIMEOUT,
    PROMETHEUS_URL,
    REQUESTS_VERIFY,
)

logger = logging.getLogger(__name__)

# Disable SSL warnings only if TLS verification is disabled (dev/local).
if not REQUESTS_VERIFY:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Constants
SLEEP_INTERVAL = 0.05  # seconds between requests
MAX_RETRIES = 15
RETRY_BACKOFF_FACTOR = 2


def create_session_with_retries() -> requests.Session:
    """Create a requests session with automatic retry logic for failed requests."""
    session = requests.Session()
    retries = Retry(
        total=10,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=[502, 503, 504],
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def query_prometheus(
    query: str,
    timestamp: float,
    session: requests.Session,
    sleep_interval: float = SLEEP_INTERVAL,
) -> list[dict[str, Any]] | None:
    """
    Query Prometheus API with retry logic.

    Args:
        query: PromQL query string
        timestamp: Unix timestamp for the query
        session: Requests session with retry configuration
        sleep_interval: Base sleep interval between requests

    Returns:
        List of query results or None if all retries failed
    """
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = session.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": query, "time": timestamp},
                verify=REQUESTS_VERIFY,
                timeout=PROMETHEUS_TIMEOUT,
            )
            response.raise_for_status()

            data = response.json()
            if "data" not in data or "result" not in data["data"]:
                logger.warning(f"Unexpected response format: {data}")
                return None

            results = data["data"]["result"]
            return results
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Error querying Prometheus (attempt {retries + 1}/{MAX_RETRIES}): {e}"
            )
            retries += 1
            if retries < MAX_RETRIES:
                sleep_time = sleep_interval * (
                    RETRY_BACKOFF_FACTOR**retries
                )  # Exponential backoff
                logger.debug(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error("Max retries reached for Prometheus query")
                return None
        finally:
            time.sleep(sleep_interval)  # Sleep interval between requests


def convert_to_timestamp(dt: datetime.datetime) -> int:
    """Convert datetime to Unix timestamp (seconds)."""
    return int(time.mktime(dt.timetuple()))


def convert_to_timestamp_with_milliseconds(dt: datetime.datetime) -> float:
    """Convert datetime to Unix timestamp with milliseconds precision."""
    return dt.timestamp()


def is_numeric(value: str) -> bool:
    """Check if a string value can be converted to float."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def process_results(
    results: list[dict[str, Any]], data_type: str
) -> list[dict[str, Any]]:
    """
    Process Prometheus query results into a standardized format.

    Args:
        results: List of result dictionaries from Prometheus
        data_type: Type of metric (e.g., 'cpu', 'memory')

    Returns:
        List of processed data dictionaries with timestamp and metric value
    """
    processed_data = []
    for result in results:
        if "value" in result:
            timestamp, value = result["value"]
            if is_numeric(value) and value not in ["+Inf", "Inf", "inf", "-Inf"]:
                processed_data.append(
                    {
                        "timestamp": datetime.datetime.fromtimestamp(timestamp),
                        data_type: float(value),
                    }
                )
        else:
            logger.debug(f"Unexpected result format: {result}")
    return processed_data


def fill_missing_data(
    data: list[dict[str, Any]],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> pd.DataFrame:
    """
    Fill missing timestamps in data with NaN values at 30-second intervals.

    Args:
        data: List of data dictionaries with timestamps
        start_time: Start of the time range
        end_time: End of the time range

    Returns:
        DataFrame with filled missing timestamps
    """
    all_timestamps = pd.date_range(start=start_time, end=end_time, freq="30s")
    df = pd.DataFrame(data)
    df.drop_duplicates(subset="timestamp", inplace=True)  # Remove duplicate timestamps
    df.set_index("timestamp", inplace=True)
    df = df.reindex(all_timestamps, fill_value=pd.NA)  # Fill missing values with NaN
    df.reset_index(inplace=True)
    df.rename(columns={"index": "timestamp"}, inplace=True)
    return df


def download_prometheus_data(
    task: str, service_name: str, data_type: str, timestamps: list[str], active_set: str
) -> None:
    logger.info(f"Downloading Prometheus data for {service_name} ({data_type})...")
    if data_type == "cpu":
        QUERY_TEMPLATE = f"""
        round(
            sum(irate(container_cpu_usage_seconds_total{{container="{service_name}"}}[1m])) by (container) /
            (sum by (container) (container_spec_cpu_quota{{container="{service_name}"}} / container_spec_cpu_period{{container="{service_name}"}})) * 100,
            2
        )
        """

    elif data_type == "memory":
        QUERY_TEMPLATE = f"""
        round(
            sum(container_memory_usage_bytes{{container="{service_name}"}}) by (container) /
            sum(container_spec_memory_limit_bytes{{container="{service_name}"}}) by (container) * 100,
            2
        )
        """
    elif data_type == "disk":  # Not used currently
        QUERY_TEMPLATE = f"""
        round(
            avg(disk_free_bytes{{service="{service_name}"}}/disk_total_bytes{{service="{service_name}"}})*100,
            2
        )
        """
    elif data_type == "latency":  # standard is to use percentiles
        QUERY_TEMPLATE = f"""
        round(
            sum(http_server_requests_seconds_sum{{service="{service_name}"}})/sum(http_server_requests_seconds_count{{service="{service_name}"}})*1000,
            2
        )
        """
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    # Create a session with retries
    session = create_session_with_retries()

    all_processed_data = []
    failed_timestamps_count = 0

    logger.debug(f"Querying Prometheus with query: {QUERY_TEMPLATE}")

    try:
        for timestamp in timestamps:
            try:
                timestamp_unix = convert_to_timestamp_with_milliseconds(
                    datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
                )
            except ValueError:
                logger.warning(f"Invalid timestamp format: {timestamp}")
                failed_timestamps_count += 1
                continue

            query = QUERY_TEMPLATE
            results = query_prometheus(query, timestamp_unix, session)

            if results:
                processed_data = process_results(results, data_type)
                all_processed_data.extend(processed_data)
            else:
                logger.debug(f"No results obtained for timestamp {timestamp}")
                failed_timestamps_count += 1

        if failed_timestamps_count > 0:
            logger.warning(
                f"Failed to get results for {failed_timestamps_count}/{len(timestamps)} "
                f"timestamps for {service_name} ({data_type})"
            )

        if all_processed_data:
            # Convert processed data to DataFrame
            df = pd.DataFrame(all_processed_data)

            # Replace hyphens with underscores in the service name
            service_name_cleaned = service_name.replace("-", "_")

            # Create directory path
            directory = (
                Path("output")
                / active_set
                / f"prometheus_data_{task}"
                / service_name_cleaned
            )
            directory.mkdir(parents=True, exist_ok=True)

            # Save to JSON
            json_file = directory / f"{data_type}-{service_name_cleaned}.json"
            df.to_json(json_file, orient="records")
            logger.info(f"✓ Data saved to {json_file} ({len(df)} records)")
        else:
            logger.warning(
                f"No results obtained from Prometheus for {service_name} ({data_type})"
            )
    finally:
        # Close the session to free resources
        session.close()


def read_splunk_data(file_path: str) -> list[dict[str, Any]]:
    """
    Read Splunk data from a JSON file.

    Args:
        file_path: Path to the Splunk JSON file

    Returns:
        List of Splunk data entries
    """
    try:
        with open(file_path) as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        logger.error(f"Splunk data file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing Splunk JSON file {file_path}: {e}")
        raise


def get_timestamps_for_service(
    splunk_data: list[dict[str, Any]], service_name: str
) -> list[str]:
    """
    Extract unique timestamps for a specific service from Splunk data.

    Args:
        splunk_data: List of Splunk data entries
        service_name: Name of the service to filter by

    Returns:
        List of unique timestamps for the service
    """
    timestamps = {
        entry["timestamp"]
        for entry in splunk_data
        if entry.get("service") == service_name
    }
    return list(timestamps)


def main(task: str, active_set: str) -> None:
    """
    Main function to download Prometheus data for all services and metrics.

    Args:
        task: Task name (e.g., 'training', 'inference')
        active_set: Active service set name
    """
    # Read Splunk data from file
    splunk_file = Path("output") / active_set / f"splunk_results_{task}.json"
    splunk_data = read_splunk_data(str(splunk_file))

    # Get unique services
    services = {entry["service"] for entry in splunk_data if "service" in entry}

    if not services:
        logger.warning(f"No services found in Splunk data from {splunk_file}")
        return

    logger.info(f"Downloading Prometheus data for {len(services)} services")

    for service_name in services:
        # Get timestamps for the specific service
        timestamps = get_timestamps_for_service(splunk_data, service_name)

        if not timestamps:
            logger.warning(f"No timestamps found for service {service_name}")
            continue

        for metric_type in conf.get_metrics(active_set):
            download_prometheus_data(
                task, service_name, metric_type, timestamps, active_set
            )


if __name__ == "__main__":
    main()
