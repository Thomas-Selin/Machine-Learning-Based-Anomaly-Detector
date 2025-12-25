import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import urllib3

import ml_monitoring_service.configuration as conf
from ml_monitoring_service.constants import (
    DATA_SAMPLING,
    DATA_SUBSET,
    REQUESTS_VERIFY,
    SPLUNK_AUTH_TOKEN,
    SPLUNK_TIMEOUT,
    SPLUNK_URL,
)

# Disable SSL warnings only if TLS verification is disabled (dev/local).
if not REQUESTS_VERIFY:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# Constants
BATCH_LIMIT = 50000  # Splunk's max_events_per_bucket limit
JOB_DELETION_DELAY = 10  # seconds between job deletions
MAX_JOB_STATUS_CHECKS = 100  # Maximum number of status checks before giving up
JOB_STATUS_CHECK_INTERVAL = 5  # Initial seconds between status checks
MAX_STATUS_CHECK_INTERVAL = 30  # Maximum seconds between status checks


def _get_auth_headers() -> dict[str, str]:
    """
    Get authentication headers with the current SPLUNK_AUTH_TOKEN from environment.
    This function reads the token at runtime rather than at module import time.

    Returns:
        Dictionary of HTTP headers for Splunk API requests
    """
    return {
        "Authorization": f"Bearer {SPLUNK_AUTH_TOKEN}",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }


def check_authentication() -> bool:
    """
    Verify that authentication with JWT token works correctly.

    Returns:
        True if authentication is successful, False otherwise
    """
    if not SPLUNK_AUTH_TOKEN:
        logger.error("SPLUNK_AUTH_TOKEN environment variable not set")
        return False

    logger.debug("Checking JWT authentication...")
    try:
        response = requests.get(
            f"{SPLUNK_URL}/services/authentication/current-context?output_mode=json",
            headers=_get_auth_headers(),
            verify=REQUESTS_VERIFY,
            timeout=SPLUNK_TIMEOUT,
        )
        if response.status_code == 200:
            logger.debug("Authentication verified.")
            return True
        else:
            logger.error(
                f"Authentication failed with status code {response.status_code}: {response.text}"
            )
            return False
    except requests.RequestException:
        logger.exception("Error checking authentication.")
        return False


def start_search(search_query: str) -> str | None:
    """
    Start a Splunk search and return the search job ID.

    Args:
        search_query: Splunk search query string

    Returns:
        Job ID (sid) if successful, None otherwise
    """
    data = {
        "search": search_query,
        "output_mode": "json",
        "time_format": "%Y-%m-%d %H:%M:%S.%f",
    }

    logger.debug(f"Starting search with query: {search_query}")
    try:
        response = requests.post(
            f"{SPLUNK_URL}/services/search/jobs?output_mode=json",
            headers=_get_auth_headers(),
            data=data,
            verify=REQUESTS_VERIFY,
            timeout=SPLUNK_TIMEOUT,
        )
        logger.debug(f"Search started, status code: {response.status_code}")
        response.raise_for_status()

        try:
            job_data = response.json()
            job_id = job_data.get("sid")
            if not job_id:
                logger.error(
                    "No job ID received. Check the search format and authentication."
                )
                logger.debug(f"Response: {job_data}")
                return None
            logger.info(f"Search started with job ID: {job_id}")
            return job_id
        except json.JSONDecodeError:
            logger.error("JSONDecodeError - raw response from server:")
            logger.debug(response.text)
            return None
    except requests.RequestException:
        logger.exception("Error starting search.")
        return None


def parse_raw_data(raw_data: str) -> dict[str, str]:
    """
    Parse the raw data string to extract specific key-value pairs.

    Args:
        raw_data: Raw data string in format "key1=value1, key2=value2, ..."

    Returns:
        Dictionary of parsed key-value pairs
    """
    parsed_data = {}
    try:
        # Split the raw data string by commas
        key_value_pairs = raw_data.split(", ")
        for pair in key_value_pairs:
            # Split each pair by the first '=' to separate key and value
            if "=" in pair:
                key, value = pair.split("=", 1)
                # Remove any surrounding quotes from the value
                value = value.strip('"')
                parsed_data[key] = value
    except Exception as e:
        logger.debug(f"Failed to parse raw data: {raw_data[:100]}... Error: {e}")
    return parsed_data


def get_search_results(job_id: str) -> list[dict[str, Any]] | None:
    """
    Retrieve results for a Splunk search based on the job ID.

    Args:
        job_id: Splunk search job ID

    Returns:
        List of parsed result dictionaries, or None if failed
    """
    wait_time = JOB_STATUS_CHECK_INTERVAL
    error_count = 0
    status_check_count = 0

    # Fields to exclude from results
    EXCLUDED_FIELDS = {
        "info_min_time",
        "info_max_time",
        "info_search_time",
        "search_name",
        "search_now",
    }

    try:
        # Poll for job completion
        while status_check_count < MAX_JOB_STATUS_CHECKS:
            try:
                response = requests.get(
                    f"{SPLUNK_URL}/services/search/jobs/{job_id}?output_mode=json",
                    headers=_get_auth_headers(),
                    verify=REQUESTS_VERIFY,
                    timeout=SPLUNK_TIMEOUT,
                )
                response.raise_for_status()
                job_data = response.json()

                job_status = job_data["entry"][0]["content"]["dispatchState"]

                if job_status == "DONE":
                    logger.info(f"Search job {job_id} completed successfully")
                    break
                elif job_status == "FAILED":
                    logger.error(f"Search job {job_id} failed")
                    return None

                logger.debug(
                    f"Job status: {job_status}, waiting {wait_time}s before next check..."
                )
                time.sleep(wait_time)
                wait_time = min(MAX_STATUS_CHECK_INTERVAL, wait_time * 2)
                status_check_count += 1

            except json.JSONDecodeError:
                logger.error("Failed to parse job status response")
                logger.debug(f"Response text: {response.text[:500]}")
                error_count += 1
                if error_count > 5:
                    logger.error("Too many JSONDecodeErrors. Aborting.")
                    return None
                time.sleep(wait_time)

        if status_check_count >= MAX_JOB_STATUS_CHECKS:
            logger.error(
                f"Job {job_id} did not complete after {MAX_JOB_STATUS_CHECKS} status checks"
            )
            return None

        # Retrieve results
        request_data = {
            "output_mode": "json",
            "time_format": "%Y-%m-%d %H:%M:%S.%f",
            "count": 0,  # Set to 0 to retrieve all results
        }

        response = requests.get(
            f"{SPLUNK_URL}/services/search/jobs/{job_id}/results",
            headers=_get_auth_headers(),
            params=request_data,
            verify=REQUESTS_VERIFY,
            timeout=300,
        )
        response.raise_for_status()

        try:
            results = response.json().get("results", [])
            logger.info(f"Retrieved {len(results)} results from Splunk job {job_id}")

            if not results:
                logger.warning("No results found for this search")
                return []

            all_parsed_data = []
            for result in results:
                raw_data = result.get("_raw")
                if raw_data:
                    parsed_data = parse_raw_data(raw_data)
                    # Exclude specific columns
                    for field in EXCLUDED_FIELDS:
                        parsed_data.pop(field, None)
                    all_parsed_data.append(parsed_data)

            return all_parsed_data
        except json.JSONDecodeError:
            logger.error("JSONDecodeError retrieving search results")
            logger.debug(f"Response text: {response.text[:500]}")
            return None

    except requests.RequestException:
        logger.exception(f"Error retrieving search results for job {job_id}")
        return None


def delete_search_job(job_id: str) -> bool:
    """
    Delete a Splunk search job to clean up resources.

    Args:
        job_id: The ID of the search job to delete

    Returns:
        True if the job was successfully deleted, False otherwise
    """
    logger.debug(f"Deleting search job: {job_id}")
    try:
        response = requests.delete(
            f"{SPLUNK_URL}/services/search/jobs/{job_id}?output_mode=json",
            headers=_get_auth_headers(),
            verify=REQUESTS_VERIFY,
            timeout=SPLUNK_TIMEOUT,
        )
        if response.status_code == 200:
            logger.debug(f"Successfully deleted job {job_id}")
            return True
        else:
            logger.warning(
                f"Failed to delete job {job_id}. Status: {response.status_code}"
            )
            return False
    except Exception as e:
        logger.warning(f"Error deleting job {job_id}: {e}")
        return False


def cleanup_all_search_jobs() -> int:
    """
    Clean up all existing search jobs for the current user.
    This helps prevent job buildup and resource consumption on the Splunk server.
    Only deletes jobs where the current user is the author.

    Returns:
        The number of jobs successfully deleted
    """
    logger.info("Cleaning up existing Splunk search jobs for current user...")

    try:
        # Get current user information
        user_response = requests.get(
            f"{SPLUNK_URL}/services/authentication/current-context?output_mode=json",
            headers=_get_auth_headers(),
            verify=REQUESTS_VERIFY,
            timeout=SPLUNK_TIMEOUT,
        )

        if user_response.status_code != 200:
            logger.warning(
                f"Failed to retrieve current user. Status: {user_response.status_code}"
            )
            return 0

        try:
            user_data = user_response.json()
            username = (
                user_data.get("entry", [{}])[0].get("content", {}).get("username")
            )

            if not username:
                logger.warning("Could not determine current username")
                return 0

            logger.info(f"Cleaning up jobs for user: {username}")

            # Get all jobs for the current user
            response = requests.get(
                f"{SPLUNK_URL}/services/search/jobs?output_mode=json&username={username}",
                headers=_get_auth_headers(),
                verify=REQUESTS_VERIFY,
                timeout=SPLUNK_TIMEOUT,
            )

            if response.status_code != 200:
                logger.warning(
                    f"Failed to retrieve jobs list. Status: {response.status_code}"
                )
                return 0

            job_data = response.json()
            jobs = job_data.get("entry", [])

            if not jobs:
                logger.info(f"No existing jobs found for user {username}")
                return 0

            logger.info(f"Found {len(jobs)} existing jobs to clean up")
            deleted_count = 0

            for i, job in enumerate(jobs, 1):
                job_id = job.get("content", {}).get("sid")
                job_author = job.get("author")

                # Only delete jobs where the author matches the current username
                if job_id and job_author == username:
                    logger.debug(f"Deleting job {i}/{len(jobs)}: {job_id}")
                    if delete_search_job(job_id):
                        deleted_count += 1
                        # Add delay between job deletions to avoid overwhelming Splunk
                        if i < len(jobs):  # Don't wait after the last job
                            time.sleep(JOB_DELETION_DELAY)

            logger.info(
                f"Successfully cleaned up {deleted_count}/{len(jobs)} jobs for user {username}"
            )
            return deleted_count

        except json.JSONDecodeError:
            logger.error("Failed to parse user/jobs response")
            return 0

    except Exception as e:
        logger.error(f"Request failed: {e}")
        return 0


def parse_and_format_timestamp(timestamp_str: str) -> str:
    """
    Parse timestamp string and convert to Splunk format (MM/DD/YYYY:HH:MM:SS).

    Args:
        timestamp_str: Timestamp string in format "YYYY-MM-DD HH:MM:SS.ffffff"

    Returns:
        Formatted timestamp string in Splunk format
    """
    try:
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
        return dt.strftime("%m/%d/%Y:%H:%M:%S")
    except ValueError as e:
        logger.warning(f"Failed to parse timestamp '{timestamp_str}': {e}")
        return timestamp_str


def download_splunk_data_in_batches(
    task: str, active_set: str, timestamp_latest_data: str | None = None
) -> list[dict[str, Any]] | None:
    """
    Download data from Splunk in batches to handle the 50,000 event limit per search.
    This function iteratively calls Splunk, checking for the max event limit and continuing
    where the previous batch left off.

    Args:
        task: Task name (e.g., 'training', 'inference')
        active_set: Active service set name
        timestamp_latest_data: Optional timestamp of latest existing data (MM/DD/YYYY:HH:MM:SS format)

    Returns:
        List of all collected Splunk results, or None if failed
    """
    # Clean up any existing search jobs before starting a new batch search
    cleanup_all_search_jobs()

    service_set_config = conf.config.get_config(active_set)
    training_lookback_hours = service_set_config.training_lookback_hours
    inference_lookback_minutes = service_set_config.inference_lookback_minutes

    # Set initial time range
    if timestamp_latest_data:
        if isinstance(timestamp_latest_data, str):
            try:
                # Parse the timestamp in MM/DD/YYYY:HH:MM:SS format
                from datetime import timedelta

                latest_data_dt = datetime.strptime(
                    timestamp_latest_data, "%m/%d/%Y:%H:%M:%S"
                )

                # Calculate how far back we would go with training_lookback_hours
                current_time = datetime.now()
                training_lookback_time = current_time - timedelta(
                    hours=training_lookback_hours
                )

                # Use training_lookback_hours if it goes further back
                if training_lookback_time > latest_data_dt:
                    logger.info(
                        f"Using training_lookback_hours ({training_lookback_hours}h) "
                        f"as it goes further back than latest data timestamp"
                    )
                    earliest_time = f"-{training_lookback_hours}h"
                else:
                    logger.info(f"Using latest data timestamp: {timestamp_latest_data}")
                    earliest_time = timestamp_latest_data
            except ValueError as e:
                logger.warning(
                    f"Failed to parse timestamp '{timestamp_latest_data}': {e}. "
                    f"Defaulting to training_lookback_hours."
                )
                earliest_time = f"-{training_lookback_hours}h"
        else:
            earliest_time = timestamp_latest_data
    else:
        earliest_time = f"-{training_lookback_hours}h"

    latest_time = f"-{inference_lookback_minutes}m"

    all_results = []
    batch_count = 1

    # Continue fetching batches until we get a batch smaller than the limit
    while True:
        logger.info(
            f"Fetching batch {batch_count}: earliest={earliest_time}, latest={latest_time}"
        )

        # Create time range for this batch
        if isinstance(earliest_time, str) and earliest_time.startswith("-"):
            time_range_spl = f"earliest={earliest_time} latest={latest_time}"
        else:
            time_range_spl = f'earliest="{earliest_time}" latest={latest_time}'

        search_query = (
            f"search index=ml_monitoring source=ml_monitoring_{active_set}_1_percent_sampling_rate "  # Adjust Splunk Query (SPL) as needed
            f"{time_range_spl} | sort 0 _time | spath input=_raw"
        )

        if not check_authentication():
            logger.error("JWT authentication failed, check token and permissions")
            return None

        job_id = start_search(search_query)
        if not job_id:
            logger.error("Failed to start search")
            return None

        batch_results = get_search_results(job_id)

        # Delete the job after getting results to clean up resources
        delete_search_job(job_id)

        if not batch_results:
            logger.warning(f"No results found for batch {batch_count}")
            break

        batch_size = len(batch_results)
        logger.info(f"Retrieved {batch_size} results in batch {batch_count}")

        all_results.extend(batch_results)

        # If we got fewer results than the limit, we've retrieved all data
        if batch_size < BATCH_LIMIT:
            logger.info(
                f"Completed: {batch_count} batches, {len(all_results)} total records"
            )
            break

        # Otherwise, find the timestamp of the last record and continue from there
        try:
            last_record = batch_results[-1]

            # Try to extract timestamp from various fields
            timestamp = last_record.get("_time") or last_record.get("timestamp")

            if not timestamp:
                logger.error(
                    "Could not find timestamp in last record to continue batch processing"
                )
                break

            # Convert timestamp to Splunk format
            if isinstance(timestamp, str):
                earliest_time = parse_and_format_timestamp(timestamp)
            else:
                earliest_time = timestamp

            logger.debug(f"Continuing next batch from timestamp: {earliest_time}")
            batch_count += 1
        except (IndexError, KeyError) as e:
            logger.error(f"Error extracting timestamp from batch results: {e}")
            break

    # Save all collected results
    if not all_results:
        logger.error("No results could be retrieved from any batch")
        return None

    # Apply subsetting or sampling if configured
    if DATA_SUBSET:
        try:
            subset_size = int(DATA_SUBSET)
            all_results = all_results[:subset_size]
            logger.info(f"Applied subset: limited to {subset_size} results")
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Invalid DATA_SUBSET value '{DATA_SUBSET}': {e}. Using all results"
            )
    elif DATA_SAMPLING:
        try:
            import random

            sampling_rate = float(DATA_SAMPLING)
            if 0 < sampling_rate < 1:
                sample_size = int(len(all_results) * sampling_rate)
                sampled_results = random.sample(all_results, sample_size)
                logger.info(
                    f"Applied sampling: {sampling_rate * 100:.1f}% of data "
                    f"({sample_size}/{len(all_results)} results)"
                )
                all_results = sampled_results
            else:
                logger.warning(
                    f"DATA_SAMPLING must be between 0 and 1. Got {sampling_rate}. Using all results"
                )
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Invalid DATA_SAMPLING value '{DATA_SAMPLING}': {e}. Using all results"
            )

    # Save results to file
    output_path = Path("output") / active_set / f"splunk_results_{task}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w") as json_file:
            json.dump(all_results, json_file, indent=2)
        logger.info(f"✓ Saved {len(all_results)} results to {output_path}")
        return all_results
    except OSError as e:
        logger.error(f"Failed to save results to {output_path}: {e}")
        return None


def download_splunk_data(
    task: str, active_set: str, timestamp_latest_data: str | None = None
) -> list[dict[str, Any]] | None:
    """
    Download data from Splunk, using batch processing to handle large result sets.

    Args:
        task: Task name (e.g., 'training', 'inference')
        active_set: Active service set name
        timestamp_latest_data: Optional timestamp of latest existing data

    Returns:
        List of Splunk results, or None if failed
    """
    return download_splunk_data_in_batches(task, active_set, timestamp_latest_data)


if __name__ == "__main__":
    # Updated to use the batched download approach
    download_splunk_data("test", "default")
