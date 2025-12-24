import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from ml_monitoring_service.data_gathering import get_splunk_data


@pytest.fixture
def mock_requests():
    """Mock requests library for testing"""
    with patch(
        "ml_monitoring_service.data_gathering.get_splunk_data.requests"
    ) as mock_req:
        yield mock_req


@pytest.fixture
def mock_time():
    """Mock time.sleep to avoid delays in tests"""
    with patch(
        "ml_monitoring_service.data_gathering.get_splunk_data.time"
    ) as mock_time:
        yield mock_time


class TestDeleteSearchJob:
    """Tests for the delete_search_job function"""

    def test_successful_deletion(self, mock_requests):
        """Test successful job deletion"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests.delete.return_value = mock_response

        # Call the function
        job_id = "dummy_job_id"
        result = get_splunk_data.delete_search_job(job_id)

        # Assertions
        assert result is True
        # Verify delete was called (don't check exact URL due to dynamic headers)
        assert mock_requests.delete.called

    def test_failed_deletion(self, mock_requests):
        """Test failed job deletion"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Job not found"
        mock_requests.delete.return_value = mock_response

        # Call the function
        job_id = "nonexistent_job_id"
        result = get_splunk_data.delete_search_job(job_id)

        # Assertions
        assert result is False
        mock_requests.delete.assert_called_once()

    def test_exception_during_deletion(self, mock_requests):
        """Test exception handling during job deletion"""
        # Setup mock to raise exception
        mock_requests.delete.side_effect = requests.RequestException("Connection error")

        # Call the function
        job_id = "error_job_id"
        result = get_splunk_data.delete_search_job(job_id)

        # Assertions
        assert result is False
        mock_requests.delete.assert_called_once()


class TestCleanupAllSearchJobs:
    """Tests for the cleanup_all_search_jobs function"""

    def test_successful_cleanup(self, mock_requests, mock_time):
        """Test successful cleanup of all jobs"""
        # Setup mock for user context
        user_response = MagicMock()
        user_response.status_code = 200
        user_response.json.return_value = {
            "entry": [{"content": {"username": "test_user"}}]
        }

        # Setup mock for jobs list
        jobs_response = MagicMock()
        jobs_response.status_code = 200
        jobs_response.json.return_value = {
            "entry": [
                {"content": {"sid": "job1"}, "author": "test_user"},
                {"content": {"sid": "job2"}, "author": "test_user"},
                {
                    "content": {"sid": "job3"},
                    "author": "other_user",
                },  # Should be skipped
            ]
        }

        # Setup mock for job deletion
        delete_response = MagicMock()
        delete_response.status_code = 200

        # Configure mock requests
        mock_requests.get.side_effect = [user_response, jobs_response]
        mock_requests.delete.return_value = delete_response

        # Call the function
        result = get_splunk_data.cleanup_all_search_jobs()

        # Assertions
        assert result == 2  # Only 2 jobs should be deleted (the ones by test_user)
        assert mock_requests.get.call_count == 2
        assert mock_requests.delete.call_count == 2
        assert mock_time.sleep.call_count == 2

    def test_failed_user_retrieval(self, mock_requests):
        """Test failure to retrieve current user"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 401  # Unauthorized
        mock_requests.get.return_value = mock_response

        # Call the function
        result = get_splunk_data.cleanup_all_search_jobs()

        # Assertions
        assert result == 0
        mock_requests.get.assert_called_once()
        mock_requests.delete.assert_not_called()

    def test_failed_jobs_retrieval(self, mock_requests):
        """Test failure to retrieve jobs list"""
        # Setup mock for user context
        user_response = MagicMock()
        user_response.status_code = 200
        user_response.json.return_value = {
            "entry": [{"content": {"username": "test_user"}}]
        }

        # Setup mock for jobs list with failure
        jobs_response = MagicMock()
        jobs_response.status_code = 500  # Server error

        # Configure mock requests
        mock_requests.get.side_effect = [user_response, jobs_response]

        # Call the function
        result = get_splunk_data.cleanup_all_search_jobs()

        # Assertions
        assert result == 0
        assert mock_requests.get.call_count == 2
        mock_requests.delete.assert_not_called()

    def test_no_jobs_found(self, mock_requests):
        """Test case when no jobs are found"""
        # Setup mock for user context
        user_response = MagicMock()
        user_response.status_code = 200
        user_response.json.return_value = {
            "entry": [{"content": {"username": "test_user"}}]
        }

        # Setup mock for empty jobs list
        jobs_response = MagicMock()
        jobs_response.status_code = 200
        jobs_response.json.return_value = {"entry": []}

        # Configure mock requests
        mock_requests.get.side_effect = [user_response, jobs_response]

        # Call the function
        result = get_splunk_data.cleanup_all_search_jobs()

        # Assertions
        assert result == 0
        assert mock_requests.get.call_count == 2
        mock_requests.delete.assert_not_called()

    def test_json_decode_error(self, mock_requests):
        """Test JSON decode error handling"""
        # Setup mock for user context
        user_response = MagicMock()
        user_response.status_code = 200
        user_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        # Configure mock requests
        mock_requests.get.return_value = user_response

        # Call the function
        result = get_splunk_data.cleanup_all_search_jobs()

        # Assertions
        assert result == 0
        mock_requests.get.assert_called_once()
        mock_requests.delete.assert_not_called()

    def test_request_exception(self, mock_requests):
        """Test request exception handling"""
        # Setup mock to raise exception
        mock_requests.get.side_effect = requests.RequestException("Connection error")

        # Call the function
        result = get_splunk_data.cleanup_all_search_jobs()

        # Assertions
        assert result == 0
        mock_requests.get.assert_called_once()
        mock_requests.delete.assert_not_called()
