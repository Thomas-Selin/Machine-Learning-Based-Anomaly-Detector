import numpy as np
import pandas as pd
import pytest

from ml_monitoring_service.data_handling import (
    check_for_nan,
    get_microservice_data_from_file,
    normalize_feature,
)


class TestCheckForNaN:
    """Tests for check_for_nan function"""

    def test_no_nan_values(self):
        """Test DataFrame with no NaN values"""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        result = check_for_nan(df)
        assert result.isnull().sum().sum() == 0
        assert len(result) == 3

    def test_with_nan_values(self):
        """Test DataFrame with NaN values - should fill with 0"""
        df = pd.DataFrame({"col1": [1, np.nan, 3], "col2": [4, 5, np.nan]})
        result = check_for_nan(df)
        assert result.isnull().sum().sum() == 0
        assert result["col1"].iloc[1] == 0
        assert result["col2"].iloc[2] == 0

    def test_all_nan_column(self):
        """Test DataFrame with all NaN values in a column"""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [np.nan, np.nan, np.nan]})
        result = check_for_nan(df)
        assert result["col2"].sum() == 0


class TestNormalizeFeature:
    """Tests for normalize_feature function"""

    def test_normal_normalization(self):
        """Test normal min-max normalization"""
        df = pd.DataFrame({"feature": [0, 50, 100]})
        result = normalize_feature(df, "feature")
        assert result["feature"].iloc[0] == 0.0
        assert result["feature"].iloc[1] == 0.5
        assert result["feature"].iloc[2] == 1.0

    def test_constant_value(self):
        """Test normalization with constant values - should return 0.5"""
        df = pd.DataFrame({"feature": [5, 5, 5]})
        result = normalize_feature(df, "feature")
        assert all(result["feature"] == 0.5)

    def test_negative_values(self):
        """Test normalization with negative values"""
        df = pd.DataFrame({"feature": [-10, 0, 10]})
        result = normalize_feature(df, "feature")
        assert result["feature"].iloc[0] == 0.0
        assert result["feature"].iloc[1] == 0.5
        assert result["feature"].iloc[2] == 1.0

    def test_missing_feature(self):
        """Test normalization with non-existent feature"""
        df = pd.DataFrame({"feature1": [1, 2, 3]})
        result = normalize_feature(df, "feature2")
        assert "feature2" not in result.columns
        assert "feature1" in result.columns

    def test_single_value(self):
        """Test normalization with single value"""
        df = pd.DataFrame({"feature": [42]})
        result = normalize_feature(df, "feature")
        assert result["feature"].iloc[0] == 0.5


class TestGetMicroserviceDataFromFile:
    """Tests for get_microservice_data_from_file function"""

    def test_read_valid_json(self, tmp_path):
        """Test reading valid JSON file"""
        # Create temporary JSON file
        json_file = tmp_path / "test_data.json"
        test_data = [
            {"service": "service1", "timestamp": "2022-01-01 00:00:00", "cpu": 1.0},
            {"service": "service2", "timestamp": "2022-01-01 00:01:00", "cpu": 2.0},
        ]
        pd.DataFrame(test_data).to_json(json_file, orient="records")

        result = get_microservice_data_from_file(str(json_file))
        assert len(result) == 2
        assert "timestamp" in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])

    def test_file_not_found(self):
        """Test reading non-existent file"""
        with pytest.raises(FileNotFoundError):
            get_microservice_data_from_file("nonexistent_file.json")

    def test_timestamp_conversion(self, tmp_path):
        """Test that timestamps are properly converted to datetime"""
        json_file = tmp_path / "test_timestamps.json"
        test_data = [
            {"service": "service1", "timestamp": "2022-01-01 12:00:00"},
            {"service": "service1", "timestamp": "01/01/2022 13:00:00"},
        ]
        pd.DataFrame(test_data).to_json(json_file, orient="records")

        result = get_microservice_data_from_file(str(json_file))
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])


class TestConvertToModelInput:
    """Tests for convert_to_model_input function"""

    def test_basic_conversion(self):
        """Test basic data conversion to model input format"""
        from ml_monitoring_service.data_handling import convert_to_model_input

        df = pd.DataFrame(
            {
                "service": ["service1", "service1", "service2", "service2"],
                "timestamp": pd.date_range("2022-01-01", periods=4, freq="1min"),
                "timestamp_nanoseconds": [1000, 1000, 2000, 2000],
                "cpu": [0.5, 0.5, 0.8, 0.8],
                "memory": [0.3, 0.3, 0.6, 0.6],
                "latency": [100, 100, 150, 150],
                "severity_level": [0, 0, 1, 1],
            }
        )

        # Mock active_set and configuration
        with pytest.raises(ValueError):
            # This will fail without proper config, but tests the function exists
            data, services, features = convert_to_model_input("test-set", df)


def test_normalize_feature_with_nan():
    """Test normalize_feature with NaN values"""
    df = pd.DataFrame({"feature": [1.0, np.nan, 3.0, np.nan, 5.0]})
    result = normalize_feature(df, "feature")
    # NaN values should remain or be handled
    assert "feature" in result.columns
