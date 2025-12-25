"""Tests for memory management utilities."""

import time
from pathlib import Path

from ml_monitoring_service.memory_management import (
    check_disk_space,
    cleanup_old_files,
    format_bytes,
    get_directory_size,
)


class TestMemoryManagement:
    """Test memory management functions."""

    def test_format_bytes(self):
        """Test byte formatting."""
        assert format_bytes(0) == "0.00 B"
        assert format_bytes(1023) == "1023.00 B"
        assert format_bytes(1024) == "1.00 KB"
        assert format_bytes(1024 * 1024) == "1.00 MB"
        assert format_bytes(1024 * 1024 * 1024) == "1.00 GB"
        assert format_bytes(1536) == "1.50 KB"

    def test_get_directory_size(self, tmp_path):
        """Test directory size calculation."""
        # Create test files
        file1 = tmp_path / "file1.txt"
        file1.write_text("a" * 100)
        file2 = tmp_path / "file2.txt"
        file2.write_text("b" * 200)

        size = get_directory_size(tmp_path)
        assert size == 300

    def test_get_directory_size_with_subdirs(self, tmp_path):
        """Test directory size with subdirectories."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        file1 = tmp_path / "file1.txt"
        file1.write_text("a" * 100)
        file2 = subdir / "file2.txt"
        file2.write_text("b" * 200)

        size = get_directory_size(tmp_path)
        assert size == 300

    def test_cleanup_old_files_dry_run(self, tmp_path):
        """Test cleanup in dry-run mode."""
        # Create old file
        old_file = tmp_path / "old.txt"
        old_file.write_text("old content")

        # Make it old by modifying timestamp
        old_time = time.time() - (35 * 24 * 60 * 60)  # 35 days ago
        Path(old_file).touch()
        import os

        os.utime(old_file, (old_time, old_time))

        # Create new file
        new_file = tmp_path / "new.txt"
        new_file.write_text("new content")

        # Dry run shouldn't delete anything
        deleted, freed = cleanup_old_files(tmp_path, max_age_days=30, dry_run=True)
        assert deleted == 1
        assert freed == len("old content")
        assert old_file.exists()  # File should still exist

    def test_cleanup_old_files_actual_deletion(self, tmp_path):
        """Test actual file deletion."""
        # Create old file
        old_file = tmp_path / "old.txt"
        old_file.write_text("old content")

        # Make it old
        old_time = time.time() - (35 * 24 * 60 * 60)
        Path(old_file).touch()
        import os

        os.utime(old_file, (old_time, old_time))

        # Create new file
        new_file = tmp_path / "new.txt"
        new_file.write_text("new content")

        # Actual deletion
        deleted, freed = cleanup_old_files(tmp_path, max_age_days=30, dry_run=False)
        assert deleted == 1
        assert freed == len("old content")
        assert not old_file.exists()  # File should be deleted
        assert new_file.exists()  # New file should remain

    def test_cleanup_old_files_with_pattern(self, tmp_path):
        """Test cleanup with file pattern."""
        # Create old .txt and .log files
        old_txt = tmp_path / "old.txt"
        old_txt.write_text("old txt")
        old_log = tmp_path / "old.log"
        old_log.write_text("old log")

        # Make them old
        old_time = time.time() - (35 * 24 * 60 * 60)
        import os

        for f in [old_txt, old_log]:
            Path(f).touch()
            os.utime(f, (old_time, old_time))

        # Clean only .txt files
        deleted, freed = cleanup_old_files(
            tmp_path, max_age_days=30, pattern="*.txt", dry_run=False
        )
        assert deleted == 1
        assert freed == len("old txt")
        assert not old_txt.exists()
        assert old_log.exists()  # .log file should remain

    def test_check_disk_space(self, tmp_path):
        """Test disk space checking."""
        # This should work on any system
        result = check_disk_space(tmp_path, threshold_percent=100.0)
        assert result is False  # Disk usage should be below 100%

        result = check_disk_space(tmp_path, threshold_percent=0.0)
        assert result is True  # Disk usage should be above 0%

    def test_cleanup_nonexistent_directory(self):
        """Test cleanup on nonexistent directory."""
        deleted, freed = cleanup_old_files("/nonexistent/path", dry_run=False)
        assert deleted == 0
        assert freed == 0
