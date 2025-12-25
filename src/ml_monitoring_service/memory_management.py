"""Memory management utilities for cleaning up old models and artifacts."""

import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def get_directory_size(path: Path) -> int:
    """Calculate total size of a directory in bytes.

    Args:
        path: Path to directory

    Returns:
        Total size in bytes
    """
    total_size = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total_size += entry.stat().st_size
    except (OSError, PermissionError) as e:
        logger.warning(f"Error calculating directory size for {path}: {e}")
    return total_size


def format_bytes(size_bytes: int) -> str:
    """Format bytes as human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "123.45 MB")
    """
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def cleanup_old_files(
    directory: str | Path,
    max_age_days: int = 30,
    pattern: str = "*",
    dry_run: bool = False,
) -> tuple[int, int]:
    """Remove files older than specified age.

    Args:
        directory: Directory to clean
        max_age_days: Maximum age in days before deletion
        pattern: Glob pattern for files to consider
        dry_run: If True, only log what would be deleted without deleting

    Returns:
        Tuple of (files_deleted, bytes_freed)
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        logger.warning(f"Directory {directory} does not exist")
        return 0, 0

    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    files_deleted = 0
    bytes_freed = 0

    try:
        for file_path in directory_path.rglob(pattern):
            if not file_path.is_file():
                continue

            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                file_size = file_path.stat().st_size
                if dry_run:
                    logger.info(
                        f"[DRY RUN] Would delete: {file_path} "
                        f"(age: {file_age / 86400:.1f} days, size: {format_bytes(file_size)})"
                    )
                else:
                    logger.info(
                        f"Deleting old file: {file_path} "
                        f"(age: {file_age / 86400:.1f} days, size: {format_bytes(file_size)})"
                    )
                    file_path.unlink()
                files_deleted += 1
                bytes_freed += file_size
    except (OSError, PermissionError) as e:
        logger.error(f"Error during cleanup of {directory}: {e}")

    if files_deleted > 0:
        logger.info(
            f"Cleanup complete: {files_deleted} files deleted, "
            f"{format_bytes(bytes_freed)} freed"
        )
    return files_deleted, bytes_freed


def cleanup_old_mlruns(
    mlruns_dir: str | Path = "mlruns",
    max_age_days: int = 90,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Remove old MLflow experiment runs.

    Args:
        mlruns_dir: Path to MLflow runs directory
        max_age_days: Maximum age in days before deletion
        dry_run: If True, only log what would be deleted

    Returns:
        Tuple of (runs_deleted, bytes_freed)
    """
    mlruns_path = Path(mlruns_dir)
    if not mlruns_path.exists():
        logger.warning(f"MLflow directory {mlruns_dir} does not exist")
        return 0, 0

    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    runs_deleted = 0
    bytes_freed = 0

    try:
        # Look for experiment directories (numeric IDs)
        for exp_dir in mlruns_path.iterdir():
            if not exp_dir.is_dir() or exp_dir.name in ["models", "0", ".trash"]:
                continue

            # Look for run directories within each experiment
            for run_dir in exp_dir.iterdir():
                if not run_dir.is_dir() or run_dir.name in ["meta.yaml", "tags"]:
                    continue

                # Check run age using meta.yaml modification time
                meta_file = run_dir / "meta.yaml"
                if not meta_file.exists():
                    continue

                run_age = current_time - meta_file.stat().st_mtime
                if run_age > max_age_seconds:
                    run_size = get_directory_size(run_dir)
                    if dry_run:
                        logger.info(
                            f"[DRY RUN] Would delete run: {run_dir} "
                            f"(age: {run_age / 86400:.1f} days, size: {format_bytes(run_size)})"
                        )
                    else:
                        logger.info(
                            f"Deleting old run: {run_dir} "
                            f"(age: {run_age / 86400:.1f} days, size: {format_bytes(run_size)})"
                        )
                        shutil.rmtree(run_dir)
                    runs_deleted += 1
                    bytes_freed += run_size
    except (OSError, PermissionError) as e:
        logger.error(f"Error during MLflow cleanup: {e}")

    if runs_deleted > 0:
        logger.info(
            f"MLflow cleanup complete: {runs_deleted} runs deleted, "
            f"{format_bytes(bytes_freed)} freed"
        )
    return runs_deleted, bytes_freed


def check_disk_space(path: str | Path = ".", threshold_percent: float = 90.0) -> bool:
    """Check if disk usage is above threshold.

    Args:
        path: Path to check disk usage for
        threshold_percent: Threshold percentage (0-100)

    Returns:
        True if disk usage is above threshold, False otherwise
    """
    try:
        stat = shutil.disk_usage(path)
        usage_percent = (stat.used / stat.total) * 100
        logger.info(
            f"Disk usage: {format_bytes(stat.used)} / {format_bytes(stat.total)} "
            f"({usage_percent:.1f}%)"
        )
        return usage_percent > threshold_percent
    except (OSError, PermissionError) as e:
        logger.error(f"Error checking disk space: {e}")
        return False


def perform_maintenance_cleanup(
    output_dir: str | Path = "output",
    mlruns_dir: str | Path = "mlruns",
    max_age_days: int = 30,
    mlflow_max_age_days: int = 90,
    disk_threshold_percent: float = 85.0,
    dry_run: bool = False,
) -> None:
    """Perform full maintenance cleanup of old artifacts.

    Args:
        output_dir: Path to output directory
        mlruns_dir: Path to MLflow runs directory
        max_age_days: Maximum age for output files
        mlflow_max_age_days: Maximum age for MLflow runs
        disk_threshold_percent: Disk usage threshold for cleanup
        dry_run: If True, only log what would be deleted
    """
    logger.info("Starting maintenance cleanup...")

    # Check disk usage
    if check_disk_space(threshold_percent=disk_threshold_percent):
        logger.warning(
            f"Disk usage above {disk_threshold_percent}% threshold, "
            "performing aggressive cleanup"
        )

    # Clean up old model files
    logger.info(f"Cleaning up old files in {output_dir}...")
    cleanup_old_files(
        output_dir, max_age_days=max_age_days, pattern="*.pth", dry_run=dry_run
    )
    cleanup_old_files(
        output_dir, max_age_days=max_age_days, pattern="*.json", dry_run=dry_run
    )
    cleanup_old_files(
        output_dir, max_age_days=max_age_days, pattern="*.npy", dry_run=dry_run
    )

    # Clean up old MLflow runs
    logger.info(f"Cleaning up old MLflow runs in {mlruns_dir}...")
    cleanup_old_mlruns(mlruns_dir, max_age_days=mlflow_max_age_days, dry_run=dry_run)

    logger.info("Maintenance cleanup complete")
