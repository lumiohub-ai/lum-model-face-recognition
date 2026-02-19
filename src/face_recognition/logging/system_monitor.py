"""System performance monitoring with periodic CSV logging.

Tracks FPS statistics (avg/min/max) and process memory usage,
buffering samples in memory and flushing to daily CSV log files
on a configurable interval (default: 1 hour).
"""

import csv
import os
import time
from datetime import date, datetime
from typing import Dict, List

import psutil
from loguru import logger


class SystemMonitor:
    """Monitors system performance metrics and writes periodic CSV logs.

    Design:
    - IO efficient: buffers samples in memory, flushes once per interval
    - Daily file rotation: one file per day via date in filename
    - Minimal overhead: no blocking IO in the hot path
    """

    CSV_HEADER = ["timestamp", "avg_fps", "min_fps", "max_fps", "memory_mb"]

    def __init__(
        self,
        log_dir: str,
        flush_interval_seconds: float = 3600.0,
        file_prefix: str = "system_metrics",
    ):
        """Initialize the system monitor.

        Args:
            log_dir: Directory for CSV log files.
            flush_interval_seconds: How often to flush buffer to disk (default 1h).
            file_prefix: Prefix for log file names.
        """
        self.log_dir = log_dir
        self.flush_interval_seconds = flush_interval_seconds
        self.file_prefix = file_prefix

        self._buffer: List[Dict] = []
        self._last_flush_time: float = time.time()
        self._process = psutil.Process()

        os.makedirs(self.log_dir, exist_ok=True)

        logger.info(
            f"SystemMonitor initialized: dir={log_dir}, "
            f"flush_interval={flush_interval_seconds}s"
        )

    def record_sample(
        self,
        avg_fps: float,
        min_fps: float,
        max_fps: float,
    ) -> None:
        """Record a performance sample into the in-memory buffer.

        Automatically flushes to disk when the flush interval elapses.

        Args:
            avg_fps: Average FPS for this sampling interval.
            min_fps: Minimum FPS for this sampling interval.
            max_fps: Maximum FPS for this sampling interval.
        """
        memory_mb = self._get_process_memory_mb()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self._buffer.append({
            "timestamp": timestamp,
            "avg_fps": round(avg_fps, 2),
            "min_fps": round(min_fps, 2),
            "max_fps": round(max_fps, 2),
            "memory_mb": round(memory_mb, 1),
        })

        current_time = time.time()
        if current_time - self._last_flush_time >= self.flush_interval_seconds:
            self.flush()
            self._last_flush_time = current_time

    def flush(self) -> None:
        """Flush buffered samples to the daily CSV file.

        Safe to call at any time (e.g. on shutdown). No-op if buffer is empty.
        """
        if not self._buffer:
            return

        file_path = self._get_weekly_file_path()

        try:
            write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0

            with open(file_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_HEADER)
                if write_header:
                    writer.writeheader()
                writer.writerows(self._buffer)

            logger.info(
                f"SystemMonitor flushed {len(self._buffer)} samples to {file_path}"
            )
            self._buffer.clear()

        except OSError as e:
            logger.error(f"SystemMonitor flush failed: {e}")

    def _get_weekly_file_path(self) -> str:
        """Get the file path for this week's log file.

        Uses ISO week number so files rotate every Monday.
        Filename format: system_metrics_2026-W05.csv
        """
        today = date.today()
        year, week, _ = today.isocalendar()
        filename = f"{self.file_prefix}_{year}-W{week:02d}.csv"
        return os.path.join(self.log_dir, filename)

    def _get_process_memory_mb(self) -> float:
        """Get current process RSS memory in MB."""
        try:
            return self._process.memory_info().rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
