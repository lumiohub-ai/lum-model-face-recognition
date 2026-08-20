"""SQLite-backed metrics history store.

Persists MetricsCollector snapshots on a configurable interval so you can
review past-day resource usage from the monitoring dashboard.

Schema (one row per snapshot):
    ts              Unix timestamp (float)
    cpu_percent     float
    ram_percent     float
    ram_used_gb     float
    gpu_util        float  (NULL if no GPU)
    gpu_mem_percent float  (NULL if no GPU)
    yolo_ms         float
    arcface_ms      float
    cameras_json    TEXT   JSON {"0": {"fps": 32.1, "drops": 0, "read_ms": .., "decode_ms": ..}, ...}
    process_json    TEXT   JSON {"rss_gb": .., "threads": .., "open_fds": .., "uptime_sec": ..}
    pipeline_json   TEXT   JSON {"tracks": .., "crops": .., "read_ms_avg": .., ...} - whatever
                           gauges are registered; stored as a blob rather than
                           fixed columns since the gauge set changes over time.
"""

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from loguru import logger

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    cpu_percent REAL,
    ram_percent REAL,
    ram_used_gb REAL,
    gpu_util    REAL,
    gpu_mem_pct REAL,
    yolo_ms     REAL,
    arcface_ms  REAL,
    cameras_json TEXT,
    process_json TEXT,
    pipeline_json TEXT,
    arcface_det_ms REAL,
    arcface_embed_ms REAL
);
CREATE INDEX IF NOT EXISTS idx_metrics_ts ON metrics(ts);
"""

# Columns added after the initial release - CREATE TABLE IF NOT EXISTS above
# is a no-op on a DB that already has the `metrics` table, so an existing DB
# needs these added explicitly. _init_db applies them, ignoring "duplicate
# column" if they're already there.
_MIGRATIONS = [
    "ALTER TABLE metrics ADD COLUMN process_json TEXT",
    "ALTER TABLE metrics ADD COLUMN pipeline_json TEXT",
    # LSO-117: split of arcface_ms (see MetricsCollector.record_arcface_ms's
    # docstring) - detection (unbatched) vs. embedding (batched) time.
    "ALTER TABLE metrics ADD COLUMN arcface_det_ms REAL",
    "ALTER TABLE metrics ADD COLUMN arcface_embed_ms REAL",
]

_INSERT_SQL = """
INSERT INTO metrics
    (ts, cpu_percent, ram_percent, ram_used_gb, gpu_util, gpu_mem_pct,
     yolo_ms, arcface_ms, cameras_json, process_json, pipeline_json,
     arcface_det_ms, arcface_embed_ms)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
"""

# Keep 30 days of data; prune rows older than this on startup
_RETENTION_DAYS = 30


class MetricsStore:
    """Persists metrics snapshots to SQLite and provides history queries.

    Runs its own background writer thread that calls MetricsCollector.snapshot()
    every `interval_sec` seconds.

    Usage::

        store = MetricsStore(metrics, db_path="/data/metrics.db", interval_sec=30)
        store.start()
        ...
        store.stop()

        # Query a day
        rows = store.get_day("2026-03-27")  # returns list of dicts
    """

    def __init__(
        self,
        metrics,
        db_path: str = "metrics.db",
        interval_sec: float = 30.0,
        camera_indices: Optional[List[int]] = None,
    ):
        self._metrics = metrics
        self._db_path = db_path
        self._interval = interval_sec
        self._camera_indices = camera_indices or []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="metrics-store"
        )
        self._thread.start()
        logger.info(f"MetricsStore started (db={self._db_path}, interval={self._interval}s)")

    def stop(self) -> None:
        self._running = False
        logger.info("MetricsStore stopped")

    def update_camera_indices(self, indices: List[int]) -> None:
        self._camera_indices = indices

    # ── Writer loop ───────────────────────────────────────────────────────────

    def _writer_loop(self) -> None:
        while self._running:
            try:
                self._write_snapshot()
            except Exception as e:
                logger.debug(f"MetricsStore write error: {e}")
            time.sleep(self._interval)

    def _write_snapshot(self) -> None:
        snap = self._metrics.snapshot(self._camera_indices)
        gpu = snap.get("gpu") or {}
        inf = snap.get("inference") or {}
        mem = snap.get("memory") or {}

        row = (
            snap["timestamp"],
            snap.get("cpu_percent"),
            mem.get("percent"),
            mem.get("used_gb"),
            gpu.get("util_percent"),
            gpu.get("mem_percent"),
            inf.get("yolo_avg_ms"),
            inf.get("arcface_avg_ms"),
            json.dumps(snap.get("cameras", {})),
            json.dumps(snap.get("process") or {}),
            json.dumps(snap.get("pipeline") or {}),
            inf.get("arcface_det_avg_ms"),
            inf.get("arcface_embed_avg_ms"),
        )

        with self._connect() as conn:
            conn.execute(_INSERT_SQL, row)

    # ── Query API ─────────────────────────────────────────────────────────────

    def get_day(self, date_str: str) -> List[Dict[str, Any]]:
        """Return all rows for a given date (YYYY-MM-DD, local time).

        Each row is a dict with keys matching column names plus a decoded
        `cameras` dict (from cameras_json).
        """
        import datetime
        try:
            d = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return []

        day_start = d.timestamp()
        day_end = (d + datetime.timedelta(days=1)).timestamp()

        sql = """
        SELECT ts, cpu_percent, ram_percent, ram_used_gb,
               gpu_util, gpu_mem_pct, yolo_ms, arcface_ms, cameras_json,
               process_json, pipeline_json, arcface_det_ms, arcface_embed_ms
        FROM metrics
        WHERE ts >= ? AND ts < ?
        ORDER BY ts ASC
        """

        def _load(blob: Optional[str]) -> Dict[str, Any]:
            try:
                return json.loads(blob) if blob else {}
            except Exception:
                return {}

        rows = []
        with self._connect() as conn:
            for r in conn.execute(sql, (day_start, day_end)):
                rows.append({
                    "ts": r[0],
                    "cpu_percent": r[1],
                    "ram_percent": r[2],
                    "ram_used_gb": r[3],
                    "gpu_util": r[4],
                    "gpu_mem_pct": r[5],
                    "yolo_ms": r[6],
                    "arcface_ms": r[7],
                    "cameras": _load(r[8]),
                    # Rows written before this field existed have no process/
                    # pipeline data - {} rather than a missing key, so callers
                    # can use .get() uniformly across old and new rows.
                    "process": _load(r[9]),
                    "pipeline": _load(r[10]),
                    "arcface_det_ms": r[11],
                    "arcface_embed_ms": r[12],
                })
        return rows

    def available_dates(self) -> List[str]:
        """Return sorted list of YYYY-MM-DD strings that have data."""
        import datetime
        sql = "SELECT DISTINCT CAST(ts/86400 AS INTEGER)*86400 FROM metrics ORDER BY 1 DESC"
        dates = []
        with self._connect() as conn:
            for (epoch_day,) in conn.execute(sql):
                try:
                    dates.append(
                        datetime.datetime.utcfromtimestamp(epoch_day).strftime("%Y-%m-%d")
                    )
                except Exception:
                    pass
        return dates

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_CREATE_SQL)
            for stmt in _MIGRATIONS:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
            # Prune old data
            cutoff = time.time() - _RETENTION_DAYS * 86400
            conn.execute("DELETE FROM metrics WHERE ts < ?", (cutoff,))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
