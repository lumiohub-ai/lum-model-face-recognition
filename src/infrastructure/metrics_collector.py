"""System and pipeline metrics collector.

Collects:
  - CPU utilization     (psutil)
  - System RAM          (psutil)
  - GPU utilization + VRAM  (pynvml / nvidia-ml-py)
  - Per-camera FPS      (rolling 10-second window)
  - YOLO / ArcFace inference latency (rolling average)
  - Frame drop counts   (queue-full events)

Designed to be low-overhead: data is only aggregated when snapshot() is called.
"""

import threading
import time
from collections import deque
from typing import Dict, List, Optional

import psutil
from loguru import logger

# ── GPU monitoring via pynvml (nvidia-ml-py) — optional ──────────────────────
_pynvml = None
_gpu_handle = None
_gpu_name: Optional[str] = None

try:
    import pynvml

    pynvml.nvmlInit()
    _gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    _pynvml = pynvml
    _raw_name = pynvml.nvmlDeviceGetName(_gpu_handle)
    _gpu_name = _raw_name.decode() if isinstance(_raw_name, bytes) else _raw_name
    logger.info(f"GPU monitoring enabled: {_gpu_name}")
except Exception as _e:
    logger.warning(f"GPU monitoring unavailable (install nvidia-ml-py for GPU stats): {_e}")


class MetricsCollector:
    """Thread-safe collector for CPU, GPU, RAM and pipeline performance metrics.

    Usage:
        # Create once and pass to workers
        metrics = MetricsCollector()

        # In camera worker — call each processed detection-frame
        metrics.record_frame(camera_idx)

        # In GPU worker — call after each batch inference
        metrics.record_yolo_ms(elapsed_ms)
        metrics.record_arcface_ms(elapsed_ms)

        # On queue-full frame drop
        metrics.record_drop(camera_idx)

        # To get a full snapshot (for logging / Redis publishing)
        snap = metrics.snapshot(camera_indices=[0, 1, 2])
    """

    FPS_WINDOW_SEC: float = 10.0   # rolling window for FPS calculation
    FPS_DEQUE_MAX: int = 120        # max timestamps kept per camera
    LATENCY_BUFFER: int = 50        # rolling buffer size for latency averages

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Per-camera: monotonic timestamps of processed detection frames
        self._frame_ts: Dict[int, deque] = {}

        # Per-camera: cumulative frame-drop counter (GPU queue full)
        self._frame_drops: Dict[int, int] = {}

        # Inference latency buffers (milliseconds)
        self._yolo_ms: deque = deque(maxlen=self.LATENCY_BUFFER)
        self._arcface_ms: deque = deque(maxlen=self.LATENCY_BUFFER)

    # ── FPS / frame tracking ──────────────────────────────────────────────────

    def record_frame(self, camera_idx: int) -> None:
        """Record a processed detection-frame timestamp for this camera."""
        now = time.monotonic()
        with self._lock:
            if camera_idx not in self._frame_ts:
                self._frame_ts[camera_idx] = deque(maxlen=self.FPS_DEQUE_MAX)
            self._frame_ts[camera_idx].append(now)

    def record_drop(self, camera_idx: int) -> None:
        """Record a dropped frame (GPU queue was full)."""
        with self._lock:
            self._frame_drops[camera_idx] = self._frame_drops.get(camera_idx, 0) + 1

    def get_fps(self, camera_idx: int) -> float:
        """Return per-camera FPS over the rolling FPS_WINDOW_SEC window."""
        with self._lock:
            ts = self._frame_ts.get(camera_idx)
            if not ts or len(ts) < 2:
                return 0.0
            cutoff = time.monotonic() - self.FPS_WINDOW_SEC
            window = [t for t in ts if t >= cutoff]
            if len(window) < 2:
                return 0.0
            return (len(window) - 1) / (window[-1] - window[0])

    def get_drops(self, camera_idx: int) -> int:
        """Return cumulative dropped frame count for this camera."""
        with self._lock:
            return self._frame_drops.get(camera_idx, 0)

    # ── Inference latency ─────────────────────────────────────────────────────

    def record_yolo_ms(self, ms: float) -> None:
        """Record one YOLO batch inference duration in milliseconds."""
        with self._lock:
            self._yolo_ms.append(ms)

    def record_arcface_ms(self, ms: float) -> None:
        """Record one ArcFace batch inference duration in milliseconds."""
        with self._lock:
            self._arcface_ms.append(ms)

    # ── System resource stats (static helpers) ────────────────────────────────

    @staticmethod
    def cpu_percent() -> float:
        """Non-blocking CPU usage % (averaged since last call)."""
        return psutil.cpu_percent(interval=None)

    @staticmethod
    def memory() -> Dict:
        """System RAM stats."""
        m = psutil.virtual_memory()
        return {
            "used_gb": round(m.used / 1e9, 2),
            "total_gb": round(m.total / 1e9, 2),
            "percent": round(m.percent, 1),
        }

    @staticmethod
    def gpu() -> Optional[Dict]:
        """GPU utilization and VRAM stats. Returns None if pynvml unavailable."""
        if _pynvml is None or _gpu_handle is None:
            return None
        try:
            util = _pynvml.nvmlDeviceGetUtilizationRates(_gpu_handle)
            mem = _pynvml.nvmlDeviceGetMemoryInfo(_gpu_handle)
            return {
                "name": _gpu_name,
                "util_percent": util.gpu,
                "mem_used_mb": round(mem.used / 1e6, 1),
                "mem_total_mb": round(mem.total / 1e6, 1),
                "mem_percent": round(mem.used / mem.total * 100, 1),
            }
        except Exception as e:
            logger.debug(f"GPU metrics read error: {e}")
            return None

    # ── Full snapshot ─────────────────────────────────────────────────────────

    def snapshot(self, camera_indices: Optional[List[int]] = None) -> Dict:
        """Return a complete metrics snapshot dict.

        Args:
            camera_indices: list of camera indices to include; defaults to all seen.

        Returns a dict with keys: timestamp, cpu_percent, memory, gpu, cameras, inference.
        """
        indices = camera_indices if camera_indices is not None else list(self._frame_ts.keys())

        with self._lock:
            yolo_avg = sum(self._yolo_ms) / len(self._yolo_ms) if self._yolo_ms else 0.0
            arcface_avg = sum(self._arcface_ms) / len(self._arcface_ms) if self._arcface_ms else 0.0

        return {
            "timestamp": time.time(),
            "cpu_percent": self.cpu_percent(),
            "memory": self.memory(),
            "gpu": self.gpu(),
            "cameras": {
                str(idx): {
                    "fps": round(self.get_fps(idx), 2),
                    "frame_drops": self.get_drops(idx),
                }
                for idx in indices
            },
            "inference": {
                "yolo_avg_ms": round(yolo_avg, 1),
                "arcface_avg_ms": round(arcface_avg, 1),
            },
        }

    def log_summary(self, camera_indices: Optional[List[int]] = None) -> None:
        """Log a compact metrics summary line to loguru (INFO level)."""
        snap = self.snapshot(camera_indices)
        gpu = snap["gpu"]
        gpu_str = (
            f"GPU={gpu['util_percent']}% VRAM={gpu['mem_used_mb']:.0f}/{gpu['mem_total_mb']:.0f}MB({gpu['mem_percent']:.0f}%)"
            if gpu else "GPU=N/A"
        )
        fps_parts = [
            f"cam{idx}={snap['cameras'][str(idx)]['fps']:.1f}fps"
            f"(drops={snap['cameras'][str(idx)]['frame_drops']})"
            for idx in (camera_indices or [])
        ]
        logger.info(
            f"[Metrics] CPU={snap['cpu_percent']:.0f}% "
            f"RAM={snap['memory']['used_gb']:.1f}/{snap['memory']['total_gb']:.1f}GB({snap['memory']['percent']:.0f}%) "
            f"{gpu_str} | "
            + (", ".join(fps_parts) if fps_parts else "no cameras yet")
            + f" | YOLO={snap['inference']['yolo_avg_ms']:.0f}ms ArcFace={snap['inference']['arcface_avg_ms']:.0f}ms"
        )

    def check_alerts(
        self,
        camera_indices: Optional[List[int]] = None,
        fps_threshold: float = 1.0,
        gpu_mem_threshold: float = 90.0,
        ram_threshold: float = 90.0,
    ) -> List[Dict]:
        """Check for critical conditions and return a list of alert dicts.

        Args:
            camera_indices:   cameras to check
            fps_threshold:    alert if FPS drops below this (0 = camera just started)
            gpu_mem_threshold: alert if GPU VRAM % exceeds this
            ram_threshold:    alert if system RAM % exceeds this
        """
        alerts = []
        snap = self.snapshot(camera_indices)

        for idx in (camera_indices or []):
            fps = snap["cameras"].get(str(idx), {}).get("fps", 0.0)
            # Only alert once we've seen some frames (fps > 0 means active)
            if 0 < fps < fps_threshold:
                alerts.append({
                    "type": "LOW_FPS",
                    "camera_idx": idx,
                    "fps": fps,
                    "threshold": fps_threshold,
                    "message": f"Camera {idx} FPS={fps:.2f} is below threshold ({fps_threshold})",
                })

        gpu = snap["gpu"]
        if gpu and gpu["mem_percent"] > gpu_mem_threshold:
            alerts.append({
                "type": "GPU_MEMORY_HIGH",
                "mem_percent": gpu["mem_percent"],
                "threshold": gpu_mem_threshold,
                "message": f"GPU VRAM at {gpu['mem_percent']:.0f}% — risk of OOM",
            })

        if snap["memory"]["percent"] > ram_threshold:
            alerts.append({
                "type": "RAM_HIGH",
                "percent": snap["memory"]["percent"],
                "threshold": ram_threshold,
                "message": f"System RAM at {snap['memory']['percent']:.0f}%",
            })

        return alerts

    def cleanup(self) -> None:
        """Shut down pynvml if it was initialised."""
        if _pynvml is not None:
            try:
                _pynvml.nvmlShutdown()
            except Exception:
                pass
