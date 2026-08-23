"""System and pipeline metrics collector.

Collects:
  - CPU utilization     (psutil)
  - System RAM          (psutil)
  - GPU utilization + VRAM  (pynvml / nvidia-ml-py)
  - Per-camera FPS      (rolling 10-second window)
  - YOLO / ArcFace inference latency (rolling average)
  - Action recognition latency + outcome counts (Ollama VLM)
  - Frame drop counts   (queue-full events)

Designed to be low-overhead: data is only aggregated when snapshot() is called.
"""

import threading
import time
from collections import Counter, deque
from typing import Any, Callable, Dict, List, Optional

import psutil
from loguru import logger

# This process — used for our own RSS/threads/fds (see MetricsCollector.process).
_PROC = psutil.Process()

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
        metrics.record_frame(camera_id)

        # In GPU worker — call after each batch inference, with the real
        # batch size (frame/crop count), not left at the batch_size=1 default
        metrics.record_yolo_ms(elapsed_ms, batch_size=len(frames))
        metrics.record_arcface_ms(elapsed_ms, batch_size=len(person_rois))

        # On queue-full frame drop
        metrics.record_drop(camera_id)

        # To get a full snapshot (for logging / Redis publishing)
        snap = metrics.snapshot(camera_ids=[0, 1, 2])
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
        # arcface_ms is the WHOLE _run_arcface_batch call (detect + embed +
        # Python overhead); these two split it (LSO-117). See
        # record_arcface_det_ms/record_arcface_embed_ms for why.
        self._arcface_det_ms: deque = deque(maxlen=self.LATENCY_BUFFER)
        self._arcface_embed_ms: deque = deque(maxlen=self.LATENCY_BUFFER)

        # Action recognition (Ollama VLM): latency plus per-outcome counters
        self._action_ms: deque = deque(maxlen=self.LATENCY_BUFFER)
        self._action_counts: Counter = Counter()
        self._action_queue_depth: int = 0

        # Live gauges: name -> zero-arg callable, polled at snapshot() time.
        # Lets pipeline components (track managers, stream handlers, GPU worker)
        # expose current sizes without this module importing them.
        self._gauges: Dict[str, Callable[[], Any]] = {}
        # Per-camera gauges: name -> fn(camera_id) -> value
        self._camera_gauges: Dict[str, Callable[[int], Any]] = {}

    # ── Live gauges ───────────────────────────────────────────────────────────

    def register_gauge(self, name: str, fn: Callable[[], Any]) -> None:
        """Register a live value polled on every snapshot.

        A failing gauge must never break metrics collection, so poll errors are
        swallowed and reported as None rather than propagating.
        """
        with self._lock:
            self._gauges[name] = fn

    def register_camera_gauge(self, name: str, fn: Callable[[int], Any]) -> None:
        """Register a per-camera gauge: fn(camera_id) -> value.

        Kept separate from register_gauge so per-camera values land inside each
        camera's block rather than as a flat pile of `decode_ms_0`, `decode_ms_1`…
        """
        with self._lock:
            self._camera_gauges[name] = fn

    def _read_camera_gauge(self, camera_id: int, name: str) -> Any:
        with self._lock:
            fn = self._camera_gauges.get(name)
        if fn is None:
            return 0.0 if name.endswith("_ms") else None
        try:
            return fn(camera_id)
        except Exception as e:
            logger.debug(f"camera gauge '{name}'[{camera_id}] read error: {e}")
            return 0.0 if name.endswith("_ms") else None

    def _read_gauges(self) -> Dict[str, Any]:
        with self._lock:
            gauges = dict(self._gauges)
        out: Dict[str, Any] = {}
        for name, fn in gauges.items():
            try:
                out[name] = fn()
            except Exception as e:  # a broken gauge must not kill the snapshot
                logger.debug(f"gauge '{name}' read error: {e}")
                out[name] = None
        return out

    # ── FPS / frame tracking ──────────────────────────────────────────────────

    def record_frame(self, camera_id: int) -> None:
        """Record a processed detection-frame timestamp for this camera."""
        now = time.monotonic()
        with self._lock:
            if camera_id not in self._frame_ts:
                self._frame_ts[camera_id] = deque(maxlen=self.FPS_DEQUE_MAX)
            self._frame_ts[camera_id].append(now)

    def record_drop(self, camera_id: int) -> None:
        """Record a dropped frame (GPU queue was full)."""
        with self._lock:
            self._frame_drops[camera_id] = self._frame_drops.get(camera_id, 0) + 1

    def get_fps(self, camera_id: int) -> float:
        """Return per-camera FPS over the rolling FPS_WINDOW_SEC window."""
        with self._lock:
            ts = self._frame_ts.get(camera_id)
            if not ts or len(ts) < 2:
                return 0.0
            cutoff = time.monotonic() - self.FPS_WINDOW_SEC
            window = [t for t in ts if t >= cutoff]
            if len(window) < 2:
                return 0.0
            return (len(window) - 1) / (window[-1] - window[0])

    def get_drops(self, camera_id: int) -> int:
        """Return cumulative dropped frame count for this camera."""
        with self._lock:
            return self._frame_drops.get(camera_id, 0)

    # ── Inference latency ─────────────────────────────────────────────────────

    def record_yolo_ms(self, ms: float, batch_size: int = 1) -> None:
        """Record one YOLO batch inference duration in milliseconds.

        batch_size is how many camera frames were in that call (1-7 here,
        whatever was ready when the loop collected the batch) - without it,
        the raw ms is a per-call total that swings with batch size and
        can't be compared to a per-frame number like decode time.
        """
        with self._lock:
            self._yolo_ms.append((ms, max(1, batch_size)))

    def record_arcface_ms(self, ms: float, batch_size: int = 1) -> None:
        """Record one ArcFace batch inference duration in milliseconds.

        batch_size is how many person crops (across all cameras) were
        processed sequentially in that call - same per-call-total caveat
        as record_yolo_ms.

        This is the WHOLE _run_arcface_batch call - detection plus embedding
        (both per-item, see gpu_worker) plus Python overhead. Kept unchanged
        in meaning across LSO-117 so history stays comparable: a baseline
        recorded before that work is still a fair comparison against this
        field today. Use record_arcface_det_ms / record_arcface_embed_ms
        below to see which phase a given change actually moved - detection
        currently dominates, at roughly 85-90% of this total.
        """
        with self._lock:
            self._arcface_ms.append((ms, max(1, batch_size)))

    def record_arcface_det_ms(self, ms: float, batch_size: int = 1) -> None:
        """Record ArcFace face-detection+alignment time (LSO-117).

        Still N separate per-ROI detect_and_align() calls (SCRFD has no batch
        path - LSO-118), so batch_size here is "how many ROIs were looped
        over," not a real batch. ms is the summed wall time across that loop
        for one _run_arcface_batch cycle.
        """
        with self._lock:
            self._arcface_det_ms.append((ms, max(1, batch_size)))

    def record_arcface_embed_ms(self, ms: float, batch_size: int = 1) -> None:
        """Record ArcFace embedding time (LSO-117).

        N separate embed_batch() calls, each of exactly one crop - a varying
        batch size makes onnxruntime re-plan and costs ~30x (see the embed
        loop in gpu_worker). batch_size is "how many faces were embedded this
        cycle," not a real batch, so ms should scale roughly linearly with it.
        """
        with self._lock:
            self._arcface_embed_ms.append((ms, max(1, batch_size)))

    def record_action_inference(
        self, ms: float, status: str, queue_depth: int = 0
    ) -> None:
        """Record one action-recognition inference.

        Args:
            ms: Wall-clock duration in milliseconds
            status: One of "ok", "error", "timeout", "parse_fail"
            queue_depth: Pending requests at the time of recording
        """
        with self._lock:
            self._action_ms.append(ms)
            self._action_counts[status] += 1
            self._action_queue_depth = queue_depth

    # ── System resource stats (static helpers) ────────────────────────────────

    @staticmethod
    def cpu_percent() -> float:
        """Non-blocking CPU usage % (averaged since last call)."""
        return psutil.cpu_percent(interval=None)

    @staticmethod
    def memory() -> Dict:
        """System RAM stats (whole host, not this process)."""
        m = psutil.virtual_memory()
        return {
            "used_gb": round(m.used / 1e9, 2),
            "total_gb": round(m.total / 1e9, 2),
            "percent": round(m.percent, 1),
        }

    @staticmethod
    def process() -> Dict:
        """This process's own footprint.

        `memory()` above is host-wide, so it cannot show whether *we* are the
        thing growing — which is exactly the question during a leak. rss_gb is
        the number to watch/alert on; threads and fds catch leaks of those too.
        """
        # rss_gb is the field that actually matters here (it's what a leak
        # investigation watches); num_fds() in particular can fail on
        # non-Linux/sandboxed environments. Read it separately so a failure
        # there doesn't take rss_gb down with it.
        out: Dict[str, Any] = {}
        try:
            with _PROC.oneshot():
                mem = _PROC.memory_info()
                out["rss_gb"] = round(mem.rss / 1e9, 2)
                out["vms_gb"] = round(mem.vms / 1e9, 2)
                out["threads"] = _PROC.num_threads()
                out["uptime_sec"] = round(time.time() - _PROC.create_time())
        except Exception as e:
            logger.debug(f"process metrics read error: {e}")
        try:
            out["open_fds"] = _PROC.num_fds()
        except Exception as e:
            logger.debug(f"process open_fds read error: {e}")
        return out

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

    def snapshot(self, camera_ids: Optional[List[int]] = None) -> Dict:
        """Return a complete metrics snapshot dict.

        Args:
            camera_ids: list of camera indices to include; defaults to all seen.

        Returns a dict with keys: timestamp, cpu_percent, memory, gpu, cameras, inference.
        """
        indices = camera_ids if camera_ids is not None else list(self._frame_ts.keys())

        def _batch_stats(buf: deque) -> Dict[str, float]:
            """Per-call average ms/batch-size, plus the honest per-item cost
            (total ms / total items) - the per-call average alone hides
            whether a slow reading is "GPU is slow" or "batch was big"."""
            if not buf:
                return {"avg_call_ms": 0.0, "avg_batch_size": 0.0, "avg_item_ms": 0.0}
            total_ms = sum(ms for ms, _ in buf)
            total_items = sum(n for _, n in buf)
            return {
                "avg_call_ms": total_ms / len(buf),
                "avg_batch_size": total_items / len(buf),
                "avg_item_ms": (total_ms / total_items) if total_items else 0.0,
            }

        with self._lock:
            yolo_stats = _batch_stats(self._yolo_ms)
            arcface_stats = _batch_stats(self._arcface_ms)
            arcface_det_stats = _batch_stats(self._arcface_det_ms)
            arcface_embed_stats = _batch_stats(self._arcface_embed_ms)
            action_avg = sum(self._action_ms) / len(self._action_ms) if self._action_ms else 0.0
            action_counts = dict(self._action_counts)
            action_queue_depth = self._action_queue_depth

        return {
            "timestamp": time.time(),
            "cpu_percent": self.cpu_percent(),
            "memory": self.memory(),
            "process": self.process(),
            "pipeline": self._read_gauges(),
            "gpu": self.gpu(),
            "cameras": {
                str(idx): {
                    "fps": round(self.get_fps(idx), 2),
                    "frame_drops": self.get_drops(idx),
                    # read_ms = blocked waiting for the next frame (network/
                    # demux stall). decode_ms = actual CPU cost of decoding a
                    # frame that already arrived - the dominant CPU consumer
                    # at high camera counts. Keep these separate: a high
                    # read_ms means the CAMERA is slow, a high decode_ms means
                    # WE are slow.
                    "read_ms": round(self._read_camera_gauge(idx, "read_ms"), 1),
                    "decode_ms": round(self._read_camera_gauge(idx, "decode_ms"), 1),
                    "stream_state": self._read_camera_gauge(idx, "stream_state"),
                }
                for idx in indices
            },
            "inference": {
                # Kept for existing consumers (metrics_server dashboard,
                # metrics_store) - per-call total, swings with batch size.
                "yolo_avg_ms": round(yolo_stats["avg_call_ms"], 1),
                "arcface_avg_ms": round(arcface_stats["avg_call_ms"], 1),
                # Normalized per-item cost - the actually comparable number.
                "yolo_avg_batch_size": round(yolo_stats["avg_batch_size"], 1),
                "yolo_ms_per_frame": round(yolo_stats["avg_item_ms"], 1),
                "arcface_avg_batch_size": round(arcface_stats["avg_batch_size"], 1),
                "arcface_ms_per_face": round(arcface_stats["avg_item_ms"], 1),
                # LSO-117 split of the arcface_* fields above: detection is
                # still per-ROI/unbatched (LSO-118), embedding is batched.
                # An offline benchmark (lum-model-vision#16) found detection
                # at ~93% of total ArcFace time at N=362 - these two fields
                # are what would confirm or update that under real load.
                "arcface_det_avg_ms": round(arcface_det_stats["avg_call_ms"], 1),
                "arcface_det_avg_rois": round(arcface_det_stats["avg_batch_size"], 1),
                "arcface_det_ms_per_roi": round(arcface_det_stats["avg_item_ms"], 1),
                "arcface_embed_avg_ms": round(arcface_embed_stats["avg_call_ms"], 1),
                "arcface_embed_avg_batch_size": round(arcface_embed_stats["avg_batch_size"], 1),
                "arcface_embed_ms_per_face": round(arcface_embed_stats["avg_item_ms"], 1),
            },
            "action": {
                "avg_ms": round(action_avg, 1),
                "queue_depth": action_queue_depth,
                "ok": action_counts.get("ok", 0),
                "error": action_counts.get("error", 0),
                "timeout": action_counts.get("timeout", 0),
                "parse_fail": action_counts.get("parse_fail", 0),
            },
        }

    def log_summary(self, camera_ids: Optional[List[int]] = None) -> None:
        """Log a compact metrics summary line to loguru (INFO level)."""
        snap = self.snapshot(camera_ids)
        gpu = snap["gpu"]
        gpu_str = (
            f"GPU={gpu['util_percent']}% VRAM={gpu['mem_used_mb']:.0f}/{gpu['mem_total_mb']:.0f}MB({gpu['mem_percent']:.0f}%)"
            if gpu else "GPU=N/A"
        )
        fps_parts = [
            f"cam{idx}={snap['cameras'][str(idx)]['fps']:.1f}fps"
            f"(drops={snap['cameras'][str(idx)]['frame_drops']},"
            f"read={snap['cameras'][str(idx)]['read_ms']:.0f}ms,"
            f"dec={snap['cameras'][str(idx)]['decode_ms']:.0f}ms)"
            for idx in (camera_ids or [])
        ]
        proc = snap.get("process") or {}
        proc_str = (
            f"RSS={proc['rss_gb']:.2f}GB thr={proc['threads']} fds={proc['open_fds']} "
            if proc else ""
        )
        pipe = snap.get("pipeline") or {}
        # Only render gauges that reported a value, so a broken one is visibly
        # absent rather than silently logged as 0.
        pipe_str = (
            "| " + " ".join(f"{k}={v}" for k, v in sorted(pipe.items()) if v is not None) + " "
            if pipe else ""
        )
        logger.info(
            f"[Metrics] CPU={snap['cpu_percent']:.0f}% "
            f"RAM={snap['memory']['used_gb']:.1f}/{snap['memory']['total_gb']:.1f}GB({snap['memory']['percent']:.0f}%) "
            f"{proc_str}{gpu_str} {pipe_str}| "
            + (", ".join(fps_parts) if fps_parts else "no cameras yet")
            + f" | YOLO={snap['inference']['yolo_avg_ms']:.0f}ms/batch"
            f"(avg {snap['inference']['yolo_avg_batch_size']:.1f} frames,"
            f" {snap['inference']['yolo_ms_per_frame']:.0f}ms/frame)"
            f" ArcFace={snap['inference']['arcface_avg_ms']:.0f}ms/batch"
            f"(avg {snap['inference']['arcface_avg_batch_size']:.1f} faces,"
            f" {snap['inference']['arcface_ms_per_face']:.0f}ms/face)"
            f" [det={snap['inference']['arcface_det_avg_ms']:.0f}ms"
            f"/{snap['inference']['arcface_det_avg_rois']:.1f}rois"
            f" embed={snap['inference']['arcface_embed_avg_ms']:.0f}ms"
            f"/{snap['inference']['arcface_embed_avg_batch_size']:.1f}faces]"
        )

    def check_alerts(
        self,
        camera_ids: Optional[List[int]] = None,
        fps_threshold: float = 1.0,
        gpu_mem_threshold: float = 90.0,
        ram_threshold: float = 90.0,
    ) -> List[Dict]:
        """Check for critical conditions and return a list of alert dicts.

        Args:
            camera_ids:   cameras to check
            fps_threshold:    alert if FPS drops below this (0 = camera just started)
            gpu_mem_threshold: alert if GPU VRAM % exceeds this
            ram_threshold:    alert if system RAM % exceeds this
        """
        alerts = []
        snap = self.snapshot(camera_ids)

        for idx in (camera_ids or []):
            fps = snap["cameras"].get(str(idx), {}).get("fps", 0.0)
            # Only alert once we've seen some frames (fps > 0 means active)
            if 0 < fps < fps_threshold:
                alerts.append({
                    "type": "LOW_FPS",
                    "camera_id": idx,
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
