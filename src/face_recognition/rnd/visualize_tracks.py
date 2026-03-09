"""Drawing helpers for the track visualization debugger."""

import cv2
import numpy as np

_PANEL_W = 280       # width of the side panel in pixels
_GT_LEAD_TOLERANCE = 10  # frames before gt_start a track can still belong to that person


def gt_active_at(annotation: dict, frame_num: int) -> list:
    """Return person IDs whose GT window covers frame_num."""
    return [
        pid for pid, v in annotation.items()
        if int(v["start"]) <= frame_num <= int(v["end"])
    ]


def find_gt_id_for_track(annotation: dict, first_frame_num: int):
    """Return the GT person ID whose window covers first_frame_num (with lead tolerance)."""
    for pid, v in annotation.items():
        if int(v["start"]) - _GT_LEAD_TOLERANCE <= first_frame_num <= int(v["end"]):
            return pid
    return None


def draw_gt_labels(
    frame: np.ndarray,
    active_tracks: list,
    track_registry: dict,
) -> None:
    """Draw GT person IDs above bounding boxes for real-detection tracks (in-place)."""
    for t in active_tracks:
        if t.time_since_update > 0:
            continue
        if not t.history_observations:
            continue
        info = track_registry.get(t.id, {})
        gt_id = info.get("gt_id")
        label = gt_id if gt_id else f"?{t.id}"
        box = t.history_observations[-1]
        x1, y1 = int(box[0]), int(box[1])
        cv2.putText(frame, label, (x1 + 2, max(y1 - 8,  14)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),       3, cv2.LINE_AA)
        cv2.putText(frame, label, (x1,     max(y1 - 10, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 220), 2, cv2.LINE_AA)


def draw_frame_overlays(
    frame: np.ndarray,
    frame_num: int,
    active_persons: list,
    annotation: dict,
) -> np.ndarray:
    """Draw frame number and GT annotation info onto the frame (in-place).

    Args:
        frame: BGR image to draw on (modified in-place)
        frame_num: current frame number
        active_persons: list of person IDs active at this frame
        annotation: full annotation dict {pid: {start, end}}

    Returns:
        The same frame with overlays drawn.
    """
    h, w = frame.shape[:2]

    # Frame number — top-left, white with black shadow for readability
    label = f"Frame: {frame_num}"
    cv2.putText(frame, label, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0),       4, cv2.LINE_AA)
    cv2.putText(frame, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # GT active persons — top-right, cyan with black shadow
    for i, pid in enumerate(active_persons):
        v = annotation[pid]
        text = f"GT: {pid}  [{v['start']} - {v['end']}]"
        x = max(w - 380, 10)
        y = 36 + i * 32
        cv2.putText(frame, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0),     3, cv2.LINE_AA)
        cv2.putText(frame, text, (x,     y),      cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    return frame


def draw_track_table(
    height: int,
    track_registry: dict,
    active_ids: set,
) -> np.ndarray:
    """Render a side panel showing per-track first/last frame log.

    Args:
        height: panel height (should match the video frame height)
        track_registry: {track_id: {"first": int, "last": int}}
        active_ids: set of track IDs currently active this frame

    Returns:
        BGR image of shape (height, _PANEL_W, 3)
    """
    panel = np.full((height, _PANEL_W, 3), 25, dtype=np.uint8)

    # header
    cv2.putText(panel, "TRACK LOG", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.line(panel, (0, 36), (_PANEL_W, 36), (80, 80, 80), 1)

    # column headers
    cv2.putText(panel, "GT ID", (8,   54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(panel, "First", (85,  54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(panel, "Last",  (178, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.line(panel, (0, 60), (_PANEL_W, 60), (60, 60, 60), 1)

    # rows — sorted by first appearance, most recent visible if overflow
    row_h = 22
    max_rows = (height - 70) // row_h
    sorted_tracks = sorted(track_registry.items(), key=lambda x: x[1]["first"])
    visible = sorted_tracks[-max_rows:] if len(sorted_tracks) > max_rows else sorted_tracks

    for i, (tid, info) in enumerate(visible):
        y = 76 + i * row_h
        color = (80, 220, 80) if tid in active_ids else (160, 160, 160)
        gt_label = info.get("gt_id") or f"?{tid}"
        cv2.putText(panel, gt_label,           (8,   y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        cv2.putText(panel, str(info["first"]), (85,  y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        cv2.putText(panel, str(info["last"]),  (178, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    # footer
    cv2.line(panel, (0, height - 28), (_PANEL_W, height - 28), (60, 60, 60), 1)
    summary = f"active: {len(active_ids)}  total: {len(track_registry)}"
    cv2.putText(panel, summary, (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

    return panel
