"""Frame annotation utilities for person tracking visualization.

This module provides the FrameAnnotator class for drawing bounding boxes,
keypoints, labels, and other visual elements on video frames.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# COCO skeleton connections for pose visualization
COCO_SKELETON = [
    # Head connections
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Arms
    (5, 7), (7, 9),   # Left arm
    (6, 8), (8, 10),  # Right arm
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Legs
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]

# Keypoint colors (BGR)
KEYPOINT_COLORS = {
    'head': (0, 255, 255),      # Yellow
    'shoulder': (255, 0, 255),  # Magenta
    'arm': (255, 255, 0),       # Cyan
    'torso': (0, 255, 0),       # Green
    'leg': (0, 165, 255),       # Orange
}

# Bone colors for skeleton
SKELETON_COLORS = [
    (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255),  # Head
    (255, 255, 0), (255, 255, 0),  # Left arm
    (255, 255, 0), (255, 255, 0),  # Right arm
    (255, 0, 255), (0, 255, 0), (0, 255, 0), (0, 255, 0),  # Torso
    (0, 165, 255), (0, 165, 255),  # Left leg
    (0, 165, 255), (0, 165, 255),  # Right leg
]


class FrameAnnotator:
    """Annotates frames with person tracking visualization.

    This class provides methods for drawing bounding boxes, keypoints,
    labels, and other visual elements on video frames.
    """

    def __init__(
        self,
        draw_skeleton: bool = True,
        draw_trajectory: bool = False,
        font_scale: float = 0.5,
        line_thickness: int = 2,
        keypoint_radius: int = 4,
        action_ttl_seconds: int = 30,
    ):
        """Initialize the FrameAnnotator."""
        self.draw_skeleton = draw_skeleton
        self.draw_trajectory = draw_trajectory
        self.font_scale = font_scale
        self.line_thickness = line_thickness
        self.keypoint_radius = keypoint_radius
        self.action_ttl_seconds = action_ttl_seconds

        # Color scheme
        self.colors = {
            'locked': (0, 255, 0),       # Green - locked identity
            'tentative': (0, 255, 255),  # Yellow - tentative identity
            'unknown': (0, 0, 255),      # Red - unknown
            'text': (255, 255, 255),     # White - text
            'background': (0, 0, 0),     # Black - label background
        }

    def annotate_frame(
        self,
        frame: np.ndarray,
        person_states: List[Dict],
        fps: float = 0.0,
        show_stats: bool = True,
        roi_active: bool = False,
        virtual_lines_stats: Optional[List[Dict]] = None,
    ) -> np.ndarray:
        """Annotate frame with all detection results.

        Args:
            frame: Input frame (BGR format)
            person_states: List of person state dictionaries
            fps: Current FPS to display
            show_stats: Whether to show statistics overlay
            roi_active: Whether to draw orange ROI border
            virtual_lines_stats: List of virtual line stat dicts (one per configured line)

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Draw ROI border when camera has an active ROI configured
        if roi_active:
            self._draw_roi_border(annotated)

        # Draw all virtual lines
        if virtual_lines_stats:
            for vls in virtual_lines_stats:
                self.draw_virtual_line(annotated, vls)

        # Draw each person
        for state in person_states:
            # Skip ghost bboxes: only draw if person is in current frame
            if not state.get('in_current_frame', True):
                continue
            self.draw_person(annotated, state)

        # Draw statistics overlay
        if show_stats:
            self.draw_stats(
                annotated,
                fps=fps,
                num_persons=len(person_states)
            )

        return annotated

    def _draw_roi_border(self, frame: np.ndarray) -> None:
        """Draw an orange border to indicate the frame is a cropped ROI."""
        h, w = frame.shape[:2]
        t = max(3, h // 120)
        color = (0, 140, 255)  # orange
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, t)
        cv2.putText(frame, "ROI", (t + 5, t + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def draw_virtual_line(self, frame: np.ndarray, vls: Dict) -> None:
        """Draw a virtual line and its counters on the frame.

        For person_counting lines: shows name + IN/OUT only.
        For fitting_room lines: also shows occupancy and optional timer.
        """
        points = vls.get("points", [])
        if len(points) < 2:
            return

        pt1 = tuple(map(int, points[0]))
        pt2 = tuple(map(int, points[1]))

        name          = vls.get("name", "")
        line_type     = vls.get("line_type", "person_counting")
        timer_enabled = bool(vls.get("timer_enabled", False))
        zone_in       = vls.get("in", 0)
        zone_out      = vls.get("out", 0)
        occupancy     = vls.get("occupancy", 0)
        duration      = vls.get("duration")
        is_fitting    = line_type == "fitting_room" or timer_enabled

        CYAN   = (255, 255, 0)
        GREEN  = (0, 220, 0)
        RED    = (0, 0, 220)
        WHITE  = (255, 255, 255)
        BLACK  = (0, 0, 0)
        YELLOW = (0, 220, 220)

        # Main line
        cv2.line(frame, pt1, pt2, CYAN, 2, cv2.LINE_AA)

        # Direction arrow at mid-point
        mx, my = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
        dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
        length = max(1, (dx**2 + dy**2) ** 0.5)
        nx, ny = -dy / length, dx / length
        tip = (int(mx + nx * 14), int(my + ny * 14))
        cv2.arrowedLine(frame, (mx, my), tip, CYAN, 2, cv2.LINE_AA, tipLength=0.5)

        def fmt(secs: float) -> str:
            s = int(secs)
            return f"{s // 60}:{s % 60:02d}"

        # Panel content differs by line type
        if is_fitting:
            header = f"{name}  IN:{zone_in}  OUT:{zone_out}  Room:{occupancy}"
            panel_lines = [(header, WHITE, 0.55, 1)]
            if timer_enabled and duration is not None:
                panel_lines.append((f"  Time: {fmt(duration)}", YELLOW, 0.5, 1))
        else:
            header = f"{name}  IN:{zone_in}  OUT:{zone_out}"
            panel_lines = [(header, WHITE, 0.55, 1)]

        panel_x, panel_y = mx - 60, my - 55
        for i, (text, color, fscale, thick) in enumerate(panel_lines):
            ty = panel_y + i * 22
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fscale, thick)
            cv2.rectangle(frame, (panel_x - 4, ty - th - 4), (panel_x + tw + 4, ty + 4), BLACK, -1)
            cv2.putText(frame, text, (panel_x, ty), cv2.FONT_HERSHEY_SIMPLEX, fscale, color, thick, cv2.LINE_AA)

        # Endpoint badges
        for badge_pt, label, color in [(pt1, f"IN {zone_in}", GREEN), (pt2, f"OUT {zone_out}", RED)]:
            (bw, bh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            bx, by = badge_pt[0] - bw // 2, badge_pt[1] - 8
            cv2.rectangle(frame, (bx - 3, by - bh - 3), (bx + bw + 3, by + 3), BLACK, -1)
            cv2.putText(frame, label, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def draw_person(
        self,
        frame: np.ndarray,
        state: Dict[str, Any]
    ) -> None:
        """Draw a single person with bbox, keypoints, and labels.

        Args:
            frame: Frame to draw on
            state: Person state dictionary with keys:
                - track_id: int
                - bbox: [x1, y1, x2, y2]
                - keypoints: np.ndarray (17, 3) or None
                - identity: str or None
                - identity_locked: bool
                - trajectory: list of (x, y) points or None
                - track_age: int (0 if in current frame, >0 if aging)
                - in_current_frame: bool (True if detected in current frame)
        """
        track_id = state.get('track_id', 0)
        global_id = state.get('global_id')
        bbox = state.get('bbox', [0, 0, 0, 0])
        keypoints = state.get('keypoints')
        identity = state.get('identity')
        identity_locked = state.get('identity_locked', False)
        trajectory = state.get('trajectory')

        # Safety check: skip if track_age > 0 (person not in current frame)
        track_age = state.get('track_age', 0)
        if track_age > 0:
            return  # Don't draw bbox for aging tracks

        # Determine color based on identity state
        # Priority: Identity locked > Tentative identity > Unknown
        if identity_locked:
            color = self.colors['locked']  # Green - locked identity
        elif identity:
            color = (255, 0, 255)  # Magenta - tentative identity (changed from yellow)
        else:
            color = self.colors['unknown']  # Red - unknown

        # Draw bounding box
        self.draw_bbox(frame, bbox, color)

        # Draw face bounding box + detection score if available
        face_bbox = state.get('face_bbox')
        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = map(int, face_bbox)
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
            face_det_score = state.get('face_det_score')
            if face_det_score is not None:
                cv2.putText(
                    frame, f"{face_det_score:.2f}",
                    (fx1, max(fy1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
                )

        # Draw InsightFace 5-point face landmarks (eyes, nose, mouth corners)
        face_landmarks = state.get('face_landmarks')
        if face_landmarks is not None:
            # Colors: left eye, right eye, nose, left mouth, right mouth
            lm_colors = [
                (0, 255, 0),    # left eye - green
                (0, 0, 255),    # right eye - red
                (0, 255, 255),  # nose - yellow
                (255, 0, 0),    # left mouth - blue
                (255, 0, 255),  # right mouth - magenta
            ]
            for i, pt in enumerate(face_landmarks):
                c = lm_colors[i] if i < len(lm_colors) else (255, 255, 255)
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, c, -1)

        # Draw keypoints and skeleton
        if keypoints is not None and self.draw_skeleton:
            self.draw_keypoints(frame, keypoints)
            self.draw_pose_skeleton(frame, keypoints)

        # Draw trajectory
        if trajectory and self.draw_trajectory:
            self.draw_person_trajectory(frame, trajectory, color)

        # Draw label — hide action if older than TTL
        action = state.get('last_detected_action')
        last_action_time = state.get('last_action_time', 0.0)
        if action and last_action_time > 0 and (time.time() - last_action_time) > self.action_ttl_seconds:
            action = None
        self.draw_person_label(
            frame,
            bbox,
            track_id,
            global_id,
            identity,
            identity_locked,
            color,
            action=action,
            vote_count=state.get('vote_count', 0),
            required_votes=state.get('required_votes', 0),
            gender=state.get('gender'),
            age=state.get('age'),
        )

    def draw_bbox(
        self,
        frame: np.ndarray,
        bbox: List[float],
        color: Tuple[int, int, int]
    ) -> None:
        """Draw bounding box on frame.

        Args:
            frame: Frame to draw on
            bbox: Bounding box [x1, y1, x2, y2]
            color: BGR color tuple
        """
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)

    def draw_keypoints(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        min_confidence: float = 0.3
    ) -> None:
        """Draw keypoints on frame.

        Args:
            frame: Frame to draw on
            keypoints: Keypoints array (17, 3)
            min_confidence: Minimum confidence to draw keypoint
        """
        for i, kp in enumerate(keypoints):
            x, y, conf = kp
            if conf < min_confidence:
                continue

            # Determine color based on keypoint type
            if i <= 4:  # Head
                color = KEYPOINT_COLORS['head']
            elif i <= 6:  # Shoulders
                color = KEYPOINT_COLORS['shoulder']
            elif i <= 10:  # Arms
                color = KEYPOINT_COLORS['arm']
            elif i <= 12:  # Torso
                color = KEYPOINT_COLORS['torso']
            else:  # Legs
                color = KEYPOINT_COLORS['leg']

            cv2.circle(
                frame,
                (int(x), int(y)),
                self.keypoint_radius,
                color,
                -1
            )

    def draw_pose_skeleton(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        min_confidence: float = 0.3
    ) -> None:
        """Draw pose skeleton connections.

        Args:
            frame: Frame to draw on
            keypoints: Keypoints array (17, 3)
            min_confidence: Minimum confidence for both keypoints
        """
        for idx, (start_idx, end_idx) in enumerate(COCO_SKELETON):
            if (keypoints[start_idx][2] < min_confidence or
                keypoints[end_idx][2] < min_confidence):
                continue

            start = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
            end = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))

            color = SKELETON_COLORS[idx] if idx < len(SKELETON_COLORS) else (0, 255, 255)
            cv2.line(frame, start, end, color, 1)

    def draw_person_label(
        self,
        frame: np.ndarray,
        bbox: List[float],
        track_id: int,
        global_id: Optional[int],
        identity: Optional[str],
        identity_locked: bool,
        color: Tuple[int, int, int],
        action: Optional[str] = None,
        vote_count: int = 0,
        required_votes: int = 0,
        gender: Optional[str] = None,
        age: Optional[int] = None,
    ) -> None:
        """Draw label above person bounding box.

        Args:
            frame: Frame to draw on
            bbox: Bounding box [x1, y1, x2, y2]
            track_id: Local track ID
            global_id: Global track ID (cross-camera) or None
            identity: Person identity or None
            identity_locked: Whether identity is locked
            color: Background color
            action: Detected action (e.g. 'sleeping', 'using phone') or None
        """
        x1, y1 = int(bbox[0]), int(bbox[1])

        # Build label text - show both local and global IDs for debugging
        if global_id is not None:
            parts = [f"L:{track_id} G:{global_id}"]
        else:
            parts = [f"ID:{track_id}"]

        if identity_locked and identity:
            parts.append(identity)

        if gender or age is not None:
            ga_parts = []
            if gender:
                ga_parts.append("M" if gender.lower() in ("male", "m") else "F")
            if age is not None:
                ga_parts.append(str(age))
            parts.append(" ".join(ga_parts))

        if action:
            parts.append(f"[{action}]")

        label = " | ".join(parts)

        # Calculate label size
        (label_w, label_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            self.line_thickness
        )

        # Draw background rectangle
        cv2.rectangle(
            frame,
            (x1, y1),
            (x1 + label_w + 4, y1 + label_h + 10), # Adjusted to be inside the bbox, at the top
            color,
            -1
        )

        # Draw text
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 + label_h + 5), # Adjusted to be inside the bbox, with padding
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            self.colors['text'],
            self.line_thickness
        )

    def draw_person_trajectory(
        self,
        frame: np.ndarray,
        trajectory: List[Tuple[float, float]],
        color: Tuple[int, int, int],
        max_points: int = 30
    ) -> None:
        """Draw person movement trajectory.

        Args:
            frame: Frame to draw on
            trajectory: List of (x, y) points
            color: Line color
            max_points: Maximum points to draw
        """
        if len(trajectory) < 2:
            return

        points = trajectory[-max_points:]

        for i in range(1, len(points)):
            # Fade older points
            alpha = i / len(points)
            thickness = max(1, int(self.line_thickness * alpha))

            pt1 = (int(points[i-1][0]), int(points[i-1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))

            cv2.line(frame, pt1, pt2, color, thickness)

    def draw_stats(
        self,
        frame: np.ndarray,
        fps: float,
        num_persons: int,
        position: str = 'top_left'
    ) -> None:
        """Draw statistics overlay on frame."""
        entries = [
            (f"Persons: {num_persons}", 0.9, 2, (0, 255, 0)),
        ]

        if position == 'top_left':
            x, y = 10, 22
        else:
            x, y = frame.shape[1] - 180, 22

        cursor_y = y
        for text, fscale, thick, color in entries:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fscale, thick)
            cv2.rectangle(
                frame,
                (x - 5, cursor_y - th - 5),
                (x + tw + 5, cursor_y + 5),
                (0, 0, 0),
                -1,
            )
            cv2.putText(frame, text, (x, cursor_y), cv2.FONT_HERSHEY_SIMPLEX, fscale, color, thick, cv2.LINE_AA)
            cursor_y += th + 12
