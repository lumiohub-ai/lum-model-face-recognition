"""Frame annotation utilities for person tracking visualization.

This module provides the FrameAnnotator class for drawing bounding boxes,
keypoints, labels, and other visual elements on video frames.
"""

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
        keypoint_radius: int = 4
    ):
        """Initialize the FrameAnnotator.

        Args:
            draw_skeleton: Whether to draw pose skeleton
            draw_trajectory: Whether to draw movement trajectory
            font_scale: Font scale for text labels
            line_thickness: Thickness for lines and boxes
            keypoint_radius: Radius for keypoint circles
        """
        self.draw_skeleton = draw_skeleton
        self.draw_trajectory = draw_trajectory
        self.font_scale = font_scale
        self.line_thickness = line_thickness
        self.keypoint_radius = keypoint_radius

        # Color scheme
        self.colors = {
            'locked': (0, 255, 0),       # Green - locked identity
            'tentative': (0, 255, 255),  # Yellow - tentative identity
            'unknown': (0, 0, 255),      # Red - unknown
            'phone': (255, 0, 0),        # Blue - phone
            'text': (255, 255, 255),     # White - text
            'background': (0, 0, 0),     # Black - label background
        }

    def annotate_frame(
        self,
        frame: np.ndarray,
        person_states: List[Dict],
        phones: Optional[List[Dict]] = None,
        fps: float = 0.0,
        show_stats: bool = True
    ) -> np.ndarray:
        """Annotate frame with all detection results.

        Args:
            frame: Input frame (BGR format)
            person_states: List of person state dictionaries
            phones: List of detected phone dictionaries
            fps: Current FPS to display
            show_stats: Whether to show statistics overlay

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Draw phones first (behind persons)
        if phones:
            for phone in phones:
                self.draw_phone(annotated, phone)

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
                num_persons=len(person_states),
                num_phones=len(phones) if phones else 0
            )

        return annotated

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
                - using_phone: bool
                - trajectory: list of (x, y) points or None
                - track_age: int (0 if in current frame, >0 if aging)
                - in_current_frame: bool (True if detected in current frame)
        """
        track_id = state.get('track_id', 0)
        bbox = state.get('bbox', [0, 0, 0, 0])
        keypoints = state.get('keypoints')
        identity = state.get('identity')
        identity_locked = state.get('identity_locked', False)
        using_phone = state.get('using_phone', False)
        trajectory = state.get('trajectory')

        # Safety check: skip if track_age > 0 (person not in current frame)
        track_age = state.get('track_age', 0)
        if track_age > 0:
            return  # Don't draw bbox for aging tracks

        # Determine color based on phone usage FIRST, then identity state
        # Priority: Phone usage > Identity locked > Tentative identity > Unknown
        if using_phone:
            color = (0, 255, 255)  # Yellow - phone usage (highest priority)
        elif identity_locked:
            color = self.colors['locked']  # Green - locked identity
        elif identity:
            color = (255, 0, 255)  # Magenta - tentative identity (changed from yellow)
        else:
            color = self.colors['unknown']  # Red - unknown

        # Draw bounding box
        self.draw_bbox(frame, bbox, color)

        # Draw keypoints and skeleton
        if keypoints is not None and self.draw_skeleton:
            self.draw_keypoints(frame, keypoints)
            self.draw_pose_skeleton(frame, keypoints)

        # Draw trajectory
        if trajectory and self.draw_trajectory:
            self.draw_person_trajectory(frame, trajectory, color)

        # Draw label
        self.draw_person_label(
            frame,
            bbox,
            track_id,
            identity,
            identity_locked,
            using_phone,
            color
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
        identity: Optional[str],
        identity_locked: bool,
        using_phone: bool,
        color: Tuple[int, int, int]
    ) -> None:
        """Draw label above person bounding box.

        Args:
            frame: Frame to draw on
            bbox: Bounding box [x1, y1, x2, y2]
            track_id: Track ID
            identity: Person identity or None
            identity_locked: Whether identity is locked
            using_phone: Whether person is using phone
            color: Background color
        """
        x1, y1 = int(bbox[0]), int(bbox[1])

        # Build label text
        parts = [f"ID:{track_id}"]

        if identity:
            # Show friendly message based on phone usage
            if using_phone:
                parts.append(f"{identity} is using phone")
            else:
                parts.append(identity)

            # Add lock status indicator only if not locked (for debugging)
            if not identity_locked:
                parts.append("?")
        elif using_phone:
            # Show phone usage even without identity
            parts.append("PHONE")

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
            (x1, y1 - label_h - 10),
            (x1 + label_w + 4, y1),
            color,
            -1
        )

        # Draw text
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 5),
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

    def draw_phone(
        self,
        frame: np.ndarray,
        phone: Dict[str, Any]
    ) -> None:
        """Draw phone detection box.

        Args:
            frame: Frame to draw on
            phone: Phone dictionary with 'bbox' key
        """
        bbox = phone.get('bbox', [0, 0, 0, 0])
        confidence = phone.get('confidence', 0.0)

        x1, y1, x2, y2 = map(int, bbox[:4])

        # Draw box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            self.colors['phone'],
            self.line_thickness
        )

        # Draw label
        label = f"Phone {confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale * 0.8,
            self.colors['phone'],
            1
        )

    def draw_stats(
        self,
        frame: np.ndarray,
        fps: float,
        num_persons: int,
        num_phones: int,
        position: str = 'top_left'
    ) -> None:
        """Draw statistics overlay on frame.

        Args:
            frame: Frame to draw on
            fps: Current FPS
            num_persons: Number of tracked persons
            num_phones: Number of detected phones
            position: Position of overlay ('top_left', 'top_right')
        """
        lines = [
            f"FPS: {fps:.1f}",
            f"Persons: {num_persons}",
            f"Phones: {num_phones}"
        ]

        # Calculate position
        if position == 'top_left':
            x, y = 10, 25
        else:
            x, y = frame.shape[1] - 150, 25

        # Draw each line
        for i, line in enumerate(lines):
            text_y = y + i * 25

            # Draw background
            (w, h), _ = cv2.getTextSize(
                line,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )
            cv2.rectangle(
                frame,
                (x - 5, text_y - h - 5),
                (x + w + 5, text_y + 5),
                (0, 0, 0),
                -1
            )

            # Draw text
            cv2.putText(
                frame,
                line,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    def draw_event_notification(
        self,
        frame: np.ndarray,
        event_text: str,
        duration_frames: int = 30
    ) -> None:
        """Draw event notification banner.

        Args:
            frame: Frame to draw on
            event_text: Event text to display
            duration_frames: How long to show (not implemented here)
        """
        h, w = frame.shape[:2]

        # Draw banner at bottom
        banner_height = 40
        cv2.rectangle(
            frame,
            (0, h - banner_height),
            (w, h),
            (50, 50, 50),
            -1
        )

        # Draw text
        cv2.putText(
            frame,
            event_text,
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    @staticmethod
    def create_grid_view(
        frames: List[np.ndarray],
        grid_size: Tuple[int, int] = (2, 2),
        cell_size: Tuple[int, int] = (640, 480)
    ) -> np.ndarray:
        """Create a grid view of multiple camera frames.

        Args:
            frames: List of frames
            grid_size: (rows, cols) of grid
            cell_size: (width, height) of each cell

        Returns:
            Combined grid frame
        """
        rows, cols = grid_size
        cell_w, cell_h = cell_size

        # Create empty grid
        grid = np.zeros(
            (rows * cell_h, cols * cell_w, 3),
            dtype=np.uint8
        )

        # Place each frame
        for idx, frame in enumerate(frames):
            if idx >= rows * cols:
                break

            row = idx // cols
            col = idx % cols

            # Resize frame to cell size
            resized = cv2.resize(frame, cell_size)

            # Place in grid
            y1 = row * cell_h
            y2 = y1 + cell_h
            x1 = col * cell_w
            x2 = x1 + cell_w

            grid[y1:y2, x1:x2] = resized

        return grid
