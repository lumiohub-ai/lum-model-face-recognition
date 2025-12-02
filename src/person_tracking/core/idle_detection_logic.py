"""
Idle Detection Logic using Head Pose and Screen Orientation.

Detects when a person is idle (not looking at their screen) based on:
1. Screen detection (monitor/laptop) in proximity
2. Head pose analysis (gaze direction)
3. Spatial relationship between person and screen

A person is considered IDLE when:
- No screen detected nearby OR
- Person's head is oriented away from screen

A person is considered WORKING when:
- Screen detected nearby AND
- Person's head is oriented toward the screen
"""

from typing import Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from loguru import logger


class IdleDetectionLogic:
    """
    Idle detector using head pose and screen orientation.

    Detection criteria:
    1. Detect screens (monitors/laptops) in frame
    2. Calculate person's head orientation (gaze direction)
    3. Check if person is facing toward or away from screen
    """

    # COCO keypoint indices
    NOSE_IDX = 0
    LEFT_EYE_IDX = 1
    RIGHT_EYE_IDX = 2
    LEFT_EAR_IDX = 3
    RIGHT_EAR_IDX = 4
    LEFT_SHOULDER_IDX = 5
    RIGHT_SHOULDER_IDX = 6

    def __init__(
        self,
        screen_distance_threshold: float = 300.0,  # pixels
        head_orientation_threshold: float = 60.0,  # degrees
        min_keypoint_confidence: float = 0.3
    ):
        """
        Initialize idle detection logic.

        Args:
            screen_distance_threshold: Max distance from person to screen (pixels)
            head_orientation_threshold: Max angle deviation for "facing screen" (degrees)
            min_keypoint_confidence: Minimum confidence for keypoint visibility
        """
        self.screen_distance_threshold = screen_distance_threshold
        self.head_orientation_threshold = head_orientation_threshold
        self.min_keypoint_confidence = min_keypoint_confidence

        logger.info(
            f"IdleDetectionLogic initialized: screen_dist={screen_distance_threshold}px, "
            f"head_angle_threshold={head_orientation_threshold}°"
        )

    def detect_idle(
        self,
        person_keypoints: NDArray,
        person_bbox: NDArray,
        screen_bbox: Optional[NDArray] = None,
        screen_distance: Optional[float] = None
    ) -> Dict:
        """
        Detect if person is idle (not looking at screen).

        Args:
            person_keypoints: Pose keypoints (17, 3) [x, y, confidence]
            person_bbox: Person bounding box [x1, y1, x2, y2]
            screen_bbox: Screen bounding box [x1, y1, x2, y2] (optional)
            screen_distance: Pre-calculated distance to screen (optional)

        Returns:
            Dictionary with detection result:
            {
                'is_idle': bool,  # True if person is NOT looking at screen
                'confidence': float (0.0-1.0),
                'method': str,
                'details': {
                    'screen_detected': bool,
                    'screen_nearby': bool,
                    'facing_screen': bool,
                    'head_angle': float,  # degrees
                    'distance_to_screen': float
                }
            }
        """
        # Default: idle if no screen detected
        if screen_bbox is None:
            return {
                'is_idle': True,
                'confidence': 0.9,
                'method': 'no_screen_detected',
                'details': {
                    'screen_detected': False,
                    'screen_nearby': False,
                    'facing_screen': False,
                    'head_angle': None,
                    'distance_to_screen': None
                }
            }

        # Calculate distance to screen if not provided
        if screen_distance is None:
            person_center = self._get_bbox_center(person_bbox)
            screen_center = self._get_bbox_center(screen_bbox)
            screen_distance = np.linalg.norm(person_center - screen_center)

        # Check if screen is nearby
        screen_nearby = screen_distance <= self.screen_distance_threshold

        # If screen too far, person is idle
        if not screen_nearby:
            return {
                'is_idle': True,
                'confidence': 0.85,
                'method': 'screen_too_far',
                'details': {
                    'screen_detected': True,
                    'screen_nearby': False,
                    'facing_screen': False,
                    'head_angle': None,
                    'distance_to_screen': float(screen_distance)
                }
            }

        # Calculate head pose and check if facing screen
        facing_screen, head_angle = self._check_facing_screen(
            person_keypoints,
            person_bbox,
            screen_bbox
        )

        # Determine idle status
        is_idle = not facing_screen  # Idle if NOT facing screen

        # Calculate confidence
        if facing_screen:
            # High confidence that person is working
            if head_angle is not None:
                angle_factor = 1.0 - (abs(head_angle) / self.head_orientation_threshold)
                confidence = 0.7 + (0.3 * max(0, angle_factor))
            else:
                confidence = 0.7  # Default confidence when head angle unavailable
            method = 'facing_screen_working'
        else:
            # High confidence that person is idle
            if head_angle is not None:
                confidence = 0.8 + (0.2 * min(1.0, abs(head_angle) / 90.0))
            else:
                confidence = 0.75  # Default confidence when head angle unavailable
            method = 'not_facing_screen_idle'

        return {
            'is_idle': is_idle,
            'confidence': confidence,
            'method': method,
            'details': {
                'screen_detected': True,
                'screen_nearby': True,
                'facing_screen': facing_screen,
                'head_angle': float(head_angle) if head_angle is not None else None,
                'distance_to_screen': float(screen_distance)
            }
        }

    def _check_facing_screen(
        self,
        keypoints: NDArray,
        person_bbox: NDArray,
        screen_bbox: NDArray
    ) -> Tuple[bool, Optional[float]]:
        """
        Check if person's head is oriented toward the screen.

        Uses head pose estimation from nose, eyes, ears, and shoulders.

        Args:
            keypoints: Pose keypoints (17, 3)
            person_bbox: Person bbox [x1, y1, x2, y2]
            screen_bbox: Screen bbox [x1, y1, x2, y2]

        Returns:
            Tuple of (facing_screen, head_angle_degrees)
        """
        # Get head keypoints
        nose = self._get_keypoint(keypoints, self.NOSE_IDX)
        left_eye = self._get_keypoint(keypoints, self.LEFT_EYE_IDX)
        right_eye = self._get_keypoint(keypoints, self.RIGHT_EYE_IDX)
        left_ear = self._get_keypoint(keypoints, self.LEFT_EAR_IDX)
        right_ear = self._get_keypoint(keypoints, self.RIGHT_EAR_IDX)
        left_shoulder = self._get_keypoint(keypoints, self.LEFT_SHOULDER_IDX)
        right_shoulder = self._get_keypoint(keypoints, self.RIGHT_SHOULDER_IDX)

        # Need at least nose and one shoulder for basic orientation
        if nose is None or (left_shoulder is None and right_shoulder is None):
            logger.debug("Insufficient keypoints for head pose estimation")
            return False, None

        # Calculate head center and face direction
        face_center = self._calculate_face_center(
            nose, left_eye, right_eye, left_ear, right_ear
        )

        # Calculate body orientation (from shoulders)
        body_center = self._calculate_body_center(left_shoulder, right_shoulder)

        # Calculate screen center
        screen_center = self._get_bbox_center(screen_bbox)

        # Method 1: Vector from face to screen
        face_to_screen_vector = screen_center - face_center
        face_to_screen_angle = np.degrees(
            np.arctan2(face_to_screen_vector[1], face_to_screen_vector[0])
        )

        # Method 2: Body orientation
        if body_center is not None and left_shoulder is not None and right_shoulder is not None:
            # Shoulder line vector
            shoulder_vector = right_shoulder - left_shoulder
            shoulder_angle = np.degrees(
                np.arctan2(shoulder_vector[1], shoulder_vector[0])
            )

            # Forward direction is perpendicular to shoulder line
            # Assuming person faces "up" when shoulders are horizontal
            forward_angle = shoulder_angle - 90.0

            # Calculate angle difference between forward direction and screen direction
            angle_diff = self._angle_difference(forward_angle, face_to_screen_angle)

        else:
            # Fallback: use nose-to-screen direction only
            angle_diff = abs(face_to_screen_angle)

        # Normalize angle to -180 to 180
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360

        # Check if within threshold
        facing_screen = abs(angle_diff) <= self.head_orientation_threshold

        logger.debug(
            f"Head pose: angle_diff={angle_diff:.1f}°, "
            f"facing_screen={facing_screen}"
        )

        return facing_screen, angle_diff

    def _calculate_face_center(
        self,
        nose: Optional[NDArray],
        left_eye: Optional[NDArray],
        right_eye: Optional[NDArray],
        left_ear: Optional[NDArray],
        right_ear: Optional[NDArray]
    ) -> NDArray:
        """
        Calculate face center from available keypoints.

        Args:
            nose, left_eye, right_eye, left_ear, right_ear: Face keypoints

        Returns:
            Face center point [x, y]
        """
        points = []
        if nose is not None:
            points.append(nose)
        if left_eye is not None:
            points.append(left_eye)
        if right_eye is not None:
            points.append(right_eye)
        if left_ear is not None:
            points.append(left_ear)
        if right_ear is not None:
            points.append(right_ear)

        if len(points) == 0:
            return np.array([0, 0])

        points = np.array(points)
        return np.mean(points, axis=0)

    def _calculate_body_center(
        self,
        left_shoulder: Optional[NDArray],
        right_shoulder: Optional[NDArray]
    ) -> Optional[NDArray]:
        """
        Calculate body center from shoulders.

        Args:
            left_shoulder, right_shoulder: Shoulder keypoints

        Returns:
            Body center [x, y] or None
        """
        if left_shoulder is not None and right_shoulder is not None:
            return (left_shoulder + right_shoulder) / 2
        elif left_shoulder is not None:
            return left_shoulder
        elif right_shoulder is not None:
            return right_shoulder
        return None

    @staticmethod
    def _get_bbox_center(bbox: NDArray) -> NDArray:
        """
        Calculate center point of bounding box.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Center point [cx, cy]
        """
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return np.array([cx, cy])

    @staticmethod
    def _angle_difference(angle1: float, angle2: float) -> float:
        """
        Calculate the smallest difference between two angles.

        Args:
            angle1: First angle in degrees
            angle2: Second angle in degrees

        Returns:
            Angle difference in degrees (-180 to 180)
        """
        diff = angle2 - angle1
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff

    def _get_keypoint(
        self,
        keypoints: NDArray,
        index: int
    ) -> Optional[NDArray]:
        """
        Get keypoint if visible.

        Args:
            keypoints: Keypoints array (17, 3)
            index: Keypoint index

        Returns:
            Keypoint [x, y] or None if not visible
        """
        if keypoints[index, 2] >= self.min_keypoint_confidence:
            return keypoints[index, :2]
        return None

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"IdleDetectionLogic(screen_dist={self.screen_distance_threshold}px, "
            f"head_angle={self.head_orientation_threshold}°)"
        )
