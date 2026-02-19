"""
Phone Usage Spatial Logic using Pose Keypoints.

Implements pose-based spatial association to determine if a person
is using a phone based on:
1. Phone in upper body zone (shoulders to above head)
2. Hand proximity to phone (wrist keypoints)
3. Phone proximity to head (nose keypoint for calling)

Requires 2 of 3 checks to pass for phone usage confirmation.
"""

from typing import Dict, Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray
from loguru import logger


class BoundingBox:
    """Simple bounding box helper class."""

    def __init__(self, bbox: NDArray):
        """
        Initialize bounding box.

        Args:
            bbox: [x1, y1, x2, y2]
        """
        self.x1, self.y1, self.x2, self.y2 = bbox

    @property
    def center(self) -> NDArray:
        """Get center point."""
        return np.array([(self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2])

    @property
    def width(self) -> float:
        """Get width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Get height."""
        return self.y2 - self.y1

    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if this bbox intersects with another."""
        return not (
            self.x2 < other.x1 or
            self.x1 > other.x2 or
            self.y2 < other.y1 or
            self.y1 > other.y2
        )

    def contains_point(self, point: NDArray) -> bool:
        """Check if point is inside bbox."""
        return (
            self.x1 <= point[0] <= self.x2 and
            self.y1 <= point[1] <= self.y2
        )


class PhoneUsageSpatialLogic:
    """
    Determines phone usage using pose keypoints and spatial relationships.

    Uses COCO keypoint format (17 points):
    0: Nose, 5: Left Shoulder, 6: Right Shoulder,
    9: Left Wrist, 10: Right Wrist
    """

    # COCO keypoint indices
    NOSE_IDX = 0
    LEFT_SHOULDER_IDX = 5
    RIGHT_SHOULDER_IDX = 6
    LEFT_WRIST_IDX = 9
    RIGHT_WRIST_IDX = 10

    def __init__(
        self,
        hand_distance_threshold: float = 0.20,
        head_distance_threshold: float = 0.25,
        upper_body_zone_margin: float = 0.30,
        required_checks: int = 2,
        min_keypoint_confidence: float = 0.3
    ):
        """
        Initialize Phone Usage Spatial Logic.

        Args:
            hand_distance_threshold: Max distance (meters) from hand to phone
            head_distance_threshold: Max distance (meters) from head to phone
            upper_body_zone_margin: Margin (meters) above head for upper body zone
            required_checks: Number of checks required to pass (out of 3)
            min_keypoint_confidence: Minimum confidence for keypoint visibility
        """
        self.hand_distance_threshold = hand_distance_threshold
        self.head_distance_threshold = head_distance_threshold
        self.upper_body_zone_margin = upper_body_zone_margin
        self.required_checks = required_checks
        self.min_keypoint_confidence = min_keypoint_confidence

        logger.info(
            f"PhoneUsageSpatialLogic initialized: hand_dist={hand_distance_threshold}m, "
            f"head_dist={head_distance_threshold}m, required_checks={required_checks}/3"
        )

    def calculate_phone_usage_score(
        self,
        person_keypoints: NDArray,
        phone_bbox: NDArray,
        person_height_pixels: Optional[float] = None
    ) -> Dict:
        """
        Calculate phone usage score for a person-phone pair.

        Args:
            person_keypoints: Pose keypoints (17, 3) [x, y, confidence]
            phone_bbox: Phone bounding box [x1, y1, x2, y2]
            person_height_pixels: Estimated person height in pixels for scaling

        Returns:
            Dictionary with phone usage decision:
            {
                'using_phone': bool,
                'confidence': float (0.0-1.0),
                'checks_passed': int,
                'details': {
                    'zone_check': bool,
                    'hand_check': bool,
                    'head_check': bool,
                    'hand_distance': float,
                    'head_distance': float
                }
            }
        """
        # Get relevant keypoints
        nose = self._get_keypoint(person_keypoints, self.NOSE_IDX)
        left_shoulder = self._get_keypoint(person_keypoints, self.LEFT_SHOULDER_IDX)
        right_shoulder = self._get_keypoint(person_keypoints, self.RIGHT_SHOULDER_IDX)
        left_wrist = self._get_keypoint(person_keypoints, self.LEFT_WRIST_IDX)
        right_wrist = self._get_keypoint(person_keypoints, self.RIGHT_WRIST_IDX)

        # Need at least nose and one shoulder
        if nose is None or (left_shoulder is None and right_shoulder is None):
            return self._create_no_phone_result()

        # Estimate pixel-to-meter conversion (rough approximation)
        # Assume average person height is 1.7 meters
        if person_height_pixels is None:
            # Estimate from shoulders to nose
            if left_shoulder is not None and right_shoulder is not None:
                shoulder_center = (left_shoulder + right_shoulder) / 2
                torso_height = np.linalg.norm(nose - shoulder_center)
                # Head + torso is roughly 0.6 of total height
                person_height_pixels = torso_height / 0.6
            else:
                person_height_pixels = 300  # Default estimate

        pixel_to_meter = 1.7 / person_height_pixels

        # Phone bbox center
        phone_center = np.array([
            (phone_bbox[0] + phone_bbox[2]) / 2,
            (phone_bbox[1] + phone_bbox[3]) / 2
        ])

        # Check 1: Phone in upper body zone
        zone_check = self._check_upper_body_zone(
            nose, left_shoulder, right_shoulder, phone_bbox, pixel_to_meter
        )

        # Check 2: Hand near phone
        hand_check, hand_distance = self._check_hand_proximity(
            left_wrist, right_wrist, phone_center, pixel_to_meter
        )

        # Check 3: Phone near head (calling)
        head_check, head_distance = self._check_head_proximity(
            nose, phone_center, pixel_to_meter
        )

        # Count checks passed
        checks = [zone_check, hand_check, head_check]
        checks_passed = sum(checks)

        # Determine if using phone
        using_phone = checks_passed >= self.required_checks

        # Calculate confidence (0.0 to 1.0)
        confidence = checks_passed / 3.0

        return {
            'using_phone': using_phone,
            'confidence': confidence,
            'checks_passed': checks_passed,
            'details': {
                'zone_check': zone_check,
                'hand_check': hand_check,
                'head_check': head_check,
                'hand_distance': hand_distance,
                'head_distance': head_distance
            }
        }

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

    def _check_upper_body_zone(
        self,
        nose: NDArray,
        left_shoulder: Optional[NDArray],
        right_shoulder: Optional[NDArray],
        phone_bbox: NDArray,
        pixel_to_meter: float
    ) -> bool:
        """
        Check if phone is in upper body zone.

        Args:
            nose: Nose keypoint [x, y]
            left_shoulder: Left shoulder [x, y] or None
            right_shoulder: Right shoulder [x, y] or None
            phone_bbox: Phone bbox [x1, y1, x2, y2]
            pixel_to_meter: Conversion factor

        Returns:
            True if phone in upper body zone
        """
        # Calculate shoulder positions
        if left_shoulder is not None and right_shoulder is not None:
            shoulder_center = (left_shoulder + right_shoulder) / 2
            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        elif left_shoulder is not None:
            shoulder_center = left_shoulder
            shoulder_width = 0.4 / pixel_to_meter  # Assume 40cm
        elif right_shoulder is not None:
            shoulder_center = right_shoulder
            shoulder_width = 0.4 / pixel_to_meter
        else:
            return False

        # Define upper body zone
        margin_pixels = self.upper_body_zone_margin / pixel_to_meter
        padding = 0.2 / pixel_to_meter  # 20cm horizontal padding

        zone_x1 = shoulder_center[0] - (shoulder_width / 2) - padding
        zone_x2 = shoulder_center[0] + (shoulder_width / 2) + padding
        zone_y1 = nose[1] - margin_pixels  # Above head
        zone_y2 = shoulder_center[1] + (0.3 / pixel_to_meter)  # Below shoulders

        zone = BoundingBox(np.array([zone_x1, zone_y1, zone_x2, zone_y2]))
        phone_box = BoundingBox(phone_bbox)

        return zone.intersects(phone_box)

    def _check_hand_proximity(
        self,
        left_wrist: Optional[NDArray],
        right_wrist: Optional[NDArray],
        phone_center: NDArray,
        pixel_to_meter: float
    ) -> Tuple[bool, float]:
        """
        Check if hand is near phone.

        Args:
            left_wrist: Left wrist [x, y] or None
            right_wrist: Right wrist [x, y] or None
            phone_center: Phone center [x, y]
            pixel_to_meter: Conversion factor

        Returns:
            Tuple of (check_passed, distance_in_meters)
        """
        distances = []

        if left_wrist is not None:
            dist = np.linalg.norm(left_wrist - phone_center) * pixel_to_meter
            distances.append(dist)

        if right_wrist is not None:
            dist = np.linalg.norm(right_wrist - phone_center) * pixel_to_meter
            distances.append(dist)

        if not distances:
            return False, float('inf')

        min_distance = min(distances)
        check_passed = min_distance < self.hand_distance_threshold

        return check_passed, min_distance

    def _check_head_proximity(
        self,
        nose: NDArray,
        phone_center: NDArray,
        pixel_to_meter: float
    ) -> Tuple[bool, float]:
        """
        Check if phone is near head (calling detection).

        Args:
            nose: Nose keypoint [x, y]
            phone_center: Phone center [x, y]
            pixel_to_meter: Conversion factor

        Returns:
            Tuple of (check_passed, distance_in_meters)
        """
        distance = np.linalg.norm(nose - phone_center) * pixel_to_meter
        check_passed = distance < self.head_distance_threshold

        return check_passed, distance

    def _create_no_phone_result(self) -> Dict:
        """Create result for no phone usage detected."""
        return {
            'using_phone': False,
            'confidence': 0.0,
            'checks_passed': 0,
            'details': {
                'zone_check': False,
                'hand_check': False,
                'head_check': False,
                'hand_distance': float('inf'),
                'head_distance': float('inf')
            }
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PhoneUsageSpatialLogic(hand={self.hand_distance_threshold}m, "
            f"head={self.head_distance_threshold}m, required={self.required_checks}/3)"
        )
