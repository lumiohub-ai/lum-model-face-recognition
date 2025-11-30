"""
Simplified Phone Usage Detection Logic.

Combines:
1. Bounding box overlap between phone and hands/arms
2. Pose-based activity recognition (phone usage postures)
3. No strict spatial zone requirements

A person is considered using a phone when:
- Phone detected anywhere in frame
- Phone overlaps with hand/arm regions OR
- Person exhibits phone usage posture (arms raised, hands near upper body)
"""

from typing import Dict, Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray
from loguru import logger


class PhoneUsageDetectorV2:
    """
    Simplified phone usage detector using overlap detection only.

    Detection criteria (OR logic):
    1. Phone overlaps with hand/wrist bounding boxes
    2. Phone overlaps with arm (shoulder-wrist) bounding boxes
    """

    # COCO keypoint indices
    NOSE_IDX = 0
    LEFT_EYE_IDX = 1
    RIGHT_EYE_IDX = 2
    LEFT_EAR_IDX = 3
    RIGHT_EAR_IDX = 4
    LEFT_SHOULDER_IDX = 5
    RIGHT_SHOULDER_IDX = 6
    LEFT_ELBOW_IDX = 7
    RIGHT_ELBOW_IDX = 8
    LEFT_WRIST_IDX = 9
    RIGHT_WRIST_IDX = 10
    LEFT_HIP_IDX = 11
    RIGHT_HIP_IDX = 12

    def __init__(
        self,
        hand_bbox_size: float = 80.0,  # pixels
        overlap_iou_threshold: float = 0.01,  # very low - just need any overlap
        min_keypoint_confidence: float = 0.3
    ):
        """
        Initialize phone usage detector with overlap detection only.

        Args:
            hand_bbox_size: Size of bounding box around wrist (pixels)
            overlap_iou_threshold: Minimum IoU for overlap detection
            min_keypoint_confidence: Minimum confidence for keypoint visibility
        """
        self.hand_bbox_size = hand_bbox_size
        self.overlap_iou_threshold = overlap_iou_threshold
        self.min_keypoint_confidence = min_keypoint_confidence

        logger.info(
            f"PhoneUsageDetectorV2 initialized: hand_bbox={hand_bbox_size}px, "
            f"overlap_iou={overlap_iou_threshold} (overlap-only mode)"
        )

    def detect_phone_usage(
        self,
        person_keypoints: NDArray,
        phone_bbox: NDArray,
        person_bbox: Optional[NDArray] = None
    ) -> Dict:
        """
        Detect if person is using phone using overlap detection only.

        Args:
            person_keypoints: Pose keypoints (17, 3) [x, y, confidence]
            phone_bbox: Phone bounding box [x1, y1, x2, y2]
            person_bbox: Person bounding box [x1, y1, x2, y2] (optional)

        Returns:
            Dictionary with detection result:
            {
                'using_phone': bool,
                'confidence': float (0.0-1.0),
                'method': str,  # 'hand_overlap', 'arm_overlap', or 'hand_arm_overlap'
                'details': {
                    'hand_overlap': bool,
                    'arm_overlap': bool,
                    'hand_overlap_iou': float,
                    'arm_overlap_iou': float
                }
            }
        """
        # Check 1: Phone overlaps with hands
        hand_overlap, hand_iou = self._check_hand_overlap(person_keypoints, phone_bbox)

        # Check 2: Phone overlaps with arms
        arm_overlap, arm_iou = self._check_arm_overlap(person_keypoints, phone_bbox)

        # Determine usage (OR logic - either overlap method succeeds)
        using_phone = hand_overlap or arm_overlap

        # Calculate confidence based on what passed
        confidence = 0.0
        method = 'none'

        if hand_overlap and arm_overlap:
            confidence = 0.95
            method = 'hand_arm_overlap'
        elif hand_overlap:
            confidence = 0.85
            method = 'hand_overlap'
        elif arm_overlap:
            confidence = 0.75
            method = 'arm_overlap'

        return {
            'using_phone': using_phone,
            'confidence': confidence,
            'method': method,
            'details': {
                'hand_overlap': hand_overlap,
                'arm_overlap': arm_overlap,
                'hand_overlap_iou': hand_iou,
                'arm_overlap_iou': arm_iou
            }
        }

    def _check_hand_overlap(
        self,
        keypoints: NDArray,
        phone_bbox: NDArray
    ) -> Tuple[bool, float]:
        """
        Check if phone overlaps with hand bounding boxes.

        Creates bounding boxes around wrist keypoints and checks overlap.

        Args:
            keypoints: Pose keypoints (17, 3)
            phone_bbox: Phone bbox [x1, y1, x2, y2]

        Returns:
            Tuple of (overlap_detected, max_iou)
        """
        left_wrist = self._get_keypoint(keypoints, self.LEFT_WRIST_IDX)
        right_wrist = self._get_keypoint(keypoints, self.RIGHT_WRIST_IDX)

        max_iou = 0.0

        for wrist in [left_wrist, right_wrist]:
            if wrist is None:
                continue

            # Create bounding box around wrist
            half_size = self.hand_bbox_size / 2
            hand_bbox = np.array([
                wrist[0] - half_size,
                wrist[1] - half_size,
                wrist[0] + half_size,
                wrist[1] + half_size
            ])

            # Calculate IoU
            iou = self._calculate_iou(hand_bbox, phone_bbox)
            max_iou = max(max_iou, iou)

        overlap = max_iou > self.overlap_iou_threshold
        return overlap, max_iou

    def _check_arm_overlap(
        self,
        keypoints: NDArray,
        phone_bbox: NDArray
    ) -> Tuple[bool, float]:
        """
        Check if phone overlaps with arm regions.

        Creates bounding boxes for arm segments (shoulder -> elbow -> wrist).

        Args:
            keypoints: Pose keypoints (17, 3)
            phone_bbox: Phone bbox [x1, y1, x2, y2]

        Returns:
            Tuple of (overlap_detected, max_iou)
        """
        # Get arm keypoints
        left_shoulder = self._get_keypoint(keypoints, self.LEFT_SHOULDER_IDX)
        right_shoulder = self._get_keypoint(keypoints, self.RIGHT_SHOULDER_IDX)
        left_elbow = self._get_keypoint(keypoints, self.LEFT_ELBOW_IDX)
        right_elbow = self._get_keypoint(keypoints, self.RIGHT_ELBOW_IDX)
        left_wrist = self._get_keypoint(keypoints, self.LEFT_WRIST_IDX)
        right_wrist = self._get_keypoint(keypoints, self.RIGHT_WRIST_IDX)

        max_iou = 0.0

        # Check left arm
        if left_shoulder is not None and left_wrist is not None:
            arm_bbox = self._create_arm_bbox(left_shoulder, left_elbow, left_wrist)
            iou = self._calculate_iou(arm_bbox, phone_bbox)
            max_iou = max(max_iou, iou)

        # Check right arm
        if right_shoulder is not None and right_wrist is not None:
            arm_bbox = self._create_arm_bbox(right_shoulder, right_elbow, right_wrist)
            iou = self._calculate_iou(arm_bbox, phone_bbox)
            max_iou = max(max_iou, iou)

        overlap = max_iou > self.overlap_iou_threshold
        return overlap, max_iou

    def _check_phone_usage_posture(
        self,
        keypoints: NDArray,
        phone_bbox: NDArray
    ) -> Tuple[bool, float]:
        """
        Detect phone usage posture using pose analysis.

        Phone usage indicators:
        1. Arms raised (wrists above hips)
        2. Arms bent (elbow angle < 150°)
        3. Hands near upper body (wrists close to shoulders/head)
        4. Phone in proximity to person

        Args:
            keypoints: Pose keypoints (17, 3)
            phone_bbox: Phone bbox [x1, y1, x2, y2]

        Returns:
            Tuple of (is_phone_pose, confidence_score)
        """
        score = 0.0
        max_score = 5.0

        # Get keypoints
        nose = self._get_keypoint(keypoints, self.NOSE_IDX)
        left_shoulder = self._get_keypoint(keypoints, self.LEFT_SHOULDER_IDX)
        right_shoulder = self._get_keypoint(keypoints, self.RIGHT_SHOULDER_IDX)
        left_elbow = self._get_keypoint(keypoints, self.LEFT_ELBOW_IDX)
        right_elbow = self._get_keypoint(keypoints, self.RIGHT_ELBOW_IDX)
        left_wrist = self._get_keypoint(keypoints, self.LEFT_WRIST_IDX)
        right_wrist = self._get_keypoint(keypoints, self.RIGHT_WRIST_IDX)
        left_hip = self._get_keypoint(keypoints, self.LEFT_HIP_IDX)
        right_hip = self._get_keypoint(keypoints, self.RIGHT_HIP_IDX)

        # Check 1: Arms raised (wrists above hips)
        if self._arms_raised(left_wrist, right_wrist, left_hip, right_hip):
            score += 1.5

        # Check 2: Arms bent (elbow angle indicates holding something)
        bent_score = self._check_arm_bend(
            left_shoulder, left_elbow, left_wrist,
            right_shoulder, right_elbow, right_wrist
        )
        score += bent_score

        # Check 3: Hands near upper body
        if self._hands_near_upper_body(
            left_wrist, right_wrist, nose, left_shoulder, right_shoulder
        ):
            score += 1.5

        # Check 4: Phone within person's reach
        if self._phone_in_reach(keypoints, phone_bbox):
            score += 1.0

        confidence = score / max_score
        is_phone_pose = confidence >= self.phone_usage_pose_threshold

        return is_phone_pose, confidence

    def _arms_raised(
        self,
        left_wrist: Optional[NDArray],
        right_wrist: Optional[NDArray],
        left_hip: Optional[NDArray],
        right_hip: Optional[NDArray]
    ) -> bool:
        """Check if arms are raised (wrists above hips)."""
        if left_hip is None and right_hip is None:
            return False

        hip_y = None
        if left_hip is not None and right_hip is not None:
            hip_y = (left_hip[1] + right_hip[1]) / 2
        elif left_hip is not None:
            hip_y = left_hip[1]
        else:
            hip_y = right_hip[1]

        # Check if any wrist is above hips
        raised = False
        if left_wrist is not None and left_wrist[1] < hip_y:
            raised = True
        if right_wrist is not None and right_wrist[1] < hip_y:
            raised = True

        return raised

    def _check_arm_bend(
        self,
        left_shoulder: Optional[NDArray],
        left_elbow: Optional[NDArray],
        left_wrist: Optional[NDArray],
        right_shoulder: Optional[NDArray],
        right_elbow: Optional[NDArray],
        right_wrist: Optional[NDArray]
    ) -> float:
        """
        Check arm bend angles.

        Returns score (0-1) based on how bent the arms are.
        """
        score = 0.0

        # Check left arm
        if all(kp is not None for kp in [left_shoulder, left_elbow, left_wrist]):
            angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
            # Phone usage typically has elbow angle between 45-120 degrees
            if 45 <= angle <= 120:
                score += 0.5

        # Check right arm
        if all(kp is not None for kp in [right_shoulder, right_elbow, right_wrist]):
            angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
            if 45 <= angle <= 120:
                score += 0.5

        return score

    def _hands_near_upper_body(
        self,
        left_wrist: Optional[NDArray],
        right_wrist: Optional[NDArray],
        nose: Optional[NDArray],
        left_shoulder: Optional[NDArray],
        right_shoulder: Optional[NDArray]
    ) -> bool:
        """Check if hands are near upper body (head/shoulders)."""
        if nose is None or (left_shoulder is None and right_shoulder is None):
            return False

        # Calculate upper body center
        if left_shoulder is not None and right_shoulder is not None:
            upper_body_center = (left_shoulder + right_shoulder) / 2
            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        elif left_shoulder is not None:
            upper_body_center = left_shoulder
            shoulder_width = 100  # pixels
        else:
            upper_body_center = right_shoulder
            shoulder_width = 100

        # Define "near" as within 2x shoulder width
        threshold = shoulder_width * 2.0

        # Check if any wrist is near upper body
        for wrist in [left_wrist, right_wrist]:
            if wrist is None:
                continue

            # Check distance to nose
            if np.linalg.norm(wrist - nose) < threshold:
                return True

            # Check distance to upper body center
            if np.linalg.norm(wrist - upper_body_center) < threshold:
                return True

        return False

    def _phone_in_reach(
        self,
        keypoints: NDArray,
        phone_bbox: NDArray
    ) -> bool:
        """
        Check if phone is within person's reach.

        Simply checks if phone center is reasonably close to any body keypoint.
        """
        phone_center = np.array([
            (phone_bbox[0] + phone_bbox[2]) / 2,
            (phone_bbox[1] + phone_bbox[3]) / 2
        ])

        # Check distance to all visible keypoints
        for i in range(len(keypoints)):
            kp = self._get_keypoint(keypoints, i)
            if kp is None:
                continue

            distance = np.linalg.norm(kp - phone_center)
            # If phone within 400 pixels of any keypoint, it's in reach
            if distance < 400:
                return True

        return False

    def _create_arm_bbox(
        self,
        shoulder: NDArray,
        elbow: Optional[NDArray],
        wrist: NDArray
    ) -> NDArray:
        """
        Create bounding box encompassing arm segment.

        Args:
            shoulder: Shoulder keypoint [x, y]
            elbow: Elbow keypoint [x, y] or None
            wrist: Wrist keypoint [x, y]

        Returns:
            Bounding box [x1, y1, x2, y2]
        """
        points = [shoulder, wrist]
        if elbow is not None:
            points.append(elbow)

        points = np.array(points)

        x_coords = points[:, 0]
        y_coords = points[:, 1]

        # Add padding
        padding = 40  # pixels

        return np.array([
            np.min(x_coords) - padding,
            np.min(y_coords) - padding,
            np.max(x_coords) + padding,
            np.max(y_coords) + padding
        ])

    def _calculate_angle(
        self,
        p1: NDArray,
        p2: NDArray,
        p3: NDArray
    ) -> float:
        """
        Calculate angle at p2 formed by p1-p2-p3.

        Args:
            p1: First point [x, y]
            p2: Middle point (vertex) [x, y]
            p3: Third point [x, y]

        Returns:
            Angle in degrees (0-180)
        """
        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angle = np.degrees(np.arccos(cos_angle))
        return angle

    def _calculate_iou(
        self,
        bbox1: NDArray,
        bbox2: NDArray
    ) -> float:
        """
        Calculate Intersection over Union (IoU) between two bboxes.

        Args:
            bbox1: First bbox [x1, y1, x2, y2]
            bbox2: Second bbox [x1, y1, x2, y2]

        Returns:
            IoU score (0.0-1.0)
        """
        # Intersection coordinates
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        # Intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        if intersection == 0:
            return 0.0

        # Calculate areas
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

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
            f"PhoneUsageDetectorV2(hand_bbox={self.hand_bbox_size}px, "
            f"overlap_iou={self.overlap_iou_threshold})"
        )
