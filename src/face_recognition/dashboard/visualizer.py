"""Visualization utilities for displaying and drawing on video frames."""

import cv2
import numpy as np

class Visualization():
    """Class for visualizing tracking results and other information on video frames."""
    def __init__(self):
        """Initialize the visualization class with default parameters."""
        super().__init__()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5

    @staticmethod
    def concat_frames(frame1: np.ndarray, frame2: np.ndarray, mode: str = "horizontal") -> np.ndarray:
        """Concatenate two frames horizontally or vertically.

        Args:
            frame1: First frame
            frame2: Second frame
            mode: Concatenation mode ("horizontal" or "vertical")

        Returns:
            Concatenated frame
        """
        # Resize frames to window size whcih is 720x1920
        frame1 = cv2.resize(frame1, (1280, 720))
        frame2 = cv2.resize(frame2, (1280, 720))

        if mode == "horizontal":
            if frame1.shape[0] != frame2.shape[0]:
                frame2 = cv2.resize(frame2, (int(frame2.shape[1] * (frame1.shape[0] / frame2.shape[0])), frame1.shape[0]))
            concated_frame =cv2.hconcat([frame1, frame2])

        elif mode == "vertical":
            if frame1.shape[1] != frame2.shape[1]:
                frame2 = cv2.resize(frame2, (frame1.shape[1], int(frame2.shape[0] * (frame1.shape[1] / frame2.shape[1]))))
            concated_frame = cv2.vconcat([frame1, frame2])

        return cv2.resize(concated_frame, (1920, 720))  # Resize to window size