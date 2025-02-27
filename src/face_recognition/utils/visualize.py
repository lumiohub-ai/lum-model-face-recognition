import random
import cv2
import numpy as np
from typing import List, Tuple, Optional


class Visualization:
    def __init__(self, image: Optional[np.ndarray] = None):
        """
        Initializes the Visualization class.

        Args:
            image (Optional[np.ndarray]): Image on which regions can be drawn.
        """
        self.image = image
    
    @staticmethod
    def generate_random_color() -> Tuple[int, int, int]:
        """
        Generates a random color in BGR format.
        
        Returns:
            Tuple[int, int, int]: Random BGR color.
        """
        return tuple(random.randint(0, 255) for _ in range(3))
    
    @staticmethod
    def display(frame: np.ndarray, window_name: str = "Frame") -> bool:
        """
        Displays a frame in a window.
        
        Args:
            frame (np.ndarray): The image frame to be displayed.
            window_name (str): Name of the display window.
        
        Returns:
            bool: True if 'q' is pressed, otherwise False.
        """
        cv2.imshow(window_name, frame)
        return cv2.waitKey(1) & 0xFF == ord("q")
    
    def draw_region(self,
                    image: np.ndarray,
                    reg_pts: List[Tuple[int, int]], 
                    color: Tuple[int, int, int] = (0, 255, 0), 
                    thickness: int = 5) -> None:
        """
        Draws a region using the given points on the stored image.

        Args:
            reg_pts (List[Tuple[int, int]]): List of region points (2 points for a line, 4 points for a region).
            color (Tuple[int, int, int]): Color of the region (default is green).
            thickness (int): Thickness of the region lines.
        """
        if image is None:
            raise ValueError("No image provided for drawing.")

        cv2.polylines(image, [np.array(reg_pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)
        
        for point in reg_pts:
            cv2.circle(image, tuple(point), thickness * 2, color, -1)  # Draw small filled circles at corner points
    
    @staticmethod
    def concat_frames(frame1: np.ndarray, frame2: np.ndarray, mode: str = "horizontal") -> np.ndarray:
        """
        Concatenates two frames either horizontally or vertically.
        If necessary, the second frame is resized to match the first.

        Args:
            frame1 (np.ndarray): First image frame.
            frame2 (np.ndarray): Second image frame.
            mode (str): "horizontal" or "vertical" concatenation.

        Returns:
            np.ndarray: Concatenated image.
        """
        if mode == "horizontal":
            if frame1.shape[0] != frame2.shape[0]:
                frame2 = cv2.resize(frame2, (int(frame2.shape[1] * (frame1.shape[0] / frame2.shape[0])), frame1.shape[0]))
            return cv2.hconcat([frame1, frame2])
        elif mode == "vertical":
            if frame1.shape[1] != frame2.shape[1]:
                frame2 = cv2.resize(frame2, (frame1.shape[1], int(frame2.shape[0] * (frame1.shape[1] / frame2.shape[1]))))
            return cv2.vconcat([frame1, frame2])
        else:
            raise ValueError("Unsupported mode. Choose 'horizontal' or 'vertical'.")
