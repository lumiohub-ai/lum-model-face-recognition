import random
import cv2
import numpy as np
from typing import List, Tuple

class Visualization():
    def __init__(self):
        super().__init__()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.color_track_id = {}
    
    def define_color(self, track_id: int) -> Tuple[int, int, int]:
        if track_id not in self.color_track_id:
            self.color_track_id[track_id] = self.generate_random_color()
        return self.color_track_id[track_id]
    
    @staticmethod
    def generate_random_color() -> Tuple[int, int, int]:
        return tuple(random.randint(0, 255) for _ in range(3))
    
    @staticmethod
    def display(frame: np.ndarray, window_name: str = "Frame") -> bool:
        cv2.imshow(window_name, frame)
        return cv2.waitKey(1) & 0xFF == ord("q")
    
    def draw_region(self,
                    image: np.ndarray,
                    reg_pts: List[Tuple[int, int]], 
                    color: Tuple[int, int, int] = (0, 255, 0), 
                    thickness: int = 5) -> None:
        if image is None:
            raise ValueError("No image provided for drawing.")

        cv2.polylines(image, [np.array(reg_pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)
        
        for point in reg_pts:
            cv2.circle(image, tuple(point), thickness * 2, color, -1)  # Draw small filled circles at corner points
    
    @staticmethod
    def concat_frames(frame1: np.ndarray, frame2: np.ndarray, mode: str = "horizontal") -> np.ndarray:
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


class ShapeDrawer:
    def __init__(self, image):
        """
        Initializes the ShapeDrawer with an image.
        :param image_path: Path to the input image
        """
        self.image = image
        self.clone = self.image.copy()
        self.drawing = False
        self.shapes = []  # Stores all drawn shapes (lines/polygons)
        self.current_shape = []  # Stores points of the current shape
    
    @staticmethod
    def get_first_frame(source):
        cap = cv2.VideoCapture(source)
        _, frame = cap.read()
        cap.release()
        
        return frame

    @staticmethod
    def initialize_shape_drawer(frame):
        return ShapeDrawer(frame).run()[0]

    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback function for drawing shapes interactively.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_shape.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and self.current_shape:
            self.shapes.append(self.current_shape.copy())
            self.current_shape = []

    def draw_shapes(self):
        """
        Draws all stored shapes on a copy of the image.
        """
        temp_image = self.clone.copy()
        
        # Draw completed shapes
        for shape in self.shapes:
            if len(shape) > 1:
                cv2.polylines(temp_image, [np.array(shape, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Draw current shape (in progress)
        if len(self.current_shape) > 1:
            cv2.polylines(temp_image, [np.array(self.current_shape, np.int32)], isClosed=False, color=(0, 0, 255), thickness=2)
        
        return temp_image

    def run(self):
        """
        Runs the interactive shape drawing process.
        """
        cv2.namedWindow("Shape Drawer")
        cv2.setMouseCallback("Shape Drawer", self.mouse_callback)

        while True:
            display_image = self.draw_shapes()
            cv2.imshow("Shape Drawer", display_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Press "Enter" to save current shape
                if self.current_shape:
                    self.shapes.append(self.current_shape.copy())
                    self.current_shape = []
            elif key == 27:  # Press "Esc" to exit
                break
        
        cv2.destroyAllWindows()
        return self.shapes