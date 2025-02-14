import cv2
import numpy as np
import os

class RectangleManager:
    def __init__(self, annotation_file="annotations.txt"):
        self.annotation_file = annotation_file
        self.rectangles = []
        self.start_point = None
        self.drawing = False

    def load_annotations(self, image_height, image_width):
        """Load annotations from a file if it exists."""
        if os.path.exists(self.annotation_file) and os.path.getsize(self.annotation_file) > 0:
            with open(self.annotation_file, 'r') as file:
                for line in file.readlines():
                    # Parse and convert saved annotations
                    points = list(map(float, line.strip().split()[1:]))
                    rectangle = [
                        (int(points[0] * image_width), int(points[1] * image_height)),  # top-left corner
                        (int(points[2] * image_width), int(points[3] * image_height))   # bottom-right corner
                    ]
                    self.rectangles.append(rectangle)
            return True
        else:
            return False

    def save_annotations(self, image_height, image_width):
        """Save annotations to a file in normalized format."""
        with open(self.annotation_file, 'w') as file:
            for i, rect in enumerate(self.rectangles):
                # Normalize the points
                normalized_points = [(x / image_width, y / image_height) for (x, y) in rect]
                points_str = " ".join(f"{x} {y}" for x, y in normalized_points)
                file.write(f"{i} {points_str}\n")

    def add_rectangle(self, rectangle):
        """Add a new rectangle."""
        self.rectangles.append(rectangle)

    def get_rectangles(self):
        return self.rectangles

    def get_annotations(self):
        """Return the current list of rectangles."""
        if os.path.exists(self.annotation_file) and os.path.getsize(self.annotation_file) > 0:
            with open(self.annotation_file, 'r') as file:
                annotations = file.readlines()
        return annotations

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left-click: Start drawing rectangle
            self.start_point = (x, y)
            self.drawing = True

        elif event == cv2.EVENT_LBUTTONUP:
            # Left-click release: Define the rectangle with start point and current point
            self.drawing = False
            self.add_rectangle([self.start_point, (x, y)])
            self.start_point = None

    def draw_rectangles(self, image):
        temp_image = image.copy()

        # Draw completed rectangles
        for rect in self.get_rectangles():
            cv2.rectangle(temp_image, rect[0], rect[1], (0, 255, 0), 2)

        # Draw current rectangle (in progress)
        if self.drawing and self.start_point:
            cv2.rectangle(temp_image, self.start_point, (cv2.mouseCallback.x, cv2.mouseCallback.y), (0, 0, 255), 2)

        return temp_image

    def set_rectangles(self, cap, add_rectangle=False):
        # Get first frame of the video to determine rectangles
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Error reading video file")
        
        # Get image width and height
        image_height, image_width = frame.shape[:2]

        # Load saved annotations if available
        annot_exist = self.load_annotations(image_height, image_width)

        if annot_exist and not add_rectangle:
            return self.get_annotations()
        
        # Create window and set mouse callback
        cv2.namedWindow("Rectangle Drawer")
        cv2.setMouseCallback("Rectangle Drawer", self.mouse_callback)

        while True:
            display_image = self.draw_rectangles(frame)
            cv2.imshow("Rectangle Drawer", display_image)

            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Press "Enter" to finish drawing
                if self.start_point:
                    self.add_rectangle([self.start_point, (cv2.mouseCallback.x, cv2.mouseCallback.y)])
                    self.start_point = None
            elif key == 27:  # Press "Esc" to exit
                break
        
        self.save_annotations(image_height, image_width)
        cv2.destroyAllWindows()

        return self.get_annotations()
