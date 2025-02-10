import cv2
import numpy as np
import os

class PolygonManager:
    def __init__(self, annotation_file="annotations.txt"):
        self.annotation_file = annotation_file
        self.polygons = []
        self.current_polygon = []
        self.drawing = True

    def load_annotations(self, image_height, image_width):
        """Load annotations from a file if it exists."""
        if os.path.exists(self.annotation_file) and os.path.getsize(self.annotation_file) > 0:
            with open(self.annotation_file, 'r') as file:
                for line in file.readlines():
                    # Parse and convert saved annotations
                    points = list(map(float, line.strip().split()[1:]))
                    polygon = [(int(points[i] * image_width), int(points[i+1] * image_height)) 
                               for i in range(0, len(points), 2)]
                    self.polygons.append(polygon)
            return True
        else:
            return False

    def save_annotations(self, image_height, image_width):
        """Save annotations to a file in normalized format."""
        with open(self.annotation_file, 'w') as file:
            for i, polygon in enumerate(self.polygons):
                normalized_points = [(x / image_width, y / image_height) for (x, y) in polygon]
                points_str = " ".join(f"{x} {y}" for x, y in normalized_points)
                file.write(f"{i} {points_str}\n")

    def add_polygon(self, polygon):
        """Add a new polygon."""
        self.polygons.append(polygon)

    def get_polygons(self):
        return self.polygons
    
    def get_annotations(self):
        """Return the current list of polygons."""
        if os.path.exists(self.annotation_file) and os.path.getsize(self.annotation_file) > 0:
            with open(self.annotation_file, 'r') as file:
                annotations = file.readlines()
        return annotations

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left-click: Add point to current polygon
            self.current_polygon.append((x, y))

        elif event == cv2.EVENT_RBUTTONDOWN and self.current_polygon:
            # Right-click: Close the polygon and save it
            self.add_polygon(self.current_polygon.copy())
            self.current_polygon = []  # Reset for next polygon

    def draw_polygons(self, image):
        temp_image = image.copy()

        # Draw completed polygons
        for polygon in self.get_polygons():
            if len(polygon) > 1:
                cv2.polylines(temp_image, [np.array(polygon, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw current polygon (in progress)
        if len(self.current_polygon) > 1:
            cv2.polylines(temp_image, [np.array(self.current_polygon, np.int32)], isClosed=False, color=(0, 0, 255), thickness=2)

        return temp_image
    
    def set_polygons(self, cap, add_polygon=False):
        # Get first frame of the video to determine polygons
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Error reading video file")
        
        # Get image width and height
        image_height, image_width = frame.shape[:2]

        # Load saved annotations if available
        annot_exist = self.load_annotations(image_height, image_width)

        if annot_exist and not add_polygon:
            return self.get_annotations()
        
        # Create window and set mouse callback
        cv2.namedWindow("Polygon Drawer")
        cv2.setMouseCallback("Polygon Drawer", self.mouse_callback)

        while True:
            display_image = self.draw_polygons(frame)
            cv2.imshow("Polygon Drawer", display_image)

            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Press "Enter" to finish drawing
                if self.current_polygon:
                    self.add_polygon(self.current_polygon.copy())
                    self.current_polygon = []
            elif key == 27:  # Press "Esc" to exitq
                break
        
        self.save_annotations(image_height, image_width)
        cv2.destroyAllWindows()

        return self.get_annotations()
