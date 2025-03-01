import cv2
import numpy as np

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