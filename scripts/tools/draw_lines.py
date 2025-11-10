import cv2
import numpy as np

# Example usage
image_path = "IN.jpg"  # Replace with your image path
image = cv2.imread(image_path)
drawer = ShapeDrawer(image)
shapes = drawer.run()

print("Collected Shapes:", shapes)
