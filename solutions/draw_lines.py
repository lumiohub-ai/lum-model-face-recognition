import cv2
import numpy as np

# Global variables
drawing = False
current_polygon = []
polygons = []

def mouse_callback(event, x, y, flags, param):
    global drawing, current_polygon

    if event == cv2.EVENT_LBUTTONDOWN:
        # Left-click: Add point to current polygon
        current_polygon.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN and current_polygon:
        # Right-click: Close the polygon and save it
        polygons.append(current_polygon.copy())
        current_polygon = []  # Reset for next polygon

def draw_polygons(image):
    temp_image = image.copy()

    # Draw completed polygons
    for polygon in polygons:
        if len(polygon) > 1:
            cv2.polylines(temp_image, [np.array(polygon, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw current polygon (in progress)
    if len(current_polygon) > 1:
        cv2.polylines(temp_image, [np.array(current_polygon, np.int32)], isClosed=False, color=(0, 0, 255), thickness=2)

    return temp_image

def normalize_and_format_polygons(polygons, image_width, image_height):
    annotations = ""
    for i, polygon in enumerate(polygons):
        normalized_points = [
            (x / image_width, y / image_height) for (x, y) in polygon
        ]
        # Flatten and format points
        points_str = " ".join(f"{x} {y}" for x, y in normalized_points)
        annotations += f"{i} {points_str}\n"
    return annotations.strip()

# Load image
image_path = "frame.jpg"  # Change this to your image path
image = cv2.imread(image_path)
image_height, image_width = image.shape[:2]

cv2.namedWindow("Polygon Drawer")
cv2.setMouseCallback("Polygon Drawer", mouse_callback)

while True:
    display_image = draw_polygons(image)
    cv2.imshow("Polygon Drawer", display_image)

    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Press "Enter" to finish drawing
        if current_polygon:
            polygons.append(current_polygon.copy())
            current_polygon = []
    elif key == 27:  # Press "Esc" to exit
        break

cv2.destroyAllWindows()

# Normalize and format the collected polygons
print("Collected Polygons:\n", polygons)
annotations = normalize_and_format_polygons(polygons, image_width, image_height)

# Print and save the output
print("Formatted Annotations:\n", annotations)
