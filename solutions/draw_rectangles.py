import cv2

def draw_rectangle_interactive(image_path):
    """
    Opens an image and allows the user to draw a rectangle interactively.
    Prints the bounding box coordinates in x1 y1 x2 y2 format.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return
    
    clone = image.copy()
    bbox = []
    
    def draw_bbox(event, x, y, flags, param):
        nonlocal bbox, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            bbox = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            bbox.append((x, y))
            x1, y1 = bbox[0]
            x2, y2 = bbox[1]
            cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Draw Bounding Box", clone)
            print(f"Bounding Box: {x1} {y1} {x2} {y2}")
    
    cv2.imshow("Draw Bounding Box", image)
    cv2.setMouseCallback("Draw Bounding Box", draw_bbox)
    
    while True:
        cv2.imshow("Draw Bounding Box", clone)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press ESC to exit
            break
    
    cv2.destroyAllWindows()

# Example usage
image_path = "frame.jpg"  # Replace with your image path
draw_rectangle_interactive(image_path)


# In bbox
# 372 30 1241 720
# Out bbox
# 2 114 547 717