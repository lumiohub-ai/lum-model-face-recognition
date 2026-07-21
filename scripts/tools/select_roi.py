import cv2
import numpy as np

def draw_rectangle(event, x, y, flags, param):
    global drawing, ix, iy, display_img, original_img, scale_factor_x, scale_factor_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = display_img.copy()
            cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Image', temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        w, h = abs(x - ix), abs(y - iy)
        x_display, y_display = min(ix, x), min(iy, y)
        
        # Draw on display image
        cv2.rectangle(display_img, (x_display, y_display), (x_display + w, y_display + h), (0, 255, 0), 2)
        cv2.imshow('Image', display_img)
        
        # Calculate coordinates in original image
        x1 = int(x_display * scale_factor_x)
        y1 = int(y_display * scale_factor_y)
        x2 = int((x_display + w) * scale_factor_x)
        y2 = int((y_display + h) * scale_factor_y)
        
        print(f"{x1},{y1},{x2},{y2}")
        

# Original image dimensions
ORIGINAL_WIDTH = 2560
ORIGINAL_HEIGHT = 1440

# Target display size (adjust to what works on your screen)
TARGET_HEIGHT = 800  # Adjust this value based on your screen

# Load image
original_img = cv2.imread('frame_in.jpg')

# Calculate scaling factors
scale_factor_y = ORIGINAL_HEIGHT / TARGET_HEIGHT
scale_factor_x = ORIGINAL_WIDTH / (TARGET_HEIGHT * ORIGINAL_WIDTH / ORIGINAL_HEIGHT)

# Resize image for display
display_width = int(ORIGINAL_WIDTH / scale_factor_x)
display_height = TARGET_HEIGHT
display_img = cv2.resize(original_img, (display_width, display_height))

# Setup drawing variables
drawing = False
ix = iy = 0

# Line drawing variables
line_drawing = False
line_ix = line_iy = 0
line_drawn = False
roi_coords = [0, 0, 0, 0]

cv2.imshow('Image', display_img)
cv2.setMouseCallback('Image', draw_rectangle)
print("First, select ROI by drawing a rectangle.")
cv2.waitKey(0)
cv2.destroyAllWindows()


