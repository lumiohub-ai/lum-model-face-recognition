import cv2

def draw_rectangle(event, x, y, flags, param):
    global drawing, ix, iy, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = img.copy()
            cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Image', temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        w, h = abs(x - ix), abs(y - iy)
        x, y = min(ix, x), min(iy, y)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(f"Rectangle Coordinates: x={x}, y={y}, w={w}, h={h}")
        cv2.imshow('Image', img)

# Load image
img = cv2.imread('image.jpg')  # Change to your image path
roi_coordinates = (528, 41, 714, 659)
img = img[roi_coordinates[1]:roi_coordinates[1] + roi_coordinates[3], roi_coordinates[0]:roi_coordinates[0] + roi_coordinates[2]]

cv2.imshow('Image', img)
drawing = False
ix = iy = 0

cv2.setMouseCallback('Image', draw_rectangle)
cv2.waitKey(0)
cv2.destroyAllWindows()


