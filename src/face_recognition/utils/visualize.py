import random
import cv2
from typing import Any, Tuple


def generate_random_color():
    return tuple(random.randint(0, 255) for _ in range(3))


def display(frame, status):
    cv2.imshow(status, frame)
    # out.write(frame)
    
    return cv2.waitKey(1) & 0xFF == ord("q")

def concat_frames(frame1: Any, frame2: Any, mode: str = "horizontal") -> Any:
    """
    Concatenates two frames either horizontally or vertically.
    If necessary, the second frame is resized to match the first.
    
    Args:
        frame1: First image frame (numpy array).
        frame2: Second image frame (numpy array).
        mode: "horizontal" or "vertical" concatenation.
    
    Returns:
        The concatenated image.
    """
    if mode == "horizontal":
        # Resize frame2 if heights differ
        if frame1.shape[0] != frame2.shape[0]:
            height = frame1.shape[0]
            scale = height / frame2.shape[0]
            frame2 = cv2.resize(frame2, (int(frame2.shape[1] * scale), height))
        concatenated = cv2.hconcat([frame1, frame2])
    elif mode == "vertical":
        # Resize frame2 if widths differ
        if frame1.shape[1] != frame2.shape[1]:
            width = frame1.shape[1]
            scale = width / frame2.shape[1]
            frame2 = cv2.resize(frame2, (width, int(frame2.shape[0] * scale)))
        concatenated = cv2.vconcat([frame1, frame2])
    else:
        raise ValueError("Unsupported mode. Choose 'horizontal' or 'vertical'.")
    return concatenated
