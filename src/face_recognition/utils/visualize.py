import random
import cv2


def generate_random_color():
    return tuple(random.randint(0, 255) for _ in range(3))


def display(frame, out):
    cv2.imshow("Face Recognition", frame)
    out.write(frame)
    return cv2.waitKey(1) & 0xFF == ord("q")
