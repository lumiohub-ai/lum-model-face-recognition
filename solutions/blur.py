import cv2

def blur_video(input_path, output_path, x, y, w, h):

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract ROI and apply blur
        roi = frame[y:y+h, x:x+w]
        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
        frame[y:y+h, x:x+w] = blurred_roi

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
blur_video('output.mp4', 'output_blurred.mp4', 361, 145, 350, 513)
