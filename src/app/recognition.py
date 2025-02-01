import cv2
import face_recognition
import os
import pickle

# Load stored face encodings and names
if os.path.exists("face_data.pkl"):
    with open("face_data.pkl", "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    known_face_encodings = []
    known_face_names = []

# Load video file
video_path = "rtsp://admin:hbai2024@172.30.1.63:554/Streaming/Channels/1"  # Change this to your video file path
video_capture = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec if needed
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
        else:
            label = input("Who is this? ")
            if label:
                known_face_encodings.append(face_encoding)
                known_face_names.append(label)
                with open("face_data.pkl", "wb") as f:
                    pickle.dump((known_face_encodings, known_face_names), f)
                name = label

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Video", frame)
    out.write(frame)  # Write processed frame to output video

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as {output_path}")
