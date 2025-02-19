import streamlit as st
import cv2
import os
import tempfile
from cfg import Config
from engine import FaceRecognitionModel, TrackManager
from utils import EntryLogger, generate_random_color, display


def process_video(video_source, cfg):
    cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        st.error("Error: Unable to open video stream.")
        return

    stop = False
    stframe = st.empty()  # Create a single placeholder for the video frame

    while cap.isOpened():
        model.check_new_faces()

        ret, frame = cap.read()
        if not ret or stop:
            break

        # x, y, w, h = cfg.camera_roi_coordinates
        # frame = frame[y : y + h, x : x + w]

        detections = model.detector.track(
            frame,
            conf=cfg.detection_threshold,
            verbose=False,
            imgsz=cfg.imgsz,
            persist=True,
        )

        if detections[0].boxes.id is None:
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            continue

        boxes = detections[0].boxes.data.cpu().tolist()
        track_ids = detections[0].boxes.id.cpu().tolist()

        for det, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2, _, _, _ = map(int, det)
            h, w, _ = frame.shape
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]
            track.track_frame_count[track_id] += 1

            if track.track_frame_count[track_id] >= cfg.check_interval:
                track.name_to_track_id.pop(track_id, None)
                track.track_frame_count[track_id] = 0

            name = track.name_to_track_id.get(track_id, "Detecting...")
            
            if name == "Detecting...":
                face_emb = model.compute_embeddings(face)
                name = model.recognize_face(face_emb)
                if name != "Detecting...":
                    track.name_to_track_id[track_id] = name

            entry_logger.log_person_entry(name)

            if name not in track.name_to_color:
                track.name_to_color[name] = (
                    (0, 0, 0) if name == "Detecting..." else generate_random_color()
                )

            color = track.name_to_color[name]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2
            )

        entry_logger.visualize_entries(frame)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Display frame in RGB format
    
    cap.release()
    cv2.destroyAllWindows()

def init_session_state():
    default_values = {'video_path': None}
    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def main():
    st.title("Face Recognition Streamlit App")
    st.sidebar.header("Input Options")
    
    input_type = st.sidebar.radio("Choose Input Type", ("Video File", "RTSP Stream"))
    init_session_state()
    
    cfg = Config()
    global model, entry_logger, track
    model = FaceRecognitionModel(cfg.device, cfg.face_crops_path, cfg.match_threshold)
    entry_logger = EntryLogger(cfg.logging_path)
    track = TrackManager()
    
    if input_type == "Video File":
        uploaded_file = st.sidebar.file_uploader("Upload a Video File", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded_file.read())
            # Open session state for video file
            st.session_state.video_path = temp_file.name
            
            process_video(st.session_state.video_path, cfg)
            os.remove(temp_file.name)
    
    elif input_type == "RTSP Stream":
        rtsp_url = st.sidebar.text_input("Enter RTSP URL")
        if st.sidebar.button("Start Stream") and rtsp_url:
            process_video(rtsp_url, cfg)

if __name__ == "__main__":
    main()
