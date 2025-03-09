import streamlit as st
import cv2
import os
import tempfile
from cfg import Config
from models import FaceRecognitionModel, TrackManager
from utils import Visualization, EntryLogger
from ultralytics import YOLO
import os


def process_video(video_source):
    cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
    
    output_path = f'output_{os.path.basename(video_source)}.avi'
    print(f"Output path: {output_path}")
    
    cfg = Config()
    
    model = FaceRecognitionModel(cfg.device, cfg.face_crops_path, 
                                 cfg.in_region_points, cfg.out_region_points)
    detector = YOLO('yolov8n-face.pt')

    project_comparison_name = 'comparison_server_video_3'
    os.makedirs(project_comparison_name, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    visualize = Visualization()
    track = TrackManager()
    entry_logger = EntryLogger(cfg.logging_path)

    if not cap.isOpened():
        st.error("Error: Unable to open video stream.")
        return

    stop = False
    stframe = st.empty()  # Create a single placeholder for the video frame

    frame_count = 0
    while cap.isOpened():
        frame_count += 1
        model.check_new_faces()

        ret, frame = cap.read()
        if not ret or stop:
            break

        detections = detector.track(
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
        person_names = []

        det_num = 0
        for det, track_id in zip(boxes, track_ids):
            det_num += 1
            x1, y1, x2, y2, _, _, _ = map(int, det)
            h, w, _ = frame.shape
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]
            # track.track_frame_count[track_id] += 1

            # if track.track_frame_count[track_id] >= cfg.check_interval:
            #     track.name_to_track_id.pop(track_id, None)
            #     track.track_frame_count[track_id] = 0
            name = track.name_to_track_id.get(track_id, "Detecting...")
            
            if name == "Detecting...":
                face_emb = model.compute_embeddings(face)
                name = model.recognize_face(face_emb)
                
                if name != "Detecting...":
                    track.name_to_track_id[track_id] = name
                    person_names.append(name)
                    
                    person_path = os.path.join(cfg.face_crops_path, f"{name}.jpg.jpg")
                    person = cv2.imread(person_path)

                    if person is not None:
                        face_resized = cv2.resize(face, (person.shape[1], person.shape[0]))
                        comparison_image = cv2.hconcat([person, face_resized])
                        save_path = os.path.join(project_comparison_name, f"{name}_comparison_{det_num}_{frame_count}.jpg")
                        cv2.imwrite(save_path, comparison_image)

                    entry_logger.log_person_entry(name, 'IN')

            if name != "Detecting...":
                color = track.name_to_color.setdefault(name, visualize.generate_random_color())
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2)

        entry_logger.visualize_entries(frame)
        out.write(frame)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Display frame in RGB format
    
    st.write("Person names detected in the video:", set(person_names))

    cap.release()
    out.release()
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
    
    if input_type == "Video File":
        uploaded_file = st.sidebar.file_uploader("Upload a Video File", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded_file.read())
            # Open session state for video file
            st.session_state.video_path = temp_file.name
            
            process_video(st.session_state.video_path)
    
    elif input_type == "RTSP Stream":
        rtsp_url = st.sidebar.text_input("Enter RTSP URL")
        if st.sidebar.button("Start Stream") and rtsp_url:
            process_video(rtsp_url)

if __name__ == "__main__":
    main()
