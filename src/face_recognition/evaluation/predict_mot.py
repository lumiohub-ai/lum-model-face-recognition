import cv2
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set

import pandas as pd
from ultralytics import YOLO

# Add required paths to system path
sys.path.append(os.curdir)
sys.path.append(os.path.join(os.curdir, 'src/face_recognition'))

from models import FaceRecognitionModel


def set_paths(video_name: str, alg_name: str) -> Tuple[str, str]:
    """
    Set up output paths for recognition and tracking results and ensure directories exist.
    
    Args:
        video_name: Name of the video being processed
        alg_name: Name of the algorithm being used
        
    Returns:
        Tuple of (recognition_path, tracking_path)
    """
    output_recognition_path = f'results/{video_name}_recognition_{alg_name}.txt'
    output_tracking_path = f'TrackEval/data/trackers/mot_challenge/hbface-train/{alg_name}/data/{video_name}.txt'
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_tracking_path), exist_ok=True)

    # Remove existing files if they exist
    for path in [output_recognition_path, output_tracking_path]:
        if os.path.exists(path):
            os.remove(path)
            
    return output_recognition_path, output_tracking_path


def process_detections(
    frame: Any,
    detections: Any, 
    frame_num: int,
    track_crops_frame: Dict[int, Dict[int, Any]],
    track_boxes_frame: Dict[int, Dict[int, List]],
    all_tracks: List[int]
) -> None:
    """
    Process detection results for the current frame.
    
    Args:
        frame: Current video frame
        detections: Detection results from YOLO
        frame_num: Current frame number
        track_crops_frame: Dictionary to store face crops by track ID and frame
        track_boxes_frame: Dictionary to store bounding boxes by track ID and frame
        all_tracks: List to keep track of all track IDs
    """
    if detections[0].boxes.id is None:
        return
        
    boxes = detections[0].boxes.data.cpu().tolist()
    track_ids = detections[0].boxes.id.cpu().tolist()
    
    for det, track_id in zip(boxes, track_ids):
        track_id = int(track_id)
        
        all_tracks.append(track_id)
        conf = det[5]
        x1, y1, x2, y2, _, _, _ = map(int, det)
        w = x2 - x1
        h = y2 - y1

        face = frame[y1:y2, x1:x2]
        
        if track_id not in track_crops_frame:
            track_crops_frame[track_id] = {}
            track_boxes_frame[track_id] = {}

        track_crops_frame[track_id][frame_num] = face
        track_boxes_frame[track_id][frame_num] = [x1, y1, w, h, conf]


def process_removed_tracks(
    detections: Any,
    passed_tracks: List[int],
    track_crops_frame: Dict[int, Dict[int, Any]],
    track_boxes_frame: Dict[int, Dict[int, List]],
    model: Any,
    frame: Any,
    mot_results: List[Dict[str, Any]],
    name_to_track_id: Dict[int, str],
    name_to_consistent_id: Dict[str, int],
    id_mapping: Dict[int, int]
) -> List[int]:
    """
    Process tracks that have been removed.
    
    Args:
        detections: Detection results from YOLO
        passed_tracks: List of already processed track IDs
        track_crops_frame: Dictionary of face crops by track ID and frame
        track_boxes_frame: Dictionary of bounding boxes by track ID and frame
        model: Face recognition model
        frame: Current video frame
        mot_results: List to store MOT results
        name_to_track_id: Dictionary to map track IDs to recognized names
        name_to_consistent_id: Dictionary to map names to consistent IDs
        id_mapping: Dictionary to map original track IDs to consistent IDs
        
    Returns:
        Updated list of passed track IDs
    """
    removed_tracks = detections[0].removed_tracks.tolist()
    removed_tracks = [track_id for track_id in removed_tracks if track_id not in passed_tracks]

    for track_id in removed_tracks:
        passed_tracks.append(track_id)
        face_embeddings = model.compute_embeddings(track_crops_frame[track_id].values())
        
        # After recognition, delete crops to save memory
        name = model.recognize_face(face_embeddings)
        del track_crops_frame[track_id]
        
        if name != "Unknown":
            # Assign a consistent ID for this name if not already assigned
            if name not in name_to_consistent_id:
                name_to_consistent_id[name] = track_id
                consistent_id = track_id
            else:
                consistent_id = name_to_consistent_id[name]
                
            # Map the original track ID to the consistent ID
            id_mapping[track_id] = consistent_id
            
            # Use consistent ID for display and result storage
            display_id = consistent_id
            
            print(f'Original ID {track_id} recognized as {name}, using consistent ID {display_id}')
            
            # Add to MOT results with the consistent ID
            for frame_num in track_boxes_frame[track_id]:
                box = track_boxes_frame[track_id][frame_num]
                mot_results.append({
                    'frame': frame_num,
                    'id': display_id,  # Use consistent ID in results
                    'x': box[0],
                    'y': box[1],
                    'w': box[2],
                    'h': box[3],
                    'conf': box[4],
                })

            name_to_track_id[track_id] = name
            text_show = f'{display_id} recognized as {name}'
            cv2.putText(frame, text_show, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return passed_tracks


def process_final_frame(
    frame_num: int,
    last_frame: int,
    all_tracks: List[int],
    passed_tracks: List[int],
    track_crops_frame: Dict[int, Dict[int, Any]],
    track_boxes_frame: Dict[int, Dict[int, List]],
    model: Any,
    frame: Any,
    mot_results: List[Dict[str, Any]],
    name_to_track_id: Dict[int, str],
    name_to_consistent_id: Dict[str, int],
    id_mapping: Dict[int, int]
) -> None:
    """
    Process the final frame to handle any remaining tracks.
    
    Args:
        frame_num: Current frame number
        last_frame: Last frame number in the video
        all_tracks: List of all track IDs
        passed_tracks: List of processed track IDs
        track_crops_frame: Dictionary of face crops by track ID and frame
        track_boxes_frame: Dictionary of bounding boxes by track ID and frame
        model: Face recognition model
        frame: Current video frame
        mot_results: List to store MOT results
        name_to_track_id: Dictionary to map track IDs to recognized names
        name_to_consistent_id: Dictionary to map names to consistent IDs
        id_mapping: Dictionary to map original track IDs to consistent IDs
    """
    if frame_num != last_frame:
        return
        
    for track_id in all_tracks:
        if track_id not in passed_tracks and track_id in track_crops_frame:
            face_embeddings = model.compute_embeddings(track_crops_frame[track_id].values())
            name = model.recognize_face(face_embeddings)
            
            del track_crops_frame[track_id]
            name_to_track_id[track_id] = name
            
            if name != "Unknown" and track_id in track_boxes_frame:
                # Assign a consistent ID for this name if not already assigned
                if name not in name_to_consistent_id:
                    name_to_consistent_id[name] = track_id
                    consistent_id = track_id
                else:
                    consistent_id = name_to_consistent_id[name]
                    
                # Map the original track ID to the consistent ID
                id_mapping[track_id] = consistent_id
                
                # Use consistent ID for display and result storage
                display_id = consistent_id
                
                for frame_num_track in track_boxes_frame[track_id]:
                    if frame_num_track in track_boxes_frame[track_id]:
                        box = track_boxes_frame[track_id][frame_num_track]
                        mot_results.append({
                            'frame': frame_num_track,
                            'id': display_id,  # Use consistent ID in results
                            'x': box[0],
                            'y': box[1],
                            'w': box[2],
                            'h': box[3],
                            'conf': box[4],
                        })
                
                text_show = f'{display_id} recognized as {name}'
                cv2.putText(frame, text_show, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def save_mot_results(mot_results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save MOT results to a file.
    
    Args:
        mot_results: List of MOT results
        output_path: Path to save results to
    """
    if not mot_results:
        print("Warning: No MOT results to save.")
        return
        
    df = pd.DataFrame(mot_results)
    
    # Sort by frame and id
    df = df.sort_values(by=['frame', 'id'])
    
    # Remove duplicate entries (same frame, same ID, different original track IDs)
    df = df.drop_duplicates(subset=['frame', 'id', 'x', 'y', 'w', 'h'])
    
    with open(output_path, 'w') as f:
        for _, row in df.iterrows():
            mot_format_text = f"{int(row['frame'])},{int(row['id'])},{int(row['x'])},{int(row['y'])},{int(row['w'])},{int(row['h'])},{row['conf']},-1,-1,-1,-1"
            f.write(mot_format_text + '\n')


def predict(
    video_path: str, 
    video_name: str, 
    alg_name: str,
    model_arch: str, 
    confidence_threshold: float, 
    imgsz: int, 
    tracker: str, 
    match_threshold: int,
    show: bool = True
) -> None:
    """
    Main prediction function that processes the video for face recognition and tracking.
    
    Args:
        video_path: Path to input video
        video_name: Name of the video 
        alg_name: Name of the algorithm
        model_arch: Model architecture for detection
        confidence_threshold: Confidence threshold for detection
        imgsz: Image size for processing
        tracker: Tracker type
        match_threshold: Match threshold for face recognition
        show: Whether to display the processed video
    """
    _, output_tracking_path = set_paths(video_name, alg_name)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit(1)

    # Configure detector and model
    cfg = {
        'model_arch': model_arch,
        'conf': confidence_threshold,
        'imgsz': imgsz,
        'persist': True,
        'tracker': tracker,
        'match_threshold': match_threshold,
    }
    
    detector = YOLO(cfg['model_arch'])
    model = FaceRecognitionModel(match_threshold=cfg['match_threshold'])

    # Initialize tracking variables
    frame_num = 0
    track_crops_frame = {}
    passed_tracks = []
    name_to_track_id = {}  # Maps track IDs to names
    name_to_consistent_id = {}  # Maps names to a consistent ID for the same person
    id_mapping = {}  # Maps original track IDs to consistent IDs
    all_tracks = []
    track_boxes_frame = {}
    mot_results = []
    last_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_num += 1

        # Run detection and tracking
        detections = detector.track(
            frame,
            verbose=False,
            conf=cfg['conf'],
            imgsz=cfg['imgsz'],
            persist=cfg['persist'],
            tracker=cfg['tracker'],
        )

        # Skip if no detections
        if detections[0].boxes.id is None:
            if show:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        # Process detections
        process_detections(
            frame, 
            detections, 
            frame_num, 
            track_crops_frame, 
            track_boxes_frame, 
            all_tracks
        )
        
        # Process removed tracks
        passed_tracks = process_removed_tracks(
            detections,
            passed_tracks,
            track_crops_frame,
            track_boxes_frame,
            model,
            frame,
            mot_results,
            name_to_track_id,
            name_to_consistent_id,
            id_mapping
        )
        
        # Process final frame to handle remaining tracks
        process_final_frame(
            frame_num,
            last_frame,
            all_tracks,
            passed_tracks,
            track_crops_frame,
            track_boxes_frame,
            model,
            frame,
            mot_results,
            name_to_track_id,
            name_to_consistent_id,
            id_mapping
        )
        
        # Display frame if requested
        if show:    
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Consolidate results by consistent IDs
    consolidated_results = []
    for result in mot_results:
        original_id = result['id']
        # Use the consistent ID from mapping if available
        if original_id in id_mapping:
            result['id'] = id_mapping[original_id]
        consolidated_results.append(result)

    # Save consolidated results
    save_mot_results(consolidated_results, output_tracking_path)
    cap.release()
    cv2.destroyAllWindows()


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Face Recognition and Tracking")
    
    parser.add_argument('-p', '--video_path', type=str, required=True,
                        help='Path to the video file')
    parser.add_argument('-v', '--video_name', type=str, required=True,
                        help='Name of the video')
    parser.add_argument('-a', '--alg_name', type=str, required=True,
                        help='Name of the algorithm')
    parser.add_argument('-m', '--model_arch', type=str, required=True,
                        help='Model architecture')
    parser.add_argument('-c', '--confidence_threshold', type=float, required=True,
                        help='Confidence threshold')
    parser.add_argument('-i', '--imgsz', type=int, required=True,
                        help='Image size')
    parser.add_argument('-t', '--tracker', type=str, required=True,
                        help='Tracker')
    parser.add_argument('-mt', '--match_threshold', type=int, required=True,
                        help='Match threshold')
    parser.add_argument('-s', '--show', type=bool, default=True,
                        help='Show video')
                        
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    predict(
        args.video_path, 
        args.video_name, 
        args.alg_name,
        args.model_arch, 
        args.confidence_threshold, 
        args.imgsz, 
        args.tracker, 
        args.match_threshold,
        show=args.show
    )