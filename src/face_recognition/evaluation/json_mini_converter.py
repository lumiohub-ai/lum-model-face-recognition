import argparse
import json
import cv2
import os
from moviepy.editor import VideoFileClip
import pandas as pd

def linear_interpolation(start_seq, end_seq, label, start_frame, end_frame,
                         image_width, image_height):
    frame_diff = end_frame - start_frame
    
    interpolated_boxes = {}
    
    for frame in range(start_frame, end_frame + 1):
        if frame == start_frame:
            t = 0
        elif frame == end_frame:
            t = 1
        else:
            t = (frame - start_frame) / frame_diff
        
        interpolated_box = {
            'label': label,
            'x': int((start_seq['x'] + t * (end_seq['x'] - start_seq['x'])) * image_width / 100),
            'y': int((start_seq['y'] + t * (end_seq['y'] - start_seq['y'])) * image_height / 100),
            'width': int((start_seq['width'] + t * (end_seq['width'] - start_seq['width'])) * image_width / 100),
            'height': int((start_seq['height'] + t * (end_seq['height'] - start_seq['height'])) * image_height / 100)            
        }
        
        interpolated_boxes[frame] = interpolated_box
    
    return interpolated_boxes

def process_video_annotation(video_annotation, video_dir, labels_dict, label_studio_fps):
    if isinstance(video_annotation, list):
        for video in video_annotation:
            video_annotation = video
    else:
        video_annotation = video_annotation

    video_path = video_dir
    video_name = os.path.basename(video_path)
    
    print(f"Processing annotation for video: {video_name}")
    
    if video_path:
        vidcap = cv2.VideoCapture(str(video_path))

        image_height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        image_width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_rate_ratio = 1

    else:
        print("Error: Video file not found.")
        return
    
    # Gather labels from this video
    for subject in video_annotation['box']:
        
        for label in subject['labels']:
            if label not in labels_dict:
                labels_dict[label] = len(labels_dict)

    total_frames = video_annotation['box'][0]['framesCount']
    boxes_dict = {frame: [] for frame in range(1, int(total_frames) + 1)}
    
    for idx, subject in enumerate(video_annotation['box']):
        label = idx
        sequences = sorted(subject['sequence'], key=lambda x: x['frame'])
        
        for i, seq in enumerate(sequences):
            frame = int(seq['frame'] * frame_rate_ratio)
            
            if seq['enabled']:
                next_seq = sequences[i+1] if i+1 < len(sequences) else None
                
                if next_seq:
                    end_frame = int(next_seq['frame'] * frame_rate_ratio) - 1
                else:
                    end_frame = total_frames
                
                current_box = {k: float(seq[k]) for k in ('x', 'y', 'width', 'height')}

                if next_seq and next_seq['enabled']:
                    next_box = {k: float(next_seq[k]) for k in ('x', 'y', 'width', 'height')}
                else:
                    next_box = current_box
                
                interpolated_boxes = linear_interpolation(current_box, next_box, label, frame, end_frame,
                                                          image_width, image_height)
                for f, box in interpolated_boxes.items():
                    boxes_dict[f].append(box)
            else:
                box = {'label': label, 
                       'x': int(seq['x'] * image_width / 100),
                       'y': int(seq['y'] * image_height / 100),
                       'width': int(seq['width'] * image_width / 100),
                       'height': int(seq['height'] * image_height / 100)}
                
                boxes_dict[frame].append(box)
    
    return boxes_dict


def process_video_fps(video_path, label_studio_fps):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found - {video_path}")
        return

    try:
        # Load video
        video_clip = VideoFileClip(video_path)
        output_path = os.path.basename(video_path).split('.')[0] + '_processed_fps.mp4'

        # Set the FPS
        video_clip = video_clip.set_fps(label_studio_fps)

        # Write the output video
        video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

        # Close the video file properly
        video_clip.close()
        print("Video processing complete.")

    except Exception as e:
        print(f"Error processing video: {e}")
    
    return output_path

def show_video_annotations(output_video_path, output_txt_path):
    mot_data_path = output_txt_path
    
    # Read MOT data
    columns = ["frame", "track_id", "x", "y", "w", "h", "confidence", "class", "visibility"]
    df = pd.read_csv(mot_data_path, names=columns)

    cap = cv2.VideoCapture(output_video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'{output_video_path}_visualized.avi', fourcc, fps, (image_width, image_height))

    frame_count = 0  # Track frame index

    while True:
        # Read frame from video
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends

        frame_count += 1  # Update frame index

        # Get tracking data for the current frame
        frame_data = df[df["frame"] == frame_count]

        # Draw bounding boxes for all objects in this frame
        for _, row in frame_data.iterrows():
            track_id = row["track_id"]
            x, y, w, h = row["x"], row["y"], row["w"], row["h"]

            # Draw bounding box
            color = (0, 255, 0)  # Green box
            thickness = 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

            # Display track ID
            cv2.putText(frame, f"ID: {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show frame
        cv2.imshow("Tracking Verification", frame)
        out.write(frame)

        # Control playback speed and allow exit
        key = cv2.waitKey(30) & 0xFF  # Adjust the speed by changing 30 (milliseconds delay)
        if key == ord('q'):  # Press 'q' to exit
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    out.release()

def main(json_path, video_dir, output_txt_path, label_studio_fps, verify=True):
    print("Parsing annotations from JSON")
    with open(json_path) as f:
        video_annotations = json.load(f)
    
    labels_dict = {}
    data = []
    boxes_dict = process_video_annotation(
            video_annotations, video_dir, labels_dict, label_studio_fps
    )
    data.append(boxes_dict)

    print(f"Writing annotations to {output_txt_path}")

    for boxes_dict in data:
        for frame, boxes in boxes_dict.items():
            if not boxes:
                continue

            for box in boxes:
                # make integer every value
                box['x'] = int(box['x'])
                box['y'] = int(box['y'])
                box['width'] = int(box['width'])
                box['height'] = int(box['height'])
                box['label'] = int(box['label'])
                frame = int(frame)

                saving_txt = f"{frame},{box['label'] + 1},{box['x']},{box['y']},{box['width']},{box['height']},1,1,1\n"

                with open(output_txt_path, 'a') as f:
                    f.write(saving_txt)
    
    output_video_path = process_video_fps(video_dir, label_studio_fps)
    
    # Video saved to output_video_path
    print(f"Output video saved to {output_video_path}")

    # Verify the output video
    if verify:
        show_video_annotations(output_video_path, output_txt_path)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""This script processes video annotations exported from Label Studio
        in JSON-MIN format, converting them directly into the COCO dataset format. 
        The script supports interpolation of bounding boxes for intermediate frames based 
        on key-frame annotations and exports these labels along with corresponding frames.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-j", "--json_path", required=True, help="Path to JSON annotations")
    parser.add_argument("-v", "--video_dir", help="Path to directory containing video files")
    parser.add_argument("--label_studio_fps", type=float, default=25, help="Label Studio FPS")
    parser.add_argument("--output_txt_path", default='output_mot_10.txt', help="Path to output txt file")
    parser.add_argument("--verify", action='store_true', default=True, help="Verify the output video")
    args = parser.parse_args()

    main(args.json_path, args.video_dir, 
        args.output_txt_path, args.label_studio_fps, args.verify)