import os
import sys
import argparse
import subprocess
from pathlib import Path
import cv2


def get_video_info(video_path, info_type):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            sys.exit(1)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        return {"width": width, "height": height, "frames": frames}.get(info_type)

def create_seqinfo_ini(video_name, output_dir, width, height, frames):
    """Create seqinfo.ini file with video properties"""
    seqinfo_path = os.path.join(output_dir, "seqinfo.ini")
    
    with open(seqinfo_path, 'w') as f:
        f.write(f"[Sequence]\n")
        f.write(f"name={video_name}\n")
        f.write(f"imDir=img1\n")
        f.write(f"frameRate=25\n")
        f.write(f"seqLength={frames}\n")
        f.write(f"imWidth={width}\n")
        f.write(f"imHeight={height}\n")
        f.write(f"imExt=.jpg\n")
    
    print(f"Created seqinfo.ini for {video_name}")

def process_video(video_path, label_dir, base_output_dir):
    """Process a single video and organize output"""
    video_name = os.path.basename(video_path).replace('.webm', '')
    json_file = os.path.join(label_dir, f"{video_name}.json")

    # Create output directory structure
    video_output_dir = os.path.join(base_output_dir, video_name)
    gt_output_dir = os.path.join(video_output_dir, "gt")
    
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(gt_output_dir, exist_ok=True)
    
    # Get video properties
    width = get_video_info(video_path, "width")
    height = get_video_info(video_path, "height")
    frames = get_video_info(video_path, "frames")
    
    # Create seqinfo.ini
    create_seqinfo_ini(video_name, video_output_dir, width, height, frames)
    
    # Check if JSON file exists
    if os.path.isfile(json_file):
        print(f"Processing {video_name}...")
        
        # Run the Python script to process the video
        output_txt_path = os.path.join(gt_output_dir, "gt.txt")
        video_dir = os.path.dirname(video_path)
        
        cmd = [
            "python3", "src/face_recognition/evaluation/json_mini_converter.py",
            "-j", json_file,
            "-v", video_path,
            "--output_txt_path", output_txt_path,
            "--verify"
        ]
        print(cmd)
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Completed processing {video_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {video_name}: {e}")
    else:
        print(f"⚠️ Warning: JSON label file not found for {video_name}")

def main():
    parser = argparse.ArgumentParser(description="Process video files and organize output for TrackEval")
    parser.add_argument("--video_dir", default="./videos", help="Directory containing video files")
    parser.add_argument("--label_dir", default="./labels", help="Directory containing label JSON files")
    parser.add_argument("--output_dir", default="TrackEval/data/gt/mot_challenge/ilhan-train", 
                        help="Base output directory")
    
    args = parser.parse_args()
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # List of videos to process
    videos = [
        os.path.join(args.video_dir, "videoa1-1.webm"),
        os.path.join(args.video_dir, "videoa1-2.webm"),
        os.path.join(args.video_dir, "videoa1-3.webm"),
        os.path.join(args.video_dir, "videoa1-4.webm"),
        os.path.join(args.video_dir, "videoa1-5.webm")
    ]
    
    # Process each video
    print("Starting video processing...")
    for video_path in videos:
        if os.path.isfile(video_path):
            process_video(video_path, args.label_dir, args.output_dir)
        else:
            print(f"⚠️ Warning: Video file not found: {video_path}")
    
    print("All processing complete!")

if __name__ == "__main__":
    main()