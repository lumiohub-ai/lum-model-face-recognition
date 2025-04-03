import sys
import os
import time
from datetime import datetime
import argparse

sys.path.append(os.curdir)
from src.face_recognition.hbface import HBFace

def parse_roi(roi_str):
    """
    Parse ROI string of the format "x,y,width,height" into a tuple of integers.
    """
    try:
        roi = tuple(map(int, roi_str.split(',')))
        if len(roi) != 4:
            raise ValueError("ROI must have 4 values: x,y,width,height")
        return roi
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid ROI format: {e}")
    
def get_arguments():
    parser = argparse.ArgumentParser(description="Multi-camera face recognition using HBFace")
    
    parser.add_argument("--tz", type=str, default="Asia/Tashkent", help="Timezone (default: Asia/Tashkent)")
    parser.add_argument("--in_camera", type=str, default="rtsp://admin:Namhbai01@192.168.13.17:554",
                        help="RTSP stream URL for IN camera")
    parser.add_argument("--out_camera", type=str, 
                        default="rtsp://admin:Namhbai01@192.168.13.16:554",
                        help="RTSP stream URL for OUT camera")
    parser.add_argument("--roi_in", type=parse_roi, default="383,53,1015,709",
                        help="ROI for IN camera in format x,y,width,height")
    parser.add_argument("--roi_out", type=parse_roi, default="639,1,1276,717",
                        help="ROI for OUT camera in format x,y,width,height")
    parser.add_argument("--show", action="store_true",
                        help="Display the processed video (default is off)")
    parser.add_argument("--match_threshold", type=float, default=0.6,
                        help="Face matching threshold")
    parser.add_argument("--detection_threshold", type=float, default=0.5,
                        help="Face detection threshold")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Image size for processing")
    parser.add_argument("--padding_ratio", type=float, default=0.05,
                        help="Padding ratio for ROI")
    parser.add_argument("--db_path", type=str, default="data/hb-uzb",
                        help="Path to face database")
    parser.add_argument("--backend_url", type=str, default="http://backend:4000/graphql", 
                        help="Backend URL for API calls")
    
    return parser.parse_args()

def main():
    args = get_arguments()
    
    # Set timezone based on command-line argument
    os.environ['TZ'] = args.tz
    time.tzset()

    print("Starting face recognition...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize HBFace for multi-camera setup with parsed arguments
    face_engine_multi = HBFace(
        cam_type="IN",
        video_path=args.in_camera,
        #roi=[args.roi_in],
        multi_camera=False,
        show=args.show,
        match_threshold=args.match_threshold,
        detection_threshold=args.detection_threshold,
        imgsz=args.imgsz,
        padding_ratio=args.padding_ratio,
        db_path=args.db_path,
        backend_url=args.backend_url,
    )
    
    # Run the face recognition system
    face_engine_multi.run()
    
    print("Face recognition completed.")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
