import os
import argparse
import logging
import csv
from evaluation.predict_mot import FaceRecognitionSystem
from evaluation.reca import eval_recognition
from evaluation.eval_mot import eval_mot

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run face recognition on a video")
    parser.add_argument("--model_arch", type=str, required=True, help="Model architecture")
    parser.add_argument("--alg_name", type=str, required=True, help="Name of the algorithm")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--match_threshold", type=int, default=0.7, help="Match threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--tracker", type=str, default="default_tracker", help="Tracker name")
    return parser.parse_args()

def process_videos(face_system, video_paths, args):
    """Processes videos and evaluates recognition results."""
    results = {}
    
    for video_name, video_path in video_paths.items():
        logging.info(f"Processing {video_name} - {video_path}")

        face_system.process_video(
            video_path=video_path,
            video_name=video_name,
            alg_name=args.alg_name,
        )

    return results

def evaluate_mot(benchmark, args, results):
    """Evaluates multiple object tracking (MOT) performance."""
    mot_results = eval_mot(benchmark, args.alg_name)
    
    for video_name, res in mot_results.items():
        if video_name in results:
            results[video_name].update({
                "DetA": res.get("DetA", 0),
                "DetRe": res.get("DetRe", 0),
                "DetPr": res.get("DetPr", 0),
                "LocA": res.get("LocA", 0)
            })

    return results

def save_results(results, alg_name):
    """Saves per-video results to a CSV file."""
    output_path = f"results/algs/{alg_name}_results.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logging.info(f"Saving results to {output_path}")

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["video_name", "precision", "recall", "accuracy", "DetA", "DetRe", "DetPr", "LocA"])
        
        for video_name, res in results.items():
            writer.writerow([
                video_name, 
                res.get("precision", 0), res.get("recall", 0), res.get("accuracy", 0),
                res.get("DetA", 0), res.get("DetRe", 0), res.get("DetPr", 0), res.get("LocA", 0)
            ])

def calculate_and_save_avg(results, main_path, alg_name):
    """Calculates and appends average results for the algorithm to a CSV file."""
    if not results:
        logging.warning("No results to process for averaging.")
        return

    avg_results = {
        key: sum(res.get(key, 0) for res in results.values()) / len(results)
        for key in ["precision", "recall", "accuracy", "DetA", "DetRe", "DetPr", "LocA"]
    }

    # Prepare row to append
    text_alg = f"{alg_name},{avg_results['precision']:.4f},{avg_results['recall']:.4f},{avg_results['accuracy']:.4f},{avg_results['DetA']:.4f},{avg_results['DetRe']:.4f},{avg_results['DetPr']:.4f},{avg_results['LocA']:.4f}\n"

    # Append the average results to the main CSV file
    os.makedirs(os.path.dirname(main_path), exist_ok=True)

    with open(main_path, "a") as f:
        f.write(text_alg)
    
    logging.info(f"Updated {main_path} with average results for {alg_name}")

def main():
    args = parse_arguments()

    face_system = FaceRecognitionSystem(
        model_arch=args.model_arch,
        confidence_threshold=args.conf,
        imgsz=args.imgsz,
        tracker=args.tracker,
        match_threshold=args.match_threshold,
        show=True,
        eval=True
    )

    video_paths = {
        "video10": "videos/output_10_processed_fps.mp4",
        "video11": "videos/output_11_processed_fps.mp4",
    }

    benchmark = "hbface"
    main_path = "results.csv"

    results = process_videos(face_system, video_paths, args)
    results = evaluate_mot(benchmark, args, results)
    
    save_results(results, args.alg_name)
    calculate_and_save_avg(results, main_path, args.alg_name)

if __name__ == "__main__":
    main()
