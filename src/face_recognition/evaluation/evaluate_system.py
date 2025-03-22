import os
import logging
import csv
import subprocess
import sys
import pandas as pd
from hbface import HBFace
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
def evaluate_recognition(video_name, alg_name, benchmark="hbface"):
    id_to_name_path = f'annotations/id_to_name_{video_name}.json'
    pred_path = f'results/{video_name}_recognition_{alg_name}-{benchmark}.txt'

    # Check all files exist
    for path in [id_to_name_path, pred_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    with open(id_to_name_path, "r") as file:
        id_to_name = json.load(file)

    id_to_name = {int(k): v for k, v in id_to_name.items()}

    # Load and prepare data 
    pred_df = pd.read_csv(pred_path, names=['frame', 'x', 'y', 'w', 'h', 'name'])

    # Get all unique GT names (even those not matched)
    all_gt_names = id_to_name.values()
    all_gt_names = [name for name in all_gt_names if name != "Unknown"]

    # Get all unique predicted names (even those not matched)
    all_pred_names = sorted(pred_df['name'].unique())

    tp = 0
    fp = 0
    fn = 0

    tp_names = []
    fp_names = []
    fn_names = []

    for gt_name in all_gt_names:
        # Check if the name is predicted
        if gt_name in all_pred_names:
            tp += 1
            tp_names.append(gt_name)
        else:
            fn += 1
            fn_names.append(gt_name)
    
    for pred_name in all_pred_names:
        # Check if the name is a false positive
        if pred_name not in all_gt_names:
            fp += 1
            fp_names.append(pred_name)
        
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    accuracy = tp / (tp + fn + fp) if tp + fn + fp > 0 else 0

    # Print only person-based metrics
    print(f"\n---- Person-based Metrics ----")
    print(f"TP: {tp} (People correctly identified)")
    print(f"FP: {fp} (Incorrect identifications)")
    print(f"FN: {fn} (People never correctly identified)")

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # print(f"-------------------------------\n")
    # print(f"\nTP Names: {tp_names}")
    # print(f"FP Names: {fp_names}")
    # print(f"FN Names: {fn_names}")
    # print("GT Names: ", all_gt_names)
    # print("Pred Names: ", all_pred_names)
    

    return precision, recall, accuracy

# Function to extract data from a given path
def extract_data(summary_path):
    data = {}
    
    if os.path.exists(summary_path):
        # Read the CSV file
        df = pd.read_csv(summary_path)

        for _, row in df.iterrows():
            data[row.get("seq", "unknown")] = {
                "DetA": row.get("DetA___AUC", 0),
                "DetRe": row.get("DetRe___AUC", 0),
                "DetPr": row.get("DetPr___AUC", 0),
                "LocA": row.get("LocA___AUC", 0),
            }
    else:
        print(f"Error: Summary file not found at {summary_path}")
        return {}
    
    return data

def evaluate_mot(benchmark, tracker_to_eval, split_to_eval="train"):
    subprocess.run([
        sys.executable, "TrackEval/scripts/run_mot_challenge.py",
        "--BENCHMARK", benchmark,
        "--SPLIT_TO_EVAL", split_to_eval,
        "--TRACKERS_TO_EVAL", tracker_to_eval,
        "--METRICS", "HOTA", "CLEAR", "Identity", "VACE",
        "--USE_PARALLEL", "False",
        "--NUM_PARALLEL_CORES", "1"
    ])

    # From results directory of trackeval, get the results of the evaluation
    results_dir = f'TrackEval/data/trackers/mot_challenge/{benchmark}-{split_to_eval}/{tracker_to_eval}/pedestrian_detailed.csv'
    return extract_data(results_dir)

def save_inference_results(mot_results, output_recognition_path, output_tracking_path):
    """Saves recognition and tracking results to text files."""
    print(f"Saving results to {output_recognition_path} and {output_tracking_path}")
    if not mot_results:
        logging.warning("No results to save.")
        return
        
    df = pd.DataFrame(mot_results)
    
    # Sort by frame and id
    df = df.sort_values(by=['frame', 'id'])
    
    # Remove duplicate entries (same frame, same ID, different original track IDs)
    df = df.drop_duplicates(subset=['frame', 'id'])
    
    with open(output_tracking_path, 'w') as f:
        for _, row in df.iterrows():
            mot_format_text = f"{int(row['frame'])},{int(row['id'])},{int(row['x'])},{int(row['y'])},{int(row['w'])},{int(row['h'])},{row['conf']},-1,-1,-1,-1"
            
            f.write(mot_format_text + '\n')
    
    with open(output_recognition_path, 'w') as r:
        # Format sample 'frame', 'x', 'y', 'w', 'h', 'name'
        for _, row in df.iterrows():
            recognition_format_text = f"{int(row['frame'])},{int(row['x'])},{int(row['y'])},{int(row['w'])},{int(row['h'])},{row['name']}"
            
            r.write(recognition_format_text + '\n')

def inference(results, video_paths, alg_name, benchmark):
    """Processes videos and evaluates recognition results."""
    for video_name, video_path in video_paths.items():
        logging.info(f"Processing {video_name} - {video_path}")
        output_recognition_path, output_tracking_path = set_paths(video_name, alg_name, benchmark)

        # Process the video
        streamer = HBFace(cam_type="IN", video_path=video_path, eval=True, 
                        db_path="data/ilhan-aligned", show=True,
                        match_threshold=0.6, detection_threshold=0.5, imgsz=1280,
                        roi=(302, 82, 986, 976), line_points=[(129, 241), (1799, 267)])
                
        streamer.run()

        mot_results = streamer.mot_results

        # Save MOT results
        save_inference_results(mot_results, output_recognition_path, output_tracking_path)

        precision, recall, accuracy = evaluate_recognition(video_name, alg_name, benchmark=benchmark)
        results[video_name] = {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy
        }

        logging.info(f"Finished processing {video_name}")

    return results

def evaluate_tracking(results, alg_name, benchmark):
    """Evaluates multiple object tracking (MOT) performance."""
    mot_results = evaluate_mot(benchmark, alg_name)
    
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
        writer.writerow(["precision", "recall", "accuracy", "DetA", "DetRe", "DetPr", "LocA"])
        
        for video_name, res in results.items():
            writer.writerow([
                video_name, 
                res.get("precision", 0), res.get("recall", 0), res.get("accuracy", 0),
                res.get("DetA", 0), res.get("DetRe", 0), res.get("DetPr", 0), res.get("LocA", 0)
            ])
        
    logging.info(f"Results saved successfully to {output_path}.")

def set_paths(video_name, alg_name, benchmark):
    output_recognition_path = f'results/{video_name}_recognition_{alg_name}-{benchmark}.txt'
    output_tracking_path = f'TrackEval/data/trackers/mot_challenge/{benchmark}-train/{alg_name}/data/{video_name}.txt'
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_tracking_path), exist_ok=True)

    # Remove existing files if they exist
    for path in [output_recognition_path, output_tracking_path]:
        if os.path.exists(path):
            os.remove(path)
            
    return output_recognition_path, output_tracking_path

def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run inference and evaluate tracking on videos.')
    
    # Add arguments
    parser.add_argument('--videos', nargs='+', help='List of video names to process', default=["videoa1-1", "videoa1-2", "videoa1-3", "videoa1-4", "videoa1-5"])
    parser.add_argument('--video_dir', default='client/pred_videos', help='Directory containing videos')
    parser.add_argument('--alg_name', default='alg11', help='Algorithm name')
    parser.add_argument('--benchmark', default='ilhan', help='Benchmark name') # hbface or ilhan
    parser.add_argument('--output', default=None, help='Output filename for results CSV')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up video paths dictionary
    video_paths = {}
    
    # If specific videos are provided, use those
    if args.videos:
        for video in args.videos:
            video_paths[video] = os.path.join(args.video_dir, f"{video}_eval.mp4")
    else:
        # Default videos if none specified
        default_videos = ["videoa1-1", "videoa1-2", "videoa1-3", "videoa1-4", "videoa1-5"]
        for video in default_videos:
            video_paths[video] = os.path.join(args.video_dir, f"{video}_eval.mp4")
    
    # Process the videos
    results = {}
    results = inference(results, video_paths, args.alg_name, args.benchmark)
    results = evaluate_tracking(results, args.alg_name, args.benchmark)
    
    # Determine output filename
    output_filename = args.output if args.output else f"{args.alg_name}_results.csv"
    
    # Save results to a CSV file
    save_results(results, output_filename)
    
    print(f"Processing complete. Results saved to {output_filename}")

if __name__ == "__main__":
    main()
