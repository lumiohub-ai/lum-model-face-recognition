import os
import logging
import csv
import subprocess
import sys
import pandas as pd
from hbface import HBFace

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

def save_mot_results(mot_results, output_path):
    if not mot_results:
        print("Warning: No MOT results to save.")
        return
        
    df = pd.DataFrame(mot_results)
    
    # Sort by frame and id
    df = df.sort_values(by=['frame', 'id'])
    
    # Remove duplicate entries (same frame, same ID, different original track IDs)
    df = df.drop_duplicates(subset=['frame', 'id'])
    
    with open(output_path, 'w') as f:
        for _, row in df.iterrows():
            mot_format_text = f"{int(row['frame'])},{int(row['id'])},{int(row['x'])},{int(row['y'])},{int(row['w'])},{int(row['h'])},{row['conf']},-1,-1,-1,-1"
            
            f.write(mot_format_text + '\n')

def process_videos(video_paths, alg_name, benchmark):
    """Processes videos and evaluates recognition results."""
    for video_name, video_path in video_paths.items():
        logging.info(f"Processing {video_name} - {video_path}")
        _, output_tracking_path = set_paths(video_name, alg_name, benchmark)

        # Process the video
        streamer = HBFace(video_path, cam_type="IN", annot=True, eval=True)
        streamer.run()

        mot_results = streamer.mot_results

        # Save MOT results
        save_mot_results(mot_results, output_tracking_path)       

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
        writer.writerow(["DetA", "DetRe", "DetPr", "LocA"])
        
        for video_name, res in results.items():
            writer.writerow([
                video_name, 
                res.get("precision", 0), res.get("recall", 0), res.get("accuracy", 0),
                res.get("DetA", 0), res.get("DetRe", 0), res.get("DetPr", 0), res.get("LocA", 0)
            ])

def set_paths(video_name, alg_name, benchmark):
    output_recognition_path = f'results/{video_name}_recognition_{alg_name}.txt'
    output_tracking_path = f'TrackEval/data/trackers/mot_challenge/{benchmark}-train/{alg_name}/data/{video_name}.txt'
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_tracking_path), exist_ok=True)

    # Remove existing files if they exist
    for path in [output_recognition_path, output_tracking_path]:
        if os.path.exists(path):
            os.remove(path)
            
    return output_recognition_path, output_tracking_path

def main():
    video_paths = {
        "videoa1-1": "client/pred_videos/videoa1-1_eval.mp4",
        "videoa1-2": "client/pred_videos/videoa1-2_eval.mp4",
        "videoa1-3": "client/pred_videos/videoa1-3_eval.mp4",
        "videoa1-4": "client/pred_videos/videoa1-4_eval.mp4",
        "videoa1-5": "client/pred_videos/videoa1-5_eval.mp4",
    }

    benchmark = "ilhan"
    alg_name = "alg1-ilhan"
    
    results = {}
    process_videos(video_paths, alg_name, benchmark)
    results = evaluate_tracking(results, alg_name, benchmark)

    # Save results to a CSV file
    save_results(results, alg_name)


if __name__ == "__main__":
    main()
