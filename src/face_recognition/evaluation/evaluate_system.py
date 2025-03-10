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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def evaluate_recognition(video_name, alg_name):
    id_to_name_path = f'annotations/id_to_name_{video_name}.json'
    gt_path = f'TrackEval/data/gt/mot_challenge/hbface-train/{video_name}/gt/gt.txt'
    pred_path = f'results/{video_name}_recognition_{alg_name}.txt'

    # Check all files exist
    for path in [id_to_name_path, gt_path, pred_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    with open(id_to_name_path, "r") as file:
        id_to_name = json.load(file)

    id_to_name = {int(k): v for k, v in id_to_name.items()}

    # Load and prepare data 
    gt_df = pd.read_csv(gt_path, names=['frame', 'track_id', 'x', 'y', 'w', 'h', 'confidence', 'class', 'visibility']) 
    pred_df = pd.read_csv(pred_path, names=['frame', 'x', 'y', 'w', 'h', 'name'])

    # Clean ground truth data 
    gt_df = gt_df[['frame', 'track_id', 'x', 'y', 'w', 'h']] 
    gt_df['name'] = gt_df['track_id'].map(id_to_name) 
    gt_df = gt_df[['frame', 'x', 'y', 'w', 'h', 'name']]

    # Lists to store ground truth and predicted names 
    true_names = [] # Ground truth 
    pred_names = [] # Predictions

    # Calculate total_frames from the maximum frame number in both dataframes
    total_frames = max(gt_df['frame'].max(), pred_df['frame'].max())

    # Function to calculate IOU (Intersection over Union)
    def calculate_iou(box1, box2):
        # Extract coordinates
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate coordinates of the intersection box
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        # Check if there is an intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate area of intersection
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate area of both bounding boxes
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        # Calculate IOU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou

    # Variables to count metrics
    id_tp = 0  # Identity True Positive
    id_fp = 0  # Identity False Positive
    id_fn = 0  # Identity False Negative

    # Lists to store ground truth and predicted names (only for matched detections)
    true_names = []
    pred_names = []

    # Process each frame
    for frame in range(1, total_frames + 1):
        # Get data for current frame
        frame_gt = gt_df[gt_df['frame'] == frame]
        frame_pred = pred_df[pred_df['frame'] == frame]
        
        # Skip if either ground truth or predictions are empty
        if frame_gt.empty or frame_pred.empty:
            if not frame_gt.empty:
                # All ground truth detections in this frame are missed (FN)
                id_fn += len(frame_gt)
                
            elif not frame_pred.empty:
                # All predicted detections in this frame are false positives (FP)
                id_fp += len(frame_pred)
            continue
        
        # Keep track of matched ground truth and predictions
        matched_gt = set()
        matched_pred = set()
        
        # Check each ground truth against each prediction
        for gt_idx, gt_row in frame_gt.iterrows():
            gt_box = [gt_row['x'], gt_row['y'], gt_row['w'], gt_row['h']]
            gt_name = gt_row['name']
            best_iou = 0
            best_pred_idx = None
            
            for pred_idx, pred_row in frame_pred.iterrows():
                if pred_idx in matched_pred:
                    continue  # Skip already matched predictions
                    
                pred_box = [pred_row['x'], pred_row['y'], pred_row['w'], pred_row['h']]
                pred_name = pred_row['name']
                
                # Calculate IOU between ground truth and prediction
                iou = calculate_iou(gt_box, pred_box)
                
                # Keep track of best match
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx
            
            # If we found a match with IOU > 0.5
            if best_iou > 0.5 and best_pred_idx is not None:
                matched_gt.add(gt_idx)
                matched_pred.add(best_pred_idx)
                
                # Get the name of the matched prediction
                pred_name = frame_pred.loc[best_pred_idx, 'name']
                
                # Store names for confusion matrix
                true_names.append(gt_name)
                pred_names.append(pred_name)
                
                # Check if the predicted name is correct
                if pred_name == gt_name:
                    id_tp += 1  # Identity True Positive
                else:
                    id_fp += 1  # Identity False Positive
        
        # Count unmatched ground truth as false negatives
        id_fn += len(frame_gt) - len(matched_gt)
        
        # Count unmatched predictions as false positives
        id_fp += len(frame_pred) - len(matched_pred)


    # Calculate ID precision, recall, F1, and accuracy
    id_precision = id_tp / (id_tp + id_fp) if (id_tp + id_fp) > 0 else 0
    id_recall = id_tp / (id_tp + id_fn) if (id_tp + id_fn) > 0 else 0
    id_f1 = 2 * (id_precision * id_recall) / (id_precision + id_recall) if (id_precision + id_recall) > 0 else 0
    id_accuracy = id_tp / (id_tp + id_fp + id_fn) if (id_tp + id_fp + id_fn) > 0 else 0
    
    print(f"Precision: {id_precision:.4f}")
    print(f"Recall: {id_recall:.4f}")
    print(f"F1 Score: {id_f1:.4f}")
    print(f"Accuracy: {id_accuracy:.4f}")

    if true_names and pred_names:
        # Get unique names (classes)
        unique_names = sorted(list(set(true_names) | set(pred_names)))

        # Calculate confusion matrix
        cm = confusion_matrix(true_names, pred_names, labels=unique_names)

        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Identity Confusion Matrix')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'results/plots/{video_name}_confusion_matrix_{alg_name}.png')
    
    else:
        print("No matches found for confusion matrix.")

    return id_precision, id_recall, id_accuracy

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

def save_results(mot_results, output_recognition_path, output_tracking_path):
    """Saves recognition and tracking results to text files."""
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
        streamer = HBFace(video_path, cam_type="IN", annot=True, eval=True)
        streamer.run()

        mot_results = streamer.mot_results

        # Save MOT results
        save_results(mot_results, output_recognition_path, output_tracking_path)

        precision, recall, accuracy = evaluate_recognition(video_name, alg_name)
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
    results = inference(results, video_paths, alg_name, benchmark)
    results = evaluate_tracking(results, alg_name, benchmark)

    # Save results to a CSV file
    save_results(results, alg_name)


if __name__ == "__main__":
    main()
