import os
import csv
import subprocess
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

from hbface import HBFace
from utils import ColorLogger

class Evaluator:
    """Class for evaluating face recognition and tracking performance."""
    def __init__(self, 
                 alg_name: str = "default_alg", 
                 benchmark: str = "ilhan",
                 video_dir: str = "client/pred_videos",
                 results_dir: str = "results",
                 tracking_data_dir: str = "TrackEval/data/trackers/mot_challenge",
                db_path: str = "data/hb-kor",
                 **kwargs
                 ):
        self.logger = ColorLogger()
        self.alg_name = alg_name
        self.benchmark = benchmark
        self.video_dir = video_dir
        self.results_dir = results_dir
        self.tracking_data_dir = tracking_data_dir
        self.db_path = db_path

        self.imgsz = kwargs.get("imgsz", 1280)
        self.match_threshold = kwargs.get("match_threshold", 0.7)
        self.detection_threshold = kwargs.get("detection_threshold", 0.5)
        self.roi = kwargs.get("roi", (302, 82, 986, 976))
        self.line_points = kwargs.get("line_points", [(129, 241), (1799, 267)])
        self.show = kwargs.get("show", False)
        self.padding_ratio = kwargs.get("padding_ratio", 0.2)

        # Create results directory if it doesn't exist
        os.makedirs(os.path.join(results_dir, "algs"), exist_ok=True)
        
        # Dictionary to store evaluation results
        self.results = {}
    
    def set_paths(self, video_name: str) -> Tuple[str, str]:
        output_recognition_path = os.path.join(
            self.results_dir, 
            f"{video_name}_recognition_{self.alg_name}-{self.benchmark}.txt"
        )
        
        output_tracking_path = os.path.join(
            self.tracking_data_dir,
            f"{self.benchmark}-train",
            self.alg_name,
            "data",
            f"{video_name}.txt"
        )
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(output_tracking_path), exist_ok=True)

        # Remove existing files if they exist
        for path in [output_recognition_path, output_tracking_path]:
            if os.path.exists(path):
                os.remove(path)
                
        return output_recognition_path, output_tracking_path
    
    def save_inference_results(self, 
                              mot_results: List[Dict], 
                              output_recognition_path: str, 
                              output_tracking_path: str) -> None:
        self.logger.info(f"Saving results to {output_recognition_path} and {output_tracking_path}")
        
        if not mot_results:
            self.logger.warning("No results to save.")
            return
            
        df = pd.DataFrame(mot_results)
        
        # Sort by frame and id
        df = df.sort_values(by=['frame', 'id'])
        
        # Remove duplicate entries (same frame, same ID, different original track IDs)
        df = df.drop_duplicates(subset=['frame', 'id'])
        
        # Save tracking results in MOT format
        with open(output_tracking_path, 'w') as f:
            for _, row in df.iterrows():
                mot_format_text = (
                    f"{int(row['frame'])},{int(row['id'])},{int(row['x'])},{int(row['y'])},"
                    f"{int(row['w'])},{int(row['h'])},{row['conf']},-1,-1,-1,-1"
                )
                f.write(mot_format_text + '\n')
        
        # Save recognition results
        with open(output_recognition_path, 'w') as r:
            for _, row in df.iterrows():
                recognition_format_text = (
                    f"{int(row['frame'])},{int(row['x'])},{int(row['y'])},"
                    f"{int(row['w'])},{int(row['h'])},{row['name']}"
                )
                r.write(recognition_format_text + '\n')
    
    def evaluate_recognition(self, video_name: str) -> Tuple[float, float, float]:
        id_to_name_path = f'annotations/id_to_name_{video_name}.json'
        pred_path = f'{self.results_dir}/{video_name}_recognition_{self.alg_name}-{self.benchmark}.txt'

        # Check all files exist
        for path in [id_to_name_path, pred_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

        # Load ground truth mapping
        with open(id_to_name_path, "r") as file:
            id_to_name = json.load(file)

        id_to_name = {int(k): v for k, v in id_to_name.items()}

        # Load predictions
        pred_df = pd.read_csv(pred_path, names=['frame', 'x', 'y', 'w', 'h', 'name'])

        # Get all unique GT names (even those not matched)
        all_gt_names = list(id_to_name.values())
        all_gt_names = [name for name in all_gt_names if name != "Unknown"]

        # Get all unique predicted names
        all_pred_names = sorted(pred_df['name'].unique())

        # Calculate metrics
        tp = 0  # True positives - correctly identified persons
        fp = 0  # False positives - incorrect identifications
        fn = 0  # False negatives - missed persons

        tp_names = []  # List of correctly identified names
        fp_names = []  # List of falsely identified names
        fn_names = []  # List of missed names

        # Calculate true positives and false negatives
        for gt_name in all_gt_names:
            if gt_name in all_pred_names:
                tp += 1
                tp_names.append(gt_name)
            else:
                fn += 1
                fn_names.append(gt_name)
        
        # Calculate false positives
        for pred_name in all_pred_names:
            if pred_name not in all_gt_names:
                fp += 1
                fp_names.append(pred_name)
            
        # Calculate metrics
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        accuracy = tp / (tp + fn + fp) if tp + fn + fp > 0 else 0

        # Print person-based metrics
        print(f"\n---- Person-based Metrics for {video_name} ----")
        print(f"TP: {tp} (People correctly identified)")
        print(f"FP: {fp} (Incorrect identifications)")
        print(f"FN: {fn} (People never correctly identified)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        return precision, recall, accuracy
    
    def extract_tracking_data(self, summary_path: str) -> Dict:
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
            self.logger.error(f"❌ Summary file not found at {summary_path}")
            return {}
        
        return data
    
    def evaluate_mot(self, split_to_eval: str = "train") -> Dict:
        self.logger.info(f"🔍 Evaluating MOT performance for {self.alg_name} on {self.benchmark}")
        
        try:
            subprocess.run([
                sys.executable, "TrackEval/scripts/run_mot_challenge.py",
                "--BENCHMARK", self.benchmark,
                "--SPLIT_TO_EVAL", split_to_eval,
                "--TRACKERS_TO_EVAL", self.alg_name,
                "--METRICS", "HOTA", "CLEAR", "Identity", "VACE",
                "--USE_PARALLEL", "False",
                "--NUM_PARALLEL_CORES", "1"
            ], check=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Error running MOT evaluation: {e}")
            return {}

        # Get results from TrackEval
        results_dir = os.path.join(
            self.tracking_data_dir,
            f"{self.benchmark}-{split_to_eval}",
            self.alg_name,
            "pedestrian_detailed.csv"
        )
        
        return self.extract_tracking_data(results_dir)
    
    def process_video(self, video_name: str, video_path: str) -> None:
        self.logger.info(f"🎥 Processing {video_name} - {video_path}")
        
        output_recognition_path, output_tracking_path = self.set_paths(video_name)

        # Create face recognition system with evaluation enabled
        streamer = HBFace(
            cam_type=["IN"], 
            video_path=video_path, 
            eval=True,
            db_path=self.db_path, 
            show=self.show,
            imgsz=self.imgsz,
            match_threshold=self.match_threshold,
            detection_threshold=self.detection_threshold,
            roi=self.roi,
            line_points=self.line_points,
            padding_ratio=self.padding_ratio
        )
                
        # Process the video
        streamer.run()

        # Save MOT results
        self.save_inference_results(
            streamer.engines[0].mot_results, 
            output_recognition_path, 
            output_tracking_path
        )

        # Evaluate recognition performance
        precision, recall, accuracy = self.evaluate_recognition(video_name)
        
        # Store results
        self.results[video_name] = {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy
        }

        self.logger.info(f"✅ Finished processing {video_name}")
    
    def save_results(self, output_path: Optional[str] = None) -> None:
        if output_path is None:
            output_path = os.path.join(self.results_dir, "algs", f"{self.alg_name}_results.csv")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.logger.info(f"💾 Saving results to {output_path}")

        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "video", "precision", "recall", "accuracy", 
                "DetA", "DetRe", "DetPr", "LocA"
            ])
            
            for video_name, res in self.results.items():
                writer.writerow([
                    video_name, 
                    res.get("precision", 0), 
                    res.get("recall", 0), 
                    res.get("accuracy", 0),
                    res.get("DetA", 0), 
                    res.get("DetRe", 0), 
                    res.get("DetPr", 0), 
                    res.get("LocA", 0)
                ])
            
        self.logger.info(f"✅ Results saved successfully to {output_path}.")
    
    def run_evaluation(self, video_names: List[str]) -> Dict:
        # Create video paths dictionary
        video_paths = {
            video: os.path.join(self.video_dir, f"{video}_eval.mp4")
            for video in video_names
        }
        
        # Process each video
        for video_name, video_path in video_paths.items():
            if not os.path.exists(video_path):
                self.logger.warning(f"⚠️ Video not found: {video_path} - skipping")
                continue
                
            self.process_video(video_name, video_path)
        
        # Evaluate tracking performance
        tracking_results = self.evaluate_mot()
        
        # Merge tracking results with recognition results
        for video_name, track_res in tracking_results.items():
            if video_name in self.results:
                self.results[video_name].update({
                    "DetA": track_res.get("DetA", 0),
                    "DetRe": track_res.get("DetRe", 0),
                    "DetPr": track_res.get("DetPr", 0),
                    "LocA": track_res.get("LocA", 0)
                })
        
        # Save final results
        self.save_results()
        
        return self.results
    
    def visualize_results(self) -> None:
        """Generate visualization of evaluation results."""
        if not self.results:
            self.logger.warning("⚠️ No results to visualize")
            return
            
        # Extract metrics
        videos = list(self.results.keys())
        precision = [self.results[v].get("precision", 0) for v in videos]
        recall = [self.results[v].get("recall", 0) for v in videos]
        accuracy = [self.results[v].get("accuracy", 0) for v in videos]
        
        # Create figure with subplots
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(videos))
        width = 0.25
        
        # Plot bars
        ax.bar(x - width, precision, width, label='Precision')
        ax.bar(x, recall, width, label='Recall')
        ax.bar(x + width, accuracy, width, label='Accuracy')
        
        # Add labels and title
        ax.set_xlabel('Videos')
        ax.set_ylabel('Score')
        ax.set_title(f'Recognition Performance - {self.alg_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(videos, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(self.results_dir, "visualizations", f"{self.alg_name}_results.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"📊 Results visualization saved to {output_path}")
