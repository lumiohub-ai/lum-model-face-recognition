import subprocess
import os
import sys
import pandas as pd


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

def eval_mot(benchmark, tracker_to_eval, split_to_eval="train"):
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate MOT challenge')
    parser.add_argument('--benchmark', type=str, help='Name of the benchmark')
    parser.add_argument('--alg_name', type=str, help='Name of the tracker to evaluate')
    parser.add_argument('--split_to_eval', type=str, default="train", help='Split to evaluate')
    args = parser.parse_args()

    eval_mot(args.benchmark, args.tracker_to_eval)