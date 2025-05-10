import os
import sys
sys.path.append(os.curdir)
from tests.evaluate_system import Evaluator
import argparse

parser = argparse.ArgumentParser(description='Evaluate face recognition system')
parser.add_argument('--alg_name', type=str, default='alg-11', help='Algorithm name')
parser.add_argument('--benchmark', type=str, default='ilhan', help='Benchmark dataset')
parser.add_argument('--videos', nargs='+', help='List of video names to process', 
                       default=["videoa1-1", "videoa1-2", "videoa1-3", "videoa1-4", "videoa1-5"])
parser.add_argument('--video_dir', default='data/tests/ilhan_videos', help='Directory containing videos')
parser.add_argument('--db_path', type=str, default='data/embeddings/ilhan.pkl', help='Path to face database')

parser.add_argument('--output', type=bool, default='results/', help='Output path for results')
parser.add_argument('--match_threshold', type=float, default=0.3, help='Face matching threshold')
parser.add_argument('--show', action='store_true', default=True, help='Show video stream')

args = parser.parse_args()

evaluator = Evaluator(
        alg_name=args.alg_name,
        benchmark=args.benchmark,
        video_dir=args.video_dir,
        db_path=args.db_path,
        match_threshold=args.match_threshold,
        show=False,
)

evaluator.run_evaluation(args.videos)

# Visualize results
evaluator.visualize_results()


# Example usage:
# python eval.py --alg_name alg-11 --benchmark ilhan --videos videoa1-1 videoa1-2 videoa1-3 videoa1-4 videoa1-5 --video_dir client/pred_videos --db_path data/hb-kor --match_threshold 0.6 --detection_threshold 0.5 --imgsz 1280 --padding_ratio 0.2 --show