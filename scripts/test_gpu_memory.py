#!/usr/bin/env python3
"""Test script to validate GPU memory management fixes.

This script monitors GPU memory usage over time to verify that
the memory leak fixes are working correctly.
"""

import os
import sys
import time
import csv
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    print("⚠️  PyTorch not available, GPU monitoring disabled")


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if not CUDA_AVAILABLE:
        return 0.0
    
    return torch.cuda.memory_allocated() / 1024**2


def log_gpu_stats(output_file, interval_seconds=10, duration_hours=1):
    """Log GPU memory stats to CSV file.
    
    Args:
        output_file: Path to output CSV file
        interval_seconds: Logging interval in seconds
        duration_hours: Test duration in hours
    """
    print(f"📊 Starting GPU memory monitoring")
    print(f"   Output: {output_file}")
    print(f"   Interval: {interval_seconds}s")
    print(f"   Duration: {duration_hours}h")
    print(f"   CUDA Available: {CUDA_AVAILABLE}")
    print()
    
    if not CUDA_AVAILABLE:
        print("❌ CUDA not available, cannot monitor GPU memory")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # CSV headers
    headers = ['timestamp', 'elapsed_seconds', 'allocated_mb', 'reserved_mb', 'growth_mb']
    
    start_time = time.time()
    end_time = start_time + (duration_hours * 3600)
    baseline_memory = get_gpu_memory_mb()
    
    print(f"📈 Baseline GPU memory: {baseline_memory:.1f}MB")
    print(f"🕐 Test will run until: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Monitoring... (Press Ctrl+C to stop)")
    print()
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        try:
            while time.time() < end_time:
                current_time = time.time()
                elapsed = current_time - start_time
                
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                growth = allocated - baseline_memory
                
                # Write to CSV
                writer.writerow({
                    'timestamp': datetime.now().isoformat(),
                    'elapsed_seconds': int(elapsed),
                    'allocated_mb': round(allocated, 2),
                    'reserved_mb': round(reserved, 2),
                    'growth_mb': round(growth, 2)
                })
                f.flush()
                
                # Print status
                print(f"[{int(elapsed):>6}s] "
                      f"Allocated: {allocated:>7.1f}MB | "
                      f"Reserved: {reserved:>7.1f}MB | "
                      f"Growth: {growth:>+7.1f}MB")
                
                # Warning if growth is high
                if growth > 100:  # More than 100MB growth
                    print(f"⚠️  WARNING: Significant memory growth detected: {growth:.1f}MB")
                
                time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
    
    # Final summary
    final_allocated = get_gpu_memory_mb()
    total_growth = final_allocated - baseline_memory
    growth_rate = total_growth / (elapsed / 3600)  # MB per hour
    
    print()
    print("="*60)
    print("📊 Test Summary")
    print("="*60)
    print(f"Duration: {elapsed/3600:.2f} hours")
    print(f"Baseline Memory: {baseline_memory:.1f}MB")
    print(f"Final Memory: {final_allocated:.1f}MB")
    print(f"Total Growth: {total_growth:+.1f}MB")
    print(f"Growth Rate: {growth_rate:+.2f}MB/hour")
    print()
    
    if abs(growth_rate) < 5:
        print("✅ PASS: Memory growth rate is acceptable (<5MB/hour)")
    elif abs(growth_rate) < 20:
        print("⚠️  WARNING: Memory growth rate is moderate (5-20MB/hour)")
    else:
        print("❌ FAIL: Memory growth rate is high (>20MB/hour)")
    
    print()
    print(f"Results saved to: {output_file}")


def analyze_results(csv_file):
    """Analyze GPU memory test results.
    
    Args:
        csv_file: Path to CSV file with test results
    """
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return
    
    print(f"📊 Analyzing results from: {csv_file}")
    print()
    
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'elapsed': int(row['elapsed_seconds']),
                'allocated': float(row['allocated_mb']),
                'growth': float(row['growth_mb'])
            })
    
    if not data:
        print("❌ No data found in CSV file")
        return
    
    # Calculate statistics
    duration_hours = data[-1]['elapsed'] / 3600
    initial_memory = data[0]['allocated']
    final_memory = data[-1]['allocated']
    total_growth = data[-1]['growth']
    growth_rate = total_growth / duration_hours if duration_hours > 0 else 0
    
    # Find peak
    peak_memory = max(d['allocated'] for d in data)
    peak_growth = max(d['growth'] for d in data)
    
    print(f"Duration: {duration_hours:.2f} hours")
    print(f"Initial Memory: {initial_memory:.1f}MB")
    print(f"Final Memory: {final_memory:.1f}MB")
    print(f"Peak Memory: {peak_memory:.1f}MB")
    print(f"Total Growth: {total_growth:+.1f}MB")
    print(f"Peak Growth: {peak_growth:+.1f}MB")
    print(f"Growth Rate: {growth_rate:+.2f}MB/hour")
    print()
    
    # Verdict
    if abs(growth_rate) < 5:
        print("✅ PASS: Memory is stable")
    elif abs(growth_rate) < 20:
        print("⚠️  WARNING: Moderate memory growth")
    else:
        print("❌ FAIL: Significant memory leak detected")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test GPU memory management')
    parser.add_argument('--output', '-o', default='logs/gpu_memory_test.csv',
                        help='Output CSV file')
    parser.add_argument('--interval', '-i', type=int, default=10,
                        help='Logging interval in seconds')
    parser.add_argument('--duration', '-d', type=float, default=1.0,
                        help='Test duration in hours')
    parser.add_argument('--analyze', '-a', type=str,
                        help='Analyze existing CSV file')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_results(args.analyze)
    else:
        log_gpu_stats(args.output, args.interval, args.duration)
