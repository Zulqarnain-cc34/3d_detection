#!/usr/bin/env python3
"""
SP1 3D Object Detection Pipeline - Complete Test Script

This script tests the full pipeline with various scenarios and produces
comprehensive visualizations and evaluation metrics.

Usage:
    python run_test.py                    # Run with default test image
    python run_test.py --image path.jpg   # Run with custom image
    python run_test.py --benchmark        # Run benchmark mode
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import SP1Pipeline, PipelineResult
from src.visualizer import PipelineVisualizer, quick_visualize
from src.evaluation import PipelineEvaluator, create_synthetic_ground_truth
from src.projector import CameraIntrinsics


def download_test_image(url: str, save_path: str) -> str:
    """Download test image from URL."""
    import requests
    from io import BytesIO
    
    print(f"Downloading test image from {url}...")
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image.save(save_path)
    print(f"Saved to: {save_path}")
    return save_path


def run_single_image_test(
    pipeline: SP1Pipeline,
    image_path: str,
    classes: List[str],
    output_dir: str
) -> PipelineResult:
    """Run pipeline on a single image and generate visualizations."""
    print("\n" + "=" * 60)
    print("RUNNING SINGLE IMAGE TEST")
    print("=" * 60)
    
    image = np.array(Image.open(image_path).convert('RGB'))
    print(f"Image loaded: {image_path}")
    print(f"Image shape: {image.shape}")
    print(f"Query classes: {classes}")
    
    print("\nRunning pipeline...")
    result = pipeline.detect(image, classes)
    print("\n" + result.summary())
    
    json_path = os.path.join(output_dir, "detection_results.json")
    result.save_json(json_path)
    print(f"\nResults saved to: {json_path}")
    
    print("\nGenerating visualization...")
    viz = PipelineVisualizer()
    fig = viz.visualize_pipeline_result(
        image=image,
        detections_2d=result.detections_2d,
        depth_result=result.depth_result,
        detections_3d=result.detections_3d,
        title=f"SP1 Pipeline Test - {len(result.detections_3d)} Objects Detected",
        save_path=os.path.join(output_dir, "pipeline_visualization.png")
    )
    plt.close(fig)
    
    try:
        fig_3d = viz.create_3d_scene_plot(
            result.detections_3d,
            save_path=os.path.join(output_dir, "3d_scene.png")
        )
        plt.close(fig_3d)
    except Exception as e:
        print(f"3D visualization skipped: {e}")
    
    return result


def run_benchmark(
    pipeline: SP1Pipeline,
    image_path: str,
    classes: List[str],
    num_runs: int = 10
) -> dict:
    """Run performance benchmark."""
    print("\n" + "=" * 60)
    print(f"RUNNING BENCHMARK ({num_runs} iterations)")
    print("=" * 60)
    
    image = np.array(Image.open(image_path).convert('RGB'))
    
    print("\nWarming up...")
    for _ in range(3):
        _ = pipeline.detect(image, classes)
    
    times = {'detection': [], 'depth': [], 'projection': [], 'total': []}
    
    print(f"\nRunning {num_runs} iterations...")
    for i in range(num_runs):
        result = pipeline.detect(image, classes)
        times['detection'].append(result.detection_time_ms)
        times['depth'].append(result.depth_time_ms)
        times['projection'].append(result.projection_time_ms)
        times['total'].append(result.total_time_ms)
        print(f"  Run {i+1}/{num_runs}: {result.total_time_ms:.1f}ms")
    
    stats = {}
    for key, values in times.items():
        stats[key] = {
            'mean_ms': np.mean(values),
            'std_ms': np.std(values),
            'min_ms': np.min(values),
            'max_ms': np.max(values)
        }
    stats['fps'] = 1000 / stats['total']['mean_ms']
    
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\nDetection: {stats['detection']['mean_ms']:.1f} ± {stats['detection']['std_ms']:.1f} ms")
    print(f"Depth:     {stats['depth']['mean_ms']:.1f} ± {stats['depth']['std_ms']:.1f} ms")
    print(f"Projection:{stats['projection']['mean_ms']:.1f} ± {stats['projection']['std_ms']:.1f} ms")
    print(f"\nTotal:     {stats['total']['mean_ms']:.1f} ± {stats['total']['std_ms']:.1f} ms")
    print(f"FPS:       {stats['fps']:.2f}")
    
    return stats


def run_multi_query_test(
    pipeline: SP1Pipeline,
    image_path: str,
    output_dir: str
):
    """Test with different query sets."""
    print("\n" + "=" * 60)
    print("RUNNING MULTI-QUERY TEST")
    print("=" * 60)
    
    image = np.array(Image.open(image_path).convert('RGB'))
    
    query_sets = [
        {"name": "furniture", "classes": ["chair", "sofa", "table", "desk", "bed"]},
        {"name": "electronics", "classes": ["tv", "laptop", "phone", "remote", "monitor"]},
        {"name": "household", "classes": ["lamp", "plant", "pillow", "book", "cup"]},
        {"name": "structure", "classes": ["door", "window", "wall", "floor", "ceiling"]},
    ]
    
    results_summary = []
    
    for query in query_sets:
        print(f"\nQuery: {query['name']} - {query['classes']}")
        result = pipeline.detect(image, query['classes'])
        
        detected = [d.class_name for d in result.detections_3d]
        print(f"  Detected: {detected if detected else 'None'}")
        print(f"  Time: {result.total_time_ms:.1f}ms")
        
        results_summary.append({
            "query_name": query['name'],
            "query_classes": query['classes'],
            "detected_objects": detected,
            "num_detections": len(result.detections_3d),
            "inference_time_ms": result.total_time_ms
        })
    
    summary_path = os.path.join(output_dir, "multi_query_results.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nMulti-query results saved to: {summary_path}")


def run_waypoint_test(
    pipeline: SP1Pipeline,
    image_path: str,
    target_objects: List[str]
):
    """Test navigation waypoint generation."""
    print("\n" + "=" * 60)
    print("RUNNING WAYPOINT GENERATION TEST")
    print("=" * 60)
    
    image = np.array(Image.open(image_path).convert('RGB'))
    
    for target in target_objects:
        print(f"\nTarget: '{target}'")
        waypoint = pipeline.get_waypoint(image, target, offset_distance=0.5)
        
        if waypoint:
            print(f"  Object found at: {waypoint['object_position']}")
            print(f"  Waypoint: {waypoint['waypoint_position']}")
            print(f"  Distance: {waypoint['distance_to_object']:.2f}m")
            print(f"  Confidence: {waypoint['confidence']:.2f}")
        else:
            print(f"  Object not found in scene")


def main():
    parser = argparse.ArgumentParser(description="SP1 3D Detection Pipeline Test")
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--classes', type=str, nargs='+', 
                        default=['chair', 'sofa', 'table', 'lamp', 'tv', 'door', 'window', 'plant', 'pillow'],
                        help='Classes to detect')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--benchmark-runs', type=int, default=10, help='Number of benchmark runs')
    parser.add_argument('--multi-query', action='store_true', help='Run multi-query test')
    parser.add_argument('--waypoint', action='store_true', help='Run waypoint test')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu, cuda:0)')
    parser.add_argument('--detector-model', type=str, default='yolov8s-world', help='Detector model')
    parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('./data', exist_ok=True)
    
    if args.image is None:
        image_url = "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=800"
        args.image = './data/test_room.jpg'
        if not os.path.exists(args.image):
            download_test_image(image_url, args.image)
    
    print("\n" + "=" * 60)
    print("SP1 3D OBJECT DETECTION PIPELINE TEST")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Device: {args.device}")
    print(f"  Detector: {args.detector_model}")
    print(f"  Confidence: {args.confidence}")
    print(f"  Test image: {args.image}")
    print(f"  Output dir: {args.output_dir}")
    
    print("\n" + "-" * 60)
    print("Initializing Pipeline...")
    print("-" * 60)
    
    pipeline = SP1Pipeline(
        detector_model=args.detector_model,
        device=args.device,
        confidence_threshold=args.confidence
    )
    
    result = run_single_image_test(pipeline, args.image, args.classes, args.output_dir)
    
    if args.benchmark:
        stats = run_benchmark(pipeline, args.image, args.classes, args.benchmark_runs)
        stats_path = os.path.join(args.output_dir, "benchmark_results.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=float)
        print(f"\nBenchmark results saved to: {stats_path}")
    
    if args.multi_query:
        run_multi_query_test(pipeline, args.image, args.output_dir)
    
    if args.waypoint:
        run_waypoint_test(pipeline, args.image, ['chair', 'sofa', 'table', 'door'])
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {args.output_dir}/")
    print("  - detection_results.json")
    print("  - pipeline_visualization.png")
    if args.benchmark:
        print("  - benchmark_results.json")
    if args.multi_query:
        print("  - multi_query_results.json")
    
    return result


if __name__ == "__main__":
    main()
