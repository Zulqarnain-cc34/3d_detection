"""
SP1: 3D Open-Vocabulary Object Detection Pipeline (FIXED)

RGB-Only pipeline for semantic 3D object detection in indoor environments.
Designed for deployment on Jetson Orin 8GB.

FIXES in this version:
- Proper metric depth handling (no incorrect normalization)
- depth_scale parameter for calibration
- calibrate_depth() method to find correct scale
- Better depth sampling (percentile instead of median)

Pipeline Stages:
    A. Open-Vocabulary 2D Detection (YOLO-World)
    B. Monocular Depth Estimation (Depth Anything V2 - Metric)
    C. 3D Geometric Projection

Usage:
    from src import SP1Pipeline
    
    # Initialize with depth_scale (adjust based on calibration)
    pipeline = SP1Pipeline(device='cuda:0', depth_scale=0.5)
    
    # Or calibrate first
    pipeline = SP1Pipeline(device='cuda:0')
    scale = pipeline.calibrate_depth(image, 'person', actual_distance=3.0)
    pipeline.set_depth_scale(scale)
    
    # Then detect
    result = pipeline.detect('image.jpg', classes=['chair', 'table', 'person'])
    print(result.summary())
"""

from src.pipeline import SP1Pipeline, PipelineResult
from src.detector import OpenVocabDetector, Detection2D
from src.depth_estimator import MonocularDepthEstimator, DepthResult
from src.projector import Projector3D, BoundingBox3D, CameraIntrinsics

__version__ = "1.1.0"
__author__ = "SP1 Development Team"

__all__ = [
    # Main pipeline
    "SP1Pipeline",
    "PipelineResult",
    
    # Stage A: Detection
    "OpenVocabDetector",
    "Detection2D",
    
    # Stage B: Depth
    "MonocularDepthEstimator", 
    "DepthResult",
    
    # Stage C: Projection
    "Projector3D",
    "BoundingBox3D",
    "CameraIntrinsics",
]
