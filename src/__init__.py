"""
SP1: 3D Open-Vocabulary Object Detection Pipeline

RGB-Only pipeline for semantic 3D object detection in indoor environments.
Designed for deployment on Jetson Orin 8GB.

Pipeline Stages:
    A. Open-Vocabulary 2D Detection (YOLO-World)
    B. Monocular Depth Estimation (Depth Anything V2)
    C. 3D Geometric Projection

Usage:
    from src import SP1Pipeline
    
    pipeline = SP1Pipeline(device='cuda:0')
    result = pipeline.detect('image.jpg', classes=['chair', 'table', 'cup'])
    print(result.summary())
"""

from src.pipeline import SP1Pipeline, PipelineResult
from src.detector import OpenVocabDetector, Detection2D
from src.depth_estimator import MonocularDepthEstimator, DepthResult
from src.projector import Projector3D, BoundingBox3D, CameraIntrinsics
from src.visualizer import PipelineVisualizer, quick_visualize
from src.evaluation import PipelineEvaluator, EvaluationResult

__version__ = "1.0.0"
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
    
    # Utilities
    "PipelineVisualizer",
    "quick_visualize",
    "PipelineEvaluator",
    "EvaluationResult",
]
